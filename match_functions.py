# Match decompiled code with their original code
import contextlib
import functools
import json
import os
import pickle
import random
import re
import shutil
import string
import time
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterator, List, NamedTuple, Optional, Set, Tuple

import argtyped
import flutes
import pycparser
import tqdm
from argtyped import Switch
from pycparser.c_ast import Node as ASTNode
from pycparser.c_generator import CGenerator
from pycparser.c_parser import CParser
# from pycparserext.ext_c_generator import GnuCGenerator as CGenerator
# from pycparserext.ext_c_parser import GnuCParser as CParser

import ghcc


class Arguments(argtyped.Arguments):
    archive_dir: Optional[str] = "archives/"  # directory containing repo archives
    decompile_dir: str = "decompile_output_fixed/"  # directory containing decompiled output (JSONL files)
    temp_dir: str = "repos/"
    log_file: str = "match-log.txt"
    output_dir: str = "match_output/"  # directory to save matching functions
    use_fake_libc_headers: Switch = True  # use pycparser fake libc headers to save time
    n_procs: int = 0  # number of processes
    max_repos: Optional[int] = None  # maximum number of repositories to process (ignoring non-existent)
    exit_on_exception: Switch = False
    pdb: Switch = False
    write_db: Switch = True
    skip_to: Optional[str]  # skip to specified repository
    repo_binary_info_cache_path: Optional[str]  # if specified, load/save repo_binaries instead of loading from DB
    verbose: Switch = False
    preprocess_timeout: Optional[int] = 600
    show_progress: Switch = False  # show a progress bar for each worker process; large overhead
    force_reprocess: Switch = False  # also process repos that are recorded as processed in DB


class RepoInfo(NamedTuple):
    idx: int
    repo_owner: str
    repo_name: str
    makefiles: Dict[str, Dict[str, str]]  # make_dir -> (binary_path -> binary_sha256)


class MatchedFunction(NamedTuple):
    # Source code & decompiler info
    file_path: str
    binary_hash: str
    # Code
    func_name: str
    variable_names: Dict[str, Tuple[str, str]]  # (var_id) -> (decompiled_var_name, original_var_name)
    original_tokens: List[str]  # tokenized code
    decompiled_tokens: List[str]  # tokenized code
    original_ast_json: ghcc.parse.JSONNode
    decompiled_ast_json: Optional[ghcc.parse.JSONNode]


class Result(NamedTuple):
    repo_owner: str
    repo_name: str
    matched_functions: List[MatchedFunction]
    preprocessed_original_code: Dict[str, str]  # (sha) -> code
    files_found: int
    functions_found: int
    funcs_without_asts: int


DECOMPILED_CODE_HEADER = r"""
#define __fastcall
#define __noreturn
#define __cdecl
#define __stdcall
#define __usercall

#define __int128 long long  // who cares
#define __int64 long long
#define __int32 int
#define __int16 short
#define __int8 char

// Actual types don't matter; we just need to know it's a type.
typedef int _QWORD;
typedef int _DWORD;
typedef int _WORD;
typedef int _BYTE;
typedef int _BOOL8;
typedef int _BOOL4;
typedef int WCHAR;
typedef int gcc_va_list;
typedef int (*__compar_fn_t)(const void *, const void *);
"""
with open(os.path.join(ghcc.parse.FAKE_LIBC_PATH, "_fake_gcc_ext.h")) as f:
    DECOMPILED_CODE_HEADER += f.read()


def exception_handler(e: Exception, repo_info: RepoInfo):
    flutes.log_exception(e, f"Exception occurred when processing {repo_info.repo_owner}/{repo_info.repo_name}",
                         force_console=True)


def find_matching_rbrace(tokens: List[ghcc.parse.Token], start: int) -> int:
    balance = 0
    for idx in range(start, len(tokens)):
        if tokens[idx].name == "{":
            balance += 1
        elif tokens[idx].name == "}":
            balance -= 1
            if balance == 0:
                return idx
    raise ValueError


def serialize(func_ast: ASTNode, tokens: List[ghcc.parse.Token]) -> Tuple[ghcc.parse.JSONNode, List[str]]:
    r"""Generate serialized AST and lexed tokens for a single function.

    :param func_ast:
    :param tokens:
    :return:
    """
    ast_dict = ghcc.parse.ast_to_dict(func_ast, tokens)
    # Instead of generating and lexing the code again, we find the function boundaries based on heuristics:
    # - Left boundary is given by smallest token position in tree.
    # - Right boundary is the matching right curly brace given the token position of the function body
    #   compound statement.
    inf = len(tokens)
    COORD = ghcc.parse.TOKEN_POS_ATTR
    find_min_pos_fn = lambda node, xs: min(min(xs, default=inf), node[COORD] or inf)
    left = ghcc.parse.visit_dict(find_min_pos_fn, ast_dict[ghcc.parse.CHILDREN_ATTR]["decl"])
    body_start = ghcc.parse.visit_dict(find_min_pos_fn, ast_dict[ghcc.parse.CHILDREN_ATTR]["body"])
    try:
        right = find_matching_rbrace(tokens, body_start)

        # Decrease all token positions by offset.
        def visit_fn(node: ghcc.parse.JSONNode, _) -> None:
            if node[COORD] is not None:
                node[COORD] -= left

        ghcc.parse.visit_dict(visit_fn, ast_dict)
        token_names = [tok.name for tok in tokens[left:(right + 1)]]
    except ValueError:
        # Fallback to the fail-safe method.
        token_names = ghcc.parse.LexerWrapper().lex(CGenerator().visit(func_ast))
    return ast_dict, token_names


# Match function names in raw JSON read from decompilation output.
# - Capture groups: (func_name)
JSON_FUNC_NAME_REGEX = re.compile(r'"function":\s*"([^"]+)"')
# Match variable references in decompiled code.
# - Capture groups: (var_id, decompiled_var_name, original_var_name)
DECOMPILED_VAR_REGEX = re.compile(r"@@VAR_(\d+)@@([^@]+)@@([_a-zA-Z0-9]+)")
# Match register allocation annotations in decompiled code.
DECOMPILED_REG_ALLOC_REGEX = re.compile(r"@<[a-z0-9]+>")
# Match preprocessor comments.
LINE_CONTROL_REGEX = re.compile(r'^#[^\n]*\n', flags=re.MULTILINE)  # also chomp the newline symbol

IDENTIFIER_CHARS = string.ascii_letters + string.digits + "_"


@flutes.exception_wrapper(exception_handler)
def match_functions(repo_info: RepoInfo, archive_folder: str, temp_folder: str, decompile_folder: str,
                    use_fake_libc_headers: bool = True, preprocess_timeout: Optional[int] = None,
                    *, progress_bar: Optional[flutes.ProgressBarManager.Proxy] = None) -> Result:
    # Directions:
    # 1. Clone or extract from archive.
    # 2. For each Makefile, rerun the compilation process with the flag "-E", so only the preprocessor is run.
    #    This probably won't take long as the compiler exits after running the processor, and linking would fail.
    #    Also, consider using "-nostdlib -Ipath/to/fake_libc_include" as suggested by `pycparser`.
    # 3. The .o files are now preprocessed C code. Parse them using `pycparser` to obtain a list of functions.

    start_time = time.time()
    total_files = sum(len(makefile) for makefile in repo_info.makefiles.values())
    repo_folder_name = f"{repo_info.repo_owner}_____{repo_info.repo_name}"
    repo_full_name = f"{repo_info.repo_owner}/{repo_info.repo_name}"
    archive_path = (Path(archive_folder) / f"{repo_full_name}.tar.gz").absolute()
    repo_dir = (Path(temp_folder) / repo_folder_name).absolute()
    repo_src_path = repo_dir / "src"
    repo_binary_dir = repo_dir / "bin"
    repo_binary_dir.mkdir(parents=True, exist_ok=True)
    has_error = False

    if progress_bar is not None:
        worker_id = flutes.get_worker_id()
        process_name = f"Worker {worker_id}" if worker_id is not None else "Main Process"
        progress_bar.new(total=total_files, desc=process_name + f" [{repo_full_name}]")

    flutes.log(f"Begin processing {repo_full_name} ({total_files} files)")

    if os.path.exists(archive_path):
        # Extract archive
        flutes.run_command(["tar", f"xzf", str(archive_path)], cwd=str(repo_dir))
        (repo_dir / repo_folder_name).rename(repo_src_path)
    else:
        # Clone repo
        if repo_src_path.exists():
            shutil.rmtree(repo_src_path)
        ret = ghcc.clone(repo_info.repo_owner, repo_info.repo_name, clone_folder=str(repo_dir), folder_name="src")
        if ret.error_type not in [None, ghcc.CloneErrorType.SubmodulesFailed]:
            flutes.log(f"Failed to clone {repo_full_name}: error type {ret.error_type}", "error")
            # Return a dummy result so this repo is ignored in the future.
            return Result(repo_info.repo_owner, repo_info.repo_name, [], {}, 0, 0, 0)

    # Write makefile info to pickle
    with (repo_binary_dir / "makefiles.pkl").open("wb") as f_pkl:
        pickle.dump(repo_info.makefiles, f_pkl)

    gcc_flags = "-E"
    directory_mapping = None
    if use_fake_libc_headers:
        gcc_flags = f"-E -nostdlib -I/usr/src/libc"
        directory_mapping = {ghcc.parse.FAKE_LIBC_PATH: "/usr/src/libc"}

    if progress_bar is not None:
        progress_bar.update(postfix={"status": "preprocessing"})
    makefiles = ghcc.docker_batch_compile(
        str(repo_binary_dir), str(repo_src_path), compile_timeout=preprocess_timeout,
        gcc_override_flags=gcc_flags, use_makefile_info_pkl=True, directory_mapping=directory_mapping,
        user_id=(repo_info.idx % 10000) + 30000,  # user IDs 30000 ~ 39999
        exception_log_fn=functools.partial(exception_handler, repo_info=repo_info))

    parser = CParser(lexer=ghcc.parse.CachedCLexer)
    lexer = ghcc.parse.LexerWrapper()
    decompile_path = Path(decompile_folder)
    extractor = ghcc.parse.FunctionExtractor()
    matched_functions: List[MatchedFunction] = []
    preprocessed_original_code: Dict[str, str] = {}
    files_found = 0
    functions_found = 0
    for makefile in makefiles:
        mkfile_dir = Path(makefile['directory'])
        for path, sha in zip(makefile["binaries"], makefile["sha256"]):
            # Load and parse preprocessed original code.
            code_path = str(mkfile_dir / path)
            json_path = decompile_path / (sha + ".jsonl")
            preprocessed_code_path = repo_binary_dir / sha
            if progress_bar is not None:
                progress_bar.update(1, postfix={"file": code_path})
            if not json_path.exists() or not preprocessed_code_path.exists():
                continue
            try:
                with preprocessed_code_path.open("r") as f:
                    code = f.read()
                code = LINE_CONTROL_REGEX.sub("", code)
            except UnicodeDecodeError:
                continue  # probably a real binary file
            preprocessed_original_code[sha] = code
            try:
                original_ast: ASTNode = parser.parse(code, filename=os.path.join(repo_full_name, path))
            except (pycparser.c_parser.ParseError, AssertionError) as e:
                # For some reason `pycparser` uses `assert`s in places where there should have been a check.
                flutes.log(f"{repo_full_name}: Parser error when processing file "
                           f"{code_path} ({sha}): {str(e)}", "error")
                has_error = True
                continue  # ignore parsing errors
            original_tokens = ghcc.parse.convert_to_tokens(code, parser.clex.cached_tokens)
            files_found += 1
            function_asts = extractor.find_functions(original_ast)
            functions_found += len(function_asts)

            # Collect decompiled functions with matching original code.
            with json_path.open("r") as f:
                decompiled_json = [line for line in f if line]  # don't decode, as we only need the function name
            decompiled_funcs: Dict[str, str] = {}  # (func_name) -> decompiled_code
            decompiled_var_names: Dict[str, Dict[str, Tuple[str, str]]] = {} \
                # (func_name) -> (var_id) -> (decomp_name, orig_name)
            for line_num, j in enumerate(decompiled_json):
                # Find function name from JSON line without parsing.
                match = JSON_FUNC_NAME_REGEX.search(j)
                assert match is not None
                func_name = match.group(1)
                if func_name not in function_asts:
                    continue

                try:
                    decompiled_data = json.loads(j)
                except json.JSONDecodeError as e:
                    flutes.log(f"{repo_full_name}: Decode error when reading JSON file at {json_path}: "
                               f"{str(e)}", "error")
                    continue
                decompiled_code = decompiled_data["raw_code"]
                # Store the variable names used in the function.
                # We use a random string as the identifier prefix. Sadly, C89 (and `pycparser`) doesn't support Unicode.
                for length in range(3, 10 + 1):
                    var_identifier_prefix = "v" + "".join(random.choices(string.ascii_lowercase, k=length))
                    if var_identifier_prefix not in decompiled_code:
                        break
                else:
                    # No way this is happening, right?
                    flutes.log(f"{repo_full_name}: Could not find valid identifier prefix for "
                               f"{func_name} in {code_path} ({sha})", "error")
                    continue
                variables: Dict[str, Tuple[str, str]] = {}  # (var_id) -> (decompiled_name, original_name)
                for match in DECOMPILED_VAR_REGEX.finditer(decompiled_code):
                    var_id, decompiled_name, original_name = match.groups()
                    var_id = f"{var_identifier_prefix}_{var_id}"
                    if var_id in variables:
                        assert variables[var_id] == (decompiled_name, original_name)
                    else:
                        variables[var_id] = (decompiled_name, original_name)
                decompiled_var_names[func_name] = variables
                # Remove irregularities in decompiled code to make the it parsable:
                # - Replace `@@VAR` with special identifiers (literally anything identifier that doesn't clash).
                # - Remove the register allocation indication in `var@<rdi>`.
                decompiled_code = DECOMPILED_VAR_REGEX.sub(rf"{var_identifier_prefix}_\1", decompiled_code)
                decompiled_code = DECOMPILED_REG_ALLOC_REGEX.sub("", decompiled_code)
                if func_name.startswith("_"):
                    # For some reason, Hexrays would chomp off one leading underscore from function names in their
                    # generated code, which might lead to corrupt code (`_01inverse` -> `01inverse`). Here we
                    # heuristically try to find and replace the changed function name.
                    decompiled_code = re.sub(  # replace all identifiers with matching name
                        r"(?<![a-zA-Z0-9_])" + func_name[1:] + r"(?![a-zA-Z0-9_])", func_name, decompiled_code)
                    # Note that this doesn't fix references of the function in other functions. But really, why would
                    # someone name their function `_01inverse`?
                decompiled_funcs[func_name] = decompiled_code

            # Generate code replacing original functions with decompiled functions.
            replacer = ghcc.parse.FunctionReplacer(decompiled_funcs)
            replaced_code = replacer.visit(original_ast)

            # Obtain AST for decompiled code by parsing it again.
            code_to_preprocess = DECOMPILED_CODE_HEADER + "\n" + replaced_code
            try:
                code_to_parse = ghcc.parse.preprocess(code_to_preprocess)
            except ghcc.parse.PreprocessError as e:
                msg = (f"{repo_full_name}: GCC return value nonzero for decompiled code of "
                       f"{code_path} ({sha})")
                if len(e.args) > 0:
                    msg += ":\n" + str(e)
                flutes.log(msg, "error")
                has_error = True
                continue

            try:
                decompiled_ast, code_to_parse = ghcc.parse.parse_decompiled_code(code_to_parse, lexer, parser)
                decompiled_tokens = ghcc.parse.convert_to_tokens(code_to_parse, parser.clex.cached_tokens)
            except (ValueError, pycparser.c_parser.ParseError) as e:
                flutes.log(f"{repo_full_name}: Could not parse decompiled code for "
                           f"{code_path} ({sha}): {str(e)}", "error")
                has_error = True

                # We don't have ASTs for decompiled functions, but we can still dump the code.
                # Use the dummy typedefs to extract functions.
                code_lines = code_to_parse.split("\n")
                func_begin_end: Dict[str, List[Optional[int]]] = defaultdict(lambda: [None, None])
                for idx, line in enumerate(code_lines):
                    name, is_begin = replacer.extract_func_name(line)
                    if name is not None:
                        func_begin_end[name][0 if is_begin else 1] = idx
                for func_name, (begin, end) in func_begin_end.items():
                    if begin is not None and end is not None and func_name in function_asts:
                        decompiled_func_tokens = lexer.lex("\n".join(code_lines[(begin + 1):end]))
                        original_func_ast = function_asts[func_name]
                        original_ast_json, original_func_tokens = serialize(original_func_ast, original_tokens)
                        matched_func = MatchedFunction(
                            file_path=code_path, binary_hash=sha, func_name=func_name,
                            variable_names=decompiled_var_names[func_name],
                            original_tokens=original_func_tokens, decompiled_tokens=decompiled_func_tokens,
                            original_ast_json=original_ast_json, decompiled_ast_json=None)
                        matched_functions.append(matched_func)

            else:
                # We've successfully parsed decompiled code.
                decompiled_func_asts = extractor.find_functions(decompiled_ast)
                for func_name in decompiled_funcs.keys():
                    original_func_ast = function_asts[func_name]
                    if func_name not in decompiled_func_asts:
                        # Maybe there's other Hexrays-renamed functions that we didn't fix, just ignore them.
                        continue
                    decompiled_func_ast = decompiled_func_asts[func_name]
                    original_ast_json, original_func_tokens = serialize(original_func_ast, original_tokens)
                    decompiled_ast_json, decompiled_func_tokens = serialize(decompiled_func_ast, decompiled_tokens)
                    matched_func = MatchedFunction(
                        file_path=code_path, binary_hash=sha, func_name=func_name,
                        variable_names=decompiled_var_names[func_name],
                        original_tokens=original_func_tokens, decompiled_tokens=decompiled_func_tokens,
                        original_ast_json=original_ast_json, decompiled_ast_json=decompiled_ast_json)
                    matched_functions.append(matched_func)

    # Cleanup the folders; if errors occurred, keep the preprocessed code.
    status = ("success" if not has_error and len(matched_functions) > 0 else
              ("warning" if not has_error or len(matched_functions) > 0 else
               "error"))
    shutil.rmtree(repo_dir)

    end_time = time.time()
    funcs_without_asts = sum(matched_func.decompiled_ast_json is None for matched_func in matched_functions)
    flutes.log(f"[{end_time - start_time:6.2f}s] "
               f"{repo_full_name}: "
               f"Files found: {files_found}/{total_files}, "
               f"functions matched: {len(matched_functions)}/{functions_found} "
               f"({funcs_without_asts} w/o ASTs)", status, force_console=True)
    return Result(repo_owner=repo_info.repo_owner, repo_name=repo_info.repo_name,
                  matched_functions=matched_functions, preprocessed_original_code=preprocessed_original_code,
                  files_found=files_found, functions_found=functions_found, funcs_without_asts=funcs_without_asts)


def _iter_repos(db_entries: Set[Tuple[str, str]], max_count: Optional[int] = None, skip_to: Optional[str] = None,
                cache_path: Optional[str] = None) -> Iterator[RepoInfo]:
    with contextlib.closing(ghcc.RepoDB()) as repo_db, contextlib.closing(ghcc.BinaryDB()) as binary_db:
        @flutes.cache(cache_path, name="repo binary info")
        def _get_repo_binaries_info() -> Dict[Tuple[str, str], Set[str]]:
            repo_binaries: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
            all_bins = binary_db.collection.find()
            for entry in tqdm.tqdm(all_bins, total=all_bins.count(), desc="Listing decompiled binaries"):
                if entry["success"]:
                    repo_binaries[entry["repo_owner"], entry["repo_name"]].add(entry["sha"])
            return repo_binaries

        repo_binaries = _get_repo_binaries_info()
        repo_entries: Iterator[ghcc.RepoDB.Entry] = repo_db.safe_iter(batch_size=10000, static=True)
        if skip_to is not None:
            skip_to_repo = tuple(skip_to.split("/"))
            repo_entries = flutes.drop_until(
                lambda entry: (entry["repo_owner"], entry["repo_name"]) == skip_to_repo, repo_entries)
        index = 0
        for entry in repo_entries:
            if (not entry['clone_successful'] or not entry['compiled'] or
                    len(entry['makefiles']) == 0 or entry['num_binaries'] == 0):
                # Nothing to match.
                continue
            if (entry["repo_owner"], entry["repo_name"]) in db_entries:
                # Already processed.
                continue

            binaries = repo_binaries[entry["repo_owner"], entry["repo_name"]]
            makefiles: Dict[str, Dict[str, str]] = {}
            for makefile in entry['makefiles']:
                mapping = {}
                for path, sha in zip(makefile['binaries'], makefile['sha256']):
                    if sha in binaries:
                        mapping[path] = sha
                        binaries.remove(sha)  # the same binary may be generated in multiple Makefiles
                if len(mapping) > 0:
                    makefiles[makefile['directory']] = mapping
            if len(makefiles) == 0:
                continue

            yield RepoInfo(index, entry['repo_owner'], entry['repo_name'], makefiles)
            index += 1
            if max_count is not None and index >= max_count:
                break


class DBStats(NamedTuple):
    repo_count: int
    func_count: int
    func_without_ast_count: int


def iter_repos(db: ghcc.MatchFuncDB, max_count: Optional[int] = None, skip_to: Optional[str] = None,
               cache_path: Optional[str] = None, force_reprocess: bool = False) -> Tuple[Iterator[RepoInfo], DBStats]:
    repo_count = func_count = func_without_ast_count = 0
    processed_db_entries: Set[Tuple[str, str]] = set()
    for entry in db.collection.find():
        if not force_reprocess and entry['funcs_matched_without_ast'] == 0:
            processed_db_entries.add((entry["repo_owner"], entry["repo_name"]))
        repo_count += 1
        func_count += entry["funcs_matched"]
        func_without_ast_count += entry["funcs_matched_without_ast"]

    iterator = _iter_repos(processed_db_entries, max_count, skip_to, cache_path)
    stats = DBStats(repo_count, func_count, func_without_ast_count)
    return iterator, stats


def main() -> None:
    if not ghcc.utils.verify_docker_image(verbose=True):
        exit(1)

    sys.setrecursionlimit(10000)
    args = Arguments()
    if args.pdb:
        flutes.register_ipython_excepthook()
        if args.n_procs == 0:
            globals()['match_functions'] = match_functions.__wrapped__

    if not args.verbose:
        flutes.set_logging_level("quiet", console=True, file=False)
    flutes.set_log_file(args.log_file)
    flutes.log("Running with arguments:\n" + args.to_string(), force_console=True)

    if os.path.exists(args.temp_dir):
        flutes.log(f"Removing contents of temporary folder '{args.temp_dir}'...", "warning", force_console=True)
        ghcc.utils.run_docker_command(["rm", "-rf", "/usr/src/*"], user=0,
                                      directory_mapping={args.temp_dir: "/usr/src"})

    db = ghcc.MatchFuncDB()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manager = flutes.ProgressBarManager(
        verbose=args.show_progress, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}{postfix}]")
    with flutes.safe_pool(args.n_procs, closing=[db, manager]) as pool:
        iterator, stats = iter_repos(
            db, args.max_repos, skip_to=args.skip_to, cache_path=args.repo_binary_info_cache_path,
            force_reprocess=args.force_reprocess)
        match_fn: Callable[[RepoInfo], Result] = functools.partial(
            match_functions,
            archive_folder=args.archive_dir, temp_folder=args.temp_dir, decompile_folder=args.decompile_dir,
            use_fake_libc_headers=args.use_fake_libc_headers, preprocess_timeout=args.preprocess_timeout,
            progress_bar=manager.proxy)

        repo_count = stats.repo_count
        func_count = stats.func_count
        func_without_ast_count = stats.func_without_ast_count
        for result in pool.imap_unordered(match_fn, iterator):
            if result is None:
                # Exception occurred.
                if args.exit_on_exception:
                    flutes.log(f"Exception occurred, exiting because 'exit_on_exception' is True", "warning")
                    break
                continue

            # Write the matched functions to disk.
            result: Result  # type: ignore
            repo_dir = output_dir / result.repo_owner / result.repo_name
            repo_dir.mkdir(parents=True, exist_ok=True)
            with (repo_dir / "matched_funcs.jsonl").open("w") as f:
                for matched_func in result.matched_functions:
                    f.write(json.dumps(matched_func._asdict(), separators=(',', ':')) + "\n")
            for sha, code in result.preprocessed_original_code.items():
                with (repo_dir / f"{sha}.c").open("w") as f:
                    pos = code.rfind(ghcc.parse.FAKE_LIBC_END_LINE)
                    if pos != -1:
                        code = code[(pos + len(ghcc.parse.FAKE_LIBC_END_LINE)):]
                    f.write(code)

            if args.write_db:
                db.add_repo(result.repo_owner, result.repo_name,
                            files_found=result.files_found, funcs_found=result.functions_found,
                            funcs_matched=len(result.matched_functions),
                            funcs_matched_without_ast=result.funcs_without_asts)

            repo_count += 1
            func_count += len(result.matched_functions)
            func_without_ast_count += result.funcs_without_asts
            if repo_count % 100 == 0:
                flutes.log(f"Processed {repo_count} repositories, {func_count} functions matched "
                           f"({func_without_ast_count} w/o AST)", force_console=True)


if __name__ == '__main__':
    main()
