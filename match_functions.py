# Match decompiled code with their original code
import contextlib
import functools
import json
import os
import pickle
import re
import shutil
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, NamedTuple, Optional, Set, Tuple

import argtyped
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
    preprocess_timeout: Optional[int] = 120


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
    original_tokens: List[str]  # tokenized code
    decompiled_tokens: List[str]  # tokenized code
    original_ast_json: Dict[str, Any]
    decompiled_ast_json: Optional[Dict[str, Any]]


class Result(NamedTuple):
    repo_owner: str
    repo_name: str
    matched_functions: List[MatchedFunction]
    files_found: int
    functions_found: int
    funcs_without_asts: int


# Theoretically we also need `_fake_gcc_ext.h`, but the code probably has `#include`'s.
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


def exception_handler(e: Exception, repo_info: RepoInfo):
    ghcc.utils.log_exception(e, f"Exception occurred when processing {repo_info.repo_owner}/{repo_info.repo_name}",
                             force_console=True)


@ghcc.utils.exception_wrapper(exception_handler)
def match_functions(repo_info: RepoInfo, archive_folder: str, temp_folder: str, decompile_folder: str,
                    use_fake_libc_headers: bool = True, preprocess_timeout: Optional[int] = None) -> Result:
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

    ghcc.log(f"Begin processing {repo_full_name} ({total_files} files)")

    if os.path.exists(archive_path):
        # Extract archive
        ghcc.utils.run_command(["tar", f"xzf", str(archive_path)], cwd=str(repo_dir))
        (repo_dir / repo_folder_name).rename(repo_src_path)
    else:
        # Clone repo
        if repo_src_path.exists():
            shutil.rmtree(repo_src_path)
        ret = ghcc.clone(repo_info.repo_owner, repo_info.repo_name, clone_folder=str(repo_dir), folder_name="src")
        if ret.error_type not in [None, ghcc.CloneErrorType.SubmodulesFailed]:
            ghcc.log(f"Failed to clone {repo_full_name}: error type {ret.error_type}", "error")
            # Return a dummy result so this repo is ignored in the future.
            return Result(repo_info.repo_owner, repo_info.repo_name, [], 0, 0, 0)

    # Write makefile info to pickle
    with (repo_binary_dir / "makefiles.pkl").open("wb") as f_pkl:
        pickle.dump(repo_info.makefiles, f_pkl)

    gcc_flags = "-E"
    directory_mapping = None
    if use_fake_libc_headers:
        gcc_flags = f"-E -nostdlib -I/usr/src/libc"
        directory_mapping = {ghcc.parse.FAKE_LIBC_PATH: "/usr/src/libc"}

    makefiles = ghcc.docker_batch_compile(
        str(repo_binary_dir), str(repo_src_path), compile_timeout=preprocess_timeout,
        gcc_override_flags=gcc_flags, use_makefile_info_pkl=True, directory_mapping=directory_mapping,
        user_id=(repo_info.idx % 10000) + 30000,  # user IDs 30000 ~ 39999
        exception_log_fn=functools.partial(exception_handler, repo_info=repo_info))

    parser = CParser()
    generator = CGenerator()
    lexer = ghcc.parse.LexerWrapper()
    func_name_regex = re.compile(r'"function":\s*"([^"]+)"')
    line_control_regex = re.compile(r'^#[^\n]*$', flags=re.MULTILINE)
    decompile_path = Path(decompile_folder)
    extractor = ghcc.parse.FunctionExtractor()
    matched_functions = []
    files_found = 0
    functions_found = 0
    for makefile in makefiles:
        mkfile_dir = Path(makefile['directory'])
        for path, sha in zip(makefile["binaries"], makefile["sha256"]):
            # Load and parse preprocessed original code.
            code_path = mkfile_dir / path
            json_path = decompile_path / (sha + ".jsonl")
            if not json_path.exists():
                continue
            try:
                with (repo_binary_dir / sha).open("r") as f:
                    code = f.read()
            except UnicodeDecodeError:
                continue  # probably a real binary file
            try:
                ast: ASTNode = parser.parse(code, filename=os.path.join(repo_full_name, path))
            except pycparser.c_parser.ParseError as e:
                ghcc.log(f"{repo_full_name}: Parser error when processing file "
                         f"{str(code_path)} ({sha}): {str(e)}", "error")
                has_error = True
                continue  # ignore parsing errors
            files_found += 1
            function_asts = extractor.find_functions(ast)
            functions_found += len(function_asts)

            # Collect decompiled functions with matching original code.
            with json_path.open("r") as f:
                decompiled_json = [line for line in f if line]  # don't decode, as we only need the function name
            decompiled_funcs: Dict[str, str] = {}
            for line_num, j in enumerate(decompiled_json):
                match = func_name_regex.search(j)
                assert match is not None
                func_name = match.group(1)
                if func_name not in function_asts:
                    continue

                decompiled_data = json.loads(j)
                decompiled_code = decompiled_data["raw_code"]
                decompiled_code = re.sub(  # replace @@VAR with developer-assigned names
                    r"@@VAR_\d+@@[^@]+@@([_a-zA-Z0-9]+)", r"\1", decompiled_code)
                decompiled_code = re.sub(  # remove the register allocation indication in `var@<rdi>`
                    r"@<[a-z0-9]+>", "", decompiled_code)
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
            replaced_code = replacer.visit(ast)

            # Obtain AST for decompiled code by parsing it again.
            with tempfile.TemporaryDirectory() as temp_dir:
                code_to_preprocess = DECOMPILED_CODE_HEADER + "\n" + replaced_code
                input_path = os.path.join(temp_dir, "test.c")
                output_path = os.path.join(temp_dir, "test.prep.c")
                with open(input_path, "w") as f:
                    f.write(code_to_preprocess)
                compile_ret = ghcc.utils.run_command(
                    ["gcc", "-E", "-I" + ghcc.parse.FAKE_LIBC_PATH, "-o", output_path, input_path], ignore_errors=True)
                if compile_ret.return_code != 0:
                    msg = (f"{repo_full_name}: GCC return value nonzero for decompiled code of "
                           f"{str(code_path)} ({sha})")
                    if compile_ret.captured_output is not None:
                        msg += ":\n" + compile_ret.captured_output.decode("utf-8")
                    ghcc.log(msg, "error")
                    has_error = True
                    continue

                with open(output_path, "r") as f:
                    code_to_parse = f.read()
                # Remove line control macros so we know where errors occur
                code_to_parse = line_control_regex.sub("", code_to_parse)

            def generate_output(func_ast: ASTNode) -> Tuple[ghcc.parse.JSONNode, List[str]]:
                func_code = generator.visit(func_ast)
                # `line_start[lineno - 1]` stores the `lexpos` right before the beginning of line `lineno`.
                # So `tok.lexpos - line_start[tok.lineno - 1]` gives the column of the token.
                line_start = [-1] + [i for i, ch in enumerate(func_code) if ch == "\n"]
                func_tokens = []
                token_coords = []
                for tok in lexer.lex_tokens(func_code):
                    func_tokens.append(tok.value)
                    token_coords.append(ghcc.parse.TokenCoord(tok.lineno, tok.lexpos - line_start[tok.lineno - 1]))
                ast_json = ghcc.parse.ast_to_json(func_ast, token_coords)
                return ast_json, func_tokens

            try:
                decompiled_ast = ghcc.parse.parse_decompiled_code(code_to_parse, lexer, parser)
            except (ValueError, pycparser.c_parser.ParseError) as e:
                ghcc.log(f"{repo_full_name}: Could not parse decompiled code for "
                         f"{str(code_path)} ({sha}): {str(e)}", "error")
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
                        original_ast_json, original_func_tokens = generate_output(original_func_ast)
                        matched_func = MatchedFunction(
                            file_path=path, binary_hash=sha, func_name=func_name,
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
                    original_ast_json, original_func_tokens = generate_output(original_func_ast)
                    decompiled_ast_json, decompiled_func_tokens = generate_output(decompiled_func_ast)
                    matched_func = MatchedFunction(
                        file_path=path, binary_hash=sha, func_name=func_name,
                        original_tokens=original_func_tokens, decompiled_tokens=decompiled_func_tokens,
                        original_ast_json=original_ast_json, decompiled_ast_json=decompiled_ast_json)
                    matched_functions.append(matched_func)

    # Cleanup the folders; if errors occurred, keep the preprocessed code.
    status = ("success" if not has_error and len(matched_functions) > 0 else
              ("warning" if not has_error or len(matched_functions) > 0 else
               "error"))
    if status == "success":
        shutil.rmtree(repo_src_path)
    else:
        shutil.rmtree(repo_dir)

    end_time = time.time()
    funcs_without_asts = sum(matched_func.decompiled_ast_json is None for matched_func in matched_functions)
    ghcc.log(f"[{end_time - start_time:6.2f}s] "
             f"{repo_full_name}: "
             f"Files found: {files_found}/{total_files}, "
             f"functions matched: {len(matched_functions)}/{functions_found} "
             f"({funcs_without_asts} w/o ASTs)", status, force_console=True)
    return Result(repo_owner=repo_info.repo_owner, repo_name=repo_info.repo_name,
                  matched_functions=matched_functions,
                  files_found=files_found, functions_found=functions_found, funcs_without_asts=funcs_without_asts)


def _iter_repos(db_entries: Set[Tuple[str, str]], max_count: Optional[int] = None, skip_to: Optional[str] = None,
                cache_path: Optional[str] = None) -> Iterator[RepoInfo]:
    with contextlib.closing(ghcc.RepoDB()) as repo_db, contextlib.closing(ghcc.BinaryDB()) as binary_db:
        @ghcc.utils.cache(cache_path, name="repo binary info")
        def _get_repo_binaries_info() -> Dict[Tuple[str, str], Set[str]]:
            repo_binaries: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
            all_bins = binary_db.collection.find()
            for entry in tqdm.tqdm(all_bins, total=all_bins.count(), desc="Listing decompiled binaries"):
                if entry["success"]:
                    repo_binaries[entry["repo_owner"], entry["repo_name"]].add(entry["sha"])
            return repo_binaries

        repo_binaries = _get_repo_binaries_info()
        repo_entries: Iterator[ghcc.RepoDB.Entry] = repo_db.safe_iter(static=True)
        if skip_to is not None:
            skip_to_repo = tuple(skip_to.split("/"))
            repo_entries = ghcc.utils.drop_until(
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
               cache_path: Optional[str] = None) -> Tuple[Iterator[RepoInfo], DBStats]:
    repo_count = func_count = func_without_ast_count = 0
    db_entries: Set[Tuple[str, str]] = set()
    for entry in db.collection.find():
        db_entries.add((entry["repo_owner"], entry["repo_name"]))
        repo_count += 1
        func_count += entry["funcs_matched"]
        func_without_ast_count += entry["funcs_matched_without_ast"]

    iterator = _iter_repos(db_entries, max_count, skip_to, cache_path)
    stats = DBStats(repo_count, func_count, func_without_ast_count)
    return iterator, stats


def main() -> None:
    if not ghcc.utils.verify_docker_image(verbose=True):
        exit(1)

    args = Arguments()
    if args.pdb:
        ghcc.utils.register_ipython_excepthook()
        if args.n_procs == 0:
            globals()['match_functions'] = match_functions.__wrapped__

    if not args.verbose:
        ghcc.set_logging_level("quiet", console=True, file=False)
    ghcc.set_log_file(args.log_file)
    ghcc.log("Running with arguments:\n" + args.to_string(), force_console=True)

    if os.path.exists(args.temp_dir):
        ghcc.log(f"Removing contents of temporary folder '{args.temp_dir}'...", "warning", force_console=True)
        ghcc.utils.run_docker_command(["rm", "-rf", "/usr/src/*"], user=0,
                                      directory_mapping={args.temp_dir: "/usr/src"})

    db = ghcc.MatchFuncDB()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with ghcc.utils.safe_pool(args.n_procs, closing=[db]) as pool:
        iterator, stats = iter_repos(
            db, args.max_repos, skip_to=args.skip_to, cache_path=args.repo_binary_info_cache_path)
        match_fn: Callable[[RepoInfo], Result] = functools.partial(
            match_functions,
            archive_folder=args.archive_dir, temp_folder=args.temp_dir, decompile_folder=args.decompile_dir,
            use_fake_libc_headers=args.use_fake_libc_headers, preprocess_timeout=args.preprocess_timeout)

        repo_count = stats.repo_count
        func_count = stats.func_count
        func_without_ast_count = stats.func_without_ast_count
        for result in pool.imap_unordered(match_fn, iterator):
            if result is None:
                # Exception occurred.
                if args.exit_on_exception:
                    ghcc.log(f"Exception occurred, exiting because 'exit_on_exception' is True", "warning")
                    break
                continue

            # Write the matched functions to disk.
            with (output_dir / f"{result.repo_owner}_____{result.repo_name}.jsonl").open("w") as f:
                for matched_func in result.matched_functions:
                    f.write(json.dumps(matched_func._asdict(), separators=(',', ':')) + "\n")

            if args.write_db:
                db.add_repo(result.repo_owner, result.repo_name,
                            files_found=result.files_found, funcs_found=result.functions_found,
                            funcs_matched=len(result.matched_functions),
                            funcs_matched_without_ast=result.funcs_without_asts)

            repo_count += 1
            func_count += len(result.matched_functions)
            func_without_ast_count += result.funcs_without_asts
            if repo_count % 100 == 0:
                ghcc.log(f"Processed {repo_count} repositories, {func_count} functions matched "
                         f"({func_without_ast_count} w/o AST)", force_console=True)


if __name__ == '__main__':
    main()
