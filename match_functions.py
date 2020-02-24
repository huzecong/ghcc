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
from pycparser.c_lexer import CLexer
from pycparser.c_parser import CParser
from pycparser.c_generator import CGenerator
from pycparser.c_ast import Node as ASTNode
# from pycparserext.ext_c_parser import GnuCParser as CParser
# from pycparserext.ext_c_generator import GnuCGenerator as CGenerator

import ghcc
from main import exception_handler

FAKE_LIBC_PATH = str((Path(ghcc.__file__).parent.parent / "scripts" / "fake_libc_include").absolute())


class LexToken:  # stub
    type: str
    value: str
    lineno: int
    lexpos: int


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
    repo_owner: str
    repo_name: str
    makefiles: Dict[str, Dict[str, str]]  # make_dir -> (binary_path -> binary_sha256)


class FunctionExtractor:
    r"""An AST visitor that extracts all function definitions from an AST."""

    class FuncDefVisitor(pycparser.c_ast.NodeVisitor):
        func_def_asts: Dict[str, ASTNode]

        def visit_FuncDef(self, node: pycparser.c_ast.FuncDef):
            func_name = node.decl.name
            self.func_def_asts[func_name] = node

    def __init__(self):
        self._visitor = self.FuncDefVisitor()

    def find_functions(self, node: ASTNode) -> Dict[str, ASTNode]:
        r"""Find all function definitions given an AST.

        :param node: The ``pycparser`` AST node object.
        :return: A dictionary mapping function names to function definition AST subtrees.
        """
        ret = self._visitor.func_def_asts = {}
        self._visitor.visit(node)
        del self._visitor.func_def_asts
        return ret


class FunctionReplacer(CGenerator):
    r"""An AST visitor inherited from ``pycparser.CGenerator``, but replaces FuncDef nodes with other function code."""

    BOUNDARY_PREFIX = "typedef int __func__"
    BEGIN_SUFFIX = "__begin;"
    END_SUFFIX = "__end;"

    def __init__(self, func_defs: Dict[str, str]):
        super().__init__()
        self._func_defs = func_defs

    def visit_FuncDef(self, node: pycparser.c_ast.FuncDef):
        func_name = node.decl.name
        if func_name in self._func_defs:
            # Add dummy typedefs around the function code, so we can still extract raw code even if we can't parse it.
            func_begin = self.BOUNDARY_PREFIX + func_name + self.BEGIN_SUFFIX
            func_end = self.BOUNDARY_PREFIX + func_name + self.END_SUFFIX
            return "\n".join(["", func_begin, self._func_defs[func_name], func_end, ""])
        return super().visit_FuncDef(node)

    def extract_func_name(self, line: str) -> Tuple[Optional[str], bool]:
        r"""Extracts the function name from a function boundary marker.

        :param line:
        :return: A tuple containing the function name and a ``bool`` indicating whether the marker indicates the
            beginning of a function. If the line is not a boundary marker, or the function name is not recognized,
            ``None`` is returned instead of the extracted name.
        """
        func_name = None
        is_begin = False
        if line.startswith(self.BOUNDARY_PREFIX):
            if line.endswith(self.BEGIN_SUFFIX):
                func_name = line[len(self.BOUNDARY_PREFIX):-len(self.BEGIN_SUFFIX)]
                is_begin = True
            if line.endswith(self.END_SUFFIX):
                func_name = line[len(self.BOUNDARY_PREFIX):-len(self.END_SUFFIX)]
        if func_name is not None and func_name in self._func_defs:
            return func_name, is_begin
        return None, False


class Lexer:
    @staticmethod
    def _error_func(msg, loc0, loc1):
        pass

    @staticmethod
    def _brace_func():
        pass

    @staticmethod
    def _type_lookup_func(typ):
        return False

    def __init__(self):
        self.lexer = CLexer(self._error_func, self._brace_func, self._brace_func, self._type_lookup_func)
        self.lexer.build(optimize=True, lextab='pycparser.lextab')

    def lex_tokens(self, code: str) -> List[LexToken]:
        self.lexer.input(code)
        tokens = []
        while True:
            token = self.lexer.token()
            if token is None:
                break
            tokens.append(token)
        return tokens

    def lex(self, code: str) -> List[str]:
        return [token.value for token in self.lex_tokens(code)]


class MatchedFunction(NamedTuple):
    # Source code & decompiler info
    file_path: str
    binary_hash: str
    # Code
    func_name: str
    original_code: List[str]  # tokenized code
    decompiled_code: List[str]  # tokenized code
    original_ast_json: Dict[str, Any]
    decompiled_ast_json: Optional[Dict[str, Any]]


class Result(NamedTuple):
    repo_owner: str
    repo_name: str
    matched_functions: List[MatchedFunction]
    files_found: int
    functions_found: int
    funcs_without_asts: int


RE_CHILD_ARRAY = re.compile(r'(.*)\[(.*)\]')
RE_INTERNAL_ATTR = re.compile('__.*__')


@functools.lru_cache()
def child_attrs_of(klass):
    r"""Given a Node class, get a set of child attrs.
    Memoized to avoid highly repetitive string manipulation
    """
    non_child_attrs = set(klass.attr_names)
    all_attrs = set([i for i in klass.__slots__ if not RE_INTERNAL_ATTR.match(i)])
    all_attrs -= {"coord"}
    return all_attrs - non_child_attrs


def ast_to_json(node: ASTNode) -> Dict[str, Any]:
    r"""Recursively convert an ast into dict representation.

    Adapted from ``pycparser`` example ``c_json.py``.
    """
    klass = node.__class__

    result = {}

    # Metadata
    result['_nodetype'] = klass.__name__

    # Local node attributes
    for attr in klass.attr_names:
        result[attr] = getattr(node, attr)

    # Child attributes
    for child_name, child in node.children():
        # Child strings are either simple (e.g. 'value') or arrays (e.g. 'block_items[1]')
        match = RE_CHILD_ARRAY.match(child_name)
        if match:
            array_name, array_index = match.groups()
            array_index = int(array_index)
            # arrays come in order, so we verify and append.
            result[array_name] = result.get(array_name, [])
            if array_index != len(result[array_name]):
                raise ValueError(f"Internal ast error. Array {array_name} out of order. "
                                 f"Expected index {len(result[array_name])}, got {array_index}")
            result[array_name].append(ast_to_json(child))
        else:
            result[child_name] = ast_to_json(child)

    # Any child attributes that were missing need "None" values in the json.
    for child_attr in child_attrs_of(klass):
        if child_attr not in result:
            result[child_attr] = None

    return result


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
        ret = ghcc.clone(repo_info.repo_owner, repo_info.repo_name, clone_folder=str(repo_dir), folder_name="src")
        assert ret.error_type in [None, ghcc.CloneErrorType.SubmodulesFailed]

    # Write makefile info to pickle
    with (repo_binary_dir / "makefiles.pkl").open("wb") as f_pkl:
        pickle.dump(repo_info.makefiles, f_pkl)

    gcc_flags = "-E"
    directory_mapping = None
    if use_fake_libc_headers:
        gcc_flags = f"-E -nostdlib -I/usr/src/libc"
        directory_mapping = {FAKE_LIBC_PATH: "/usr/src/libc"}

    makefiles = ghcc.docker_batch_compile(
        str(repo_binary_dir), str(repo_src_path), compile_timeout=preprocess_timeout,
        gcc_override_flags=gcc_flags, use_makefile_info_pkl=True, directory_mapping=directory_mapping,
        exception_log_fn=functools.partial(exception_handler, repo_info=repo_info))

    parser = CParser()
    generator = CGenerator()
    lexer = Lexer()
    func_name_regex = re.compile(r'"function":\s*"([^"]+)"')
    line_control_regex = re.compile(r'^#[^\n]*$', flags=re.MULTILINE)
    decompile_path = Path(decompile_folder)
    extractor = FunctionExtractor()
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
            replacer = FunctionReplacer(decompiled_funcs)
            replaced_code = replacer.visit(ast)

            # Obtain AST for decompiled code by parsing it again.
            with tempfile.TemporaryDirectory() as temp_dir:
                code_to_preprocess = DECOMPILED_CODE_HEADER + "\n" + replaced_code
                input_path = os.path.join(temp_dir, "test.c")
                output_path = os.path.join(temp_dir, "test.prep.c")
                with open(input_path, "w") as f:
                    f.write(code_to_preprocess)
                compile_ret = ghcc.utils.run_command(
                    ["gcc", "-E", "-I" + FAKE_LIBC_PATH, "-o", output_path, input_path], ignore_errors=True)
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

            try:
                decompiled_ast = parse_decompiled_code(lexer, parser, code_to_parse)
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
                        decompiled_func_code = lexer.lex("\n".join(code_lines[(begin + 1):end]))
                        original_func_ast = function_asts[func_name]
                        original_func_code = lexer.lex(generator.visit(original_func_ast))
                        original_ast_json = ast_to_json(original_func_ast)
                        matched_func = MatchedFunction(
                            file_path=path, binary_hash=sha, func_name=func_name,
                            original_code=original_func_code, decompiled_code=decompiled_func_code,
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
                    original_func_code = lexer.lex(generator.visit(original_func_ast))
                    decompiled_func_code = lexer.lex(generator.visit(decompiled_func_ast))
                    original_ast_json = ast_to_json(original_func_ast)
                    decompiled_ast_json = ast_to_json(decompiled_func_ast)
                    matched_func = MatchedFunction(
                        file_path=path, binary_hash=sha, func_name=func_name,
                        original_code=original_func_code, decompiled_code=decompiled_func_code,
                        original_ast_json=original_ast_json, decompiled_ast_json=decompiled_ast_json)
                    matched_functions.append(matched_func)
    # Cleanup the folders; if errors occurred, keep the preprocessed code.
    if has_error:
        shutil.rmtree(repo_src_path)
    else:
        shutil.rmtree(repo_dir)


    end_time = time.time()
    funcs_without_asts = sum(matched_func.decompiled_ast_json is None for matched_func in matched_functions)
    status = "success" if not has_error else ("warning" if len(matched_functions) > 0 else "error")
    ghcc.log(f"[{end_time - start_time:6.2f}s] "
             f"{repo_full_name}: "
             f"Files found: {files_found}/{total_files}, "
             f"functions matched: {len(matched_functions)}/{functions_found} "
             f"({funcs_without_asts} w/o ASTs)", status, force_console=True)
    return Result(repo_owner=repo_info.repo_owner, repo_name=repo_info.repo_name,
                  matched_functions=matched_functions,
                  files_found=files_found, functions_found=functions_found, funcs_without_asts=funcs_without_asts)


MAX_TYPE_FIX_RETRIES = 10
PARSE_ERROR_REGEX = re.compile(r'.*?:(?P<line>\d+):(?P<col>\d+): (?P<msg>.+)')


def parse_decompiled_code(lexer: Lexer, parser: CParser, code_to_parse: str) -> ASTNode:
    added_types: Set[str] = set()
    for _ in range(MAX_TYPE_FIX_RETRIES):
        try:
            decompiled_ast = parser.parse(code_to_parse)
            break
        except pycparser.c_parser.ParseError as e:
            error_match = PARSE_ERROR_REGEX.match(str(e))
            if error_match is None or not error_match.group("msg").startswith("before: "):
                raise
            before_token = ghcc.utils.remove_prefix(error_match.group("msg"), "before: ")
            error_line = code_to_parse.split("\n")[int(error_match.group("line")) - 1]
            error_pos = int(error_match.group("col")) - 1
            tokens = lexer.lex_tokens(error_line)
            try:
                error_token_idx = next(
                    idx for idx, token in enumerate(tokens)
                    if token.lexpos == error_pos and token.value == before_token)
                # There are multiple possible cases here:
                # 1. The type is the first ID-type token before the reported token (`type token`). It might not
                #    be the one immediately in front (for example, `(type) token`， `type *token`).
                # 2. The type is the token itself. This is rare and only happens in a situation like:
                #      `int func(const token var)`
                #    Replacing `const` with any combination of type qualifiers also works.
                if (error_token_idx > 0 and
                        tokens[error_token_idx - 1].type in ["CONST", "VOLATILE", "RESTRICT",
                                                             "__CONST", "__RESTRICT", "__EXTENSION__"]):
                    type_token = tokens[error_token_idx]
                else:
                    type_token = next(
                        tokens[idx] for idx in range(error_token_idx - 1, -1, -1)
                        if tokens[idx].type == "ID")
            except StopIteration:
                # If we don't catch this, it would terminate the for-loop in `main()`. Stupid design.
                raise e from None

            if type_token.value in added_types:
                raise ValueError(f"Type {type_token.value} already added (types so far: {list(added_types)})")
            added_types.add(type_token.value)
            code_to_parse = f"typedef int {type_token.value};\n" + code_to_parse
    else:
        raise ValueError(f"Type fixes exceeded limit ({MAX_TYPE_FIX_RETRIES})")
    return decompiled_ast


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

            yield RepoInfo(entry['repo_owner'], entry['repo_name'], makefiles)
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


def main():
    if not ghcc.utils.verify_docker_image(verbose=True):
        exit(1)

    args = Arguments()
    if args.pdb:
        ghcc.utils.register_ipython_excepthook()
        if args.n_procs == 0:
            globals()['match_functions'] = match_functions.__wrapped__
    if not args.verbose:
        ghcc.set_logging_level("quiet")

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
