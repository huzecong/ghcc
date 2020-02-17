# Match decompiled code with their original code
import contextlib
import functools
import json
import os
import pickle
import pprint
import re
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterator, List, NamedTuple, Optional, Set, Tuple, Any

import argtyped
from argtyped import Switch
import pycparser
import pycparser.c_generator
import tqdm
from termcolor import colored

import ghcc
from main import exception_handler

FAKE_LIBC_PATH = str((Path(ghcc.__file__).parent.parent / "scripts" / "fake_libc_include").absolute())


class Arguments(argtyped.Arguments):
    archive_dir: Optional[str] = "archives/"  # directory containing repo archives
    decompile_dir: str = "decompile_output_fixed/"  # directory containing decompiled output (JSONL files)
    temp_dir: str = "repos/"
    log_file: str = "match-log.txt"
    output_dir: str = "match_output/"  # directory to save matching functions
    use_fake_libc_headers: Switch = True  # use pycparser fake libc headers to save time
    n_procs: int = 0  # number of processes
    max_repos: Optional[int] = None  # maximum number of repositories to process (ignoring non-existent)
    pdb: Switch = False
    write_db: Switch = True


def download_fake_libc() -> bool:
    r"""Download fake libc headers for ``pycparser``, and returns whether download succeeded.
    """
    if os.path.exists(os.path.join(FAKE_LIBC_PATH, "stdlib.h")):
        return True
    ghcc.log("Downloading fake libc headers from eliben/pycparser ...", "warning")
    with tempfile.TemporaryDirectory() as tempdir:
        result = ghcc.clone("eliben", "pycparser", clone_folder=tempdir, folder_name="pycparser", timeout=20)
        if not result.success:
            return False
        src_dir = os.path.join(tempdir, "pycparser", "utils", "fake_libc_include")
        ghcc.utils.copy_tree(src_dir, FAKE_LIBC_PATH, overwrite=True)
    return True


class RepoInfo(NamedTuple):
    repo_owner: str
    repo_name: str
    makefiles: Dict[str, Dict[str, str]]  # make_dir -> (binary_path -> binary_sha256)


class FunctionExtractor:
    r"""A visitor that extracts all function definitions from an AST."""

    class FuncDefVisitor(pycparser.c_ast.NodeVisitor):
        func_defs: Dict[str, str]
        func_nodes: Dict[str, pycparser.c_ast.Node]

        def __init__(self):
            self.c_generator = pycparser.c_generator.CGenerator()

        def visit_FuncDef(self, node: pycparser.c_ast.FuncDef):
            func_name = node.decl.name
            self.func_nodes[func_name] = node
            self.func_defs[func_name] = self.c_generator.visit(node)

    def __init__(self):
        self._visitor = self.FuncDefVisitor()

    def find_functions(self, node: pycparser.c_ast.Node) -> Dict[str, str]:
        r"""Find all function definitions given an AST.

        :param node: The ``pycparser`` AST node object.
        :return: A dictionary mapping function names to function definitions.
        """
        ret = self._visitor.func_defs = {}
        self._visitor.func_nodes = {}
        self._visitor.visit(node)
        del self._visitor.func_defs
        return ret


class MatchedFunction(NamedTuple):
    # Source code & decompiler info
    file_path: str
    binary_hash: str
    line_number: int  # line number in the decompiled JSONL data
    # Code
    func_name: str
    original_code: str
    decompiled_code: str
    # No need to store the AST & stuff; we can get those from the decompiled data.


class Result(NamedTuple):
    repo_owner: str
    repo_name: str
    matched_functions: List[MatchedFunction]
    functions_found: int


@ghcc.utils.exception_wrapper(exception_handler)
def match_functions(repo_info: RepoInfo, archive_folder: str, temp_folder: str, decompile_folder: str,
                    use_fake_libc_headers: bool = True) -> Result:
    # Directions:
    # 1. Clone or extract from archive.
    # 2. For each Makefile, rerun the compilation process with the flag "-E", so only the preprocessor is run.
    #    This probably won't take long as the compiler exits after running the processor, and linking would fail.
    #    Also, consider using "-nostdlib -Ipath/to/fake_libc_include" as suggested by `pycparser`.
    # 3. The .o files are now preprocessed C code. Parse them using `pycparser` to obtain a list of functions.

    repo_folder_name = f"{repo_info.repo_owner}_____{repo_info.repo_name}"
    repo_full_name = f"{repo_info.repo_owner}/{repo_info.repo_name}"
    archive_path = (Path(archive_folder) / f"{repo_full_name}.tar.gz").absolute()
    temp_dir = (Path(temp_folder) / repo_folder_name).absolute()
    repo_path = temp_dir / "src"
    repo_binary_dir = temp_dir / "bin"
    repo_binary_dir.mkdir(parents=True, exist_ok=True)
    if os.path.exists(archive_path):
        # Extract archive
        ghcc.utils.run_command(["tar", f"xzf", str(archive_path)], cwd=str(temp_dir))
        (temp_dir / repo_folder_name).rename(repo_path)
    else:
        # Clone repo
        ret = ghcc.clone(repo_info.repo_owner, repo_info.repo_name, clone_folder=str(temp_dir), folder_name="src")
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
        str(repo_binary_dir), str(repo_path), compile_timeout=600,
        gcc_override_flags=gcc_flags, use_makefile_info_pkl=True, directory_mapping=directory_mapping,
        exception_log_fn=functools.partial(exception_handler, repo_info=repo_info))

    parser = pycparser.CParser()
    func_name_regex = re.compile(r'"function":\s*"([^"]+)"')
    decompile_path = Path(decompile_folder)
    extractor = FunctionExtractor()
    matched_functions = []
    functions_found = 0
    for makefile in makefiles:
        for path, sha in zip(makefile["binaries"], makefile["sha256"]):
            json_path = decompile_path / (sha + ".jsonl")
            if not json_path.exists():
                continue
            with (repo_binary_dir / sha).open("r") as f:
                code = f.read()
            try:
                ast: pycparser.c_ast.Node = parser.parse(code, filename=os.path.join(repo_full_name, path))
            except pycparser.c_parser.ParseError:
                continue  # ignore parsing errors
            functions = extractor.find_functions(ast)
            functions_found += len(functions)
            with json_path.open("r") as f:
                decompiled_json = [line for line in f if line]  # don't decode, as we only need the function name
            for line_num, j in enumerate(decompiled_json):
                match = func_name_regex.search(j)
                assert match is not None
                func_name = match.group(1)
                if func_name not in functions:
                    continue
                original_code = functions[func_name]
                decompiled_data = json.loads(j)
                decompiled_code = decompiled_data["raw_code"]
                pretty_decompiled_code = re.sub(  # replace @@VAR with developer-assigned names
                    r"@@VAR_\d+@@[^@]+@@([_a-zA-Z0-9]+)", r"\1", decompiled_code)

                print(colored(f"Function Name: {func_name}", "green"))
                print(colored("Original Code:", "yellow"), original_code, sep='\n')
                print(colored("Decompiled Code:", "red"), pretty_decompiled_code, sep='\n')
                input("Press any key to continue...")
                matched_func = MatchedFunction(
                    file_path=path, binary_hash=sha, line_number=line_num,
                    func_name=func_name, original_code=original_code, decompiled_code=decompiled_code)
                matched_functions.append(matched_func)

    ghcc.log(f"{repo_info.repo_owner}/{repo_info.repo_name}: "
             f"{functions_found} found, {len(matched_functions)} matched", "success")
    return Result(repo_owner=repo_info.repo_owner, repo_name=repo_info.repo_name,
                  matched_functions=matched_functions, functions_found=functions_found)


def iter_repos(db: ghcc.MatchFuncDB, max_count: Optional[int] = None) -> Iterator[RepoInfo]:
    db_entries: Set[Tuple[str, str]] = {
        (entry["repo_owner"], entry["repo_name"])
        for entry in db.collection.find()  # getting stuff in batch is much faster
    }

    with contextlib.closing(ghcc.RepoDB()) as repo_db, contextlib.closing(ghcc.BinaryDB()) as binary_db:
        repo_binaries: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
        all_bins = binary_db.collection.find()
        for entry in tqdm.tqdm(all_bins, total=all_bins.count(), desc="Listing decompiled binaries"):
            if entry["success"]:
                repo_binaries[entry["repo_owner"], entry["repo_name"]].add(entry["sha"])

        repo_entries: Iterator[ghcc.RepoDB.Entry] = repo_db.collection.find()
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


def main():
    if not ghcc.utils.verify_docker_image(verbose=True):
        exit(1)

    args = Arguments()
    if args.pdb:
        ghcc.utils.register_ipython_excepthook()
        if args.n_procs == 0:
            globals()['match_functions'] = match_functions.__wrapped__

    if args.use_fake_libc_headers and not download_fake_libc():
        ghcc.log("Failed to download fake libc headers. Consider rerunning the script, or use the"
                 "`--no-use-fake-libc-headers` flag", "error")
        exit(1)

    ghcc.log(f"Running with {args.n_procs} worker processes", "warning")

    if os.path.exists(args.temp_dir):
        ghcc.log(f"Removing contents of temporary folder '{args.temp_dir}'...", "warning", force_console=True)
        ghcc.utils.run_docker_command(["rm", "-rf", "/usr/src/*"], user=0,
                                      directory_mapping={args.temp_dir: "/usr/src"})

    db = ghcc.MatchFuncDB()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with ghcc.utils.safe_pool(args.n_procs, closing=[db]) as pool:
        iterator = iter_repos(db, args.max_repos)
        match_fn: Callable[[RepoInfo], Result] = functools.partial(
            match_functions,
            archive_folder=args.archive_dir, temp_folder=args.temp_dir, decompile_folder=args.decompile_dir,
            use_fake_libc_headers=args.use_fake_libc_headers)

        repo_count = 0
        for result in pool.imap_unordered(match_fn, iterator):
            if result is None:
                continue  # exception occurred

            # Write the matched functions to disk.
            with (output_dir / f"{result.repo_owner}_____{result.repo_name}").open("w") as f:
                for matched_func in result.matched_functions:
                    f.write(json.dumps(matched_func._asdict()) + "\n")

            if args.write_db:
                db.add_repo(result.repo_owner, result.repo_name,
                            result.functions_found, funcs_matched=len(result.matched_functions))

            repo_count += 1
            if repo_count % 100 == 0:
                ghcc.log(f"Processed {repo_count} repositories", force_console=True)


if __name__ == '__main__':
    main()
