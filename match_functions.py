# Match decompiled code with their original code
import functools
import json
import os
import pickle
import re
import tempfile
from pathlib import Path
from typing import Dict, Iterator, NamedTuple, Optional, Callable

import pycparser
import pycparser.c_generator

import ghcc
from ghcc.arguments import Switch
from main import exception_handler
from run_decompiler import get_binary_mapping

FAKE_LIBC_PATH = str((Path(ghcc.__file__).parent.parent / "scripts" / "fake_libc_include").absolute())


class Arguments(ghcc.arguments.Arguments):
    archive_dir: Optional[str] = "archives/"  # directory containing repo archives
    decompile_dir: str = "decompile_output_fixed/"  # directory containing decompiled output (JSONL files)
    temp_dir: str = "repos/"
    log_file: str = "match-log.txt"
    output_dir: str = "match_output/"  # directory to save matching functions
    binary_mapping_cache_file: Optional[str] = "binary_mapping.pkl"
    use_fake_libc_headers: Switch = True  # use pycparser fake libc headers to save time
    n_procs: int = 0  # number of processes
    max_repos: Optional[int] = None  # maximum number of repositories to process (ignoring non-existent)
    pdb: Switch = False


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
    class FuncDefVisitor(pycparser.c_ast.NodeVisitor):
        func_defs: Dict[str, str]

        def __init__(self):
            self.c_generator = pycparser.c_generator.CGenerator()

        def visit_FuncDef(self, node: pycparser.c_ast.FuncDef):
            self.func_defs[node.decl.name] = self.c_generator.visit(node)

    def __init__(self):
        self._visitor = self.FuncDefVisitor()

    def find_functions(self, node: pycparser.c_ast.Node) -> Dict[str, str]:
        ret = self._visitor.func_defs = {}
        self._visitor.visit(node)
        del self._visitor.func_defs
        return ret


@ghcc.utils.exception_wrapper(exception_handler)
def match_functions(repo_info: RepoInfo, archive_folder: str, temp_folder: str, decompile_folder: str,
                    use_fake_libc_headers: bool = True):
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
    for makefile in makefiles:
        for path, sha in zip(makefile["binaries"], makefile["sha256"]):
            json_path = decompile_path / (sha + ".jsonl")
            if not json_path.exists():
                continue
            with (repo_binary_dir / sha).open("r") as f:
                code = f.read()
            ast: pycparser.c_ast.Node = parser.parse(code, filename=os.path.join(repo_full_name, path))
            functions = extractor.find_functions(ast)
            with json_path.open("r") as f:
                decompiled_json = [line for line in f if line]  # don't decode, as we only need the function name
            for j in decompiled_json:
                match = func_name_regex.search(j)
                assert match is not None
                func_name = match.group(1)
                if func_name not in functions:
                    continue
                decompiled_code = json.loads(j)["raw_code"]
                decompiled_code = re.sub(r"@@VAR_\d+@@[^@]+@@([_a-zA-Z0-9]+)", r"\1", decompiled_code)
                print("Function Name:", func_name)
                print("Original Code:", functions[func_name], sep='\n')
                print("Decompiled Code:", decompiled_code, sep='\n')
                input("Press any key to continue...")


def iter_repos(db: ghcc.RepoDB, cache_path: Optional[str] = None, max_count: Optional[int] = None) \
        -> Iterator[RepoInfo]:
    binaries = get_binary_mapping(cache_path)

    db_entries: Iterator[ghcc.RepoDB.Entry] = db.collection.find()
    index = 0
    for entry in db_entries:
        if (not entry['clone_successful'] or not entry['compiled'] or
                len(entry['makefiles']) == 0 or entry['num_binaries'] == 0):
            continue

        prefix = f"{entry['repo_owner']}/{entry['repo_name']}"
        makefiles: Dict[str, Dict[str, str]] = {}
        for makefile in entry['makefiles']:
            mapping = {}
            for path, sha in zip(makefile['binaries'], makefile['sha256']):
                if os.path.normpath(binaries[sha][0]) == os.path.join(prefix, sha):
                    mapping[path] = sha
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

    if os.path.exists(args.temp_dir):
        ghcc.log(f"Removing contents of temporary folder '{args.temp_dir}'...", "warning", force_console=True)
        ghcc.utils.run_docker_command(["rm", "-rf", "/usr/src/*"], user=0,
                                      directory_mapping={args.temp_dir: "/usr/src"})

    db = ghcc.RepoDB()
    with ghcc.utils.safe_pool(args.n_procs, closing=[db]) as pool:
        iterator = iter_repos(db, args.binary_mapping_cache_file, args.max_repos)
        match_fn: Callable[[RepoInfo], None] = functools.partial(
            match_functions,
            archive_folder=args.archive_dir, temp_folder=args.temp_dir, decompile_folder=args.decompile_dir,
            use_fake_libc_headers=args.use_fake_libc_headers)

        for result in pool.imap_unordered(match_fn, iterator):
            pass


if __name__ == '__main__':
    main()
