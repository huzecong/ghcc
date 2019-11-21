r"""Run the cloning--compilation pipeline. What happens is:

1. Repositories are cloned from GitHub according to the given list.
2. Successfully cloned repositories are scanned for Makefiles.
3. Each Makefile will be used for compilation, and results will be gathered.
4. Compilation products are cleaned and the repository is archived to save space.
"""

import functools
import os
import pickle
import shutil
import subprocess
import traceback
from typing import Iterator, List, NamedTuple, Optional, Set, Dict, Callable

from mypy_extensions import TypedDict

import ghcc
from ghcc import CloneErrorType, RepoMakefileEntry


def parse_args():
    parser = ghcc.utils.ArgumentParser()
    parser.add_argument("--repo-list-file", type=str, default="../c-repos.csv")
    parser.add_argument("--clone-folder", type=str,
                        default="repos/")  # where cloned repositories are stored (temporarily)
    parser.add_argument("--binary-folder", type=str, default="binaries/")  # where compiled binaries are stored
    parser.add_argument("--archive-folder", type=str, default="archives/")  # where archived repositories are stored
    parser.add_argument("--n-procs", type=int, default=0)  # 0 for single-threaded execution
    parser.add_argument("--log-file", type=str, default="log.txt")
    parser.add_argument("--clone-timeout", type=int, default=600)  # wait up to 10 minutes
    parser.add_toggle_argument("--force-reclone", default=False)  # if not, use archives when possible
    parser.add_argument("--compile-timeout", type=int, default=900)  # wait up to 15 minutes
    parser.add_toggle_argument("--force-recompile", default=False)
    parser.add_toggle_argument("--docker-batch-compile", default=True)
    parser.add_argument("--compression-type", choices=["gzip", "xz"], default="gzip")
    parser.add_argument("--max-archive-size", type=int,
                        default=100 * 1024 * 1024)  # only archive repos no larger than 100MB.
    parser.add_argument("--record-libraries", type=str,
                        default=None)  # gather libraries used in Makefiles and print to the specified file
    parser.add_argument("--logging-level", choices=list(ghcc.logging.LEVEL_MAP.keys()), default="info")
    parser.add_argument("--max-repos", type=int,
                        default=-1)  # maximum number of repositories to process (ignoring non-existent)
    parser.add_toggle_argument("--recursive-clone", default=True)  # if True, use `--recursive` when `git clone`
    parser.add_toggle_argument("--write-db", default=True)  # only modify the DB when True
    parser.add_toggle_argument("--record-metainfo", default=False)  # if True, record a bunch of other stuff
    # parser.add_argument("--automake", action="store_true", default=False)  # if True, support automake
    args = parser.parse_args()
    return args


class RepoInfo(NamedTuple):
    idx: int  # `tuple` has an `index` method
    repo_owner: str
    repo_name: str
    db_result: Optional[ghcc.RepoEntry]


class PipelineMetaInfo(TypedDict):
    r"""Meta-info that might be required for experimentations."""
    num_makefiles: int  # total number of Makefiles
    has_gitmodules: bool  # whether repo contains a .gitmodules file


class PipelineResult(NamedTuple):
    repo_info: RepoInfo
    clone_success: Optional[bool] = None
    repo_size: Optional[int] = None
    makefiles: Optional[List[RepoMakefileEntry]] = None
    libraries: Optional[List[str]] = None
    meta_info: Optional[PipelineMetaInfo] = None


def contains_in_file(file_path: str, text: str) -> bool:
    r"""Check whether the file contains a specific piece of text in its first line.

    :param file_path: Path to the file.
    :param text: The piece of text to search for.
    :return: ``True`` only if the file exists and contains the text in its first line.
    """
    if not os.path.exists(file_path):
        return False
    with open(file_path, 'r') as f:
        line = f.readline()
    return text in line


def _docker_batch_compile(index: int, repo_binary_dir: str, repo_path: str,
                          compile_timeout: Optional[float], record_libraries: bool = False) -> List[RepoMakefileEntry]:
    try:
        # Don't rely on Docker timeout, but instead constrain running time in script run in Docker. Otherwise we won't
        # get the results file if any compilation task timeouts.
        ret = ghcc.utils.run_docker_command([
            "batch_make.py",
            *(["--record-libraries"] if record_libraries else []),
            *(["--compile-timeout", str(compile_timeout)] if compile_timeout is not None else [])],
            user=(index % 10000) + 30000,  # user IDs 30000 ~ 39999
            directory_mapping={repo_path: "/usr/src/repo", repo_binary_dir: "/usr/src/bin"}, return_output=True)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        raise e

    with open(os.path.join(repo_binary_dir, "log.pkl"), "rb") as f:
        makefiles: List[RepoMakefileEntry] = pickle.load(f)
    os.remove(os.path.join(repo_binary_dir, "log.pkl"))
    return makefiles


def exception_handler(e, *args, **kwargs):
    repo_info: RepoInfo = args[0] if len(args) > 0 else kwargs["repo_info"]
    exc_msg = f"<{e.__class__.__qualname__}> {e}"
    try:
        if not (isinstance(e, subprocess.CalledProcessError) and e.output is not None):
            ghcc.log(traceback.format_exc(), "error")

        ghcc.log(f"Exception occurred when processing {repo_info.repo_owner}/{repo_info.repo_name}: {exc_msg}", "error")
    except Exception as log_e:
        print(f"Exception occurred when processing {repo_info.repo_owner}/{repo_info.repo_name}: {exc_msg}")
        print(f"Another exception occurred while logging: <{log_e.__class__.__qualname__}> {log_e}")
        raise log_e


@ghcc.utils.exception_wrapper(exception_handler)
def clone_and_compile(repo_info: RepoInfo, clone_folder: str, binary_folder: str, archive_folder: str,
                      recursive_clone: bool = True,
                      clone_timeout: Optional[float] = None, compile_timeout: Optional[float] = None,
                      force_reclone: bool = False, force_recompile: bool = False, docker_batch_compile: bool = True,
                      max_archive_size: Optional[int] = None, compression_type: str = "gzip",
                      record_libraries: bool = False, record_metainfo: bool = False) -> PipelineResult:
    r"""Perform the entire pipeline.

    :param repo_info: Information about the repository.
    :param clone_folder: Path to the folder where the repository will be stored. The actual destination folder will be
        ``clone_folder/repo_owner_____repo_name``, e.g., ``clone_folder/torvalds_____linux``.
        This strange notation is used in order to have a flat directory hierarchy, so we're not left with a bunch of
        empty folders for repository owners.
    :param binary_folder: Path to the folder where compiled binaries will be stored. The actual destination folder will
        be ``binary_folder/repo_owner/repo_name``, e.g., ``binary_folder/torvalds/linux``.
    :param archive_folder: Path to the folder where archived repositories will be stored. The actual archive file will
        be ``archive_folder/repo_owner/repo_name.tar.xz``, e.g., ``archive_folder/torvalds/linux.tar.xz``.

    :param recursive_clone: If ``True``, uses ``--recursive`` when cloning.
    :param clone_timeout: Timeout for cloning, or `None` (default) for unlimited time.
    :param compile_timeout: Timeout for compilation, or `None` (default) for unlimited time.
    :param force_reclone: If ``True``, always clone a fresh copy for compilation. If ``False``, only clone when there
        are no matching archives.
    :param force_recompile: If ``True``, the repository is compiled regardless of the value in DB.
    :param docker_batch_compile: If ``True``, compile all Makefiles within a repository in a single Docker container.
    :param max_archive_size: If specified, only archive repositories whose size is not larger than the given
        value (in bytes).
    :param compression_type: The file type of the archive to produce. Valid values are ``"gzip"`` (faster) and
        ``"xz"`` (smaller).
    :param record_libraries: If ``True``, record the libraries used in compilation.
    :param record_metainfo: If ``True``, record meta-info values.

    :return: An entry to insert into the DB, or `None` if no operations are required.
    """
    repo_full_name = f"{repo_info.repo_owner}/{repo_info.repo_name}"
    repo_folder_name = f"{repo_info.repo_owner}_____{repo_info.repo_name}"
    repo_path = os.path.join(clone_folder, repo_folder_name)
    if compression_type == "xz":
        archive_extension = ".tar.xz"
        tar_type_flag = "J"
    elif compression_type == "gzip":
        archive_extension = ".tar.gz"
        tar_type_flag = "z"
    else:
        raise ValueError(f"Invalid compression type '{compression_type}'")
    archive_path = os.path.abspath(os.path.join(archive_folder, f"{repo_full_name}{archive_extension}"))

    repo_entry = repo_info.db_result
    clone_success = None

    # Skip repos that are fully processed
    if (repo_entry is not None and
            (repo_entry["clone_successful"] and not force_reclone) and
            (repo_entry["compiled"] and not force_recompile)):
        return PipelineResult(repo_info)

    # Stage 1: Cloning from GitHub.
    if not force_reclone and os.path.exists(archive_path):
        # Extract the archive instead of cloning.
        try:
            ghcc.utils.run_command(["tar", f"x{tar_type_flag}f", archive_path], timeout=clone_timeout, cwd=clone_folder)
            ghcc.log(f"{repo_full_name} extracted from archive", "success")
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
            ghcc.log(f"Unknown error when extracting {repo_full_name}. Captured output: '{e.output}'", "error")
            shutil.rmtree(repo_path)
            return PipelineResult(repo_info)  # return dummy info
        repo_size = ghcc.utils.get_folder_size(repo_path)
    elif (repo_entry is None or  # not processed
          (repo_entry["clone_successful"] and  # not compiled
           (not repo_entry["compiled"] or force_recompile) and not os.path.exists(repo_path))):
        clone_result = ghcc.clone(
            repo_info.repo_owner, repo_info.repo_name, clone_folder=clone_folder, folder_name=repo_folder_name,
            timeout=clone_timeout, skip_if_exists=False, recursive=recursive_clone)
        clone_success = clone_result.success
        if not clone_result.success:
            if clone_result.error_type is CloneErrorType.FolderExists:
                ghcc.log(f"{repo_full_name} skipped because folder exists", "warning")
            elif clone_result.error_type is CloneErrorType.PrivateOrNonexistent:
                ghcc.log(f"Failed to clone {repo_full_name} because repository is private or nonexistent", "warning")
            else:
                if clone_result.error_type is CloneErrorType.Unknown:
                    msg = f"Failed to clone {repo_full_name} with unknown error"
                else:  # CloneErrorType.Timeout
                    msg = f"Time expired ({clone_timeout}s) when attempting to clone {repo_full_name}"
                if clone_result.captured_output is not None:
                    msg += f". Captured output: '{clone_result.captured_output!r}'"
                ghcc.log(msg, "error")

                if clone_result.error_type is CloneErrorType.Unknown:
                    return PipelineResult(repo_info)  # return dummy info

            return PipelineResult(repo_info, clone_success=clone_success)

        repo_size = ghcc.utils.get_folder_size(repo_path)
        ghcc.log(f"{repo_full_name} successfully cloned ({clone_result.time:.2f}s, "
                 f"{ghcc.utils.readable_size(repo_size)})", "success")
    else:
        if not repo_entry["clone_successful"]:
            return PipelineResult(repo_info)  # return dummy info
        repo_size = ghcc.utils.get_folder_size(repo_path)

    makefiles = None
    libraries = None
    meta_info: Optional[PipelineMetaInfo] = None
    if not repo_entry or not repo_entry["compiled"] or force_recompile:
        # # SPECIAL CHECK: Do not attempt to compile OS kernels!
        # kernel_name = None
        # if contains_in_file(os.path.join(repo_path, "README"), "Linux kernel release"):
        #     kernel_name = "Linux"
        # elif contains_in_file(os.path.join(repo_path, "README"), "FreeBSD source directory"):
        #     kernel_name = "FreeBSD"
        # if kernel_name is not None:
        #     shutil.rmtree(repo_path)
        #     ghcc.log(f"Found {kernel_name} kernel in {repo_full_name}, will not attempt to compile. "
        #              f"Repository deleted", "warning")
        #     return PipelineResult(repo_info, clone_success=clone_success, makefiles=[])

        # Stage 2: Finding Makefiles.
        makefile_dirs = ghcc.find_makefiles(repo_path)
        if len(makefile_dirs) == 0:
            # Repo has no Makefiles, delete.
            shutil.rmtree(repo_path)
            ghcc.log(f"No Makefiles found in {repo_full_name}, repository deleted", "warning")
            return PipelineResult(repo_info, clone_success=clone_success, makefiles=[])
        else:
            pass

        # Stage 3: Compile each Makefile.
        repo_binary_dir = os.path.join(binary_folder, repo_full_name)
        if not os.path.exists(repo_binary_dir):
            os.makedirs(repo_binary_dir)
        ghcc.log(f"Starting compilation for {repo_full_name}...")

        if docker_batch_compile:
            makefiles = _docker_batch_compile(
                repo_info.idx, repo_binary_dir, repo_path, compile_timeout, record_libraries)
        else:
            makefiles = ghcc.compile_and_move(
                repo_binary_dir, repo_path, makefile_dirs, compile_timeout, record_libraries)
        num_succeeded = sum(makefile["success"] for makefile in makefiles)
        if record_libraries:
            library_log_path = os.path.join(repo_binary_dir, "libraries.txt")
            if os.path.exists(library_log_path):
                with open(library_log_path) as f:
                    libraries = list(set(f.read().split()))
            else:
                libraries = []
        num_binaries = sum(len(makefile["binaries"]) for makefile in makefiles)

        msg = f"{num_succeeded} ({len(makefiles)}) out of {len(makefile_dirs)} Makefile(s) " \
              f"in {repo_full_name} compiled (partially), yielding {num_binaries} binaries"
        ghcc.log(msg, "success" if num_succeeded == len(makefile_dirs) else "warning")

        if record_metainfo:
            meta_info = {
                "num_makefiles": len(makefile_dirs),
                "has_gitmodules": os.path.exists(os.path.join(repo_path, ".gitmodules")),
            }

        # Stage 4: Clean and zip repo.
        if max_archive_size is not None and repo_size > max_archive_size:
            shutil.rmtree(repo_path)
            ghcc.log(f"Removed {repo_full_name} because repository size ({ghcc.utils.readable_size(repo_size)}) "
                     f"exceeds limits", "info")
        else:
            # Repository is already cleaned in the compile stage.
            os.makedirs(os.path.split(archive_path)[0], exist_ok=True)
            compress_success = False
            try:
                ghcc.utils.run_command(["tar", f"c{tar_type_flag}f", archive_path, repo_folder_name],
                                       timeout=clone_timeout, cwd=clone_folder)
                compress_success = True
            except subprocess.TimeoutExpired:
                ghcc.log(f"Compression timeout for {repo_full_name}, giving up", "error")
            except subprocess.CalledProcessError as e:
                ghcc.log(f"Unknown error when compressing {repo_full_name}. Captured output: '{e.output}'", "error")
            if compress_success:
                shutil.rmtree(repo_path)
                ghcc.log(f"Compressed {repo_full_name}, folder removed", "info")
            elif os.path.exists(archive_path):
                os.remove(archive_path)

    return PipelineResult(repo_info, clone_success=clone_success, repo_size=repo_size,
                          makefiles=makefiles, libraries=libraries, meta_info=meta_info)


def iter_repos(db: ghcc.Database, repo_list_path: str, max_count: Optional[int] = None) -> Iterator[RepoInfo]:
    db_entries = {
        (entry["repo_owner"], entry["repo_name"]): entry
        for entry in db.collection.find()
    }
    ghcc.log(f"{len(db_entries)} entries loaded from DB")
    index = 0
    with open(repo_list_path, "r") as repo_file:
        for line in repo_file:
            if not line:
                continue
            url = line.split()[1]
            repo_owner, repo_name = url.rstrip("/").split("/")[-2:]
            # db_result = db.get(repo_owner, repo_name)
            db_result = db_entries.get((repo_owner, repo_name), None)
            yield RepoInfo(index, repo_owner, repo_name, db_result)
            index += 1
            if max_count is not None and index >= max_count:
                break


class MetaInfo:
    def __init__(self):
        self.num_repos = 0
        self.num_makefiles = 0
        self.num_binaries = 0
        self.num_gitmodules = 0
        self.fail_to_success = 0
        self.success_to_fail = 0
        self.success_makefiles = 0
        self.added_makefiles = 0
        self.missing_makefiles = 0

    def add_repo(self, result: PipelineResult) -> None:
        self.num_repos += 1
        if result.meta_info is not None:
            self.num_gitmodules += result.meta_info["has_gitmodules"]
            self.num_makefiles += result.meta_info["num_makefiles"]

        new_makefiles: Dict[str, bool] = {}
        old_makefiles: Dict[str, bool] = {}
        if result.makefiles is not None:
            new_makefiles = {makefile["directory"]: makefile["success"]
                             for makefile in result.makefiles}
            self.success_makefiles += sum(new_makefiles.values())
        if result.repo_info.db_result is not None:
            old_makefiles = {makefile["directory"]: makefile["success"]
                             for makefile in result.repo_info.db_result["makefiles"]}
        new_makefile_set = set(new_makefiles.keys())
        old_makefile_set = set(old_makefiles.keys())
        self.added_makefiles += len(new_makefile_set.difference(old_makefile_set))
        self.missing_makefiles += len(old_makefile_set.difference(new_makefile_set))
        for name in new_makefile_set.intersection(old_makefile_set):
            new_status = new_makefiles[name]
            old_status = old_makefiles[name]
            if new_status and not old_status:
                self.fail_to_success += 1
            elif old_status and not new_status:
                self.success_to_fail += 1

    def __repr__(self) -> str:
        msg = f"#Repos: {self.num_repos} ({self.num_gitmodules} with .gitmodules), #Binaries: {self.num_binaries}" \
              f"#Makefiles: {self.num_makefiles} ({self.success_makefiles} succeeded)\n" \
              f" ├─ New: {self.added_makefiles}, Missing: {self.missing_makefiles}\n" \
              f" └─ Fail->Success: {self.fail_to_success}, Success->Fail: {self.success_to_fail}"
        return msg


def main() -> None:
    if not ghcc.utils.verify_docker_image():
        ghcc.log("ERROR: Your Docker image is out-of-date. Please rebuild the image by: `docker build -t gcc-custom .`",
                 "error", force_console=True)
        exit(1)

    args = parse_args()
    print(args)
    if args.n_procs == 0:
        # Only do this on the single-threaded case.
        ghcc.utils.register_ipython_excepthook()
    ghcc.set_log_file(args.log_file)
    ghcc.set_logging_level(args.logging_level, console=True, file=False)

    if os.path.exists(args.clone_folder):
        ghcc.log(f"Removing contents of clone folder '{args.clone_folder}'...", "warning", force_console=True)
        ghcc.utils.run_docker_command(["rm", "-rf", "/usr/src/*"], user=0,
                                      directory_mapping={args.clone_folder: "/usr/src"})

    ghcc.log("Crawling starts...", "warning", force_console=True)
    pool = ghcc.utils.Pool(processes=args.n_procs)
    db = ghcc.Database()
    libraries: Set[str] = set()
    if args.record_libraries is not None and os.path.exists(args.record_libraries):
        with open(args.record_libraries, "r") as f:
            libraries = set(f.read().split())

    try:
        iterator = iter_repos(db, args.repo_list_file, args.max_repos if args.max_repos != -1 else None)
        pipeline_fn: Callable[[RepoInfo], Optional[PipelineMetaInfo]] = functools.partial(
            clone_and_compile,
            clone_folder=args.clone_folder, binary_folder=args.binary_folder, archive_folder=args.archive_folder,
            recursive_clone=args.recursive_clone,
            clone_timeout=args.clone_timeout, compile_timeout=args.compile_timeout,
            force_reclone=args.force_reclone, force_recompile=args.force_recompile,
            docker_batch_compile=args.docker_batch_compile,
            max_archive_size=args.max_archive_size, compression_type=args.compression_type,
            record_libraries=(args.record_libraries is not None), record_metainfo=args.record_metainfo)
        repo_count = 0
        meta_info = MetaInfo()
        for result in pool.imap_unordered(pipeline_fn, iterator):
            repo_count += 1
            if repo_count % 100 == 0:
                ghcc.log(f"Processed {repo_count} repositories", force_console=True)
            if result is None:
                continue
            repo_owner, repo_name = result.repo_info.repo_owner, result.repo_info.repo_name
            if args.write_db:
                if result.clone_success is not None or result.repo_info.db_result is None:
                    # There's probably an inconsistency somewhere if we didn't clone while `db_result` is None.
                    # To prevent more errors, just add it to the DB.
                    repo_size = result.repo_size or -1  # a value of zero is probably also wrong
                    db.add_repo(repo_owner, repo_name, result.clone_success or True, repo_size=repo_size)
                    ghcc.log(f"Added {repo_owner}/{repo_name} to DB")
                if result.makefiles is not None:
                    db.update_makefile(repo_owner, repo_name, result.makefiles,
                                       ignore_length_mismatch=args.force_recompile)
            if result.libraries is not None:
                libraries.update(result.libraries)
                if repo_count % 10 == 0:  # flush every 10 repos
                    assert args.record_libraries is not None
                    with open(args.record_libraries, "w") as f:
                        f.write("\n".join(libraries))

            if args.record_metainfo:
                meta_info.add_repo(result)
                if repo_count % 100 == 0:
                    ghcc.log(repr(meta_info), force_console=True)

    except KeyboardInterrupt:
        print("Press Ctrl-C again to force terminate...")
    except BlockingIOError as e:
        print(traceback.format_exc())
        pool.close()
        pool.terminate()
    except Exception as e:
        print(traceback.format_exc())
    finally:
        ghcc.log("Gracefully shutting down...", "warning", force_console=True)
        if args.record_libraries is not None:
            with open(args.record_libraries, "w") as f:
                f.write("\n".join(libraries))
        if args.n_procs > 0:
            pool.close()
            try:
                pool.join()
            except KeyboardInterrupt:
                pool.terminate()
            ghcc.utils.kill_proc_tree(os.getpid())  # commit suicide


if __name__ == '__main__':
    main()
