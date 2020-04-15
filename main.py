r"""Run the cloning--compilation pipeline. What happens is:

1. Repositories are cloned from GitHub according to the given list.
2. Successfully cloned repositories are scanned for Makefiles.
3. Each Makefile will be used for compilation, and results will be gathered.
4. Compilation products are cleaned and the repository is archived to save space.
"""

import functools
import os
import shutil
import subprocess
from typing import Callable, Iterator, List, NamedTuple, Optional, Set

import argtyped
import flutes
from argtyped import Choices, Switch
from mypy_extensions import TypedDict

import ghcc
from ghcc.database import RepoDB
from ghcc.repo import CloneErrorType


class Arguments(argtyped.Arguments):
    repo_list_file: str
    clone_folder: str = "repos/"  # where cloned repositories are stored (temporarily)
    binary_folder: str = "binaries/"  # where compiled binaries are stored
    archive_folder: str = "archives/"  # where archived repositories are stored
    n_procs: int = 0  # 0 for single-threaded execution
    log_file: str = "log.txt"
    clone_timeout: Optional[int] = 600  # wait up to 10 minutes
    force_reclone: Switch = False  # if not, use archives when possible
    compile_timeout: Optional[int] = 900  # wait up to 15 minutes
    force_recompile: Switch = False
    docker_batch_compile: Switch = True
    compression_type: Choices["gzip", "xz"] = "gzip"
    max_archive_size: Optional[int] = 100 * 1024 * 1024  # only archive repos no larger than 100MB.
    record_libraries: Optional[str] = None  # gather libraries used in Makefiles and print to the specified file
    logging_level: Choices[flutes.get_logging_levels()] = "info"
    max_repos: Optional[int] = None  # maximum number of repositories to process (ignoring non-existent)
    recursive_clone: Switch = True  # if True, use `--recursive` when `git clone`
    write_db: Switch = True  # only modify the DB when True
    record_metainfo: Switch = False  # if True, record a bunch of other stuff
    gcc_override_flags: Optional[str] = None  # GCC flags to use during compilation, e.g. "-O2 -march=x86-64"


class RepoInfo(NamedTuple):
    idx: int  # `tuple` has an `index` method
    repo_owner: str
    repo_name: str
    db_result: Optional[RepoDB.Entry]


class PipelineMetaInfo(TypedDict):
    r"""Meta-info that might be required for experimentations."""
    num_makefiles: int  # total number of Makefiles
    has_gitmodules: bool  # whether repo contains a .gitmodules file
    makefiles_using_automake: int  # how many Makefiles uses `automake`


class PipelineResult(NamedTuple):
    repo_info: RepoInfo
    clone_success: Optional[bool] = None
    repo_size: Optional[int] = None
    makefiles: Optional[List[RepoDB.MakefileEntry]] = None
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


def exception_handler(e, repo_info: RepoInfo, _return: bool = False):
    flutes.log_exception(e, f"Exception occurred when processing {repo_info.repo_owner}/{repo_info.repo_name}")
    if _return:
        # mark it as "failed to clone" so we don't deal with it anymore
        return PipelineResult(repo_info, clone_success=False)


@flutes.exception_wrapper(exception_handler)
def clone_and_compile(repo_info: RepoInfo, clone_folder: str, binary_folder: str, archive_folder: str,
                      recursive_clone: bool = True,
                      clone_timeout: Optional[float] = None, compile_timeout: Optional[float] = None,
                      force_reclone: bool = False, force_recompile: bool = False, docker_batch_compile: bool = True,
                      max_archive_size: Optional[int] = None, compression_type: str = "gzip",
                      record_libraries: bool = False, record_metainfo: bool = False,
                      gcc_override_flags: Optional[str] = None) -> PipelineResult:
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
    :param gcc_override_flags: If not ``None``, these flags will be appended to each invocation of GCC.

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
            flutes.run_command(["tar", f"x{tar_type_flag}f", archive_path], timeout=clone_timeout, cwd=clone_folder)
            flutes.log(f"{repo_full_name} extracted from archive", "success")
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
            flutes.log(f"Unknown error when extracting {repo_full_name}. Captured output: '{e.output}'", "error")
            shutil.rmtree(repo_path)
            return PipelineResult(repo_info)  # return dummy info
        repo_size = flutes.get_folder_size(repo_path)
    elif (repo_entry is None or  # not processed
          force_reclone or
          (repo_entry["clone_successful"] and  # not compiled
           (not repo_entry["compiled"] or force_recompile) and not os.path.exists(repo_path))):
        clone_result = ghcc.clone(
            repo_info.repo_owner, repo_info.repo_name, clone_folder=clone_folder, folder_name=repo_folder_name,
            timeout=clone_timeout, skip_if_exists=False, recursive=recursive_clone)
        clone_success = clone_result.success
        if not clone_result.success:
            if clone_result.error_type is CloneErrorType.FolderExists:
                flutes.log(f"{repo_full_name} skipped because folder exists", "warning")
            elif clone_result.error_type is CloneErrorType.PrivateOrNonexistent:
                flutes.log(f"Failed to clone {repo_full_name} because repository is private or nonexistent", "warning")
            else:
                if clone_result.error_type is CloneErrorType.Unknown:
                    msg = f"Failed to clone {repo_full_name} with unknown error"
                else:  # CloneErrorType.Timeout
                    msg = f"Time expired ({clone_timeout}s) when attempting to clone {repo_full_name}"
                if clone_result.captured_output is not None:
                    msg += f". Captured output: '{clone_result.captured_output!r}'"
                flutes.log(msg, "error")

                if clone_result.error_type is CloneErrorType.Unknown:
                    return PipelineResult(repo_info)  # return dummy info

            return PipelineResult(repo_info, clone_success=clone_success)

        elif clone_result.error_type is CloneErrorType.SubmodulesFailed:
            msg = f"Submodules in {repo_full_name} ignored due to error"
            if clone_result.captured_output is not None:
                msg += f". Captured output: '{clone_result.captured_output!r}'"
            flutes.log(msg, "warning")

        repo_size = flutes.get_folder_size(repo_path)
        flutes.log(f"{repo_full_name} successfully cloned ({clone_result.time:.2f}s, "
                   f"{flutes.readable_size(repo_size)})", "success")
    else:
        if not repo_entry["clone_successful"]:
            return PipelineResult(repo_info)  # return dummy info
        repo_size = flutes.get_folder_size(repo_path)

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
            flutes.log(f"No Makefiles found in {repo_full_name}, repository deleted", "warning")
            return PipelineResult(repo_info, clone_success=clone_success, makefiles=[])
        else:
            pass

        # Stage 3: Compile each Makefile.
        repo_binary_dir = os.path.join(binary_folder, repo_full_name)
        if not os.path.exists(repo_binary_dir):
            os.makedirs(repo_binary_dir)
        flutes.log(f"Starting compilation for {repo_full_name}...")

        if docker_batch_compile:
            makefiles = ghcc.docker_batch_compile(
                repo_binary_dir, repo_path, compile_timeout, record_libraries, gcc_override_flags,
                user_id=(repo_info.idx % 10000) + 30000,  # user IDs 30000 ~ 39999
                exception_log_fn=functools.partial(exception_handler, repo_info=repo_info))
        else:
            makefiles = list(ghcc.compile_and_move(
                repo_binary_dir, repo_path, makefile_dirs, compile_timeout, record_libraries, gcc_override_flags))
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
        flutes.log(msg, "success" if num_succeeded == len(makefile_dirs) else "warning")

        if record_metainfo:
            meta_info = PipelineMetaInfo({
                "num_makefiles": len(makefile_dirs),
                "has_gitmodules": os.path.exists(os.path.join(repo_path, ".gitmodules")),
                "makefiles_using_automake": sum(
                    ghcc.contains_files(directory, ["configure.ac", "configure.in"]) for directory in makefile_dirs)
            })

        # Stage 4: Clean and zip repo.
        if max_archive_size is not None and repo_size > max_archive_size:
            shutil.rmtree(repo_path)
            flutes.log(f"Removed {repo_full_name} because repository size ({flutes.readable_size(repo_size)}) "
                       f"exceeds limits", "info")
        else:
            # Repository is already cleaned in the compile stage.
            os.makedirs(os.path.split(archive_path)[0], exist_ok=True)
            compress_success = False
            try:
                flutes.run_command(["tar", f"c{tar_type_flag}f", archive_path, repo_folder_name],
                                   timeout=clone_timeout, cwd=clone_folder)
                compress_success = True
            except subprocess.TimeoutExpired:
                flutes.log(f"Compression timeout for {repo_full_name}, giving up", "error")
            except subprocess.CalledProcessError as e:
                flutes.log(f"Unknown error when compressing {repo_full_name}. Captured output: '{e.output}'", "error")
            shutil.rmtree(repo_path)
            if compress_success:
                flutes.log(f"Compressed {repo_full_name}, folder removed", "info")
            elif os.path.exists(archive_path):
                os.remove(archive_path)

    return PipelineResult(repo_info, clone_success=clone_success, repo_size=repo_size,
                          makefiles=makefiles, libraries=libraries, meta_info=meta_info)


def iter_repos(db: ghcc.RepoDB, repo_list_path: str, max_count: Optional[int] = None) -> Iterator[RepoInfo]:
    db_entries = {
        (entry["repo_owner"], entry["repo_name"]): entry
        for entry in db.collection.find()  # getting stuff in batch is much faster
    }
    flutes.log(f"{len(db_entries)} entries loaded from DB")
    index = 0
    with open(repo_list_path, "r") as repo_file:
        for line in repo_file:
            if not line:
                continue
            url = line.strip().rstrip("/")
            if url.endswith(".git"):
                url = url[:-len(".git")]
            repo_owner, repo_name = url.split("/")[-2:]
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
        self.makefiles_with_yield = 0
        self.added_makefiles = 0  # how many new success Makefiles are not in old
        self.missing_makefiles = 0  # how many old success Makefiles are not in new
        self.makefiles_using_automake = 0

    def add_repo(self, result: PipelineResult) -> None:
        self.num_repos += 1
        if result.meta_info is not None:
            self.num_gitmodules += result.meta_info["has_gitmodules"]
            self.num_makefiles += result.meta_info["num_makefiles"]
            self.makefiles_using_automake += result.meta_info["makefiles_using_automake"]

        # new_makefiles: Dict[str, bool] = {}
        # old_makefiles: Dict[str, bool] = {}
        if result.makefiles is not None:
            new_makefiles = {makefile["directory"]: makefile["success"]
                             for makefile in result.makefiles}
            self.success_makefiles += sum(new_makefiles.values())
            self.num_binaries += sum(len(makefile["binaries"]) for makefile in result.makefiles)
        elif result.repo_info.db_result is not None:
            db_result = result.repo_info.db_result
            self.success_makefiles += sum(makefile["success"] for makefile in db_result["makefiles"])
            self.num_binaries += sum(len(makefile["binaries"]) for makefile in db_result["makefiles"])

        # if result.repo_info.db_result is not None:
        #     old_makefiles = {makefile["directory"]: makefile["success"]
        #                      for makefile in result.repo_info.db_result["makefiles"]}
        # new_makefile_set = set(new_makefiles.keys())
        # old_makefile_set = set(old_makefiles.keys())
        # # Only count success Makefiles here.
        # self.added_makefiles += sum(new_makefiles[m] for m in new_makefile_set.difference(old_makefile_set))
        # self.missing_makefiles += sum(old_makefiles[m] for m in old_makefile_set.difference(new_makefile_set))
        # for name in new_makefile_set.intersection(old_makefile_set):
        #     new_status = new_makefiles[name]
        #     old_status = old_makefiles[name]
        #     if new_status and not old_status:
        #         self.fail_to_success += 1
        #     elif old_status and not new_status:
        #         self.success_to_fail += 1

    def __repr__(self) -> str:
        msg = f"#Repos: {self.num_repos} ({self.num_gitmodules} with .gitmodules), #Binaries: {self.num_binaries}\n" \
              f"#Makefiles: {self.num_makefiles} ({self.success_makefiles} succeeded, " \
              f"{self.makefiles_using_automake} using automake)"
        # f" ├─ New: {self.added_makefiles}, Missing: {self.missing_makefiles}\n" \
        # f" └─ Fail->Success: {self.fail_to_success}, Success->Fail: {self.success_to_fail}"
        return msg


def main() -> None:
    if not ghcc.utils.verify_docker_image(verbose=True):
        exit(1)

    args = Arguments()
    if args.n_procs == 0:
        # Only do this on the single-threaded case.
        flutes.register_ipython_excepthook()
    flutes.set_log_file(args.log_file)
    flutes.set_logging_level(args.logging_level, console=True, file=False)
    flutes.log("Running with arguments:\n" + args.to_string(), force_console=True)

    if os.path.exists(args.clone_folder):
        flutes.log(f"Removing contents of clone folder '{args.clone_folder}'...", "warning", force_console=True)
        ghcc.utils.run_docker_command(["rm", "-rf", "/usr/src/*"], user=0,
                                      directory_mapping={args.clone_folder: "/usr/src"})

    flutes.log("Crawling starts...", "warning", force_console=True)
    db = ghcc.RepoDB()
    libraries: Set[str] = set()
    if args.record_libraries is not None and os.path.exists(args.record_libraries):
        with open(args.record_libraries, "r") as f:
            libraries = set(f.read().split())

    def flush_libraries():
        if args.record_libraries is not None:
            with open(args.record_libraries, "w") as f:
                f.write("\n".join(libraries))

    with flutes.safe_pool(args.n_procs, closing=[db, flush_libraries]) as pool:
        iterator = iter_repos(db, args.repo_list_file, args.max_repos)
        pipeline_fn: Callable[[RepoInfo], Optional[PipelineResult]] = functools.partial(
            clone_and_compile,
            clone_folder=args.clone_folder, binary_folder=args.binary_folder, archive_folder=args.archive_folder,
            recursive_clone=args.recursive_clone,
            clone_timeout=args.clone_timeout, compile_timeout=args.compile_timeout,
            force_reclone=args.force_reclone, force_recompile=args.force_recompile,
            docker_batch_compile=args.docker_batch_compile,
            max_archive_size=args.max_archive_size, compression_type=args.compression_type,
            record_libraries=(args.record_libraries is not None), record_metainfo=args.record_metainfo,
            gcc_override_flags=args.gcc_override_flags)
        repo_count = 0
        meta_info = MetaInfo()
        for result in pool.imap_unordered(pipeline_fn, iterator):
            repo_count += 1
            if repo_count % 100 == 0:
                flutes.log(f"Processed {repo_count} repositories", force_console=True)
            if result is None:
                continue
            repo_owner, repo_name = result.repo_info.repo_owner, result.repo_info.repo_name
            if args.write_db:
                if result.clone_success is not None or result.repo_info.db_result is None:
                    # There's probably an inconsistency somewhere if we didn't clone while `db_result` is None.
                    # To prevent more errors, just add it to the DB.
                    repo_size = result.repo_size or -1  # a value of zero is probably also wrong
                    clone_success = result.clone_success if result.clone_success is not None else True
                    db.add_repo(repo_owner, repo_name, clone_success, repo_size=repo_size)
                    flutes.log(f"Added {repo_owner}/{repo_name} to DB")
                if result.makefiles is not None:
                    update_result = db.update_makefile(repo_owner, repo_name, result.makefiles,
                                                       ignore_length_mismatch=True)
                    if not update_result:
                        flutes.log(f"Makefiles of {repo_owner}/{repo_name} not saved to DB due to Unicode encoding "
                                   f"errors", "error")
            if result.libraries is not None:
                libraries.update(result.libraries)
                if repo_count % 10 == 0:  # flush every 10 repos
                    flush_libraries()

            if args.record_metainfo:
                meta_info.add_repo(result)
                if repo_count % 100 == 0:
                    flutes.log(repr(meta_info), force_console=True)

        flutes.log(repr(meta_info), force_console=True)


if __name__ == '__main__':
    main()
