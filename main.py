r"""Run the cloning--compilation pipeline. What happens is:

1. Repositories are cloned from GitHub according to the given list.
2. Successfully cloned repositories are scanned for Makefiles.
3. Each Makefile will be used for compilation, and results will be gathered.
4. Compilation products are cleaned and the repository is archived to save space.
"""

import argparse
import functools
import hashlib
import multiprocessing
import os
import shutil
import subprocess
import time
from typing import Iterator, List, NamedTuple, Optional, Tuple

from mypy_extensions import TypedDict

import ghcc
from ghcc import CloneErrorType


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-list-file", type=str, default="../c-repos.csv")
    parser.add_argument("--clone-folder", type=str,
                        default="repos/")  # where cloned repositories are stored (temporarily)
    parser.add_argument("--binary-folder", type=str, default="binaries/")  # where compiled binaries are stored
    parser.add_argument("--archive-folder", type=str, default="archives/")  # where archived repositories are stored
    parser.add_argument("--n-procs", type=int, default=32)
    parser.add_argument("--log-file", type=str, default="log.txt")
    parser.add_argument("--clone-timeout", type=int, default=600)  # wait up to 10 minutes
    parser.add_argument("--compile-timeout", type=int, default=900)  # wait up to 15 minutes
    parser.add_argument("--force-recompile", action="store_true", default=False)
    parser.add_argument("--docker-batch-compile", action="store_true", default=False)
    args = parser.parse_args()
    return args


class RepoInfo(NamedTuple):
    repo_owner: str
    repo_name: str
    db_result: Optional[ghcc.RepoEntry]


class PipelineResult(NamedTuple):
    repo_owner: str
    repo_name: str
    clone_success: Optional[bool] = None
    makefiles: Optional[List[ghcc.RepoMakefileEntry]] = None


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


class MakefileInfo(TypedDict):
    directory: str
    binaries: List[str]
    sha256: List[str]


RepoCompileResult = Tuple[int, List[MakefileInfo]]


def _compile_one_by_one(repo_binary_dir: str, makefile_dirs: List[str],
                        compile_timeout: Optional[int]) -> RepoCompileResult:
    num_succeeded = 0
    makefiles: List[MakefileInfo] = []
    remaining_time = compile_timeout
    for make_dir in makefile_dirs:
        if remaining_time <= 0.0:
            break
        start_time = time.time()
        compile_result = ghcc.docker_make(make_dir, timeout=remaining_time)
        elapsed_time = time.time() - start_time
        remaining_time -= elapsed_time
        if compile_result.success:
            num_succeeded += 1
        if len(compile_result.elf_files) > 0:
            # Failed compilations may also yield binaries.
            sha256: List[str] = []
            for path in compile_result.elf_files:
                hash_obj = hashlib.sha256()
                with open(path, "rb") as f:
                    hash_obj.update(f.read())
                digest = hash_obj.hexdigest()
                sha256.append(digest)
                shutil.move(path, os.path.join(repo_binary_dir, digest))
            makefiles.append({
                "directory": make_dir,
                "binaries": compile_result.elf_files,
                "sha256": sha256,
            })
    return num_succeeded, makefiles


def _docker_batch_compile(repo_binary_dir: str, repo_path: str, compile_timeout: Optional[int]) -> RepoCompileResult:
    try:
        ret = ghcc.run_docker_command([
            "batch_make.py",
            *(["--compile-timeout", str(compile_timeout)] if compile_timeout is not None else [])],
            directory_mapping={repo_path: "/usr/src/repo", repo_binary_dir: "/usr/src/bin"})
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        pass
    with open(os.path.join(repo_binary_dir, "log.txt")) as f:
        lines = [line.strip() for line in f if line]
    os.remove(os.path.join(repo_binary_dir, "log.txt"))
    num_succeeded = int(lines[0])
    num_makefiles = int(lines[1])
    makefiles: List[MakefileInfo] = []
    p = 2
    for _ in range(num_makefiles):
        pos = lines[p].find(' ')
        n_binaries, directory = int(lines[p][:pos]), lines[p][(pos + 1):]
        sha256, binaries = [], []
        for line in lines[(p + 1):(p + n_binaries + 1)]:
            pos = line.find(' ')
            sha, path = line[:pos], line[(pos + 1):]
            sha256.append(sha)
            binaries.append(path)
        makefiles.append({
            "directory": directory,
            "binaries": binaries,
            "sha256": sha256,
        })
    return num_succeeded, makefiles


def clone_and_compile(repo_info: RepoInfo, clone_folder: str, binary_folder: str, archive_folder: str,
                      clone_timeout: Optional[int] = None, compile_timeout: Optional[int] = None,
                      force_recompile: bool = False, docker_batch_compile: bool = True) -> Optional[PipelineResult]:
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
    :param clone_timeout: Timeout for cloning, or `None` (default) for unlimited time.
    :param compile_timeout: Timeout for compilation, or `None` (default) for unlimited time.
    :param force_recompile: If ``True``, the repository is compiled regardless of the value in DB.
    :param docker_batch_compile: If ``True``, compile all Makefiles within a repository in a single Docker container.
    :return: An entry to insert into the DB, or `None` if no operations are required.
    """
    repo_full_name = f"{repo_info.repo_owner}/{repo_info.repo_name}"
    repo_folder_name = f"{repo_info.repo_owner}_____{repo_info.repo_name}"
    repo_path = os.path.join(clone_folder, repo_folder_name)
    archive_path = os.path.join(archive_folder, f"{repo_full_name}.tar.xz")

    repo_entry = repo_info.db_result
    clone_success = None

    # Stage 1: Cloning from GitHub.
    if (repo_entry is None or  # not processed
            (repo_entry["clone_successful"] and  # not compiled
             (not repo_entry["compiled"] or force_recompile) and not os.path.exists(repo_path))):
        # TODO: Try extracting archive if it exists.
        clone_result = ghcc.clone(
            repo_info.repo_owner, repo_info.repo_name, clone_folder=clone_folder, folder_name=repo_folder_name,
            timeout=clone_timeout, skip_if_exists=False)
        clone_success = clone_result.success
        if not clone_result.success:
            if clone_result.error_type is CloneErrorType.FolderExists:
                ghcc.log(f"{repo_full_name} skipped because folder exists", "warning")
            elif clone_result.error_type is CloneErrorType.PrivateOrNonexistent:
                ghcc.log(f"Failed to clone {repo_full_name} because repository is private or nonexistent", "warning")
            else:
                if clone_result.error_type is CloneErrorType.Unknown:
                    msg = f"Failed to clone {repo_full_name} with unknown error"
                else:  # CloneErrorType.Timeout:
                    msg = f"Time expired ({clone_timeout}s) when attempting to clone {repo_full_name}"
                if clone_result.captured_output is not None:
                    msg += f". Captured output: '{clone_result.captured_output}'"
                ghcc.log(msg, "error")

                if clone_result.error_type is CloneErrorType.Unknown:
                    return None  # so we can try again in the future

            return PipelineResult(repo_info.repo_owner, repo_info.repo_name, clone_success=clone_success)
        else:
            ghcc.log(f"{repo_full_name} successfully cloned ({clone_result.time:.2f}s)", "success")
    else:
        if not repo_entry["clone_successful"]:
            return None

    makefiles = None
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
        #     return PipelineResult(repo_info.repo_owner, repo_info.repo_name,
        #                           clone_success=clone_success, makefiles=[])

        # Stage 2: Finding Makefiles.
        makefile_dirs = ghcc.find_makefiles(repo_path)
        if len(makefile_dirs) == 0:
            # Repo has no Makefiles, delete.
            shutil.rmtree(repo_path)
            ghcc.log(f"No Makefiles found in {repo_full_name}, repository deleted", "warning")
            return PipelineResult(repo_info.repo_owner, repo_info.repo_name, clone_success=clone_success, makefiles=[])
        else:
            # ghcc.log(f"Found {len(makefiles)} Makefiles in {repo_full_name}")
            pass

        # Stage 3: Compile each Makefile.
        repo_binary_dir = os.path.join(binary_folder, repo_full_name)
        if not os.path.exists(repo_binary_dir):
            os.makedirs(repo_binary_dir)
        ghcc.log(f"Starting compilation for {repo_full_name}...")

        if docker_batch_compile:
            num_succeeded, makefiles = _docker_batch_compile(repo_binary_dir, repo_path, compile_timeout)
        else:
            num_succeeded, makefiles = _compile_one_by_one(repo_binary_dir, makefile_dirs, compile_timeout)
        num_binaries = sum(len(makefile["binaries"]) for makefile in makefiles)

        msg = f"{num_succeeded} ({len(makefiles)}) out of {len(makefile_dirs)} Makefile(s) in {repo_full_name} " \
              f"compiled (partially), yielding {num_binaries} binaries"
        ghcc.log(msg, "success" if num_succeeded == len(makefile_dirs) else "warning")

        # Stage 4: Clean and zip repo.
        ghcc.clean(repo_path)
        os.makedirs(os.path.split(archive_path)[0], exist_ok=True)
        try:
            ghcc.utils.run_command(["tar", "cJf", archive_path, repo_path], timeout=clone_timeout)
            compress_success = True
        except subprocess.TimeoutExpired:
            ghcc.log(f"Compression timeout for {repo_full_name}, giving up", "error")
            compress_success = False
            if os.path.exists(archive_path):
                os.remove(archive_path)
        except subprocess.CalledProcessError as e:
            ghcc.log(f"Unknown error when compression {repo_full_name}. Captured output: '{e.output}'", "error")
            compress_success = False
        if compress_success:
            shutil.rmtree(repo_path)
            ghcc.log(f"{repo_full_name} compressed, folder removed", "info")
        elif os.path.exists(archive_path):
            os.remove(archive_path)

    return PipelineResult(repo_info.repo_owner, repo_info.repo_name,
                          clone_success=clone_success, makefiles=makefiles)


def iter_repos(db: ghcc.Database, repo_list_path: str) -> Iterator[RepoInfo]:
    db_entries = {
        (entry["repo_owner"], entry["repo_name"]): entry
        for entry in db.collection.find()
    }
    ghcc.log("All entries loaded from DB")
    with open(repo_list_path, "r") as repo_file:
        for line in repo_file:
            if not line:
                continue
            url = line.split()[1]
            repo_owner, repo_name = url.rstrip("/").split("/")[-2:]
            # db_result = db.get(repo_owner, repo_name)
            db_result = db_entries.get((repo_owner, repo_name), None)
            yield RepoInfo(repo_owner, repo_name, db_result)


def main():
    args = parse_args()
    print(args)
    ghcc.set_log_file(args.log_file)

    ghcc.log("Crawling starts...", "warning")
    pool = multiprocessing.Pool(processes=args.n_procs)
    db = ghcc.Database()

    try:
        iterator = iter_repos(db, args.repo_list_file)
        pipeline_fn = functools.partial(
            clone_and_compile,
            clone_folder=args.clone_folder, binary_folder=args.binary_folder, archive_folder=args.archive_folder,
            clone_timeout=args.clone_timeout, compile_timeout=args.compile_timeout,
            force_recompile=args.force_recompile, docker_batch_compile=args.docker_batch_compile)
        repo_count = 0
        # for result in map(pipeline_fn, iterator):
        for result in pool.imap_unordered(pipeline_fn, iterator):
            repo_count += 1
            if repo_count % 100 == 0:
                ghcc.log(f"Processed {repo_count} repositories")
            if result is None:
                continue
            result: PipelineResult
            repo_owner, repo_name = result.repo_owner, result.repo_name
            if result.clone_success is not None:
                db.add_repo(repo_owner, repo_name, result.clone_success)
            if result.makefiles is not None:
                try:
                    db.update_makefile(repo_owner, repo_name, result.makefiles)
                except ValueError as e:  # mismatching number of makefiles
                    if not args.force_recompile:
                        raise e

    except KeyboardInterrupt:
        print("Press Ctrl-C again to force terminate...")
    except BlockingIOError:
        pool.close()
        pool.terminate()
        ghcc.utils.kill_proc_tree(os.getpid())  # commit suicide
    finally:
        pool.close()
        try:
            pool.join()
        except KeyboardInterrupt:
            pool.terminate()
            ghcc.utils.kill_proc_tree(os.getpid())  # commit suicide


if __name__ == '__main__':
    # ghcc.utils.register_ipython_excepthook()
    main()
