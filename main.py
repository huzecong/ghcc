r"""Run the cloning--compilation pipeline. What happens is:

1. Repositories are cloned from GitHub according to the given list.
2. Successfully cloned repositories are scanned for Makefiles.
3. Each Makefile will be used for compilation, and results will be gathered.
4. If all Makefiles for a repository are successfully compiled, `make clean` will be called and the repository is
   zipped to save space.
"""

import argparse
import functools
import hashlib
import multiprocessing
import os
import shutil
import subprocess
from datetime import datetime

from typing import NamedTuple, Optional, Iterator, List

import ghcc
from ghcc import CloneErrorType

parser = argparse.ArgumentParser()
parser.add_argument("--repo-list-file", type=str, default="../c-repos.csv")
parser.add_argument("--clone-folder", type=str, default="c_repos/")  # where cloned repositories are stored
parser.add_argument("--binary-folder", type=str, default="binaries/")  # where compiled binaries are stored
parser.add_argument("--n-procs", type=int, default=32)
parser.add_argument("--log-file", type=str, default="clone-log.txt")
parser.add_argument("--clone-timeout", type=int, default=600)  # wait up to 10 minutes
parser.add_argument("--compile-timeout", type=int, default=900)  # wait up to 15 minutes
args = parser.parse_args()


class RepoInfo(NamedTuple):
    repo_owner: str
    repo_name: str
    db_result: Optional[ghcc.RepoEntry]


class PipelineResult(NamedTuple):
    repo_owner: str
    repo_name: str
    repo_entry: Optional[ghcc.RepoEntry] = None
    makefiles: Optional[List[ghcc.RepoMakefileEntry]] = None


def clone_and_compile(repo_info: RepoInfo, clone_folder: str, binary_folder: str,
                      clone_timeout: Optional[int] = None,
                      compile_timeout: Optional[int] = None) -> Optional[PipelineResult]:
    r"""Perform the entire pipeline.

    :param repo_info: Information about the repository.
    :param clone_folder: Path to the folder where the repository will be stored. The actual destination folder will be
        ``clone_folder/repo_owner/repo_name``, e.g., ``clone_folder/torvalds/linux``.
    :param clone_timeout: Timeout for cloning, or `None` (default) for unlimited time.
    :param compile_timeout: Timeout for compilation, or `None` (default) for unlimited time.
    :return: An entry to insert into the DB, or `None` if no operations are required.
    """
    repo_full_name = f"{repo_info.repo_owner}/{repo_info.repo_name}"
    repo_path = os.path.join(clone_folder, repo_full_name)

    repo_entry = repo_info.db_result

    # Stage 1: Cloning from GitHub.
    if repo_entry is None or (not repo_entry["compiled"] and not os.path.exists(repo_path)):
        clone_result = ghcc.clone(
            repo_info.repo_owner, repo_info.repo_name, clone_folder=clone_folder,
            timeout=clone_timeout, skip_if_exists=False)
        repo_entry_return = repo_entry = {
            "repo_owner": repo_info.repo_owner,
            "repo_name": repo_info.repo_name,
            "clone_time": datetime.now(),
            "repo_size": -1,
            "clone_successful": clone_result.success,
            "compiled": False,
            "makefiles": []
        }
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
            return PipelineResult(repo_info.repo_owner, repo_info.repo_name, repo_entry=repo_entry_return)
        else:
            ghcc.log(f"{repo_full_name} successfully cloned ({clone_result.time:.2f}s)", "success")
            repo_entry["repo_size"] = ghcc.utils.get_folder_size(repo_path)
    else:
        repo_entry_return = None
        if not repo_entry["clone_successful"]:
            return None

    if not repo_entry["compiled"]:
        # Stage 2: Finding Makefiles.
        makefiles = repo_entry["makefiles"]
        repo_entry["compiled"] = True
        if len(makefiles) == 0:
            makefiles = [{
                "directory": subdir,
                "compile_success": False,
                "binaries": [],
                "sha256": [],
            } for subdir in ghcc.find_makefiles(repo_path)]

            if len(makefiles) == 0:
                # Repo has no Makefiles, delete.
                shutil.rmtree(repo_path)
                ghcc.log(f"No Makefiles found in {repo_full_name}, repository deleted", "warning")
                return PipelineResult(repo_info.repo_owner, repo_info.repo_name,
                                      repo_entry=repo_entry_return, makefiles=[])
            else:
                # ghcc.log(f"Found {len(makefiles)} Makefiles in {repo_full_name}")
                pass

        # Stage 3: Compile each Makefile.
        repo_binary_dir = os.path.join(binary_folder, repo_full_name)
        if not os.path.exists(repo_binary_dir):
            os.makedirs(repo_binary_dir)
        succeeded_makefiles = []
        failed_makefiles = []
        for makefile in makefiles:
            compile_result = ghcc.make(makefile["directory"], timeout=compile_timeout)
            if compile_result.success:
                makefile["compile_success"] = True
                makefile["binaries"] = compile_result.elf_files
                succeeded_makefiles.append(makefile["directory"])
                # ghcc.log(f"Compiled Makefile in {makefile['directory']}, "
                #          f"yielding {len(compile_result.elf_files)} binaries", "success")
                for path in compile_result.elf_files:
                    hash_obj = hashlib.sha256()
                    with open(path, "rb") as f:
                        hash_obj.update(f.read())
                    digest = hash_obj.hexdigest()
                    makefile["sha256"].append(digest)
                    shutil.move(path, os.path.join(repo_binary_dir, digest))
            else:
                failed_makefiles.append(makefile["directory"])
                # ghcc.log(f"Failed to compile Makefile in {makefile['directory']}. "
                #          f"Captured output: '{compile_result.captured_output}'", "error")
        msg = f"{len(succeeded_makefiles)} out of {len(succeeded_makefiles) + len(failed_makefiles)} Makefile(s) " \
              f"in {repo_full_name} successfully compiled"
        if len(failed_makefiles) == 0:
            ghcc.log(msg, "success")
        else:
            ghcc.log(msg, "error")

        repo_entry['makefiles'] = makefiles

        # Stage 4: Clean and zip repo.
        ghcc.clean(repo_path)
        archive_path = os.path.join(clone_folder, f"{repo_full_name}.tar.gz")
        try:
            ghcc.utils.run_command(["tar", "cJf", archive_path, repo_path], timeout=clone_timeout)
            compress_success = True
        except subprocess.TimeoutExpired:
            ghcc.log(f"Compression timeout for {repo_full_name}, giving up", "warning")
            compress_success = False
            if os.path.exists(archive_path):
                os.remove(archive_path)
        except subprocess.CalledProcessError as e:
            ghcc.log(f"Unknown error when compression {repo_full_name}. Captured output: '{e.output}'", "warning")
            compress_success = False
        if compress_success:
            shutil.rmtree(repo_path)
            ghcc.log(f"{repo_full_name} compressed, folder removed", "info")
        elif os.path.exists(archive_path):
            os.remove(archive_path)

        return PipelineResult(repo_info.repo_owner, repo_info.repo_name,
                              repo_entry=repo_entry_return, makefiles=makefiles)

    else:
        return None


def iter_repos(db: ghcc.Database, repo_list_path: str) -> Iterator[RepoInfo]:
    with open(repo_list_path, "r") as repo_file:
        for line in repo_file:
            if not line:
                continue
            url = line.split()[1]
            repo_owner, repo_name = url.rstrip("/").split("/")[-2:]
            db_result = db.get(repo_owner, repo_name)
            # db_result = None
            yield RepoInfo(repo_owner, repo_name, db_result)


def main():
    ghcc.log("Crawling starts...", "warning")
    pool = multiprocessing.Pool(processes=args.n_procs)
    db = ghcc.Database()

    interrupted = False
    try:
        iterator = iter_repos(db, args.repo_list_file)
        pipeline_fn = functools.partial(
            clone_and_compile, clone_folder=args.clone_folder, binary_folder=args.binary_folder,
            clone_timeout=args.clone_timeout, compile_timeout=args.compile_timeout)
        repo_count = 0
        # for result in map(pipeline_fn, iterator):
        for result in pool.imap_unordered(pipeline_fn, iterator):
            if result is None:
                continue
            result: PipelineResult
            repo_owner, repo_name = result.repo_owner, result.repo_name
            if result.repo_entry is not None:
                db.add_repo(repo_owner, repo_name, result.repo_entry["clone_successful"],
                            result.repo_entry["clone_time"], result.repo_entry["repo_size"])
            if result.makefiles is not None:
                db.update_makefile(repo_owner, repo_name, result.makefiles)
            repo_count += 1
            if repo_count % 1000 == 0:
                ghcc.log(f"Processed {repo_count} repositories")

    except KeyboardInterrupt:
        interrupted = True
    finally:
        pool.close()
        if interrupted:
            print("Press Ctrl-C again to force terminate...")
            try:
                pool.join()
            except KeyboardInterrupt:
                pool.terminate()


if __name__ == '__main__':
    # ghcc.utils.register_ipython_excepthook()
    ghcc.set_log_file(args.log_file)
    main()
