r"""Run the cloning--compilation pipeline. What happens is:

1. Repositories are cloned from GitHub according to the given list.
2. Successfully cloned repositories are scanned for Makefiles.
3. Each Makefile will be used for compilation, and results will be gathered.
4. If all Makefiles for a repository are successfully compiled, `make clean` will be called and the repository is
   zipped to save space.
"""

import argparse
import functools
import multiprocessing
import os

from typing import NamedTuple, Optional, Iterator

import ghcc
from ghcc import CloneErrorType

parser = argparse.ArgumentParser()
parser.add_argument("--repo-list-file", type=str, default="../c-repos.csv")
parser.add_argument("--clone-folder", type=str, default="c_repos/")  # where cloned repositories are stored
parser.add_argument("--binary-folder", type=str, defualt="binaries/")  # where compiled binaries are stored
parser.add_argument("--n-procs", type=int, default=32)
parser.add_argument("--log-file", type=str, default="clone-log.txt")
parser.add_argument("--clone-timeout", type=int, default=600)  # wait up to 10 minutes
parser.add_argument("--compile-timeout", type=int, default=900)  # wait up to 15 minutes
args = parser.parse_args()


class RepoInfo(NamedTuple):
    repo_owner: str
    repo_name: str
    db_result: Optional[ghcc.RepoEntry]


def clone_and_compile(repo_info: RepoInfo, clone_folder: str,
                      clone_timeout: Optional[int] = None,
                      compile_timeout: Optional[int] = None) -> Optional[ghcc.RepoEntry]:
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

    # Stage 1: Cloning from GitHub.
    if repo_info.db_result is None:
        clone_result = ghcc.clone(
            repo_info.repo_owner, repo_info.repo_name, clone_folder=clone_folder,
            timeout=clone_timeout, skip_if_exists=False)
        if not clone_result.success:
            if clone_result.error_type is CloneErrorType.FolderExists:
                ghcc.log(f"{repo_full_name} skipped because folder exists", "warning")
            elif clone_result.error_type is CloneErrorType.PrivateOrNonexistent:
                ghcc.log(f"Failed to clone {repo_full_name} because repository is private or nonexistent", "error")
            else:
                if clone_result.error_type is CloneErrorType.Unknown:
                    msg = f"Failed to clone {repo_full_name} with unknown error"
                else:  # CloneErrorType.Timeout:
                    msg = f"Time expired ({clone_timeout}s) when attempting to clone {repo_full_name}"
                if clone_result.captured_output is not None:
                    msg += f". Captured output: '{clone_result.captured_output}'"
                ghcc.log(msg, "error")
            return {
                "repo_owner": repo_info.repo_owner,
                "repo_name": repo_info.repo_name,
                "clone_successful": False,
            }
        else:
            ghcc.log(f"{repo_full_name} successfully cloned ({clone_result.time:.2f}s)", "success")
    elif not repo_info.db_result.clone_successful:
        return None

    if not repo_info.db_result.compiled:
        # Stage 2: Finding Makefiles.
        makefiles = repo_info.db_result.makefiles
        if len(makefiles) == 0:
            makefiles = [{
                "directory": subdir,
                "name": name,
                "compile_succeeded": False,
                "source_files": [],
                "output_files": [],
            } for subdir, name in ghcc.find_makefiles(repo_path)]
            ghcc.log(f"Found {len(makefiles)} in {repo_full_name}")

        # Stage 3: Compile each Makefile.
        for makefile in makefiles:
            ghcc.make(makefile["directory"], makefile["name"], timeout=compile_timeout)


def iter_repos(db: ghcc.Database, repo_list_path: str) -> Iterator[RepoInfo]:
    with open(repo_list_path, "r") as repo_file:
        for line in repo_file:
            if not line:
                continue
            url = line.split()[1]
            repo_owner, repo_name = url.rstrip("/").split("/")[-2:]
            db_result = db.get(repo_owner, repo_name)
            yield RepoInfo(repo_owner, repo_name, db_result)


def main():
    ghcc.log("Crawling starts...", "warning")
    pool = multiprocessing.Pool(processes=args.n_procs)
    db = ghcc.Database()

    interrupted = False
    try:
        iterator = iter_repos(db, args.repo_list_file)
        pipeline_fn = functools.partial(
            clone_and_compile, clone_folder=args.clone_folder,
            clone_timeout=args.clone_timeout, compile_timeout=args.compile_timeout)
        for _ in pool.imap_unordered(pipeline_fn, iterator):
            pass
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
    ghcc.utils.register_ipython_excepthook()
    main()
