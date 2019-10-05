import argparse
import functools
import multiprocessing

import ghcc
from ghcc.logging import log
from ghcc.repo import CloneErrorType, CloneResult, clone

parser = argparse.ArgumentParser()
parser.add_argument("--repo-list-file", type=str, default="../c-repos.csv")
parser.add_argument("--clone-folder", type=str, default="c_repos/")
parser.add_argument("--n-procs", type=int, default=32)
parser.add_argument("--log-file", type=str, default="clone-log.txt")
parser.add_argument("--timeout", type=int, default=600)  # wait up to 10 minutes
args = parser.parse_args()


def main():
    log("Crawling starts...", "warning")
    repo_count = 0
    pool = multiprocessing.Pool(processes=args.n_procs)

    def iter_repos():
        with open(args.repo_list_file, "r") as repo_file:
            for line in repo_file:
                if not line:
                    continue
                url = line.split()[1]
                url = url.replace("https://api.github.com/repos/", "https://github.com/").rstrip("/")
                url = url + ".git"
                yield url

    interrupted = False
    try:
        clone_repo = functools.partial(clone, clone_folder=args.clone_folder, timeout=args.timeout, skip_if_exists=True)
        for result in pool.imap_unordered(clone_repo, iter_repos()):
            result: CloneResult
            repo_count += 1
            full_name = f"{result.repo_owner}/{result.repo_name}"
            if result.success:
                log(f"{full_name} successfully cloned ({result.time:.2f}s)", "success")
            elif result.error_type is CloneErrorType.FolderExists:
                log(f"{full_name} skipped because folder exists", "warning")
            elif result.error_type is CloneErrorType.PrivateOrNonexistent:
                log(f"Failed to clone {full_name} because repository has turned private or been deleted", "error")
            else:
                if result.error_type is CloneErrorType.Unknown:
                    msg = f"Failed to clone {full_name} with unknown error"
                else:  # CloneErrorType.Timeout:
                    msg = f"Time expired ({args.timeout}s) when attempting to clone clone {full_name}"
                if result.captured_output is not None:
                    msg += f". Captured output: '{result.captured_output}'"
                log(msg, "error")
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
    ghcc.utils.register_pdb_excepthook()
    main()
