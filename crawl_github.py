import argparse
import json
import multiprocessing
import time

import requests

import ghcc
from ghcc.logging import log
from ghcc.repo import CloneErrorType, CloneResult, clone

parser = argparse.ArgumentParser()
parser.add_argument("--clone-folder", type=str, default="repos/")
parser.add_argument("--repos-per-page", type=int, default=100)
parser.add_argument("--n-procs", type=int, default=32)
args = parser.parse_args()


def get_repos(n_repos=None, skip_counts=None):
    page_num = 1 if skip_counts is None else skip_counts // args.repos_per_page
    repo_count = 0
    start_idx = 0 if skip_counts is None else skip_counts % args.repos_per_page
    while n_repos is None or repo_count < n_repos:
        try:
            response = requests.get(
                "https://api.github.com/search/repositories",
                params='&'.join(f"{k}={v}" for k, v in {
                    "q": "language:C",
                    "sort": "stars",
                    "page": page_num,
                    "per_page": args.repos_per_page,
                }.items()),
                # headers={
                #     "Authorization": "token b25514f16dc55e21df8b875a4209d9a76de20ae9",
                # }
            )
        except requests.exceptions.RequestException:
            log("GitHub API failed; retry after 10s", "warning")
            time.sleep(10)
            continue
        j = json.loads(response.content)
        try:
            if n_repos is None:
                n_repos = j["total_count"]
            repo_slice = j["items"][start_idx:(start_idx + n_repos - repo_count)]
        except KeyError:
            if "message" in j:
                log("GitHub API failed with message: " + j["message"] + "; retry after 10s", "warning")
            else:
                log("GitHub API failed; retry after 10s", "warning")
            time.sleep(10)
            continue

        yield repo_slice
        repo_count += len(repo_slice)
        page_num += 1
        start_idx = 0


def log_output(result: CloneResult):
    full_name = f"{result.repo_owner}/{result.repo_name}"
    if result.success:
        log(f"{full_name} successfully cloned ({result.time:.2f}s)", "success")
    elif result.error_type is CloneErrorType.FolderExists:
        log(f"{full_name} skipped because folder exists", "warning")
    else:
        log(f"Failed to clone {full_name}: {result.output}", "error")


def clone_repo(repo):
    url = repo["clone_url"]
    default_branch = repo["default_branch"]
    return clone(url, args.clone_folder, default_branch=default_branch)


def main():
    log("Crawling starts...", "warning")
    repo_count = 0
    with open("metadata.jsonl", "r") as f:
        repos = [json.loads(line) for line in f if line]
    pool = multiprocessing.Pool(processes=args.n_procs)
    for result in pool.imap_unordered(clone_repo, repos):
        repo_count += 1
        log_output(result)

    meta_file = open("metadata.jsonl", "a")
    try:
        for repos in get_repos(None, skip_counts=len(repos)):
            for repo in repos:
                meta_file.write(json.dumps(repo) + '\n')
            meta_file.flush()
            for result in pool.imap_unordered(clone_repo, repos):
                repo_count += 1
                log_output(result)
    finally:
        meta_file.close()
        pool.close()


if __name__ == '__main__':
    ghcc.utils.register_pdb_excepthook()
    main()
