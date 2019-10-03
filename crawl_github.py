import json
import os
import shutil
import subprocess
import time

import requests

CLONE_FOLDER = "repos/"
PER_PAGE = 40


def get_repos(n_repos, skip_counts=None):
    page_num = 1 if skip_counts is None else skip_counts // 40
    repo_count = 0
    start_idx = 0 if skip_counts is None else skip_counts % 40
    while repo_count < n_repos:
        try:
            response = requests.get(
                "https://api.github.com/search/repositories",
                params='&'.join(f"{k}={v}" for k, v in {
                    "q": "language:C",
                    "sort": "stars",
                    "page": page_num,
                    "per_page": PER_PAGE,
                }.items()),
                # headers={
                #     "Authorization": "token b25514f16dc55e21df8b875a4209d9a76de20ae9",
                # }
            )
        except requests.exceptions.RequestException:
            time.sleep(10)
            continue
        # import pdb; pdb.set_trace()
        j = json.loads(response.content)

        for repo in j["items"][start_idx:]:
            yield repo
            repo_count += 1
            if repo_count >= n_repos:
                break
        page_num += 1
        start_idx = 0


def clone_repo(repo, verbose=False, skip_if_exists=False):
    url = repo["clone_url"]
    full_name = repo["full_name"]
    # clone_folder / owner_name / repo_name
    folder_path = os.path.join(CLONE_FOLDER, full_name)
    if os.path.exists(folder_path):
        if not skip_if_exists:
            shutil.rmtree(folder_path)
        else:
            print("skipped.")
            return False

    output = subprocess.check_output(
        ["git", "clone", "--depth=1", url, folder_path],
        stderr=subprocess.STDOUT)
    if verbose:
        print(output)
    return True


def main():
    meta_file = open("metadata.jsonl", "w")
    try:
        for idx, repo in enumerate(get_repos(4000)):
            print(f"{idx}: Cloning {repo['full_name']}...", end=' ', flush=True)
            meta_file.write(json.dumps(repo) + '\n')
            meta_file.flush()
            if clone_repo(repo, skip_if_exists=True):
                print("done.")
    finally:
        meta_file.close()


if __name__ == '__main__':
    main()
