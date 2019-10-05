import os
import shutil
import subprocess
import time
from enum import Enum, auto
from typing import Optional, NamedTuple

__all__ = [
    "CloneErrorType",
    "CloneResult",
    "clone"
]


class CloneErrorType(Enum):
    FolderExists = auto()
    Timeout = auto()
    PrivateOrNonexistent = auto()
    Unknown = auto()


class CloneResult(NamedTuple):
    repo_owner: str
    repo_name: str
    success: bool = False
    error_type: Optional[CloneErrorType] = None
    time: Optional[float] = None
    captured_output: Optional[bytes] = None


def clone(url: str, clone_folder: str, default_branch: Optional[str] = None,
          timeout: Optional[int] = None, skip_if_exists: bool = True) -> CloneResult:
    start_time = time.time()
    assert url.endswith(".git")
    repo_owner, repo_name = url[:-4].split("/")[-2:]
    full_name = "/".join(url.split("/")[-2:])
    # clone_folder / owner_name / repo_name
    folder_path = os.path.join(clone_folder, full_name)
    if os.path.exists(folder_path):
        if not skip_if_exists:
            shutil.rmtree(folder_path)
        else:
            return CloneResult(repo_owner, repo_name, error_type=CloneErrorType.FolderExists)

    # Certain repos might have turned private or been deleted, and git prompts for username/password when it happens.
    # Setting the environment variable `GIT_TERMINAL_PROMPT` to 0 could disable such behavior and let git fail promptly.
    # Lucky that this is introduced in version 2.3; otherwise would have to poll waiting channel of current process
    # and see if it's waiting for IO.
    # See: https://askubuntu.com/questions/19442/what-is-the-waiting-channel-of-a-process
    env = {b"GIT_TERMINAL_PROMPT": b"0"}

    def try_clone():
        # If a true git error was thrown, re-raise it and let the outer code deal with it.
        try:
            try_branch = default_branch or "master"
            # Try cloning only 'master' branch, but it's possible there's no branch named 'master'.
            subprocess.check_output(
                ["git", "clone", "--depth=1", f"--branch={try_branch}", "--single-branch", url, folder_path],
                env=env, stderr=subprocess.STDOUT, timeout=timeout)
            return
        except subprocess.CalledProcessError as err:
            expected_msg = b"fatal: Remote branch master not found in upstream origin"
            if default_branch is not None or expected_msg not in err.output:
                # If `default_branch` is specified, always re-raise the exception.
                raise err
        # 'master' branch doesn't exist; do a shallow clone of all branches.
        subprocess.check_output(
            ["git", "clone", "--depth=1", url, folder_path],
            env=env, stderr=subprocess.STDOUT)

    try:
        try_clone()
        end_time = time.time()
        elapsed_time = end_time - start_time
        return CloneResult(repo_owner, repo_name, success=True, time=elapsed_time)
    except subprocess.CalledProcessError as e:
        missing_msg = b"fatal: could not read Username for 'https://github.com': terminal prompts disabled"
        if e.output is not None and missing_msg in e.output:
            return CloneResult(repo_owner, repo_name, error_type=CloneErrorType.PrivateOrNonexistent)
        else:
            return CloneResult(repo_owner, repo_name, error_type=CloneErrorType.Unknown, captured_output=e.output)
    except subprocess.TimeoutExpired as e:
        return CloneResult(repo_owner, repo_name, error_type=CloneErrorType.Timeout, captured_output=e.output)
