import os
import shutil
import subprocess
import time
from enum import Enum, auto
from typing import NamedTuple, Optional

from ghcc.utils import run_command

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


def clean(repo_folder: str):
    r"""Clean all unversioned files in a Git repository.

    :param repo_folder: Path to the Git repository.
    """
    run_command(["git", "clean", "-f", "-x", "-d"], cwd=repo_folder)


def clone(repo_owner: str, repo_name: str, clone_folder: str, default_branch: Optional[str] = None,
          timeout: Optional[int] = None, skip_if_exists: bool = True) -> CloneResult:
    r"""Clone a repository on GitHub, for instance, ``torvalds/linux``.

    :param repo_owner: Name of the repository owner, e.g., ``torvalds``.
    :param repo_name: Name of the repository, e.g., ``linux``.
    :param clone_folder: Path to the folder where the repository will be stored. The actual destination folder will be
        ``clone_folder/repo_owner/repo_name``, e.g., ``clone_folder/torvalds/linux``.
    :param default_branch: Name of the default branch of the repository. Cloning behavior differs slightly depending on
        whether the argument is ``None``. If ``None``, then the following happens:

        1. Attempts a shallow clone on only the ``master`` branch.
        2. If error occurs, attempts a shallow clone for all branches.
        3. If error still occurs, raise the error.

        If not ``None``, then the following happens:

        1. Attempts a shallow clone on only the default branch.
        2. If error occurs, raise the error.
    :param timeout: Maximum time allowed for cloning, in seconds. Defaults to ``None`` (unlimited time).
    :param skip_if_exists: Whether to skip cloning if the destination folder already exists. If ``False``, the folder
        will be deleted.

    :return: An instance of :class:`CloneResult` indicating the result. Fields ``repo_owner``, ``repo_name``, and
        ``success`` are not ``None``.

        - If cloning succeeded, the field ``time`` is also not ``None``.
        - If cloning failed, the fields ``error_type`` and ``captured_output`` are also not ``None``.
    """
    start_time = time.time()
    url = f"https://github.com/{repo_owner}/{repo_name}.git"
    # clone_folder / repo_owner / repo_name
    folder_path = os.path.join(clone_folder, repo_owner, repo_name)
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
            run_command(
                ["git", "clone", "--depth=1", f"--branch={try_branch}", "--single-branch", url, folder_path],
                env=env, timeout=timeout)
            return
        except subprocess.CalledProcessError as err:
            expected_msg = b"fatal: Remote branch master not found in upstream origin"
            if default_branch is not None or (err.output is not None and expected_msg not in err.output):
                # If `default_branch` is specified, always re-raise the exception.
                raise err
        # 'master' branch doesn't exist; do a shallow clone of all branches.
        run_command(
            ["git", "clone", "--depth=1", url, folder_path],
            env=env, timeout=timeout)

    try:
        try_clone()
        end_time = time.time()
        elapsed_time = end_time - start_time
        return CloneResult(repo_owner, repo_name, success=True, time=elapsed_time)
    except subprocess.CalledProcessError as e:
        no_ssh_expected_msg = b"fatal: could not read Username for 'https://github.com': terminal prompts disabled"
        ssh_expected_msg = b"remote: Repository not found."
        if e.output is not None and (no_ssh_expected_msg in e.output or ssh_expected_msg in e.output):
            return CloneResult(repo_owner, repo_name, error_type=CloneErrorType.PrivateOrNonexistent)
        else:
            return CloneResult(repo_owner, repo_name, error_type=CloneErrorType.Unknown, captured_output=e.output)
    except subprocess.TimeoutExpired as e:
        return CloneResult(repo_owner, repo_name, error_type=CloneErrorType.Timeout, captured_output=e.output)
