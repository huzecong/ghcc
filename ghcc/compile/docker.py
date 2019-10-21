import os
import subprocess
from typing import Dict, List, Optional, Tuple, Union

from ghcc.utils import CommandResult, run_command

__all__ = [
    "run_docker_command",
]


def run_docker_command(command: Union[str, List[str]], cwd: Optional[str] = None,
                       user: Optional[Union[int, Tuple[int, int]]] = None,
                       directory_mapping: Optional[Dict[str, str]] = None,
                       timeout: Optional[int] = None, **kwargs) -> CommandResult:
    r"""Run a command inside a container based on the ``gcc-custom`` Docker image.

    :param command: The command to run. Should be either a `str` or a list of `str`. Note: they're treated the same way,
        because a shell is always spawn in the entry point.
    :param cwd: The working directory of the command to run. If None, uses the default (probably user home).
    :param user: The user ID to use inside the Docker container. Additionally, group ID can be specified by passing
        a tuple of two `int`\ s for this argument. If not specified, the current user and group IDs are used.
    :param directory_mapping: Mapping of host directories to container paths. Mapping is performed via "bind mount".
    :param timeout: Maximum running time for the command. If running time exceeds the specified limit,
        ``subprocess.TimeoutExpired`` is thrown.
    :param kwargs: Additional keyword arguments to pass to :meth:`ghcc.utils.run_command`.
    """
    # Validate `command` argument, and append call to `bash` if `shell` is True.
    if isinstance(command, list):
        command = ' '.join(command)
    command = f"'{command}'"

    # Assign user and group IDs based on `user` argument.
    user_id: Union[str, int] = "`id -u $USER`"
    group_id: Union[str, int] = "`id -g $USER`"
    if user is not None:
        if isinstance(user, tuple):
            user_id, group_id = user
        else:
            user_id = user

    # Construct the `docker run` command.
    docker_command = ["docker", "run", "--rm"]
    for host, container in (directory_mapping or {}).items():
        docker_command.extend(["-v", f"{os.path.abspath(host)}:{container}"])
    if cwd is not None:
        docker_command.extend(["-w", cwd])
    docker_command.extend(["-e", f"LOCAL_USER_ID={user_id}"])
    docker_command.extend(["-e", f"LOCAL_GROUP_ID={group_id}"])
    docker_command.append("gcc-custom")
    if timeout is not None:
        # Timeout is implemented by calling `timeout` inside Docker container.
        docker_command.extend(["timeout", f"{timeout}s"])
    docker_command.append(command)
    ret = run_command(' '.join(docker_command), shell=True, **kwargs)

    # Check whether exceeded timeout limit by inspecting return code.
    if ret.return_code == 124:
        raise subprocess.TimeoutExpired(ret.command, timeout, output=ret.captured_output)
    return ret
