import itertools
import os
import subprocess
from typing import Dict, List, Optional, Union

from ghcc.utils import CommandResult, run_command

__all__ = [
    "run_docker_command",
]


def run_docker_command(command: Union[str, List[str]], cwd: Optional[str] = None,
                       directory_mapping: Optional[Dict[str, str]] = None,
                       timeout: Optional[int] = None, shell: bool = False, **kwargs) -> CommandResult:
    if shell:
        if not isinstance(command, str):
            raise ValueError("'command' must be str when shell=True")
        command = ["bash", "-c", f"\"{command}\""]
    else:
        if not isinstance(command, list):
            raise ValueError("'command' must be list of str when shell=False")
    # cid_file_path = os.path.join(directory, ".docker.cid")
    ret = run_command(' '.join([
        "docker", "run", "--rm",
        *itertools.chain.from_iterable(
            ["-v", f"{os.path.abspath(host)}:{container}"] for host, container in (directory_mapping or {}).items()),
        *(["-w", cwd] if cwd is not None else []),
        # "--cidfile", cid_file_path,
        "-e", "LOCAL_USER_ID=`id -u $USER`",
        "gcc-custom",
        *(["timeout", f"{timeout}s"] if timeout is not None else []),
        *command]), shell=True, **kwargs)
    if ret.return_code == 124:  # timeout
        raise subprocess.TimeoutExpired(ret.command, timeout, output=ret.captured_output)
    return ret
