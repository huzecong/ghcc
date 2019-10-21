import subprocess
import sys
import tempfile
from typing import Any, Dict, List, NamedTuple, Optional, Type, TypeVar, Union

import psutil
import tenacity

from ghcc.logging import log

__all__ = [
    "CommandResult",
    "run_command",
    "get_folder_size",
    "get_file_lines",
    "register_ipython_excepthook",
    "to_dict",
    "to_namedtuple",
]


def _run_command_retry_logger(retry_state: tenacity.RetryCallState) -> None:
    args = retry_state.args[0] if len(retry_state.args) > 0 else retry_state.kwargs['args']
    if isinstance(args, list):
        args = ' '.join(args)
    cwd = retry_state.args[2] if len(retry_state.args) > 2 else retry_state.kwargs.get('cwd', None)
    msg = f"{retry_state.attempt_number} failed attempt(s) for command: '{args}'"
    if cwd is not None:
        msg += f" in working directory '{cwd}'"
    log(msg, "warning")


class CommandResult(NamedTuple):
    command: List[str]
    return_code: int
    captured_output: Optional[bytes]

@tenacity.retry(retry=tenacity.retry_if_exception_type(OSError), reraise=True,
                stop=tenacity.stop_after_attempt(6),  # retry 5 times
                wait=tenacity.wait_random_exponential(multiplier=2, max=60),
                before_sleep=_run_command_retry_logger)
def run_command(args: Union[str, List[str]], env: Optional[Dict[bytes, bytes]] = None, cwd: Optional[str] = None,
                timeout: Optional[int] = None, return_output: bool = False, **kwargs) -> CommandResult:
    r"""A wrapper over ``subprocess.check_output`` that prevents deadlock caused by the combination of pipes and
    timeout. Output is redirected into a temporary file and returned only on exceptions.

    In case an OSError occurs, the function will retry for a maximum for 5 times with exponential backoff. If error
    still occurs, we just throw it up.

    :param args: The command to run. Should be either a `str` or a list of `str` depending on whether ``shell`` is True.
    :param env: Environment variables to set before running the command. Defaults to None.
    :param cwd: The working directory of the command to run. If None, uses the default (probably user home).
    :param timeout: Maximum running time for the command. If running time exceeds the specified limit,
        ``subprocess.TimeoutExpired`` is thrown.
    :param return_output: If ``True``, the captured output is returned. Otherwise, the return code is returned.
    """
    with tempfile.TemporaryFile() as f:
        try:
            ret = subprocess.run(args, check=True, stdout=f, stderr=subprocess.STDOUT,
                                 timeout=timeout, env=env, cwd=cwd, **kwargs)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            f.seek(0)
            e.output = f.read()
            raise e from None
        if return_output or ret.returncode != 0:
            f.seek(0)
            return CommandResult(args, ret.returncode, f.read())
    return CommandResult(args, ret.returncode, None)


def get_folder_size(path: str) -> int:
    r"""Get disk usage of given path in bytes.

    Credit: https://stackoverflow.com/a/25574638/4909228
    """
    return int(subprocess.check_output(['du', '-bs', path]).split()[0].decode('utf-8'))


def readable_size(size: int) -> str:
    r"""Represent file size in human-readable format.

    :param size: File size in bytes.
    """
    units = ["", "K", "M", "G", "T"]
    for unit in units:
        if size < 1024:
            return f"{size:.2f}{unit}"
        size /= 1024
    return f"{size:.2f}P"  # this won't happen


def get_file_lines(path: str) -> int:
    r"""Get number of lines in text file.
    """
    return int(subprocess.check_output(['wc', '-l', path]).decode('utf-8'))


def kill_proc_tree(pid: int, including_parent: bool = True) -> None:
    r"""Kill entire process tree.
    """
    parent = psutil.Process(pid)
    children = parent.children(recursive=True)
    for child in children:
        child.kill()
    gone, still_alive = psutil.wait_procs(children, timeout=5)
    if including_parent:
        parent.kill()
        parent.wait(5)


def register_ipython_excepthook() -> None:
    r"""Register an exception hook that launches an interactive IPython session upon uncaught exceptions.
    """

    def excepthook(type, value, traceback):
        if type is KeyboardInterrupt:
            # don't capture keyboard interrupts (Ctrl+C)
            sys.__excepthook__(type, value, traceback)
        else:
            ipython_hook(type, value, traceback)

    # enter IPython debugger on exception
    from IPython.core import ultratb

    ipython_hook = ultratb.FormattedTB(mode='Context', color_scheme='Linux', call_pdb=1)
    sys.excepthook = excepthook


def to_dict(nm_tpl: NamedTuple) -> Dict[str, Any]:
    return {key: value for key, value in zip(nm_tpl._fields, nm_tpl)}


NamedTupleType = TypeVar('NamedTupleType', bound=NamedTuple)


def to_namedtuple(nm_tpl_type: Type[NamedTupleType], dic: Dict[str, Any]) -> NamedTupleType:
    return nm_tpl_type(**dic)
