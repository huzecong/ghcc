import subprocess
import sys
import tempfile
from typing import Any, Dict, NamedTuple, Type, TypeVar, List, Optional

__all__ = [
    "run_command",
    "get_folder_size",
    "get_file_lines",
    "register_ipython_excepthook",
    "to_dict",
    "to_namedtuple",
]


def run_command(args: List[str], env: Optional[Dict[bytes, bytes]] = None, cwd: Optional[str] = None,
                timeout: Optional[int] = None, return_output: bool = False, **kwargs):
    r"""A wrapper over ``subprocess.check_output`` that prevents deadlock caused by the combination of pipes and
    timeout. Output is redirected into a temporary file and returned only on exceptions.

    :param return_output: If ``True``, the captured output is returned.
    """
    with tempfile.TemporaryFile() as f:
        try:
            subprocess.run(args, check=True, stdout=f, stderr=subprocess.STDOUT,
                           timeout=timeout, env=env, cwd=cwd, **kwargs)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            f.seek(0)
            e.output = f.read()
            raise e from None
        if return_output:
            f.seek(0)
            return f.read()


def get_folder_size(path: str) -> int:
    r"""Get disk usage of given path in bytes.

    Credit: https://stackoverflow.com/a/25574638/4909228
    """
    return int(subprocess.check_output(['du', '-bs', path]).split()[0].decode('utf-8'))


def get_file_lines(path: str) -> int:
    r"""Get number of lines in text file.
    """
    return int(subprocess.check_output(['wc', '-l', path]).decode('utf-8'))


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
