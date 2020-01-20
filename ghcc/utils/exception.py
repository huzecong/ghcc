import functools
import subprocess
import sys
import traceback
from typing import Optional

from ghcc.logging import log

__all__ = [
    "register_ipython_excepthook",
    "log_exception",
    "exception_wrapper",
]


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


def log_exception(e, user_msg: Optional[str] = None):
    exc_msg = f"<{e.__class__.__qualname__}> {e}"
    if user_msg is not None:
        exc_msg = f"{user_msg}: {exc_msg}"
    try:
        if not (isinstance(e, subprocess.CalledProcessError) and e.output is not None):
            log(traceback.format_exc(), "error")
        log(exc_msg, "error")
    except Exception as log_e:
        print(exc_msg)
        print(f"Another exception occurred while logging: <{log_e.__class__.__qualname__}> {log_e}")
        raise log_e


def exception_wrapper(handler_fn):
    r"""Function decorator that calls the specified handler function when a exception occurs inside the decorated
    function.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                return handler_fn(e, *args, **kwargs)

        return wrapped

    return decorator
