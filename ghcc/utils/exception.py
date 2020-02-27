import functools
import inspect
import subprocess
import sys
import traceback
from typing import Optional

from ..log import log

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


def log_exception(e, user_msg: Optional[str] = None, **kwargs):
    exc_msg = f"<{e.__class__.__qualname__}> {e}"
    if user_msg is not None:
        exc_msg = f"{user_msg}: {exc_msg}"
    try:
        if not (isinstance(e, subprocess.CalledProcessError) and e.output is not None):
            log(traceback.format_exc(), "error", **kwargs)
        log(exc_msg, "error", **kwargs)
    except Exception as log_e:
        print(exc_msg)
        print(f"Another exception occurred while logging: <{log_e.__class__.__qualname__}> {log_e}")
        raise log_e


def exception_wrapper(handler_fn=None):
    r"""Function decorator that calls the specified handler function when a exception occurs inside the decorated
    function. By default, ``handler_fn`` is ``None``, and :meth:`log_exception` will be called to print the exception
    details.

    A custom handler function takes the following arguments:

    - A positional argument for the exception object. This must be the first argument of the method.
    - Arguments with matching names to arguments in the wrapped method. These arguments will be filled with values
      passed to the wrapped method. These arguments cannot take default values.
    - Arguments without matching names. These arguments must take default values.
    - An optional variadic keyword argument (``**kwargs``). This will be filled with remaining argument name-value pairs
      that are not captured by other arguments.

    For example:

    .. code:: python

        def handler_fn(e, three, one, args, my_arg=None, **kw): ...

        @exception_wrapper(handler_fn)
        def foo(one, two, *args, three=None, **kwargs): ...

        foo(1, "2", "arg1", "arg2", four=4)

    Assume a :exc:`ValueError` is thrown, the argument values for ``handler_fn`` would be:

    .. code::

        e:      <ValueError>
        three:  None
        one:    1
        args:   ["args1", "args2"]
        my_arg: None
        kw:     {"two": "2",
                 "kwargs": {"four": 4}}
    """

    def _unwrap(fn):
        if hasattr(fn, "__wrapped__"):
            return _unwrap(fn.__wrapped__)
        return fn

    def decorator(func):
        if handler_fn is not None:
            handler_argspec = inspect.getfullargspec(_unwrap(handler_fn))
            if len(handler_argspec.args) == 0:
                raise ValueError("Exception handler must have a positional argument for the exception object")
            if handler_argspec.varargs is not None:
                raise ValueError("Exception handler cannot have a varargs argument (*args)")
            handler_arg_names = set(handler_argspec.args[1:] + handler_argspec.kwonlyargs)
            handler_args_with_defaults = set((handler_argspec.kwonlydefaults or {}).keys())
            if handler_argspec.defaults is not None:
                handler_args_with_defaults |= set(handler_argspec.args[-len(handler_argspec.defaults):])
            handler_arg_names -= handler_args_with_defaults
            inner_signature = inspect.signature(func)
            for name in handler_arg_names:
                if name not in inner_signature.parameters:
                    raise ValueError(f"Argument '{name}' in exception handler does not match "
                                     f"any argument in wrapped method")
            for name in handler_args_with_defaults:
                if name in inner_signature.parameters:
                    raise ValueError(f"Argument '{name}' matches wrapped method argument, thus "
                                     f"cannot have default values")

        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if handler_fn is None:
                    log_exception(e)
                else:
                    # Credit: https://stackoverflow.com/questions/59831981/
                    bound_args = inner_signature.bind(*args, **kwargs)
                    bound_args.apply_defaults()
                    handler_args = {name: bound_args.arguments[name] for name in handler_arg_names}
                    if handler_argspec.varkw is not None:
                        handler_args[handler_argspec.varkw] = {
                            name: value for name, value in bound_args.arguments.values()
                            if name not in handler_arg_names}
                    return handler_fn(e, **handler_args)

        return wrapped

    return decorator
