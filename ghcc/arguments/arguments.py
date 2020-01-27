import argparse
import functools
from typing import Any, Callable, Dict, List, Optional, TypeVar

from .custom_types import Switch, is_choices, is_optional, unwrap_optional

__all__ = [
    "ArgumentParser",
    "Arguments",
]


class ArgumentParser(argparse.ArgumentParser):
    r"""A wrapper over :class:`ArgumentParser` that adds support for "toggle" arguments. A toggle argument with name
    ``"flag"`` has value ``True`` if the argument ``--flag`` exists, and ``False`` if ``--no-flag`` exists.
    """

    def add_toggle_argument(self, name: str, default: bool = False) -> None:
        if not name.startswith("--"):
            raise ValueError("Toggle arguments must begin with '--'")
        name = name[2:]
        var_name = name.replace('-', '_')
        self.add_argument(f"--{name}", action="store_true", default=default, dest=var_name)
        self.add_argument(f"--no-{name}", action="store_false", dest=var_name)


T = TypeVar('T')
ConversionFn = Callable[[str], T]


def _bool_conversion_fn(s: str) -> bool:
    if s.lower() in ["y", "yes", "true", "ok"]:
        return True
    if s.lower() in ["n", "no", "false"]:
        return False
    raise ValueError(f"Invalid value '{s}' for bool argument")


def _optional_wrapper_fn(fn: Optional[ConversionFn[T]] = None) -> ConversionFn[Optional[T]]:
    @functools.wraps(fn)  # type: ignore  # this works even if `fn` is None
    def wrapped(s: str) -> Optional[T]:
        if s.lower() == 'none':
            return None
        if fn is None:
            return s  # type: ignore
        return fn(s)

    return wrapped


class Arguments:
    r"""A typed version of ``argparse``. It's easier to illustrate using an example:

    .. code-block:: python

        from ghcc.utils import Arguments, Choices, Switch

        class MyArguments(Arguments):
            model_name: str
            hidden_size: int = 512
            activation: Choices['relu', 'tanh', 'sigmoid'] = 'relu'
            logging_level: Choices[ghcc.logging.get_levels()] = 'info'
            use_dropout: Switch = True
            dropout_prob: Optional[float] = 0.5

        args = Arguments()

    This is equivalent to the following code with Python built-in ``argparse``:

    .. code-block:: python

        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--model-name", type=str, required=True)
        parser.add_argument("--hidden-size", type=int, default=512)
        parser.add_argument("--activation", choices=["relu", "tanh", "sigmoid"], default="relu")
        parser.add_argument("--logging-level", choices=ghcc.logging.get_levels(), default="info")
        parser.add_argument("--use-dropout", action="store_true", dest="use_dropout", default=True)
        parser.add_argument("--no-use-dropout", action="store_false", dest="use_dropout")
        parser.add_argument("--dropout-prob", type=lambda s: None if s.lower() == 'none' else float(s), default=0.5)

        args = parser.parse_args()

    Suppose the following arguments are provided:

    .. code-block:: bash

        python main.py \
            --model-name LSTM \
            --activation sigmoid \
            --logging-level debug \
            --no-use-dropout \
            --dropout-prob none

    the parsed arguments will be:

    .. code-block:: bash

        Namespace(model_name="LSTM", hidden_size=512, activation="sigmoid", logging_level="debug",
                  use_dropout=False, dropout_prob=None)

    :class:`Arguments` provides the following features:

    - More concise and intuitive syntax over ``argparse``, less boilerplate code.
    - Arguments take the form of type-annotated class attributes, allowing IDEs to provide autocompletion.
    - Drop-in replacement for ``argparse``, since internally ``argparse`` is used.

    **Note:** Advanced features such as subparsers, groups, argument lists, custom actions are not supported.
    """

    _TYPE_CONVERSION_FN: Dict[type, Callable[[str], Any]] = {
        bool: _bool_conversion_fn,
    }

    def __new__(cls, args: Optional[List[str]] = None, namespace: Optional[argparse.Namespace] = None):
        annotations: Dict[str, type] = {}
        for base in reversed(cls.__mro__):  # used reversed order so derived classes can override base annotations
            if base not in [object, Arguments]:
                annotations.update(base.__dict__.get('__annotations__', {}))

        parser = ArgumentParser()
        for arg_name, arg_typ in annotations.items():
            has_default = hasattr(cls, arg_name)
            default_val = getattr(cls, arg_name, None)
            nullable = is_optional(arg_typ)
            if nullable:
                # extract the type wrapped inside `Optional`
                arg_typ = unwrap_optional(arg_typ)

            required = False
            if nullable and not has_default:
                has_default = True
                default_val = None
            elif not nullable and not has_default:
                required = True

            parser_arg_name = "--" + arg_name.replace("_", "-")
            parser_kwargs: Dict[str, Any] = {
                "required": required,
            }
            if arg_typ is Switch:
                if not isinstance(default_val, bool):
                    raise ValueError(f"Switch argument '{arg_name}' must have a default value of type bool")
                parser.add_toggle_argument(parser_arg_name, default_val)
            elif is_choices(arg_typ):
                parser_kwargs["choices"] = arg_typ.__values__  # type: ignore
                if has_default:
                    parser_kwargs["default"] = default_val
                parser.add_argument(parser_arg_name, **parser_kwargs)
            else:
                conversion_fn = None
                if arg_typ in cls._TYPE_CONVERSION_FN or callable(arg_typ):
                    conversion_fn = cls._TYPE_CONVERSION_FN.get(arg_typ, arg_typ)
                if nullable:
                    conversion_fn = _optional_wrapper_fn(conversion_fn)
                if conversion_fn is not None:
                    parser_kwargs["type"] = conversion_fn
                if has_default:
                    parser_kwargs["default"] = default_val
                parser.add_argument(parser_arg_name, **parser_kwargs)

        if cls.__module__ != "__main__":
            # Usually arguments are defined in the same script that is directly run (__main__).
            # If this is not the case, add a note in help message indicating where the arguments are defined.
            usage = parser.format_usage()
            usage += f"\nNote: Arguments defined in {cls.__module__}.{cls.__name__}"
            parser.usage = usage

        return parser.parse_args(args, namespace)
