import enum
from typing import Any, Iterable, Union, TypeVar, Optional, Type

__all__ = [
    "Choices",
    "Enum",
    "Switch",
    "is_choices",
    "is_optional",
    "unwrap_optional",
]

NoneType = type(None)
T = TypeVar('T')


class _Choices:
    def __new__(cls, values=None):
        self = super().__new__(cls)
        self.__values__ = values
        return self

    def __getitem__(self, values: Union[str, Iterable[str]]):
        if values == ():
            raise TypeError("Choices must contain at least one element")
        if isinstance(values, Iterable) and not isinstance(values, str):
            parsed_values = tuple(values)
        else:
            parsed_values = (values,)
        return self.__class__(parsed_values)


Choices: Any = _Choices()


class Enum(enum.Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name.lower()

    def __eq__(self, other):
        return self.value == other or super().__eq__(other)


Switch = Union[bool]


def is_choices(typ: type) -> bool:
    r"""Check whether a type is a choices type. This cannot be checked using traditional methods,
    since :class:`Choices` is a metaclass.
    """
    return isinstance(typ, _Choices)


def is_optional(typ: type) -> bool:
    r"""Check whether a type is `Optional`. `Optional` is internally implemented as `Union` with `type(None)`."""
    return getattr(typ, '__origin__', None) is Union and NoneType in typ.__args__  # type: ignore


def unwrap_optional(typ: Type[Optional[T]]) -> Type[T]:
    r"""Return the inner type inside an `Optional` type."""
    return next(t for t in typ.__args__ if not isinstance(t, NoneType))  # type: ignore
