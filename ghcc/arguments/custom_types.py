import enum
from typing import Any, Iterable, Union


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


def is_choices(typ) -> bool:
    r"""Check whether a type is a choices type. This cannot be checked using traditional methods,
    since :class:`Choices` is a metaclass.
    """
    return type(typ) is type(Choices)


NoneType = type(None)


class Enum(enum.Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name.lower()

    def __eq__(self, other):
        return self.value == other or super().__eq__(other)


Switch = Union[bool]
