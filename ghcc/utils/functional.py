from typing import Callable, Iterable, Iterator, TypeVar

__all__ = [
    "drop_until",
]

T = TypeVar('T')


def drop_until(pred_fn: Callable[[T], bool], iterable: Iterable[T]) -> Iterator[T]:
    iterator = iter(iterable)
    for item in iterator:
        if not pred_fn(item):
            continue
        yield item
        break
    yield from iterator
