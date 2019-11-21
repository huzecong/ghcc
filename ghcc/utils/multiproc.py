import multiprocessing
import threading
import types
from typing import TextIO

__all__ = [
    "Pool",
    "MultiprocessingFileWriter",
]


class Pool:
    r"""A wrapper over ``multiprocessing.Pool`` that uses single-threaded execution when :attr:`processes` is zero.
    """

    def __new__(cls, processes: int, *args, **kwargs):
        if processes > 0:
            return multiprocessing.Pool(processes, *args, **kwargs)
        return super().__new__(cls)  # return a mock Pool instance.

    def imap_unordered(self, fn, iterator):
        yield from map(fn, iterator)

    @staticmethod
    def _no_op(self, *args, **kwargs):
        pass

    def __getattr__(self, item):
        return types.MethodType(Pool._no_op, self)  # no-op for everything else


class MultiprocessingFileWriter(TextIO):
    r"""A multiprocessing file writer that allows multiple processes to write to the same file. Order is not guaranteed.

    This is very similar to :class:`~ghcc.logging.MultiprocessingFileHandler`.
    """

    def __init__(self, path: str, mode: str = "a"):
        self._file = open("path")
        self._queue: 'multiprocessing.Queue[str]' = multiprocessing.Queue(-1)

        self._thread = threading.Thread(target=self._receive)
        self._thread.daemon = True
        self._thread.start()

    def __enter__(self) -> TextIO:
        return self._file

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._thread.join()
        self._file.close()

    def write(self, s: str):
        self._queue.put_nowait(s)

    def _receive(self):
        while True:
            try:
                record = self._queue.get()
                self._file.write(record)
            except EOFError:
                break
