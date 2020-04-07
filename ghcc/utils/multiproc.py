import contextlib
import multiprocessing
import queue
import sys
import threading
import traceback
import types
from collections import defaultdict
from typing import Any, Callable, Dict, Iterator, List, NamedTuple, Optional, TextIO, TypeVar, Union

import psutil
from tqdm import tqdm

__all__ = [
    "get_worker_id",
    "safe_pool",
    "MultiprocessingFileWriter",
    "kill_proc_tree",
    "ProgressBarManager",
]

T = TypeVar('T')
R = TypeVar('R')


class Pool:
    r"""A wrapper over ``multiprocessing.Pool`` that uses single-threaded execution when :attr:`processes` is zero.
    """

    def __new__(cls, processes: int, *args, **kwargs):
        if processes > 0:
            return multiprocessing.Pool(processes, *args, **kwargs)
        return super().__new__(cls)  # return a mock Pool instance.

    def imap_unordered(self, fn: Callable[[T], R], iterator: Iterator[T]) -> Iterator[R]:
        yield from map(fn, iterator)

    map = imap_unordered
    imap = imap_unordered

    @staticmethod
    def _no_op(self, *args, **kwargs):
        pass

    def __getattr__(self, item):
        return types.MethodType(Pool._no_op, self)  # no-op for everything else


def get_worker_id() -> Optional[int]:
    r"""Return the ID of the pool worker process, or ``None`` if the current process is not a pool worker."""
    proc_name = multiprocessing.current_process().name
    if "PoolWorker" in proc_name:
        worker_id = int(proc_name[(proc_name.find('-') + 1):])
        return worker_id
    return None


@contextlib.contextmanager
def safe_pool(processes: int, *args, closing: Optional[List[Any]] = None, **kwargs) -> Iterator[Pool]:
    r"""A wrapper over ``multiprocessing.Pool`` that gracefully handles exceptions.

    :param processes: The number of worker processes to run. A value of 0 means single threaded execution.
    :param closing: An optional list of objects to close at exit, routines to run at exit. For each element ``obj``:

        - If it is a callable, ``obj`` is called with no arguments.
        - If it has an ``close`` method, ``obj.close()`` is invoked.
        - Otherwise, it is ignored.

    :return: A context manager that can be used in a ``with`` statement.
    """
    if closing is not None and not isinstance(closing, list):
        raise ValueError("`closing` should either be `None` or a list")

    def close_fn():
        for obj in (closing or []):
            if callable(obj):
                obj()
            elif hasattr(obj, "close") and callable(getattr(obj, "close")):
                obj.close()

    pool = Pool(processes, *args, **kwargs)
    if processes == 0:
        # Don't swallow exceptions in the single-process case.
        yield pool
        close_fn()
        return

    try:
        yield pool
    except KeyboardInterrupt:
        from ..log import log  # prevent circular import
        log("Gracefully shutting down...", "warning", force_console=True)
        print("Press Ctrl-C again to force terminate...")
        try:
            pool.join()
        except KeyboardInterrupt:
            pass
    except Exception as e:
        print(traceback.format_exc())
    finally:
        close_fn()
        if isinstance(pool, multiprocessing.pool.Pool):
            # Only required in multiprocessing scenario
            pool.close()
            pool.terminate()
            # kill_proc_tree(os.getpid())  # commit suicide


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


class NewEvent(NamedTuple):
    worker_id: Optional[int]
    kwargs: Dict[str, Any]


class UpdateEvent(NamedTuple):
    worker_id: Optional[int]
    n: int
    postfix: Optional[Dict[str, Any]]


class WriteEvent(NamedTuple):
    worker_id: Optional[int]
    message: str


class QuitEvent(NamedTuple):
    pass


Event = Union[NewEvent, UpdateEvent, WriteEvent, QuitEvent]


class ProgressBarManager:
    r"""A manager for ``tqdm`` progress bars that allows maintaining multiple bars from multiple worker processes."""

    class Proxy:
        def __init__(self, queue):
            self.queue = queue

        def new(self, **kwargs) -> None:
            r"""Construct a new progress bar."""
            self.queue.put_nowait(NewEvent(get_worker_id(), kwargs))

        def update(self, n: int, postfix: Optional[Dict[str, Any]] = None) -> None:
            self.queue.put_nowait(UpdateEvent(get_worker_id(), n, postfix))

        def write(self, message: str) -> None:
            self.queue.put_nowait(WriteEvent(get_worker_id(), message))

    def __init__(self, **kwargs):
        self.manager = multiprocessing.Manager()
        self.queue: 'queue.Queue[Event]' = self.manager.Queue(-1)
        self.progress_bars: Dict[Optional[int], tqdm] = {}
        self.worker_id_map: Dict[Optional[int], int] = defaultdict(lambda: len(self.worker_id_map))
        self.bar_kwargs = kwargs.copy()
        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True
        self.thread.start()

    @property
    def proxy(self):
        return self.Proxy(self.queue)

    def _run(self):
        while True:
            try:
                event = self.queue.get()
                if isinstance(event, NewEvent):
                    position = self.worker_id_map[event.worker_id]
                    if event.worker_id in self.progress_bars:
                        self.progress_bars[event.worker_id].close()
                        del self.progress_bars[event.worker_id]
                    kwargs = {**self.bar_kwargs, **event.kwargs, "leave": False, "position": position}
                    bar = tqdm(**kwargs)
                    self.progress_bars[event.worker_id] = bar
                elif isinstance(event, UpdateEvent):
                    bar = self.progress_bars[event.worker_id]
                    if event.postfix is not None:
                        bar.set_postfix(event.postfix, refresh=False)
                    bar.update(event.n)
                elif isinstance(event, WriteEvent):
                    tqdm.write(event.message)
                elif isinstance(event, QuitEvent):
                    break
                else:
                    assert False
            except (KeyboardInterrupt, SystemExit):
                raise
            except EOFError:
                break
            except:
                traceback.print_exc(file=sys.stderr)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self.queue.put_nowait(QuitEvent())
        self.thread.join()
        for bar in self.progress_bars.values():
            bar.close()
