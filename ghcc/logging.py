import logging
import multiprocessing
import sys
import threading
import time
import traceback

from termcolor import colored

__all__ = [
    "set_log_file",
    "log",
]

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

COLOR_MAP = {
    "success": "green",
    "warning": "yellow",
    "error": "red",
    "info": "white",
}

LOGGING_MAP = {
    "success": logger.info,
    "warning": logger.warning,
    "error": logger.error,
    "info": logger.info,
}


class MultiprocessingFileHandler(logging.Handler):
    """multiprocessing log handler

    This handler makes it possible for several processes
    to log to the same file by using a queue.

    Credit: https://mattgathu.github.io/multiprocessing-logging-in-python/
    """

    def __init__(self, path: str, mode: str = "a"):
        logging.Handler.__init__(self)

        self._handler = logging.FileHandler(path, mode=mode)
        self.queue = multiprocessing.Queue(-1)

        thrd = threading.Thread(target=self.receive)
        thrd.daemon = True
        thrd.start()

    def setFormatter(self, fmt):
        logging.Handler.setFormatter(self, fmt)
        self._handler.setFormatter(fmt)

    def receive(self):
        while True:
            try:
                record = self.queue.get()
                self._handler.emit(record)
            except (KeyboardInterrupt, SystemExit):
                raise
            except EOFError:
                break
            except:
                traceback.print_exc(file=sys.stderr)

    def send(self, s):
        self.queue.put_nowait(s)

    def _format_record(self, record):
        if record.args:
            record.msg = record.msg % record.args
            record.args = None
        if record.exc_info:
            dummy = self.format(record)
            record.exc_info = None

        return record

    def emit(self, record):
        try:
            s = self._format_record(record)
            self.send(s)
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)

    def close(self):
        self._handler.close()
        logging.Handler.close(self)


def set_log_file(path: str, fmt: str = "%(asctime)s %(levelname)s: %(message)s") -> None:
    r"""Set the path of the log file.

    :param path: Path to the log file.
    :param fmt: Logging format.
    """
    while len(logger.handlers) > 0:
        handler = logger.handlers[0]
        handler.close()
        logger.removeHandler(handler)

    handler = MultiprocessingFileHandler(path, mode="a")
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(logging.FileHandler(path, mode="a"))


def log(msg: str, level: str = "info") -> None:
    r"""Write a line of log with the specified logging level.

    :param msg: Message to log.
    :param level: Logging level. Available options are ``success``, ``warning``, ``error``, and ``info``.
    """
    if level not in LOGGING_MAP:
        raise ValueError(f"Incorrect logging level '{level}'")
    time_str = time.strftime("[%Y-%m-%d %H:%M:%S]")
    print(colored(time_str, COLOR_MAP[level]), msg, flush=True)
    LOGGING_MAP[level](msg)
