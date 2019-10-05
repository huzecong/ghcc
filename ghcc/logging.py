import logging
import time

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


def set_log_file(path: str, fmt: str = "%(asctime)s %(levelname)s: %(message)s") -> None:
    while len(logger.handlers) > 0:
        logger.removeHandler(logger.handlers[0])
    handler = logging.FileHandler(path, mode="a")
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(logging.FileHandler(path, mode="a"))


def log(msg: str, level: str = "info") -> None:
    if level not in LOGGING_MAP:
        raise ValueError(f"Incorrect logging level '{level}'")
    # lock.acquire()
    time_str = time.strftime("[%Y-%m-%d %H:%M:%S]")
    print(colored(time_str, COLOR_MAP[level]), msg, flush=True)
    LOGGING_MAP[level](msg)
    # lock.release()
