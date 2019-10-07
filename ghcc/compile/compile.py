import os
import signal
import subprocess
import sys
from typing import Iterator, NamedTuple, Optional, Set, Tuple

from . import mock_path

TIMEOUT = 1500
TIMEOUT_TICK = 0.01
FIND_MAKEFILES = True  # False
NUM_THREADS = 16

__all__ = [
    "find_makefiles",
    "make",
]


def find_makefiles(path: str) -> Iterator[Tuple[str, str]]:
    for subdir, dirs, files in os.walk(path):
        for name in files:
            if name.lower() == "makefile":  # "configure":
                yield (subdir, name)


# returns a set of all filenames (absolute path) within path
def all_files(path: str) -> Set[str]:
    output = set()
    for subdir, dirs, files in os.walk(path):
        for name in files:
            output.add(os.path.abspath(os.path.join(subdir, name)))
    return output


class CompileResult(NamedTuple):
    pass


def make(subdir: str, name: str, timeout: Optional[int] = None):
    prev_files = all_files(subdir)
    env = {
        "PATH": ":".join(sys.path + [mock_path.__file__]),
    }

    proc = None
    try:
        # Try running `./configure` if it exists.
        if os.path.isfile(os.path.join(subdir, "configure")):
            try:
                output = subprocess.check_output(
                    ["chmod", "+x", "./configure"],
                    env=env, cwd=subdir, stderr=subprocess.STDOUT)
                output = subprocess.check_output(
                    ["./configure"],
                    env=env, cwd=subdir, stderr=subprocess.STDOUT, timeout=timeout)
            except subprocess.TimeoutExpired:
                pass

        # Force make all targets.
        output = subprocess.check_output(
            ["make", "--always-make", "-k"],
            env=env, cwd=subdir, stderr=subprocess.STDOUT, timeout=timeout)
        rc = 0
    except subprocess.TimeoutExpired as e:
        rc = -102
    except OSError as e:
        rc = -103
    except subprocess.CalledProcessError as e:
        rc = e.returncode

    if proc is not None:
        proc.kill()
        # http://stackoverflow.com/questions/19447603/how-to-kill-a-python-child-process-created-with-subprocess-check-output-when-t
        # make sure process and all its children are terminated
        try:
            os.kill(proc.pid, signal.SIGTERM)
        except OSError:
            pass
        try:
            os.kill(-proc.pid, signal.SIGTERM)
        except OSError:
            pass

    after_files = all_files(subdir)
    diff = after_files.difference(prev_files)
    output.append((os.path.abspath(subdir), name, rc, list(diff)))
    print(f"{subdir}: {len(diff):d} files created, rc {rc:d}")
