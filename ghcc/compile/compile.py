import os
import subprocess
import time
from enum import Enum, auto
from typing import List, NamedTuple, Optional

from ghcc.compile.docker import run_docker_command
from ghcc.repo import clean
from ghcc.utils import run_command

MOCK_PATH = os.path.abspath(os.path.join(os.path.split(__file__)[0], "..", "..", "scripts", "mock_path"))

ELF_FILE_TAG = b"ELF"  # Linux

__all__ = [
    "find_makefiles",
    "CompileErrorType",
    "CompileResult",
    "unsafe_make",
    "docker_make",
]


def find_makefiles(path: str) -> List[str]:
    r"""Find all subdirectories under the given directory that contains Makefiles.

    :param path: Path to the directory to scan.
    :return: A list of subdirectories that contain Makefiles.
    """
    directories = []
    for subdir, dirs, files in os.walk(path):
        if any(name.lower() == "makefile" for name in files):
            directories.append(subdir)
    return directories


class CompileErrorType(Enum):
    Timeout = auto()
    CompileFailed = auto()
    Unknown = auto()


class CompileResult(NamedTuple):
    success: bool
    elf_files: List[str]  # list of paths to ELF files
    error_type: Optional[CompileErrorType] = None
    captured_output: Optional[str] = None


def _create_result(success: bool = False, elf_files: Optional[List[str]] = None,
                   error_type: Optional[CompileErrorType] = None,
                   captured_output: Optional[str] = None) -> CompileResult:
    if elf_files is None:
        elf_files = []
    return CompileResult(success, elf_files=elf_files, error_type=error_type, captured_output=captured_output)


def _make_skeleton(make_fn, directory: str, timeout: Optional[int] = None) -> CompileResult:
    directory = os.path.abspath(directory)

    try:
        # Clean unversioned files by previous compilations.
        clean(directory)

        # Call the actual function for `make`.
        make_fn(directory, timeout)
        result = _create_result(True)

    except subprocess.TimeoutExpired as e:
        # Even if exceptions occur, we still check for ELF files, just in case.
        result = _create_result(error_type=CompileErrorType.Timeout, captured_output=e.output)
    except subprocess.CalledProcessError as e:
        result = _create_result(error_type=CompileErrorType.CompileFailed, captured_output=e.output)
    except OSError as e:
        result = _create_result(error_type=CompileErrorType.Unknown, captured_output=str(e))

    try:
        # Use Git to find all unversioned files -- these would be the products of compilation.
        output = run_command(["git", "ls-files", "--others"], cwd=directory,
                             timeout=timeout, return_output=True).captured_output
        diff_files = [
            # files containing escape characters are in quotes
            file if file[0] != '"' else file[1:-1]
            for file in output.decode('unicode_escape').split("\n") if file]  # file names could contain spaces

        # Inspect each file and find ELF files.
        for file in diff_files:
            path = os.path.join(directory, file)
            output = subprocess.check_output(["file", ], timeout=10)
            output = output[len(file):]  # first part is file name
            if ELF_FILE_TAG in output:
                result.elf_files.append(file)
    except subprocess.TimeoutExpired as e:
        return _create_result(elf_files=result.elf_files, error_type=CompileErrorType.Timeout, captured_output=e.output)
    except subprocess.CalledProcessError as e:
        return _create_result(elf_files=result.elf_files, error_type=CompileErrorType.Unknown, captured_output=e.output)
    except OSError as e:
        return _create_result(elf_files=result.elf_files, error_type=CompileErrorType.Unknown, captured_output=str(e))

    return result


def _unsafe_make(directory: str, timeout: Optional[int] = None) -> None:
    env = {
        b"PATH": f"{MOCK_PATH}:{os.environ['PATH']}".encode('utf-8'),
    }
    # Try running `./configure` if it exists.
    if os.path.isfile(os.path.join(directory, "configure")):
        start_time = time.time()
        run_command(["chmod", "+x", "./configure"], env=env, cwd=directory)
        run_command(["./configure"], env=env, cwd=directory, timeout=timeout)
        end_time = time.time()
        if timeout is not None:
            timeout = max(1, timeout - int(end_time - start_time))

    # Make while ignoring errors.
    # `-B/--always-make` could give strange errors for certain Makefiles, e.g. ones containing "%:"
    # run_command(["make", "--keep-going", "-j1"], env=env, cwd=directory, timeout=timeout)
    run_command(["make", "--keep-going", "-j1"], env=env, cwd=directory, timeout=timeout)


def unsafe_make(directory: str, timeout: Optional[int] = None) -> CompileResult:
    r"""Run ``make`` in the given directory and collect compilation outputs.

    .. warning::
        This will run ``make`` on your physical machine under the same privilege granted to the Python program.
        Never run programs from unvalidated sources as malicious programs could break your system.

    :param directory: Path to the directory containing the Makefile.
    :param timeout: Maximum time allowed for compilation, in seconds. Defaults to ``None`` (unlimited time).
    :return: An instance of :class:`CompileResult` indicating the result. Fields ``success`` and ``elf_files`` are not
        ``None``.

        - If compilation failed, the fields ``error_type`` and ``captured_output`` are also not ``None``.
    """
    return _make_skeleton(_unsafe_make, directory, timeout)


def _docker_make(directory: str, timeout: Optional[int] = None):
    if os.path.isfile(os.path.join(directory, "configure")):
        # Try running `./configure` if it exists.
        run_docker_command("chmod +x configure && ./configure && make --keep-going -j1",
                           cwd="/usr/src", directory_mapping={directory: "/usr/src"},
                           timeout=timeout, shell=True)
    else:
        # Make while ignoring errors.
        # `-B/--always-make` could give strange errors for certain Makefiles, e.g. ones containing "%:"
        run_docker_command(["make", "--keep-going", "-j1"],
                           cwd="/usr/src", directory_mapping={directory: "/usr/src"},
                           timeout=timeout)


def docker_make(directory: str, timeout: Optional[int] = None) -> CompileResult:
    r"""Run ``make`` within Docker and collect compilation outputs.

    .. note::
        The ``gcc-custom`` Docker image is used. You can build this using the Dockerfile under the root directory.

    :param directory: Path to the directory containing the Makefile.
    :param timeout: Maximum time allowed for compilation, in seconds. Defaults to ``None`` (unlimited time).
    :return: An instance of :class:`CompileResult` indicating the result. Fields ``success`` and ``elf_files`` are not
        ``None``.

        - If compilation failed, the fields ``error_type`` and ``captured_output`` are also not ``None``.
    """
    return _make_skeleton(_docker_make, directory, timeout)
