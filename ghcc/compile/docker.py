import os
import subprocess
from typing import Optional

from ghcc.compile.compile import CompileErrorType, CompileResult, _create_result
from ghcc.repo import clean
from ghcc.utils import CommandResult, run_command

__all__ = [
    "docker_make",
]


def run_docker_command(command: str, directory: str, timeout: Optional[int] = None) -> CommandResult:
    # cid_file_path = os.path.join(directory, ".docker.cid")
    ret = run_command(' '.join([
        "docker", "run", "--rm",
        "-v", os.path.abspath(directory) + ":/usr/src/", "-w", "/usr/src/",
        # "--cidfile", cid_file_path,
        "-e", "LOCAL_USER_ID=`id -u $USER`",
        "gcc-custom",
        *(["timeout", f"{timeout}s"] if timeout is not None else []),
        "bash", "-c", f"\"{command}\""]), shell=True)
    if ret.return_code == 124:  # timeout
        raise subprocess.TimeoutExpired(ret.command, timeout, output=ret.captured_output)
    return ret


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
    directory = os.path.abspath(directory)

    try:
        # Clean unversioned files by previous compilations.
        clean(directory)

        if os.path.isfile(os.path.join(directory, "configure")):
            # Try running `./configure` if it exists.
            run_docker_command("chmod +x configure && ./configure && make --keep-going -j1",
                               directory, timeout=timeout)
        else:
            # Make while ignoring errors.
            # `-B/--always-make` could give strange errors for certain Makefiles, e.g. ones containing "%:"
            run_docker_command("make --keep-going -j1", directory, timeout=timeout)
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
            os.path.join(directory, file if file[0] != '"' else file[1:-1])
            for file in output.decode('unicode_escape').split("\n") if file]  # file names could contain spaces

        # Inspect each file and find ELF files.
        for file in diff_files:
            output = subprocess.check_output(["file", file], timeout=10).decode('utf-8')
            output = output[len(file):]  # first part is file name
            if "ELF" in output:
                result.elf_files.append(file)
    except subprocess.TimeoutExpired as e:
        return _create_result(elf_files=result.elf_files, error_type=CompileErrorType.Timeout, captured_output=e.output)
    except subprocess.CalledProcessError as e:
        return _create_result(elf_files=result.elf_files, error_type=CompileErrorType.Unknown, captured_output=e.output)
    except OSError as e:
        return _create_result(elf_files=result.elf_files, error_type=CompileErrorType.Unknown, captured_output=str(e))

    return result
