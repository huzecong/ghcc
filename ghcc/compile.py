import hashlib
import os
import pickle
import shutil
import subprocess
import time
from enum import Enum, auto
from typing import Callable, Dict, Iterator, List, NamedTuple, Optional

import ghcc
from ghcc.database import RepoDB
from ghcc.repo import clean
from ghcc.utils.docker import run_docker_command
from ghcc.utils.run import run_command

MOCK_PATH = os.path.abspath(os.path.join(os.path.split(__file__)[0], "..", "..", "scripts", "mock_path"))

ELF_FILE_TAG = b"ELF"  # Linux

__all__ = [
    "contains_files",
    "find_makefiles",
    "CompileErrorType",
    "CompileResult",
    "unsafe_make",
    "docker_make",
    "compile_and_move",
    "docker_batch_compile",
]


def contains_files(path: str, names: List[str]) -> bool:
    r"""Check (non-recursively) whether the directory contains at least one file with an acceptable name
    (case-insensitive).

    :param path: The directory to check for.
    :param names: List of acceptable names. Note that all names must be in lowercase!
    :return: Whether the check succeeded.
    """
    for file in os.listdir(path):
        if file.lower() in names and os.path.isfile(os.path.join(path, file)):
            return True
    return False


def find_makefiles(path: str) -> List[str]:
    r"""Find all subdirectories under the given directory that contains Makefiles.

    :param path: Path to the directory to scan.
    :return: A list of subdirectories that contain Makefiles.
    """
    directories = []
    for subdir, dirs, files in os.walk(path):
        # if any(name.lower() == "makefile" for name in files):
        # if contains_files(subdir, ["makefile", "configure.ac", "configure.in"]):
        if contains_files(subdir, ["makefile"]):
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


def _make_skeleton(make_fn, directory: str, timeout: Optional[float] = None,
                   env: Optional[Dict[str, str]] = None) -> CompileResult:
    directory = os.path.abspath(directory)

    try:
        # Clean unversioned files by previous compilations.
        clean(directory)

        # Call the actual function for `make`.
        make_fn(directory, timeout, env)
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
        assert output is not None
        diff_files = [
            # files containing escape characters are in quotes
            file if file[0] != '"' else file[1:-1]
            for file in output.decode('unicode_escape').split("\n") if file]  # file names could contain spaces

        # Inspect each file and find ELF files.
        for file in diff_files:
            path = os.path.join(directory, file)
            output = subprocess.check_output(["file", path], timeout=10)
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


def _unsafe_make(directory: str, timeout: Optional[float] = None, env: Optional[Dict[str, str]] = None) -> None:
    env = {"PATH": f"{MOCK_PATH}:{os.environ['PATH']}", **(env or {})}
    # Try GNU Automake first. Note that errors are ignored because it's possible that the original files still work.
    if contains_files(directory, ["configure.ac", "configure.in"]):
        start_time = time.time()
        if os.path.isfile(os.path.join(directory, "autogen.sh")):
            # Some projects with non-trivial build instructions provide an "autogen.sh" script.
            run_command(["chmod", "+x", "./autogen.sh"], env=env, cwd=directory)
            run_command(["./autogen.sh"], env=env, cwd=directory, timeout=timeout, ignore_errors=True)
        else:
            run_command(["autoreconf", "--force", "--install"],
                        env=env, cwd=directory, timeout=timeout, ignore_errors=True)
        end_time = time.time()
        if timeout is not None:
            timeout = max(1.0, timeout - int(end_time - start_time))

    # Try running `./configure` if it exists.
    if os.path.isfile(os.path.join(directory, "configure")):
        start_time = time.time()
        run_command(["chmod", "+x", "./configure"], env=env, cwd=directory)
        ret = run_command(
            ["./configure", "--disable-werror"], env=env, cwd=directory, timeout=timeout, ignore_errors=True)
        end_time = time.time()
        if ret.return_code != 0 and end_time - start_time <= 2:
            # The configure file might not support `--disable-werror` and died instantly. Try again without the flag.
            run_command(["./configure"], env=env, cwd=directory, timeout=timeout)
            end_time = time.time()
        if timeout is not None:
            timeout = max(1.0, timeout - int(end_time - start_time))

    # Make while ignoring errors.
    # `-B/--always-make` could give strange errors for certain Makefiles, e.g. ones containing "%:"
    try:
        run_command(["make", "--keep-going", "-j1"], env=env, cwd=directory, timeout=timeout)
    except subprocess.CalledProcessError as err:
        expected_msg = b"missing separator"
        if not (err.output is not None and expected_msg in err.output):
            raise err
        else:
            # Try again using BSD Make instead of GNU Make. Note BSD Make does not have a flag equivalent to
            # `-B/--always-make`.
            run_command(["bmake", "-k", "-j1"], env=env, cwd=directory, timeout=timeout)


def unsafe_make(directory: str, timeout: Optional[float] = None, env: Optional[Dict[str, str]] = None) -> CompileResult:
    r"""Run ``make`` in the given directory and collect compilation outputs.

    .. warning::
        This will run ``make`` on your physical machine under the same privilege granted to the Python program.
        Never run programs from unvalidated sources as malicious programs could break your system.

    :param directory: Path to the directory containing the Makefile.
    :param timeout: Maximum time allowed for compilation, in seconds. Defaults to ``None`` (unlimited time).
    :param env: The environment variables to use when calling ``make``.
    :return: An instance of :class:`CompileResult` indicating the result. Fields ``success`` and ``elf_files`` are not
        ``None``.

        - If compilation failed, the fields ``error_type`` and ``captured_output`` are also not ``None``.
    """
    return _make_skeleton(_unsafe_make, directory, timeout, env)


def _docker_make(directory: str, timeout: Optional[float] = None, env: Optional[Dict[str, str]] = None) -> None:
    if os.path.isfile(os.path.join(directory, "configure")):
        # Try running `./configure` if it exists.
        run_docker_command("chmod +x configure && ./configure && make --keep-going -j1",
                           user=0, cwd="/usr/src", directory_mapping={directory: "/usr/src"},
                           timeout=timeout, shell=True, env=env)
    else:
        # Make while ignoring errors.
        # `-B/--always-make` could give strange errors for certain Makefiles, e.g. ones containing "%:"
        run_docker_command(["make", "--keep-going", "-j1"],
                           user=0, cwd="/usr/src", directory_mapping={directory: "/usr/src"},
                           timeout=timeout, env=env)


def docker_make(directory: str, timeout: Optional[float] = None, env: Optional[Dict[str, str]] = None) -> CompileResult:
    r"""Run ``make`` within Docker and collect compilation outputs.

    .. note::
        The ``gcc-custom`` Docker image is used. You can build this using the Dockerfile under the root directory.

    .. warning::
        It is only possible to run one bash command in a (non-interactive) Docker session, the compilation heuristics
        used here is the original version. It is recommended to call `batch_make.py` instead of relying on this method.

    :param directory: Path to the directory containing the Makefile.
    :param timeout: Maximum time allowed for compilation, in seconds. Defaults to ``None`` (unlimited time).
    :param env: The environment variables to use when calling ``make``.
    :return: An instance of :class:`CompileResult` indicating the result. Fields ``success`` and ``elf_files`` are not
        ``None``.

        - If compilation failed, the fields ``error_type`` and ``captured_output`` are also not ``None``.
    """
    return _make_skeleton(_docker_make, directory, timeout, env)


def compile_and_move(repo_binary_dir: str, repo_path: str, makefile_dirs: List[str],
                     compile_timeout: Optional[float] = None, record_libraries: bool = False,
                     gcc_override_flags: Optional[str] = None,
                     compile_fn=docker_make) -> Iterator[RepoDB.MakefileEntry]:
    r"""Compile all Makefiles as provided, and move generated binaries to the binary directory.

    :param repo_binary_dir: Path to the directory where generated binaries for the repository will be stored.
    :param repo_path: Path to the repository.
    :param makefile_dirs: A list of all subdirectories containing Makefiles.
    :param compile_timeout: Maximum time allowed for compilation of all Makefiles, in seconds. Defaults to ``None``
        (unlimited time).
    :param record_libraries: If ``True``, A file named ``libraries.txt`` will be generated under
        :attr:`repo_binary_dir`, recording the libraries used in compilation. Defaults to ``False``.
    :param compile_fn: The method to call for compilation. Possible values are :meth:`ghcc.unsafe_make` and
        :meth:`ghcc.docker_make` (default).
    :param gcc_override_flags: If not ``None``, these flags will be appended to each invocation of GCC.
    :return: A list of Makefile compilation results.
    """
    env = {}
    if record_libraries:
        env["MOCK_GCC_LIBRARY_LOG"] = os.path.join(repo_binary_dir, "libraries.txt")
    if gcc_override_flags is not None:
        env["MOCK_GCC_OVERRIDE_FLAGS"] = gcc_override_flags
    remaining_time = compile_timeout
    for make_dir in makefile_dirs:
        if remaining_time is not None and remaining_time <= 0.0:
            break
        start_time = time.time()
        compile_result = compile_fn(make_dir, timeout=remaining_time, env=env)
        elapsed_time = time.time() - start_time
        if remaining_time is not None:
            remaining_time -= elapsed_time
        # Only record Makefiles that either successfully compiled or yielded binaries.
        # Successful compilations might not generate binaries, while failed compilations may also yield binaries.
        if len(compile_result.elf_files) > 0 or compile_result.success:
            sha256: List[str] = []
            for path in compile_result.elf_files:
                path = os.path.join(make_dir, path)
                hash_obj = hashlib.sha256()
                with open(path, "rb") as f:
                    hash_obj.update(f.read())
                digest = hash_obj.hexdigest()
                sha256.append(digest)
                shutil.move(path, os.path.join(repo_binary_dir, digest))
            yield {
                "directory": make_dir,
                "success": compile_result.success,
                "binaries": compile_result.elf_files,
                "sha256": sha256,
            }
    ghcc.clean(repo_path)


def docker_batch_compile(repo_binary_dir: str, repo_path: str,
                         compile_timeout: Optional[float], record_libraries: bool = False,
                         gcc_override_flags: Optional[str] = None, user_id: Optional[int] = None,
                         exception_log_fn: Optional[Callable[[Exception], None]] = None) \
        -> List[RepoDB.MakefileEntry]:
    start_time = time.time()
    try:
        # Don't rely on Docker timeout, but instead constrain running time in script run in Docker. Otherwise we won't
        # get the results file if any compilation task timeouts.
        ret = ghcc.utils.run_docker_command([
            "batch_make.py",
            *(["--record-libraries"] if record_libraries else []),
            *(["--compile-timeout", str(compile_timeout)] if compile_timeout is not None else []),
            *(["--gcc-override-flags", f'"{gcc_override_flags}"'] if gcc_override_flags is not None else [])],
            user=user_id, return_output=True,
            directory_mapping={repo_path: "/usr/src/repo", repo_binary_dir: "/usr/src/bin"})
    except subprocess.CalledProcessError as e:
        end_time = time.time()
        if ((compile_timeout is not None and end_time - start_time > compile_timeout) or
                b"Resource temporarily unavailable" in e.output):
            # Usually exceptions at this stage are due to some badly written Makefiles that gets trapped in an infinite
            # recursion. We suppress the exception and proceed normally, so we won't have to deal with it again when
            # the program is rerun.
            if exception_log_fn is not None:
                exception_log_fn(e)
        else:
            # Otherwise, it might be because Docker broke down or something.
            raise e

    log_path = os.path.join(repo_binary_dir, "log.pkl")
    makefiles: List[RepoDB.MakefileEntry] = []
    if os.path.exists(log_path):
        try:
            with open(log_path, "rb") as f:
                makefiles = pickle.load(f)
        except Exception:
            makefiles = []
        os.remove(log_path)
    return makefiles
