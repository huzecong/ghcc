import hashlib
import os
import pickle
import shutil
import subprocess
import time
from enum import Enum, auto
from typing import Callable, Dict, Iterator, List, NamedTuple, Optional

from .database import RepoDB
from .repo import clean
from .utils.docker import run_docker_command
from .utils.run import run_command

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
    :return: A list of absolute paths to subdirectories that contain Makefiles.
    """
    directories = []
    for subdir, dirs, files in os.walk(path):
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


def _check_elf_fn(directory: str, file: str) -> bool:
    r"""Checks whether the specified file is a binary file.

    :param directory: The directory containing the Makefile.
    :param file: The path to the file to check, relative to the directory.
    :return: If ``True``, the file is a binary (ELF) file.
    """
    path = os.path.join(directory, file)
    output = subprocess.check_output(["file", path], timeout=10)
    output = output[len(path):]  # first part is file name
    return ELF_FILE_TAG in output


def _make_skeleton(directory: str, timeout: Optional[float] = None,
                   env: Optional[Dict[str, str]] = None,
                   verbose: bool = True,
                   *, make_fn,
                   check_file_fn: Callable[[str, str], bool] = _check_elf_fn) -> CompileResult:
    r"""A composable routine for different compilation methods. Different routines can be composed by specifying
    different ``make_fn``\ s and ``check_file_fn``\ s.

    :param directory: The directory containing the Makefile.
    :param timeout: Maximum compilation time.
    :param env: A dictionary of environment variables.
    :param verbose: If ``True``, print out executed commands and outputs.
    :param make_fn: The function to call for compilation. The function takes as input variables ``directory``,
        ``timeout``, and ``env``.
    :param check_file_fn: A function to determine whether a generated file should be collected, i.e., whether it is a
        binary file. The function takes as input variables ``directory`` and ``file``, where ``file`` is the path of the
        file to check, relative to ``directory``. Defaults to :meth:`_check_elf_fn`, which checks whether the file is an
        ELF file.
    """
    directory = os.path.abspath(directory)

    try:
        # Clean unversioned files by previous compilations.
        clean(directory)

        # Call the actual function for `make`.
        make_fn(directory, timeout=timeout, env=env, verbose=verbose)
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
            if check_file_fn(directory, file):
                result.elf_files.append(file)
    except subprocess.TimeoutExpired as e:
        return _create_result(elf_files=result.elf_files, error_type=CompileErrorType.Timeout, captured_output=e.output)
    except subprocess.CalledProcessError as e:
        return _create_result(elf_files=result.elf_files, error_type=CompileErrorType.Unknown, captured_output=e.output)
    except OSError as e:
        return _create_result(elf_files=result.elf_files, error_type=CompileErrorType.Unknown, captured_output=str(e))

    return result


def _unsafe_make(directory: str, timeout: Optional[float] = None, env: Optional[Dict[str, str]] = None,
                 verbose: bool = False) -> None:
    env = {"PATH": f"{MOCK_PATH}:{os.environ['PATH']}", **(env or {})}
    # Try GNU Automake first. Note that errors are ignored because it's possible that the original files still work.
    if contains_files(directory, ["configure.ac", "configure.in"]):
        start_time = time.time()
        if os.path.isfile(os.path.join(directory, "autogen.sh")):
            # Some projects with non-trivial build instructions provide an "autogen.sh" script.
            run_command(["chmod", "+x", "./autogen.sh"], env=env, cwd=directory, verbose=verbose)
            run_command(["./autogen.sh"], env=env, cwd=directory, timeout=timeout, verbose=verbose, ignore_errors=True)
        else:
            run_command(["autoreconf", "--force", "--install"],
                        env=env, cwd=directory, timeout=timeout, ignore_errors=True, verbose=verbose)
        end_time = time.time()
        if timeout is not None:
            timeout = max(1.0, timeout - int(end_time - start_time))

    # Try running `./configure` if it exists.
    if os.path.isfile(os.path.join(directory, "configure")):
        start_time = time.time()
        run_command(["chmod", "+x", "./configure"], env=env, cwd=directory, verbose=verbose)
        ret = run_command(["./configure", "--disable-werror"], env=env, cwd=directory, timeout=timeout,
                          verbose=verbose, ignore_errors=True)
        end_time = time.time()
        if ret.return_code != 0 and end_time - start_time <= 2:
            # The configure file might not support `--disable-werror` and died instantly. Try again without the flag.
            run_command(["./configure"], env=env, cwd=directory, timeout=timeout, verbose=verbose)
            end_time = time.time()
        if timeout is not None:
            timeout = max(1.0, timeout - int(end_time - start_time))

    # Make while ignoring errors.
    # `-B/--always-make` could give strange errors for certain Makefiles, e.g. ones containing "%:"
    try:
        run_command(["make", "--keep-going", "-j1"], env=env, cwd=directory, timeout=timeout, verbose=verbose)
    except subprocess.CalledProcessError as err:
        expected_msg = b"missing separator"
        if not (err.output is not None and expected_msg in err.output):
            raise err
        else:
            # Try again using BSD Make instead of GNU Make. Note BSD Make does not have a flag equivalent to
            # `-B/--always-make`.
            run_command(["bmake", "-k", "-j1"], env=env, cwd=directory, timeout=timeout, verbose=verbose)


def unsafe_make(directory: str, timeout: Optional[float] = None, env: Optional[Dict[str, str]] = None,
                verbose: bool = False) -> CompileResult:
    r"""Run ``make`` in the given directory and collect compilation outputs.

    .. warning::
        This will run ``make`` on your physical machine under the same privilege granted to the Python program.
        Never run programs from unvalidated sources as malicious programs could break your system.

    :param directory: Path to the directory containing the Makefile.
    :param timeout: Maximum time allowed for compilation, in seconds. Defaults to ``None`` (unlimited time).
    :param env: The environment variables to use when calling ``make``.
    :param verbose: If ``True``, print out executed commands and outputs.
    :return: An instance of :class:`CompileResult` indicating the result. Fields ``success`` and ``elf_files`` are not
        ``None``.

        - If compilation failed, the fields ``error_type`` and ``captured_output`` are also not ``None``.
    """
    return _make_skeleton(directory, timeout, env, verbose, make_fn=_unsafe_make)


def _docker_make(directory: str, timeout: Optional[float] = None, env: Optional[Dict[str, str]] = None,
                 verbose: bool = False) -> None:
    if os.path.isfile(os.path.join(directory, "configure")):
        # Try running `./configure` if it exists.
        run_docker_command("chmod +x configure && ./configure && make --keep-going -j1",
                           user=0, cwd="/usr/src", directory_mapping={directory: "/usr/src"},
                           timeout=timeout, shell=True, env=env, verbose=verbose)
    else:
        # Make while ignoring errors.
        # `-B/--always-make` could give strange errors for certain Makefiles, e.g. ones containing "%:"
        run_docker_command(["make", "--keep-going", "-j1"],
                           user=0, cwd="/usr/src", directory_mapping={directory: "/usr/src"},
                           timeout=timeout, env=env, verbose=verbose)


def docker_make(directory: str, timeout: Optional[float] = None, env: Optional[Dict[str, str]] = None,
                verbose: bool = False) -> CompileResult:
    r"""Run ``make`` within Docker and collect compilation outputs.

    .. note::
        The ``gcc-custom`` Docker image is used. You can build this using the Dockerfile under the root directory.

    .. warning::
        It is only possible to run one bash command in a (non-interactive) Docker session, the compilation heuristics
        used here is the original version. It is recommended to call `batch_make.py` instead of relying on this method.

    :param directory: Path to the directory containing the Makefile.
    :param timeout: Maximum time allowed for compilation, in seconds. Defaults to ``None`` (unlimited time).
    :param env: The environment variables to use when calling ``make``.
    :param verbose: If ``True``, print out executed commands and outputs.
    :return: An instance of :class:`CompileResult` indicating the result. Fields ``success`` and ``elf_files`` are not
        ``None``.

        - If compilation failed, the fields ``error_type`` and ``captured_output`` are also not ``None``.
    """
    return _make_skeleton(directory, timeout, env, verbose, make_fn=_docker_make)


def _hash_file_sha256(directory: str, path: str) -> str:
    r"""Generate the SHA256 hash signature of the file located at the specified path.

    :param path: Path to the file to compute signature for.
    :return: The SHA256 signature.
    """
    path = os.path.join(directory, path)
    hash_obj = hashlib.sha256()
    with open(path, "rb") as f:
        hash_obj.update(f.read())
    return hash_obj.hexdigest()


def compile_and_move(repo_binary_dir: str, repo_path: str, makefile_dirs: List[str],
                     compile_timeout: Optional[float] = None, record_libraries: bool = False,
                     gcc_override_flags: Optional[str] = None,
                     compile_fn=docker_make, hash_fn: Callable[[str, str], str] = _hash_file_sha256) \
        -> Iterator[RepoDB.MakefileEntry]:
    r"""Compile all Makefiles as provided, and move generated binaries to the binary directory.

    :param repo_binary_dir: Path to the directory where generated binaries for the repository will be stored.
    :param repo_path: Path to the repository.
    :param makefile_dirs: A list of all subdirectories containing Makefiles.
    :param compile_timeout: Maximum time allowed for compilation of all Makefiles, in seconds. Defaults to ``None``
        (unlimited time).
    :param record_libraries: If ``True``, A file named ``libraries.txt`` will be generated under
        :attr:`repo_binary_dir`, recording the libraries used in compilation. Defaults to ``False``.
    :param gcc_override_flags: If not ``None``, these flags will be appended to each invocation of GCC.
    :param compile_fn: The method to call for compilation. Possible values are :meth:`ghcc.unsafe_make` and
        :meth:`ghcc.docker_make` (default).
    :param hash_fn: The method to call to generate a hash signature for collected binaries. The binaries will be moved
        to :attr:`repo_binary_dir` and renamed to the generated hash signature. The function takes as input variables
        ``directory`` and ``file``, where ``directory`` is the path of the directory containing the Makefile, and
        ``file`` is the path of the binary, relative to ``directory``.
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
            hashes: List[str] = []
            for path in compile_result.elf_files:
                signature = hash_fn(make_dir, path)
                hashes.append(signature)
                full_path = os.path.join(make_dir, path)
                shutil.move(full_path, os.path.join(repo_binary_dir, signature))
            yield {
                "directory": make_dir,
                "success": compile_result.success,
                "binaries": compile_result.elf_files,
                "sha256": hashes,
            }
    clean(repo_path)


def docker_batch_compile(repo_binary_dir: str, repo_path: str,
                         compile_timeout: Optional[float] = None, record_libraries: bool = False,
                         gcc_override_flags: Optional[str] = None,
                         use_makefile_info_pkl: bool = False, verbose: bool = False,
                         user_id: Optional[int] = None, directory_mapping: Optional[Dict[str, str]] = None,
                         exception_log_fn=None) -> List[RepoDB.MakefileEntry]:
    r"""Run batch compilation in Docker.

    :param repo_binary_dir: Path to store collected binaries.
    :param repo_path: Path to the code repository.
    :param compile_timeout: Timeout for compilation.
    :param record_libraries: If ``True``, libraries used during compilation are written under
        ``repo_binary_dir/libraries.txt``.
    :param gcc_override_flags: Additional flags to pass to GCC during compilation.
    :param use_makefile_info_pkl: If ``True``, the caller must prepare a file named ``makefiles.pkl`` under
        ``repo_binary_dir``, that contains a pickled object of type ``Dict[str, Dict[str, str]]``, that maps Makefile
        directories to a mapping from binary paths to SHA256 hashes.
    :param verbose: If ``True``, print out executed commands and outputs.
    :param user_id: The user ID to use inside the Docker container. See :meth:`ghcc.utils.docker.run_docker_command`.
    :param directory_mapping: Additional directory mappings for Docker. Optional.
    :param exception_log_fn: A function to log exceptions occurred in Docker. The function takes the exception object
        as input and returns nothing.
    :return: A list of Makefile entries.
    """
    start_time = time.time()
    try:
        # Don't rely on Docker timeout, but instead constrain running time in script run in Docker. Otherwise we won't
        # get the results file if any compilation task timeouts.
        cmd = [
            "batch_make.py",
            *(["--record-libraries"] if record_libraries else []),
            *(["--compile-timeout", str(compile_timeout)] if compile_timeout is not None else []),
            # We use "--flag=value" instead of "--flag value" because the GCC flags are, you know, flags, which may be
            # incorrectly interpreted by `argparse`.
            *([f'--gcc-override-flags="{gcc_override_flags}"'] if gcc_override_flags is not None else []),
            *(["--use-makefile-info-pkl"] if use_makefile_info_pkl else []),
            *(["--verbose"] if verbose else []),
        ]
        ret = run_docker_command(cmd, user=user_id, return_output=True,
                                 directory_mapping={repo_path: "/usr/src/repo", repo_binary_dir: "/usr/src/bin",
                                                    **(directory_mapping or {})})
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
