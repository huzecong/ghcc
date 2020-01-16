# Runs the decompiler to collect variable names from binaries containing
# debugging information, then strips the binaries and injects the collected
# names into that decompilation output.
# This generates an aligned, parallel corpus for training translation models.

# Requires Python 3

import base64
import datetime
import errno
import functools
import os
import pickle
import subprocess
import tempfile
import traceback
from enum import Enum, auto
from pathlib import Path
from typing import Dict, NamedTuple, Optional, Tuple

import tqdm
from termcolor import colored

import ghcc

EnvDict = Dict[str, str]


class Arguments(ghcc.arguments.Arguments):
    binaries_dir: str = "binaries/"  # directory containing binaries
    output_dir: str = "decompile_output/"  # output directory
    log_file: str = "decompile-log.txt"
    ida: str = "/data2/jlacomis/ida/idat64"  # location of the `idat64` binary
    timeout: int = 30  # decompilation timeout
    n_procs: int = 16  # number of processes


args = Arguments()
scripts_dir = Path(__file__).parent / "scripts" / "decompiler_scripts"
COLLECT = str((scripts_dir / 'collect.py').absolute())
DUMP_TREES = str((scripts_dir / 'dump_trees.py').absolute())


def make_directory(dir_path: str) -> None:
    r"""Make a directory, with clean error messages."""
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if not os.path.isdir(dir_path):
            raise NotADirectoryError(f"'{dir_path}' is not a directory")
        if e.errno != errno.EEXIST:
            raise


def write_pseudo_registry():
    encoded_registry_data = b"""
aURhNwFIaWRkZW4gTWVzc2FnZXMAIHYgcyAgcyAgcyAgIHMgVG8gcmVuZXcgaXQgIHBsZWFzZSB2
aXNpdCBvdXIgd2ViIHNpdGUgICAgcyAgVGhhbmsgeW91IGZvciB1c2luZyBJREEgAAQAAAAEAQAA
AFRoZSB0ZWNobmljYWwgc3VwcG9ydCBwZXJpb2Qgb2YgdGhlIEhleCBSYXlzIGRlY29tcGlsZXIg
aGFzIGV4cGlyZWQgIFlvdSBoYXZlIDMgbW9udGggZ3JhY2UgcGVyaW9kIHQABAAAAAQAAAAAAAFI
aXN0b3J5NjQAMAAhAAAAAS9kYXRhMy96ZWNvbmcvRElSRS9kYXRhc2V0LWdlbi9scwBBdXRvQ2hl
Y2tVcGRhdGVzAAQAAAAEAAAAAEF1dG9SZXF1ZXN0VXBkYXRlcwAEAAAABAAAAABJbmZvcm1lZEFi
b3V0VXBkYXRlczIABAAAAAQBAAAATGljZW5zZSBDTVVfIFNvZnR3YXJlIEVuZ2luZWVyaW5nIElu
c3RpdHV0ZQAEAAAABAEAAABTZWFyY2hCaW4AAAAAAAFTZWFyY2hUZXh0AAAAAAABU3VwcG9ydEV4
cGlyZWREaXNwbGF5ZWRfNy4xLjAuMTgwMjI3AAQAAAAEuncKXnVpQ29uZmlnNjQAaAAAAAP/////
/////wAAAAAAAAAAAAAAAAAAAAACAAAAAQAAAAAAAAD/////AAABAP//////////AAAAAAAAAAAA
AAAAAAAAAAIAAAAAAAAAAAAAAAAAAAAAAAAATQABBQAAAAAAAAAAAAAAAGy/rUI=
    """
    path = os.path.expanduser("~/.idapro/ida.reg")
    with open(path, "wb") as f:
        f.write(base64.decodebytes(encoded_registry_data))


def run_decompiler(file_name: str, script: str, env: Optional[EnvDict] = None,
                   timeout: Optional[int] = None):
    r"""Run a decompiler script.

    :param file_name: The binary to be decompiled.
    :param env: An `os.environ` mapping, useful for passing arguments.
    :param script: The script file to run.
    :param timeout: Timeout in seconds (default no timeout).
    """
    idacall = [args.ida, '-B', f'-S{script}', file_name]
    try:
        ghcc.utils.run_command(idacall, env=env, timeout=timeout)
    except subprocess.CalledProcessError as e:
        if b"Traceback (most recent call last):" in e.output:
            # Exception raised by Python script called by IDA, throw it up.
            raise e
        ghcc.utils.run_command(['rm', '-f', f'{file_name}.i64'])
        if b"Corrupted pseudo-registry file" in e.output:
            write_pseudo_registry()
            # Run again without try-catch; if it fails, it should crash.
            ghcc.utils.run_command(idacall, env=env, timeout=timeout)


class DecompilationStatus(Enum):
    Success = auto()
    TimedOut = auto()
    NoVariables = auto()
    UnknownError = auto()


class DecompilationResult(NamedTuple):
    binary_path: str
    original_path: str
    status: DecompilationStatus
    time: Optional[datetime.timedelta] = None


def exception_handler(e, *args, _return: bool = True, **kwargs):
    binary_path, _ = args[0] if len(args) > 0 else kwargs["binary_path"]
    exc_msg = f"<{e.__class__.__qualname__}> {e}"
    try:
        if not (isinstance(e, subprocess.CalledProcessError) and e.output is not None):
            ghcc.log(traceback.format_exc(), "error")

        ghcc.log(f"Exception occurred when processing {binary_path}: {exc_msg}", "error")
    except Exception as log_e:
        print(f"Exception occurred when processing {binary_path}: {exc_msg}")
        print(f"Another exception occurred while logging: <{log_e.__class__.__qualname__}> {log_e}")
        raise log_e


@ghcc.utils.exception_wrapper(exception_handler)
def decompile(paths: Tuple[str, str], env: Optional[EnvDict] = None) -> DecompilationResult:
    binary_path, original_path = paths

    def create_result(status: DecompilationStatus, time: Optional[datetime.timedelta] = None) -> DecompilationResult:
        return DecompilationResult(binary_path, original_path, status, time)

    start = datetime.datetime.now()
    env = (env or {}).copy()
    env['PREFIX'] = os.path.split(binary_path)[1]
    file_path = os.path.join(args.binaries_dir, binary_path)

    # Create a temporary directory, since the decompiler makes a lot of additional
    # files that we can't clean up from here.
    with tempfile.TemporaryDirectory() as tempdir:
        with tempfile.NamedTemporaryFile(dir=tempdir) as collected_vars:
            # First collect variables.
            env['COLLECTED_VARS'] = collected_vars.name
            with tempfile.NamedTemporaryFile(dir=tempdir) as orig:
                ghcc.utils.run_command(['cp', file_path, orig.name])
                # Timeout after 30 seconds for first run.
                try:
                    run_decompiler(orig.name, COLLECT, env=env, timeout=args.timeout)
                except subprocess.TimeoutExpired:
                    return create_result(DecompilationStatus.TimedOut)
                try:
                    assert pickle.load(collected_vars)  # non-empty
                except:
                    return create_result(DecompilationStatus.NoVariables)
            # Make a new stripped copy and pass it the collected vars.
            with tempfile.NamedTemporaryFile(dir=tempdir) as stripped:
                ghcc.utils.run_command(['cp', file_path, stripped.name])
                ghcc.utils.run_command(['strip', '--strip-debug', stripped.name])
                # Dump the trees.
                # No timeout here, we know it'll run in a reasonable amount of
                # time and don't want mismatched files.
                run_decompiler(stripped.name, DUMP_TREES, env=env)
    end = datetime.datetime.now()
    duration = end - start
    return create_result(DecompilationStatus.Success, duration)


def main():
    env: EnvDict = os.environ.copy()
    env['IDALOG'] = '/dev/stdout'

    # Check for/create output directories
    output_dir = os.path.abspath(args.output_dir)
    env['OUTPUT_DIR'] = output_dir

    make_directory(output_dir)

    # Use RAM-backed memory for tmp if available
    if os.path.exists('/dev/shm'):
        tempfile.tempdir = '/dev/shm'

    ghcc.set_log_file(args.log_file)
    write_pseudo_registry()

    # Obtain a list of all binaries
    db = ghcc.Database()
    all_repos = db.collection.find()
    binaries: Dict[str, Tuple[str, str]] = {}  # sha -> (path_to_bin, orig_path_in_repo)
    for repo in tqdm.tqdm(all_repos, total=all_repos.count(), ncols=120, desc="Deduplicating binaries"):
        prefix = f"{repo['repo_owner']}/{repo['repo_name']}"
        for makefile in repo['makefiles']:
            directory = f"{prefix}/" + makefile['directory'][len("/usr/src/repo/"):]
            for path, sha in zip(makefile['binaries'], makefile['sha256']):
                binaries[sha] = (f"{prefix}/{sha}", f"{directory}/{path}")

    ghcc.log(f"{len(binaries)} binaries to process.")
    ghcc.set_logging_level("error", console=True)  # prevent `ghcc.log` from writing to console, since we're using tqdm
    file_count = 1
    progress = tqdm.tqdm(binaries.values(), ncols=120)
    pool = ghcc.utils.Pool(processes=args.n_procs)
    for result in pool.imap_unordered(functools.partial(decompile, env=env), progress):
        file_count += 1
        if result is None:
            continue  # exception raised
        if result.status is DecompilationStatus.Success:
            assert result.time is not None
            status_msg, color = f"[OK {result.time.total_seconds():5.2f}s]", "green"
        elif result.status is DecompilationStatus.TimedOut:
            status_msg, color = "[TIMED OUT]", "yellow"
        else:  # DecompilationStatus.NoVariables
            status_msg, color = "[NO VARS]", "yellow"
        message = f"{file_count}: {result.original_path} ({result.binary_path})"
        progress.write(f"{colored(status_msg, color)} {message}")
        ghcc.log(f"{status_msg} {message}", "info")  # only writes to log file
    pool.close()


if __name__ == '__main__':
    main()
