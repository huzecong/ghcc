# Runs the decompiler to collect variable names from binaries containing
# debugging information, then strips the binaries and injects the collected
# names into that decompilation output.
# This generates an aligned, parallel corpus for training translation models.

import base64
import contextlib
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
from typing import Dict, Iterator, NamedTuple, Optional, Tuple

import tqdm

import ghcc

EnvDict = Dict[str, str]


class Arguments(ghcc.arguments.Arguments):
    binaries_dir: str = "binaries/"  # directory containing binaries
    output_dir: str = "decompile_output/"  # output directory
    log_file: str = "decompile-log.txt"
    ida: str = "/data2/jlacomis/ida/idat64"  # location of the `idat64` binary
    binary_mapping_cache_file: Optional[str] = "binary_mapping.pkl"
    timeout: int = 30  # decompilation timeout
    n_procs: int = 0  # number of processes


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
    hash: str
    binary_path: str
    original_path: str
    status: DecompilationStatus
    time: Optional[datetime.timedelta] = None


def exception_handler(e, paths: Tuple[str, str]):
    binary_path, _ = paths
    ghcc.utils.log_exception(e, f"Exception occurred when processing {binary_path}")


@ghcc.utils.exception_wrapper(exception_handler)
def decompile(paths: Tuple[str, str], output_dir: str, binary_dir: str,
              timeout: Optional[int] = None) -> DecompilationResult:
    binary_path, original_path = paths
    binary_hash = os.path.split(binary_path)[1]

    def create_result(status: DecompilationStatus, time: Optional[datetime.timedelta] = None) -> DecompilationResult:
        return DecompilationResult(binary_hash, binary_path, original_path, status, time)

    output_path = os.path.join(output_dir, f"{binary_hash}.jsonl")
    if os.path.exists(output_path):
        # Binary already decompiled, but for some reason it wasn't written to the DB.
        return create_result(DecompilationStatus.Success)

    start = datetime.datetime.now()
    env: EnvDict = os.environ.copy()
    env['IDALOG'] = '/dev/stdout'
    env['PREFIX'] = binary_hash
    file_path = os.path.join(binary_dir, binary_path)

    # Create a temporary directory, since the decompiler makes a lot of additional
    # files that we can't clean up from here.
    with tempfile.TemporaryDirectory() as tempdir:
        # Put the output JSONL file here as well to prevent partially-generated files.
        env['OUTPUT_DIR'] = os.path.abspath(tempdir)
        with tempfile.NamedTemporaryFile(dir=tempdir) as collected_vars:
            # First collect variables.
            env['COLLECTED_VARS'] = collected_vars.name
            with tempfile.NamedTemporaryFile(dir=tempdir) as orig:
                ghcc.utils.run_command(['cp', file_path, orig.name])
                # Timeout after 30 seconds for first run.
                try:
                    run_decompiler(orig.name, COLLECT, env=env, timeout=timeout)
                except subprocess.TimeoutExpired:
                    ghcc.log(f"[TIMED OUT] {original_path} ({binary_path})", "warning")
                    return create_result(DecompilationStatus.TimedOut)
                try:
                    assert pickle.load(collected_vars)  # non-empty
                except:
                    ghcc.log(f"[NO VARS] {original_path} ({binary_path})", "warning")
                    return create_result(DecompilationStatus.NoVariables)
            # Make a new stripped copy and pass it the collected vars.
            with tempfile.NamedTemporaryFile(dir=tempdir) as stripped:
                ghcc.utils.run_command(['cp', file_path, stripped.name])
                ghcc.utils.run_command(['strip', '--strip-debug', stripped.name])
                # Dump the trees.
                # No timeout here, we know it'll run in a reasonable amount of
                # time and don't want mismatched files.
                run_decompiler(stripped.name, DUMP_TREES, env=env)
        jsonl_path = os.path.join(tempdir, f"{binary_hash}.jsonl")
        ghcc.utils.run_command(['cp', jsonl_path, output_path])
    end = datetime.datetime.now()
    duration = end - start
    ghcc.log(f"[OK {duration.total_seconds():5.2f}s] {original_path} ({binary_path})", "success")
    return create_result(DecompilationStatus.Success, duration)


def iter_binaries(db: ghcc.BinaryDB, binaries: Dict[str, Tuple[str, str]]) -> Iterator[Tuple[str, str]]:
    binary_entries = set(entry["sha"] for entry in db.collection.find())  # getting stuff in batch is much faster
    skipped_count = 0
    for sha, paths in binaries.items():
        if sha in binary_entries:
            skipped_count += 1
            continue
        if skipped_count > 0:
            ghcc.log(f"Skipped {skipped_count} binaries that have been processed", force_console=True)
            skipped_count = 0
        yield paths


def get_binary_mapping(cache_path: Optional[str] = None) -> Dict[str, Tuple[str, str]]:
    binaries: Dict[str, Tuple[str, str]] = {}  # sha -> (path_to_bin, orig_path_in_repo)
    if cache_path is not None and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            binaries = pickle.load(f)
        ghcc.log(f"Binary mapping cache loaded from '{cache_path}'")
    else:
        with contextlib.closing(ghcc.RepoDB()) as repo_db:
            all_repos = repo_db.collection.find()
            for repo in tqdm.tqdm(all_repos, total=all_repos.count(), ncols=120, desc="Deduplicating binaries"):
                prefix = f"{repo['repo_owner']}/{repo['repo_name']}"
                for makefile in repo['makefiles']:
                    directory = f"{prefix}/" + makefile['directory'][len("/usr/src/repo/"):]
                    for path, sha in zip(makefile['binaries'], makefile['sha256']):
                        binaries[sha] = (f"{prefix}/{sha}", f"{directory}/{path}")
        if cache_path is not None:
            with open(cache_path, "wb") as f:
                pickle.dump(binaries, f)
            ghcc.log(f"Binary mapping cache saved to '{cache_path}'")
    return binaries


def main():
    if args.n_procs == 0:
        # Only do this on the single-threaded case.
        ghcc.utils.register_ipython_excepthook()
    ghcc.log(f"Running with {args.n_procs} worker processes", "warning")

    # Check for/create output directories
    make_directory(args.output_dir)

    # Use RAM-backed memory for tmp if available
    if os.path.exists('/dev/shm'):
        tempfile.tempdir = '/dev/shm'

    ghcc.set_log_file(args.log_file)
    write_pseudo_registry()

    # Obtain a list of all binaries
    binaries = get_binary_mapping(args.binary_mapping_cache_file)

    ghcc.log(f"{len(binaries)} binaries to process.")
    file_count = 0
    db = ghcc.BinaryDB()

    with ghcc.utils.safe_pool(args.n_procs, closing=[db]) as pool:
        decompile_fn = functools.partial(
            decompile, output_dir=args.output_dir, binary_dir=args.binaries_dir, timeout=args.timeout)
        for result in pool.imap_unordered(decompile_fn, iter_binaries(db, binaries)):
            result: DecompilationResult
            file_count += 1
            if result is not None:
                db.add_binary(result.hash, result.status is DecompilationStatus.Success)
            if file_count % 100 == 0:
                ghcc.log(f"Processed {file_count} binaries", force_console=True)


if __name__ == '__main__':
    main()
