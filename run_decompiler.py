# Runs the decompiler to collect variable names from binaries containing
# debugging information, then strips the binaries and injects the collected
# names into that decompilation output.
# This generates an aligned, parallel corpus for training translation models.

# Requires Python 3

import argparse
import datetime
import errno
import os
import pickle
import subprocess
import tempfile

import tqdm

import ghcc

scripts_dir = os.path.dirname(os.path.abspath(__file__))
COLLECT = os.path.join(scripts_dir, 'decompiler_scripts', 'collect.py')
DUMP_TREES = os.path.join(scripts_dir, 'decompiler_scripts', 'dump_trees.py')

parser = argparse.ArgumentParser(description="Run the decompiler to generate a corpus.")
parser.add_argument('--ida',
                    metavar='IDA',
                    help="location of the idat64 binary",
                    default='/data2/jlacomis/ida/idat64')
parser.add_argument('binaries_dir',
                    metavar='BINARIES_DIR',
                    help="directory containing binaries")
parser.add_argument('output_dir',
                    metavar='OUTPUT_DIR',
                    help="output directory")
parser.add_argument('--timeout', default=30, type=int, help="Decompilation timeout")

args = parser.parse_args()
env = os.environ.copy()
env['IDALOG'] = '/dev/stdout'

# Check for/create output directories
output_dir = os.path.abspath(args.output_dir)
env['OUTPUT_DIR'] = output_dir


def make_directory(dir_path):
    """Make a directory, with clean error messages."""
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if not os.path.isdir(dir_path):
            raise NotADirectoryError(f"'{dir_path}' is not a directory")
        if e.errno != errno.EEXIST:
            raise


make_directory(output_dir)

# Use RAM-backed memory for tmp if available
if os.path.exists('/dev/shm'):
    tempfile.tempdir = '/dev/shm'


def run_decompiler(file_name, env, script, timeout=None):
    """Run a decompiler script.

    Keyword arguments:
    file_name -- the binary to be decompiled
    env -- an os.environ mapping, useful for passing arguments
    script -- the script file to run
    timeout -- timeout in seconds (default no timeout)
    """
    idacall = [args.ida, '-B', f'-S{script}', file_name]
    try:
        output = subprocess.check_output(idacall, env=env, timeout=timeout)
    except subprocess.CalledProcessError as e:
        output = e.output
        subprocess.call(['rm', '-f', f'{file_name}.i64'])
    return output


db = ghcc.Database()
all_repos = db.collection.find()
binaries = {}
for repo in tqdm.tqdm(all_repos, total=all_repos.count(), ncols=120, desc="Deduplicating binaries"):
    prefix = f"{repo['repo_owner']}/{repo['repo_name']}"
    for makefile in repo['makefiles']:
        directory = f"{prefix}/" + makefile['directory'][len("/usr/src/repo/"):]
        for path, sha in zip(makefile['binaries'], makefile['sha256']):
            binaries[f"{prefix}/{sha}"] = f"{directory}/{path}"

# Create a temporary directory, since the decompiler makes a lot of additional
# files that we can't clean up from here
with tempfile.TemporaryDirectory() as tempdir:
    tempfile.tempdir = tempdir

    # File counts for progress output
    num_files = len(binaries)
    file_count = 1

    progress = tqdm.tqdm(binaries.items(), total=len(binaries), ncols=120)
    for binary, original_path in progress:
        progress.write(f"File {file_count}: {original_path} ({binary})")
        file_count += 1
        start = datetime.datetime.now()
        # print(f"Started: {start}")
        env['PREFIX'] = binary
        file_path = os.path.join(args.binaries_dir, binary)
        # print(f"Collecting from {file_path}")
        with tempfile.NamedTemporaryFile() as collected_vars:
            # First collect variables
            env['COLLECTED_VARS'] = collected_vars.name
            with tempfile.NamedTemporaryFile() as orig:
                subprocess.check_output(['cp', file_path, orig.name])
                # Timeout after 30 seconds for first run
                try:
                    run_decompiler(orig.name, env, COLLECT, timeout=args.timeout)
                except subprocess.TimeoutExpired:
                    progress.write("Timed out")
                    continue
                try:
                    if not pickle.load(collected_vars):
                        progress.write("No variables collected")
                        continue
                except:
                    progress.write("No variables collected")
                    continue
            # Make a new stripped copy and pass it the collected vars
            with tempfile.NamedTemporaryFile() as stripped:
                subprocess.call(['cp', file_path, stripped.name])
                subprocess.call(['strip', '--strip-debug', stripped.name])
                # print(f"{binary} stripped")
                # Dump the trees.
                # No timeout here, we know it'll run in a reasonable amount of
                # time and don't want mismatched files
                run_decompiler(stripped.name, env, DUMP_TREES)
        end = datetime.datetime.now()
        duration = end - start
        progress.write(f"Duration: {duration}")
