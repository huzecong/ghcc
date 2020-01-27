#!/usr/bin/env python3
import argparse
import functools
import multiprocessing as mp
import os
import pickle
import time
from typing import Dict, List, Optional

import ghcc

parser = argparse.ArgumentParser()
parser.add_argument("--compile-timeout", type=int, default=900)  # wait up to 15 minutes
parser.add_argument("--record-libraries", action="store_true", default=False)
parser.add_argument("--gcc-override-flags", type=str, default=None)
args = parser.parse_args()

TIMEOUT_TOLERANCE = 5  # allow worker process to run for maximum 5 seconds beyond timeout
REPO_PATH = "/usr/src/repo"
BINARY_PATH = "/usr/src/bin"


def worker(q: mp.Queue):
    makefile_dirs = ghcc.find_makefiles(REPO_PATH)

    for makefile in ghcc.compile_and_move(
            BINARY_PATH, REPO_PATH, makefile_dirs, compile_fn=ghcc.unsafe_make,
            compile_timeout=args.compile_timeout, record_libraries=args.record_libraries,
            gcc_override_flags=args.gcc_override_flags):
        # Modify makefile directory to use relative path.
        makefile.directory = os.path.relpath(makefile.directory, REPO_PATH)
        q.put(makefile)


def read_queue(makefiles: List[ghcc.RepoDB.MakefileEntry], q: 'mp.Queue[ghcc.RepoDB.MakefileEntry]'):
    try:
        while not q.empty():
            makefiles.append(q.get())
    except (OSError, ValueError):
        pass  # data in queue could be corrupt, e.g. if worker process is terminated while enqueueing


def main():
    q = mp.Queue()
    process = mp.Process(target=worker, args=(q,))
    process.start()
    start_time = time.time()

    makefiles: List[ghcc.RepoDB.MakefileEntry] = []
    while process.is_alive():
        time.sleep(2)  # no rush
        cur_time = time.time()
        if cur_time - start_time > args.compile_timeout + TIMEOUT_TOLERANCE:
            process.terminate()
            print(f"Timeout ({args.compile_timeout}s), killed", flush=True)
            ghcc.clean(REPO_PATH)  # clean up after the worker process
            break
        read_queue(makefiles, q)
    read_queue(makefiles, q)

    ghcc.utils.kill_proc_tree(os.getpid(), including_parent=False)  # make sure all subprocesses are dead
    with open(os.path.join(BINARY_PATH, "log.pkl"), "wb") as f:
        pickle.dump(makefiles, f)
    ghcc.utils.run_command(["chmod", "-R", "g+w", BINARY_PATH])
    ghcc.utils.run_command(["chmod", "-R", "g+w", REPO_PATH])


if __name__ == '__main__':
    main()
