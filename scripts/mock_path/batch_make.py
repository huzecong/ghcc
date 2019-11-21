#!/usr/bin/env python3
import argparse
import os
import pickle

import ghcc

parser = argparse.ArgumentParser()
parser.add_argument("--compile-timeout", type=int, default=900)  # wait up to 15 minutes
parser.add_argument("--record-libraries", action="store_true", default=False)
args = parser.parse_args()

SRC_REPO_PATH = "/usr/src/repo"
REPO_PATH = "/usr/src/repo_copy"
BINARY_PATH = "/usr/src/bin"


def main():
    # try:
    #     shutil.copytree(SRC_REPO_PATH, REPO_PATH)
    # except shutil.Error:
    #     pass  # `shutil.copytree` follows symlinks, which could be broken
    global REPO_PATH
    REPO_PATH = SRC_REPO_PATH
    makefile_dirs = ghcc.find_makefiles(REPO_PATH)

    makefiles = ghcc.compile_and_move(
        BINARY_PATH, REPO_PATH, makefile_dirs, compile_fn=ghcc.unsafe_make,
        compile_timeout=args.compile_timeout, record_libraries=args.record_libraries)

    with open(os.path.join(BINARY_PATH, "log.pkl"), "wb") as f:
        pickle.dump(makefiles, f)
    ghcc.utils.run_command(["chmod", "-R", "g+w", BINARY_PATH])
    ghcc.utils.run_command(["chmod", "-R", "g+w", SRC_REPO_PATH])


if __name__ == '__main__':
    main()
