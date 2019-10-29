#!/usr/bin/env python3
import argparse
import hashlib
import os
import shutil
import time

import ghcc

parser = argparse.ArgumentParser()
parser.add_argument("--compile-timeout", type=int, default=900)  # wait up to 15 minutes
parser.add_argument("--record-libraries", action="store_true", default=False)
args = parser.parse_args()

SRC_REPO_PATH = "/usr/src/repo"
REPO_PATH = "/usr/src/repo_copy"
BINARY_PATH = "/usr/src/bin"

ENV = {
    **({"MOCK_GCC_LIBRARY_LOG": os.path.join(BINARY_PATH, "libraries.txt")} if args.record_libraries else {}),
}


def main():
    # try:
    #     shutil.copytree(SRC_REPO_PATH, REPO_PATH)
    # except shutil.Error:
    #     pass  # `shutil.copytree` follows symlinks, which could be broken
    global REPO_PATH
    REPO_PATH = SRC_REPO_PATH
    makefile_dirs = ghcc.find_makefiles(REPO_PATH)

    # Stage 3: Compile each Makefile.
    num_succeeded = 0
    makefiles = []
    binary_paths = []
    remaining_time = args.compile_timeout
    for make_dir in makefile_dirs:
        if remaining_time <= 0.0:
            break
        start_time = time.time()
        compile_result = ghcc.unsafe_make(make_dir, timeout=remaining_time, env=ENV)
        elapsed_time = time.time() - start_time
        remaining_time -= elapsed_time
        if compile_result.success:
            num_succeeded += 1
        if len(compile_result.elf_files) > 0:
            sha256 = []
            for path in compile_result.elf_files:
                path = os.path.join(make_dir, path)
                hash_obj = hashlib.sha256()
                with open(path, "rb") as f:
                    hash_obj.update(f.read())
                digest = hash_obj.hexdigest()
                sha256.append(digest)
                binary_path = os.path.join(BINARY_PATH, digest)
                binary_paths.append(binary_path)
                shutil.move(path, binary_path)
            makefiles.append({
                "directory": make_dir,
                "binaries": compile_result.elf_files,
                "sha256": sha256,
            })
    ghcc.clean(REPO_PATH)

    with open(os.path.join(BINARY_PATH, "log.txt"), "w") as f:
        f.write(f"{num_succeeded}\n")
        f.write(f"{len(makefiles)}\n")
        for makefile in makefiles:
            directory = makefile["directory"]
            binaries = makefile["binaries"]
            sha256 = makefile["sha256"]
            f.write(f"{len(binaries)} {directory}\n")
            for sha, path in zip(sha256, binaries):
                f.write(f"{sha} {path}\n")
    ghcc.utils.run_command(["chmod", "-R", "g+w", BINARY_PATH])
    ghcc.utils.run_command(["chmod", "-R", "g+w", SRC_REPO_PATH])


if __name__ == '__main__':
    main()
