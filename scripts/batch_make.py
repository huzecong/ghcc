#!/usr/bin/env python3
import argparse
import hashlib
import os
import shutil
import time

import ghcc

parser = argparse.ArgumentParser()
parser.add_argument("--compile-timeout", type=int, default=900)  # wait up to 15 minutes
args = parser.parse_args()


def main():
    makefile_dirs = ghcc.find_makefiles("/usr/src/repo")

    # Stage 3: Compile each Makefile.
    num_succeeded = 0
    makefiles = []
    remaining_time = args.compile_timeout
    for make_dir in makefile_dirs:
        if remaining_time <= 0.0:
            break
        start_time = time.time()
        compile_result = ghcc.unsafe_make(make_dir, timeout=remaining_time)
        elapsed_time = time.time() - start_time
        remaining_time -= elapsed_time
        if compile_result.success:
            num_succeeded += 1
            sha256 = []
            for path in compile_result.elf_files:
                path = os.path.join(make_dir, path)
                hash_obj = hashlib.sha256()
                with open(path, "rb") as f:
                    hash_obj.update(f.read())
                digest = hash_obj.hexdigest()
                sha256.append(digest)
                shutil.move(path, os.path.join("/usr/src/bin", digest))
            makefiles.append({
                "directory": make_dir,
                "binaries": compile_result.elf_files,
                "sha256": sha256,
            })

    with open("/usr/src/bin/log.txt", "w") as f:
        f.write(f"{num_succeeded}\n")
        for makefile in makefiles:
            directory = makefile["directory"]
            binaries = makefile["binaries"]
            sha256 = makefile["sha256"]
            f.write(f"{len(binaries)} {directory}\n")
            for sha, path in zip(sha256, binaries):
                f.write(f"{sha} {path}\n")


if __name__ == '__main__':
    main()
