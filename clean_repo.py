#!/usr/bin/env python3
import argparse
import subprocess

import ghcc

parser = argparse.ArgumentParser()
parser.add_argument("folder", type=str)  # the folder to clean up
args = parser.parse_args()

try:
    ghcc.utils.run_docker_command(["rm", "-rf", "/usr/src/*"], user=0, directory_mapping={args.folder: "/usr/src"})
except subprocess.CalledProcessError as e:
    ghcc.log(f"Command failed with retcode {e.returncode}", "error")
    output = e.output.decode("utf-8")
    if len(output) > 200:
        output = output[:200] + "... (omitted)"
    ghcc.log("Captured output: " + output)
