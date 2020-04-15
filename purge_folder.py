#!/usr/bin/env python3
import argparse
import os
import subprocess

import flutes

import ghcc

parser = argparse.ArgumentParser()
parser.add_argument("folder", type=str)  # the folder to clean up
parser.add_argument("-y", action="store_true", default=False)  # yes
args = parser.parse_args()

try:
    parent = os.path.abspath(os.path.join(args.folder, ".."))
    folder = os.path.split(os.path.abspath(args.folder))[1]
    yes = args.y
    if not yes:
        confirm = input(f"This will delete {parent} / {folder}. Confirm? [y/N] ")
        yes = confirm.lower() in ["y", "yes"]
    if yes:
        ghcc.utils.run_docker_command(["rm", "-rf", f"/usr/src/{folder}"],
                                      user=0, directory_mapping={parent: "/usr/src"})
except subprocess.CalledProcessError as e:
    flutes.log(f"Command failed with retcode {e.returncode}", "error")
    output = e.output.decode("utf-8")
    if len(output) > 200:
        output = output[:200] + "... (omitted)"
    flutes.log("Captured output: " + output)
