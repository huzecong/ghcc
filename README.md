# GitHub Cloner & Compiler

This project serves as the data collection process for training neural decompilers, such as
[CMUSTRUDEL/DIRE](https://github.com/CMUSTRUDEL/DIRE).

The code for compilation is adapted from
[bvasiles/decompilationRenaming](https://github.com/bvasiles/decompilationRenaming).

## Usage

First, build the Docker image used for compiling programs by:
```bash
docker build -t gcc-custom .
```

You will need a list of GitHub repository URLs to run the code. To run, simply execute:
```bash
python main.py \
    --repo-list-file path/to/your/list \
    --clone-folder repos/ \
    --binary-folder binaries/ \
    --n-procs 32
```

If compilation is interrupted, remember to always purge leftover repositories by calling:
```bash
./clean_repos.py /path/to/clone/folder
``` 
This is because intermediate files are created under different permissions, and we need root privileges (sneakily
obtained via Docker) to purge those files. This is also performed at the beginning of the `main.py` script.

Additionally, if something messed up seriously, drop the database by:
```bash
python -m ghcc.database clear
```

If the code is modified, remember to rebuild the image since the `batch_make.py` script (run inside Docker to compile
Makefiles) depends on the library code.

## Heuristics for Compilation

The following procedure happens when compiling a Makefile:

1. Check if a file named `configure` exists in the same directory.
    - If true, call `chmod +x configure && ./configure`
2. Call `make --keep-going -j1`.

## Collecting and Installing Libraries

Most repositories require linking to external libraries. To collect libraries that are linked to in Makefiles, run the
script with the flag `--record-libraries path/to/library_log.txt`. Only libraries in commands that failed to execute
(GCC return code is non-zero) are recorded in the log file.

After gathering the library log, run `install_libraries.py path/to/library_log.txt` to resolve libraries to package
names (based on `apt-cache`). This step requires actually installing packages, so it's recommended to run it in a Docker
environment:
```bash
docker run --rm \
    -v /absolute/path/to/directory/:/usr/src/ \
    gcc-custom \
    "install_libraries.py /usr/src/library_log.txt"
```
This gives a list of packages to install. Add the list of packages to `Dockerfile` (the command that begins with
`RUN apt-get install -y --no-install-recommends`) and rebuild the image to apply changes.

## Notes on Docker Safety

Compiling random code from GitHub is basically equivalent to running `curl | bash`, and doing so in Docker would be like
`curl | sudo bash` as Docker (by default) doesn't protect you against kernel panics and fork bombs. The following notes
describe what is done to (partly) ensure safety of the host machine when compiling code.

1. Never run Docker as root. This means two things: 1) don't use `sudo docker run ...`, and 2) don't execute commands in
   Docker as the root user (default). The first goal can be achieved by create a `docker` user group, and the second
   can be achieved using a special entry-point: create a non-privileged user and use `gosu` to switch to that user and
   run commands.

   **Caveats:** When creating the non-privileged user, assign the same UID (user ID) or GID (group ID) as the host user,
   so files created inside the container can be accessed/modified by the host user.

2. Limit the number of processes. This is to prevent things like fork bombs or badly written recursive Makefiles from
   taking up the kernel memory. A simple solution is to use `ulimit -u <nprocs>` to set the maximum allowed number of
   processes, but such limits are on a per-user basis instead of a per-container or per-process-tree basis.

   What we can do is: for each container we spawn, create a user that has the same GID as the host user, but with a
   distinct UID, and call `ulimit` for that user. This serves as a workaround for per-container limits.
   
   Don't forget to `chmod g+w` for files that need to be accessed from host.
