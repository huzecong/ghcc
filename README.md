# GitHub Cloner & Compiler

This project serves as the data collection process for training neural decompilers, such as
[CMUSTRUDEL/DIRE](https://github.com/CMUSTRUDEL/DIRE).

The code for compilation is adapted from
[bvasiles/decompilationRenaming](https://github.com/bvasiles/decompilationRenaming). The code for decompilation is
adapted from [CMUSTRUDEL/DIRE](https://github.com/CMUSTRUDEL/DIRE). 


## Setup

1. Install [Docker](https://docs.docker.com/install/) and [MongoDB](https://docs.mongodb.com/manual/installation/).
2. Install required Python packages by:
   ```bash
   pip install -r requirements.txt
   ```
3. Rename `database-config-example.json` to `database-config.json`, and fill in appropriate values. This will be used
   to connect to your MongoDB server.
4. Build the Docker image used for compiling programs by:
   ```bash
   docker build -t gcc-custom .
   ```


## Usage

### Running the Compiler

You will need a list of GitHub repository URLs to run the code. The current code expects one URL per line, for example:
```
https://github.com/huzecong/ghcc.git
https://www.github.com/torvalds/linux
FFmpeg/FFmpeg
https://api.github.com/repos/pytorch/pytorch
```

To run, simply execute:
```bash
python main.py --repo-list-file path/to/your/list [arguments...]
```

The following arguments are supported:

- `--repo-list-file [path]`: Path to the list of repository URLs.
- `--clone-folder [path]`: The temporary directory to store cloned repository files. Defaults to `repos/`.
- `--binary-folder [path]`: The directory to store compiled binaries. Defaults to `binaries/`.
- `--archive-folder [path]`: The directory to store archived repository files. Defaults to `archives/`.
- `--n-procs [int]`: Number of worker processes to spawn. Defaults to 0 (single-process execution).
- `--log-file [path]`: Path to the log file. Defaults to `log.txt`.
- `--clone-timeout [int]`: Maximum cloning time (seconds) for one repository. Defaults to 600 (10 minutes).
- `--force-reclone`: If specified, all repositories are cloned regardless of whether it has been processed before or
  whether an archived version exists.
- `--compile-timeout [int]`: Maximum compilation time (seconds) for all Makefiles under a repository. Defaults to 900
  (15 minutes).
- `--force-recompile`: If specified, all repositories are compiled regardless of whether is has been processed before.
- `--docker-batch-compile`: Batch compile all Makefiles in one repository using one Docker invocation. This is on by
  default, and you almost always want this. Use the `--no-docker-batch-compile` flag to disable it. 
- `--compression-type [str]`: Format of the repository archive, available options are `gzip` (faster) and `xz`
  (smaller). Defaults to `gzip`.
- `--max-archive-size [int]`: Maximum size (bytes) of repositories to archive. Repositories with greater sizes will not
  be archived. Defaults to 104,857,600 (100MB).
- `--record-libraries [path]`: If specified, a list of libraries used during failed compilations will be written to the
  specified path. See [Collecting and Installing Libraries](#collecting-and-installing-libraries) for details.
- `--logging-level [str]`: The logging level. Defaults to `info`.
- `--max-repos [int]`: If specified, only the first `max_repos` repositories from the list will be processed.
- `--recursive-clone`: If specified, submodules in the repository will also be cloned if exists. This is on by default.
  Use the `--no-recursive-clone` flag to disable it.
- `--write-db`: If specified, compilation results will be written to database. This is on by default. Use the
  `--no-write-db` flag to disable it.
- `--record-metainfo`: If specified, additional statistics will be recorded.
- `--gcc-override-flags`: If specified, these are passed as compiler flags to GCC. By default `-O0` is used.

### Utilities

- If compilation is interrupted, there may be leftovers that cannot be removed due to privilege issues. Purge them by:
  ```bash
  ./purge_folder.py /path/to/clone/folder
  ``` 
  This is because intermediate files are created under different permissions, and we need root privileges (sneakily
  obtained via Docker) to purge those files. This is also performed at the beginning of the `main.py` script.
- If something messed up seriously, drop the database by:
  ```bash
  python -m ghcc.database clear
  ```
- If the code is modified, remember to rebuild the image since the `batch_make.py` script (executed inside Docker to
  compile Makefiles) depends on the library code. If you don't do so, well, GHCC will remind you and refuse to proceed.

### Running the Decompiler

Decompilation requires an active installation of IDA with the Hex-Rays plugin. To run, simply execute:
```bash
python run_decompiler.py --ida path/to/idat64 [arguments...]
```

The following arguments are supported:

- `--ida [path]`: Path to the `idat64` executable found under the IDA installation folder.
- `--binaries-dir [path]`: The directory where binaries are stored, i.e. the same value for `--binary-folder` in the
  compilation arguments. Defaults to `binaries/`.
- `--output-dir [path]`: The directory to store decompiled code. Defaults to `decompile_output/`. 
- `--log-file [path]`: Path to the log file. Defaults to `decompile-log.txt`.
- `--timeout [int]`: Maximum decompilation time (seconds) for one binary. Defaults to 30.
- `--n-procs [int]`: Number of worker processes to spawn. Defaults to 0 (single-process execution). 


## Advanced Topics

### Heuristics for Compilation

The following procedure happens when compiling a Makefile:

1. **Check if directory is "make"-able:** A directory is marked as "make"-able if it contains (case-insensitively) at
   least one set of files among the following:

   - *(Make)* `Makefile`
   - *(automake)* `Makefile.am`

   If the directory is not "make"-able, skip the following steps.

2. **Clean Git repository:**

   ```bash
   git reset --hard  # reset modified files
   git clean -xffd  # clean unversioned files
   # do the same for submodules
   git submodule foreach --recursive git reset --hard
   git submodule foreach --recursive git clean -xffd
   ```

   If any command fails, ignore it and continue executing the rest.

3. **Build:**

   1. If exists a file named `Makefile.am`, run `automake`:

      ```bash
      autoreconf && automake --add-missing
      ```

   2. If exists a file named `configure`, run the configuration script:

      ```bash
      chmod +x ./configure && ./configure --disable-werror
      ```

      The `--disable-werror` prevents warnings being treated as errors in cases where `-Werror` is specified.
      
      If command fails within 2 seconds, try again without `--disable-werror`.

   3. Run `make`:

      ```bash
      make --always-make --keep-going -j1
      ```
      
      The `--always-make` flag rebuilds all dependent targets even if they exist. The `--keep-going` flag allows Make to
      continue for targets if errors occur in non-dependent targets.

      If command fails within 2 seconds and the output contains `"Missing separator"`, try again with `bmake`
      *(BSD Make)*.

      **Note:** We override certain program with our "wrapped" versions by modifying the `PATH` variable. The list of
      wrapped programs are:

      - **GCC:** (`gcc`, `cc`, `clang`) Swallows unnecessary and/or error-prone flags (`-Werror`, `-march`,
        `-mlittle-endian`), records libraries used (`-l`), overrides the optimization level (`-O0`), adds override flags
        specified in the arguments, and calls the real GCC. If the real GCC fails, writes the libraries to a predefined
        path.
      - **`sudo`:** Does not prompt for the password, but instead just tries to execute the command without privileges.
      - **`pkg-config`:** Records libraries used, and calls the real `pkg-config`. If it fails (meaning packages cannot
        be resolved), write the libraries to a predefined path.

### Collecting and Installing Libraries

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

### Notes on Docker Safety

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
