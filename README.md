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
    --clone-folder c_repos/ \
    --binary-folder binaries/ \
    --n-procs 32
```
