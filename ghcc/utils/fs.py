import os
import shutil
import subprocess

__all__ = [
    "get_folder_size",
    "readable_size",
    "get_file_lines",
    "remove_prefix",
    "copy_tree",
]


def get_folder_size(path: str) -> int:
    r"""Get disk usage of given path in bytes.

    Credit: https://stackoverflow.com/a/25574638/4909228
    """
    return int(subprocess.check_output(['du', '-bs', path]).split()[0].decode('utf-8'))


def readable_size(size: float) -> str:
    r"""Represent file size in human-readable format.

    :param size: File size in bytes.
    """
    units = ["", "K", "M", "G", "T"]
    for unit in units:
        if size < 1024:
            return f"{size:.2f}{unit}"
        size /= 1024
    return f"{size:.2f}P"  # this won't happen


def get_file_lines(path: str) -> int:
    r"""Get number of lines in text file.
    """
    return int(subprocess.check_output(['wc', '-l', path]).split()[0].decode('utf-8'))


def remove_prefix(s: str, prefix: str) -> str:
    r"""Remove the specified prefix from a string. If only parts of the prefix match, then only that part is removed.
    """
    prefix_len = next((idx for idx in range(len(prefix)) if s[idx] != prefix[idx]), len(prefix))
    return s[prefix_len:]


def copy_tree(src: str, dst: str, overwrite: bool = False):
    r"""Copy contents of folder ``src`` to folder ``dst``. The ``dst`` folder can exist or whatever (looking at you,
    :meth:`shutil.copytree`).

    :param src: The source directory.
    :param dst: The destination directory. If it doesn't exist, it will be created.
    :param overwrite: If ``True``, files in ``dst`` will be overwritten if a file with the same relative path exists
        in ``src``. If ``False``, these files are not copied. Defaults to ``False``.
    """
    os.makedirs(dst, exist_ok=True)
    for file in os.listdir(src):
        src_path = os.path.join(src, file)
        dst_path = os.path.join(dst, file)
        if os.path.isdir(src_path):
            copy_tree(src_path, dst_path, overwrite=overwrite)
        elif overwrite or not os.path.exists(dst_path):
            shutil.copy2(src_path, dst_path)
    shutil.copystat(src, dst)
