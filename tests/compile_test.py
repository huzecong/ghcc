import os
import tempfile
import unittest

from typing import List

import ghcc
from main import _docker_batch_compile


class CompileTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()

        # Clone an existing repo.
        result = ghcc.clone("pjreddie", "uwimg", clone_folder=self.tempdir.name, skip_if_exists=False)
        assert result.success is True, result.captured_output

        self.directory = os.path.join(self.tempdir.name, result.repo_owner, result.repo_name)
        self.target_elfs = [
            "libuwimg.so",
            "obj/args.o",
            "obj/classifier.o",
            "obj/data.o",
            "obj/filter_image.o",
            "obj/flow_image.o",
            "obj/harris_image.o",
            "obj/list.o",
            "obj/load_image.o",
            "obj/main.o",
            "obj/matrix.o",
            "obj/panorama_image.o",
            "obj/process_image.o",
            "obj/resize_image.o",
            "obj/test.o",
            "uwimg",
        ]

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _test_debug_info(self, elf_paths: List[str]):
        # Check if binaries contain debugging information (whether mock GCC works).
        for elf in elf_paths:
            # NOTE: This doesn't work under macOS.
            ret = ghcc.utils.run_command(f"objdump --syms {elf} | grep debug | wc -l", return_output=True, shell=True)
            assert int(ret.captured_output.decode('utf-8')) > 0

    def _test_compile(self, compile_func):
        # Find Makefiles.
        makefiles = ghcc.find_makefiles(self.directory)
        self.assertEqual([self.directory], makefiles)

        # Try compile.
        result = compile_func(makefiles[0], timeout=15)
        assert result.success is True, result.captured_output
        assert set(self.target_elfs) == set(result.elf_files), result.captured_output

        elf_paths = [os.path.join(self.directory, elf) for elf in self.target_elfs]
        self._test_debug_info(elf_paths)

    def test_compile(self):
        self._test_compile(ghcc.unsafe_make)

    def test_docker_compile(self):
        self._test_compile(ghcc.docker_make)

    def test_docker_batch_compile(self):
        binary_dir = os.path.join(self.tempdir.name, "_bin")
        os.makedirs(binary_dir)
        num_succeeded, makefiles = _docker_batch_compile(0, binary_dir, self.directory, 20)
        assert num_succeeded == 1
        assert len(makefiles) == 1
        assert set(self.target_elfs) == set(makefiles[0]["binaries"])

        elf_paths = [os.path.join(binary_dir, file) for file in makefiles[0]["sha256"]]
        self._test_debug_info(elf_paths)
