import os
import subprocess
import tempfile
import unittest
from typing import List

import flutes

import ghcc


class CompileTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.repo_owner = "pjreddie"
        self.repo_name = "uwimg"

        # Clone an existing repo.
        result = ghcc.clone(self.repo_owner, self.repo_name, clone_folder=self.tempdir.name, skip_if_exists=False)
        assert result.success is True, result.captured_output

        self.directory = os.path.join(self.tempdir.name, self.repo_owner, self.repo_name)
        self.target_elfs = [
            "libuwimg.so",
            "obj/args.o",
            "obj/classifier.o",
            "obj/data.o",
            "obj/filter_image.o",
            "obj/flow_image.o",
            "obj/harris_image.o",
            "obj/image_opencv.o",
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

    def _test_debug_info(self, elf_paths: List[str]) -> None:
        # Check if binaries contain debugging information (whether mock GCC works).
        for elf in elf_paths:
            # NOTE: This doesn't work under macOS.
            ret = flutes.run_command(f"objdump --syms {elf} | grep debug | wc -l", return_output=True, shell=True)
            assert int(ret.captured_output.decode('utf-8')) > 0

    def _test_compile(self, compile_func) -> None:
        # Find Makefiles.
        makefiles = ghcc.find_makefiles(self.directory)
        self.assertEqual([self.directory], makefiles)

        # Try compile.
        result = compile_func(makefiles[0], timeout=15)
        assert result.success is True, result.captured_output
        assert set(self.target_elfs) == set(result.elf_files), result.captured_output

        elf_paths = [os.path.join(self.directory, elf) for elf in self.target_elfs]
        self._test_debug_info(elf_paths)

    def test_compile(self) -> None:
        # NOTE: This doesn't work under macOS.
        self._test_compile(ghcc.unsafe_make)

    def test_docker_compile(self) -> None:
        ghcc.utils.verify_docker_image()
        self._test_compile(ghcc.docker_make)

    def test_docker_batch_compile(self) -> None:
        ghcc.utils.verify_docker_image()
        binary_dir = os.path.join(self.tempdir.name, "_bin")
        os.makedirs(binary_dir)
        result = ghcc.docker_batch_compile(binary_dir, self.directory, 20, record_libraries=True, user_id=0)
        assert len(result) == 1
        assert set(self.target_elfs) == set(result[0]["binaries"])

        elf_paths = [os.path.join(binary_dir, file) for file in result[0]["sha256"]]
        self._test_debug_info(elf_paths)

    def test_gcc_library_log(self) -> None:
        from ghcc.compile import MOCK_PATH
        library_log_path = os.path.join(self.tempdir.name, "libraries.txt")
        env = {
            "PATH": f"{MOCK_PATH}:{os.environ['PATH']}",
            "MOCK_GCC_LIBRARY_LOG": library_log_path,
        }
        libraries = ["pthread", "m", "opencv", "openmp", "library_with_random_name"]
        try:
            flutes.run_command(
                ["gcc", *[f"-l{lib}" for lib in libraries], "nonexistent_file.c"], env=env)
        except subprocess.CalledProcessError:
            pass  # error must occur because file is nonexistent
        assert os.path.exists(library_log_path)
        with open(library_log_path) as f:
            recorded_libraries = f.read().split()
            assert set(libraries) == set(recorded_libraries)
