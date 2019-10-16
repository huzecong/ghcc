import os
import tempfile
import unittest

import ghcc


class CompileTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _test_compile(self, compile_func):
        # Clone an existing repo.
        result = ghcc.clone("pjreddie", "uwimg", clone_folder=self.tempdir.name, skip_if_exists=False)
        self.assertTrue(result.success, msg=result.captured_output)

        # Find Makefiles.
        directory = os.path.join(self.tempdir.name, result.repo_owner, result.repo_name)
        makefiles = ghcc.find_makefiles(directory)
        self.assertEqual([directory], makefiles)

        # Try compile.
        result = compile_func(makefiles[0], timeout=15)
        self.assertTrue(result.success, msg=result.captured_output)
        target_elfs = [os.path.join(directory, file) for file in [
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
        ]]
        self.assertEqual(set(target_elfs), set(result.elf_files), msg=result.captured_output)

        # Check if binaries contain debugging information (whether mock GCC works).
        for elf in target_elfs:
            # NOTE: This doesn't work under macOS either.
            ret = ghcc.utils.run_command(f"objdump --syms {elf} | grep debug | wc -l", return_output=True, shell=True)
            self.assertGreater(int(ret.captured_output.decode('utf-8')), 0)

    def test_compile(self):
        # NOTE: This test will fail under macOS, since file types are different under macOS (Mach-O).
        self._test_compile(ghcc.unsafe_make)

    def test_docker_compile(self):
        self._test_compile(ghcc.docker_make)
