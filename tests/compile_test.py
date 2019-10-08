import os
import tempfile
import unittest

import ghcc


class CompileTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_compile(self):
        # Clone an existing repo.
        result = ghcc.clone("pjreddie", "uwimg", clone_folder=self.tempdir.name)
        self.assertTrue(result.success, msg=result.captured_output)

        # Find Makefiles.
        directory = os.path.join(self.tempdir.name, result.repo_owner, result.repo_name)
        makefiles = ghcc.find_makefiles(directory)
        self.assertEqual([directory], makefiles)

        # Try compile.
        # NOTE: This test will fail under macOS, since file types are different under macOS (Mach-O).
        result = ghcc.make(makefiles[0])
        self.assertTrue(result.success, msg=result.captured_output)
        target_elfs = [
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
        self.assertEqual({os.path.join(directory, file) for file in target_elfs},
                         set(result.elf_files), msg=result.captured_output)
