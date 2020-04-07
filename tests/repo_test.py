import os
import tempfile
import unittest

import ghcc


class RepoCloneTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_clone(self) -> None:
        # Clone an existing repo.
        result = ghcc.clone("huzecong", "memes", clone_folder=self.tempdir.name)
        self.assertTrue(result.success, msg=result.captured_output)
        self.assertTrue(os.path.exists(os.path.join(self.tempdir.name, "huzecong", "memes", "Get Memes.scpt")),
                        msg=result.captured_output)

        # Non-existent repo.
        result = ghcc.clone("huzecong", "non-existent-repo", clone_folder=self.tempdir.name)
        self.assertFalse(result.success, msg=result.captured_output)
        self.assertEqual(ghcc.CloneErrorType.PrivateOrNonexistent, result.error_type, msg=result.captured_output)

        # Timeout
        result = ghcc.clone("torvalds", "linux", clone_folder=self.tempdir.name, timeout=1)
        self.assertFalse(result.success, msg=result.captured_output)
        self.assertEqual(ghcc.CloneErrorType.Timeout, result.error_type, msg=result.captured_output)
