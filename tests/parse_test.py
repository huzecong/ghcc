import tempfile
import unittest
from pathlib import Path

import pycparser
from pycparser.c_ast import Node as ASTNode

import ghcc


class ParseSerializationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _test_ast_equivalent(self, a: ASTNode, b: ASTNode) -> None:
        assert a.attr_names == b.attr_names
        for name in a.attr_names:
            assert getattr(a, name) == getattr(b, name)
        for (name_a, child_a), (name_b, child_b) in zip(a.children(), b.children()):
            assert name_a == name_b
            self._test_ast_equivalent(child_a, child_b)

    def test_serialization(self) -> None:
        # Clone the `pycparser` repo.
        result = ghcc.clone("eliben", "pycparser", clone_folder=self.tempdir.name)
        assert result.success

        parser = pycparser.CParser()
        for file in (Path(self.tempdir.name) / "eliben" / "pycparser" / "examples" / "c_files").iterdir():
            preprocessed_code = ghcc.parse.preprocess_file(str(file))
            ast = parser.parse(preprocessed_code)
            json_dict = ghcc.parse.ast_to_dict(ast)
            deserialized_ast = ghcc.parse.dict_to_ast(json_dict)
            self._test_ast_equivalent(ast, deserialized_ast)
