import tempfile
import unittest
from pathlib import Path
from typing import List, Tuple

import pycparser
from pycparser.c_ast import Node as ASTNode
from pycparser.c_generator import CGenerator

import ghcc
import match_functions

EXAMPLE_CODE: List[Tuple[str, List[int]]] = [  # [(code, [type_token_pos])]
    (r"""
typedef int some_type;

char * (*ret_func(void *(*a1)(long long))) (some_type a)
{ }

some_type * (*complete_sym(long long arg)) (long long a1, int a2) {
}
    """, [1, 4, 10, 17, 18, 23, 28, 34, 35, 40, 41, 44]),
]


class ParsingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.parser = pycparser.CParser(lexer=ghcc.parse.CachedCLexer)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _test_ast_equivalent(self, a: ASTNode, b: ASTNode) -> None:
        assert a.attr_names == b.attr_names
        for name in a.attr_names:
            assert getattr(a, name) == getattr(b, name)
        for (name_a, child_a), (name_b, child_b) in zip(a.children(), b.children()):
            assert name_a == name_b
            self._test_ast_equivalent(child_a, child_b)

    def test_pycparser(self) -> None:
        # Ensure we're using the right version of `pycparser` (and that we've cleaned generated tables from previous
        # versions) by parsing a string that takes exponential time in versions prior to 2.20.
        string = r"\xED\xFF\xFF\xEB\x04\xe0\x2d\xe5\x00\x00\x00\x00\xe0\x83\x22\xe5\xf1\x02\x03\x0e\x00\x00\xa0\xe3" \
                 r"\x02\x30\xc1\xe7\x00\x00\x53\xe3"
        code = rf'char *s = "{string}";'
        ast = self.parser.parse(code)
        assert ast.ext[0].init.value == f'"{string}"'

    def test_serialization(self) -> None:
        # Clone the `pycparser` repo.
        result = ghcc.clone("eliben", "pycparser", clone_folder=self.tempdir.name)
        assert result.success

        def _test(code: str):
            ast = self.parser.parse(code)
            json_dict = ghcc.parse.ast_to_dict(ast)
            deserialized_ast = ghcc.parse.dict_to_ast(json_dict)
            self._test_ast_equivalent(ast, deserialized_ast)

        for file in (Path(self.tempdir.name) / "eliben" / "pycparser" / "examples" / "c_files").iterdir():
            preprocessed_code = ghcc.parse.preprocess_file(str(file))
            _test(preprocessed_code)

        for code, _ in EXAMPLE_CODE:
            preprocessed_code = ghcc.parse.preprocess(code)
            _test(preprocessed_code)

    def test_token_pos(self) -> None:
        for code, type_token_pos in EXAMPLE_CODE:
            preprocessed_code = ghcc.parse.preprocess(code)
            ast = self.parser.parse(preprocessed_code)
            token_coords = ghcc.parse.convert_to_tokens(preprocessed_code, self.parser.clex.cached_tokens)
            json_dict = ghcc.parse.ast_to_dict(ast, token_coords)
            found_type_token_pos = set()

            def visit_fn(node: ghcc.parse.JSONNode, _) -> None:
                if (node[ghcc.parse.NODE_TYPE_ATTR] == "IdentifierType" and
                        node[ghcc.parse.TOKEN_POS_ATTR] is not None):
                    for idx in range(len(node["names"])):
                        found_type_token_pos.add(node[ghcc.parse.TOKEN_POS_ATTR] + idx)

            ghcc.parse.visit_dict(visit_fn, json_dict)
            assert found_type_token_pos == set(type_token_pos)


class MatchFunctionsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = pycparser.CParser(lexer=ghcc.parse.CachedCLexer)
        self.generator = CGenerator()
        self.lexer = ghcc.parse.LexerWrapper()

    def test_serialize(self) -> None:
        for code, _ in EXAMPLE_CODE:
            preprocessed_code = ghcc.parse.preprocess(code)
            ast = self.parser.parse(preprocessed_code)
            token_coords = ghcc.parse.convert_to_tokens(preprocessed_code, self.parser.clex.cached_tokens)
            functions = ghcc.parse.FunctionExtractor().find_functions(ast)
            for func_ast in functions.values():
                ast_dict, tokens = match_functions.serialize(func_ast, token_coords)
                original_code = self.lexer.lex(self.generator.visit(func_ast))
                assert tokens == original_code
