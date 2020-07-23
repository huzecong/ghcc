import os
import re
import tempfile
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import pycparser
from flutes.fs import remove_prefix
from flutes.run import run_command
from pycparser import c_ast
from pycparser.c_ast import Node as ASTNode
from pycparser.c_generator import CGenerator
from pycparser.c_parser import CParser

from .lexer import LexerWrapper

__all__ = [
    "FAKE_LIBC_PATH",
    "FAKE_LIBC_END_LINE",
    "FunctionExtractor",
    "FunctionReplacer",
    "parse_decompiled_code",
    "PreprocessError",
    "preprocess",
    "preprocess_file",
]

FAKE_LIBC_PATH = str((Path(__file__).parent.parent.parent / "scripts" / "fake_libc_include").absolute())
FAKE_LIBC_END_LINE = "typedef int __end_of_fake_libc__;"


class FunctionExtractor:
    r"""An wrapped AST visitor that extracts all function definitions from an AST."""

    class FuncDefVisitor(c_ast.NodeVisitor):
        func_def_asts: Dict[str, ASTNode]

        def visit_FuncDef(self, node: c_ast.FuncDef):
            func_name = node.decl.name
            self.func_def_asts[func_name] = node

    def __init__(self):
        self._visitor = self.FuncDefVisitor()

    def find_functions(self, node: ASTNode) -> Dict[str, ASTNode]:
        r"""Find all function definitions given an AST.

        :param node: The ``pycparser`` AST node object.
        :return: A dictionary mapping function names to function definition AST subtrees.
        """
        ret = self._visitor.func_def_asts = {}
        self._visitor.visit(node)
        del self._visitor.func_def_asts
        return ret


class FunctionReplacer(CGenerator):
    r"""An AST visitor inherited from ``pycparser.CGenerator``, but replaces ``FuncDef`` nodes with other function code.

    Use the :meth:`visit` method to generate code with functions replaced.
    """

    BOUNDARY_PREFIX = "typedef int __func__"
    BEGIN_SUFFIX = "__begin;"
    END_SUFFIX = "__end;"

    def __init__(self, func_defs: Dict[str, str]):
        super().__init__()
        self._func_defs = func_defs

    def visit_FuncDef(self, node: c_ast.FuncDef):
        func_name = node.decl.name
        if func_name in self._func_defs:
            # Add dummy typedefs around the function code, so we can still extract raw code even if we can't parse it.
            func_begin = self.BOUNDARY_PREFIX + func_name + self.BEGIN_SUFFIX
            func_end = self.BOUNDARY_PREFIX + func_name + self.END_SUFFIX
            return "\n".join(["", func_begin, self._func_defs[func_name], func_end, ""])
        return super().visit_FuncDef(node)

    def extract_func_name(self, line: str) -> Tuple[Optional[str], bool]:
        r"""Extracts the function name from a function boundary marker.

        :param line: The line of code containing the boundary marker.
        :return: A tuple containing the function name and a ``bool`` indicating whether the marker indicates the
            beginning of a function. If the line is not a boundary marker, or the function name is not recognized,
            ``None`` is returned instead of the extracted name.
        """
        func_name = None
        is_begin = False
        if line.startswith(self.BOUNDARY_PREFIX):
            if line.endswith(self.BEGIN_SUFFIX):
                func_name = line[len(self.BOUNDARY_PREFIX):-len(self.BEGIN_SUFFIX)]
                is_begin = True
            if line.endswith(self.END_SUFFIX):
                func_name = line[len(self.BOUNDARY_PREFIX):-len(self.END_SUFFIX)]
        if func_name is not None and func_name in self._func_defs:
            return func_name, is_begin
        return None, False


class PreprocessError(Exception):
    pass


LINE_CONTROL_REGEX = re.compile(r'^#[^\n]*$', flags=re.MULTILINE)


def _preprocess(input_path: str, output_path: str) -> str:
    compile_ret = run_command(
        ["gcc", "-E", "-nostdlib", "-I" + FAKE_LIBC_PATH, "-o", output_path, input_path], ignore_errors=True)

    if compile_ret.return_code != 0:
        if compile_ret.captured_output is not None:
            raise PreprocessError(compile_ret.captured_output.decode("utf-8"))
        raise PreprocessError

    with open(output_path, "r") as f:
        preprocessed_code = f.read()
    # Remove line control macros so we can programmatically locate errors.
    preprocessed_code = LINE_CONTROL_REGEX.sub("", preprocessed_code)
    return preprocessed_code


def preprocess(code: str) -> str:
    r"""Run preprocessor on code snippet by invoking GCC with ``-E`` flag.

    :raises PreprocessError: When GCC returns non-zero code.

    :return: The preprocessed code.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = os.path.join(temp_dir, "test.c")
        output_path = os.path.join(temp_dir, "test.prep.c")
        with open(input_path, "w") as f:
            f.write(code)
        return _preprocess(input_path, output_path)


def preprocess_file(path: str) -> str:
    r"""Run preprocessor on given file by invoking GCC with ``-E`` flag.

    :raises PreprocessError: When GCC returns non-zero code.

    :return: The preprocessed code.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, "test.prep.c")
        return _preprocess(path, output_path)


PARSE_ERROR_REGEX = re.compile(r'.*?:(?P<line>\d+):(?P<col>\d+): (?P<msg>.+)')


def parse_decompiled_code(code: str, lexer: LexerWrapper, parser: CParser,
                          max_type_fix_tries: int = 10) -> Tuple[ASTNode, str]:
    r"""Parse preprocessed decompiled code and heuristically fix errors caused by undefined types.

    If a parse error is encountered, we attempt to fix the code by parsing the error message and checking whether if
    could be an undefined type error. If it is, we prepend a dummy ``typedef`` and retry parsing, until either the code
    parses or we run out of tries.

    :raises ValueError: When we've run out of tries for fixing types, or the issue cannot be resolved by adding a
        ``typedef`` (i.e., getting the same error after adding ``typedef``).
    :raises pycparser.c_parser.ParseError: When we cannot identify the error.

    :param code: The preprocessed code to parse
    :param lexer: The lexer to use while parsing.
    :param parser: The parser to use while parsing.
    :param max_type_fix_tries: Maximum retries to fix type errors.
    :return: A tuple containing the parsed AST and the modified code.
    """
    added_types: Set[str] = set()
    code_lines = code.split("\n")
    for _ in range(max_type_fix_tries):
        try:
            decompiled_ast = parser.parse(code)
            break
        except pycparser.c_parser.ParseError as e:
            error_match = PARSE_ERROR_REGEX.match(str(e))
            if error_match is None or not error_match.group("msg").startswith("before: "):
                raise
            before_token = remove_prefix(error_match.group("msg"), "before: ")
            error_line = code_lines[int(error_match.group("line")) - 1]
            error_pos = int(error_match.group("col")) - 1
            tokens = list(lexer.lex_tokens(error_line))
            try:
                error_token_idx = next(
                    idx for idx, token in enumerate(tokens)
                    if token.lexpos == error_pos and token.value == before_token)
                # There are multiple possible cases here:
                # 1. The type is the first ID-type token before the reported token (`type token`). It might not
                #    be the one immediately in front (for example, `(type) token`， `type *token`).
                # 2. The type is the token itself. This is rare and only happens in a situation like:
                #      `int func(const token var)`  or  `int func(int a, token b)`
                #    Replacing `const` with any combination of type qualifiers also works.
                if (error_token_idx > 0 and
                        tokens[error_token_idx - 1].type in ["CONST", "VOLATILE", "RESTRICT",
                                                             "__CONST", "__RESTRICT", "__EXTENSION__",
                                                             "COMMA"]):
                    type_token = tokens[error_token_idx]
                else:
                    type_token = next(
                        tokens[idx] for idx in range(error_token_idx - 1, -1, -1)
                        if tokens[idx].type == "ID")
            except StopIteration:
                # If we don't catch this, it would terminate the for-loop in `main()`. Stupid design.
                raise e from None

            if type_token.value in added_types:
                raise ValueError(f"Type {type_token.value} already added (types so far: {list(added_types)})")
            added_types.add(type_token.value)
            typedef_line = f"typedef int {type_token.value};"
            code = typedef_line + "\n" + code
            code_lines.insert(0, typedef_line)
    else:
        raise ValueError(f"Type fixes exceeded limit ({max_type_fix_tries})")
    return decompiled_ast, code
