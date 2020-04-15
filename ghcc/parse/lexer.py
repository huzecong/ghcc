from typing import Iterator, List, NamedTuple

from pycparser.c_lexer import CLexer

__all__ = [
    "Token",
    "LexToken",
    "CachedCLexer",
    "convert_to_tokens",
    "LexerWrapper",
]


class Token(NamedTuple):
    name: str
    line: int
    column: int


class LexToken:  # stub
    type: str
    value: str
    lineno: int
    lexpos: int


class CachedCLexer(CLexer):
    # `ply` uses reflection to build the lexer, which somehow requires accessing the `__module__` attribute.
    __module__ = CLexer.__module__
    _cached_tokens: List[LexToken]

    def __init__(self, error_func, on_lbrace_func, on_rbrace_func, type_lookup_func) -> None:
        self._cached_tokens = []
        super().__init__(error_func, on_lbrace_func, on_rbrace_func, type_lookup_func)

    def reset_lineno(self):
        self._cached_tokens = []
        super().reset_lineno()

    def token(self) -> LexToken:
        tok = super().token()
        if tok is not None:
            self._cached_tokens.append(tok)
        return tok

    @property
    def cached_tokens(self) -> List[LexToken]:
        return self._cached_tokens


def convert_to_tokens(code: str, lex_tokens: List[LexToken]) -> List[Token]:
    # `line_start[lineno - 1]` stores the `lexpos` right before the beginning of line `lineno`.
    # So `tok.lexpos - line_start[tok.lineno - 1]` gives the column of the token.
    line_start = [-1] + [i for i, ch in enumerate(code) if ch == "\n"]
    tokens = []
    for tok in lex_tokens:
        tokens.append(Token(tok.value, tok.lineno, tok.lexpos - line_start[tok.lineno - 1]))
    return tokens


class LexerWrapper:
    @staticmethod
    def _error_func(msg, loc0, loc1):
        pass

    @staticmethod
    def _brace_func():
        pass

    @staticmethod
    def _type_lookup_func(typ):
        return False

    def __init__(self) -> None:
        self.lexer = CLexer(self._error_func, self._brace_func, self._brace_func, self._type_lookup_func)
        self.lexer.build(optimize=True, lextab='pycparser.lextab')

    def lex_tokens(self, code: str) -> Iterator[LexToken]:
        self.lexer.reset_lineno()
        self.lexer.input(code)
        while True:
            token = self.lexer.token()
            if token is None:
                break
            yield token

    def lex(self, code: str) -> List[str]:
        return [token.value for token in self.lex_tokens(code)]
