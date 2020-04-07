import functools
import re
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

from pycparser.c_ast import Node as ASTNode
from pycparser.plyparser import Coord

from .lexer import TokenCoord

__all__ = [
    "ast_to_json",
    "JSONNode",
]

T = TypeVar('T')
MaybeList = Union[T, List[T]]
JSONNode = Dict[str, Any]

RE_CHILD_ARRAY = re.compile(r'(.*)\[(.*)\]')
RE_INTERNAL_ATTR = re.compile('__.*__')


@functools.lru_cache()
def child_attrs_of(klass: Type[ASTNode]):
    r"""Given a Node class, get a set of child attrs.
    Memoized to avoid highly repetitive string manipulation
    """
    non_child_attrs = set(klass.attr_names)
    all_attrs = set([i for i in klass.__slots__ if not RE_INTERNAL_ATTR.match(i)])
    all_attrs -= {"coord"}
    return all_attrs - non_child_attrs


def ast_to_json(root: ASTNode, tokens: List[TokenCoord]) -> JSONNode:
    r"""Recursively convert an ast into dict representation. Replace position (``coord``) with the index in the lexed
    token list.

    Adapted from ``pycparser`` example ``c_json.py``.
    """
    n_lines = tokens[-1].line
    line_range: List[List[int]] = [[-1, -1] for _ in range(n_lines + 1)]  # [(l, r)]
    for idx, tok in enumerate(tokens):
        if line_range[tok.line][0] == -1:
            line_range[tok.line][0] = idx
        line_range[tok.line][1] = idx

    def find_token(line: int, column: int) -> int:
        l, r = line_range[line]
        while l < r:
            mid = (l + r + 1) >> 1
            if tokens[mid].column > column:
                r = mid - 1
            else:
                l = mid
        return l

    def traverse(node: ASTNode, depth: int = 0) -> JSONNode:
        klass = node.__class__

        result = {}

        # Metadata
        result['_nodetype'] = klass.__name__

        # Local node attributes
        for attr in klass.attr_names:
            result[attr] = getattr(node, attr)

        # Coord object
        if node.coord is not None and node.coord.line > 0:  # some nodes have coordinates of (0, 1), which is invalid
            coord: Coord = node.coord
            pos = find_token(coord.line, coord.column)
            result['coord'] = pos
        else:
            result['coord'] = None

        # node_name = (" " * (2 * depth) + klass.__name__).ljust(35)
        # if node.coord is not None:
        #     coord: Coord = node.coord
        #     pos = result['coord']
        #     print(node_name, coord.line, coord.column, pos, (tokens[pos] if pos else None), sep='\t')
        # else:
        #     print(node_name)

        # Child attributes
        children: Dict[str, Optional[MaybeList[JSONNode]]] = {}
        for child_name, child in node.children():
            child_dict = traverse(child, depth + 1)
            # Child strings are either simple (e.g. 'value') or arrays (e.g. 'block_items[1]')
            match = RE_CHILD_ARRAY.match(child_name)
            if match:
                array_name, array_index = match.groups()
                array_index = int(array_index)
                # arrays come in order, so we verify and append.
                array: List[JSONNode] = children.setdefault(array_name, [])  # type: ignore
                if array_index != len(array):
                    raise ValueError(f"Internal ast error. Array {array_name} out of order. "
                                     f"Expected index {len(array)}, got {array_index}")
                array.append(child_dict)
            else:
                children[child_name] = child_dict
        # Any child attributes that were missing need "None" values in the json.
        for child_attr in child_attrs_of(klass):
            if child_attr not in children:
                children[child_attr] = None
        result["children"] = children

        return result

    ast_json = traverse(root)
    return ast_json
