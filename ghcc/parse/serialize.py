"""
Utilities for serialization and deserialization of ``pycparser`` ASTs.
Adapted from ``pycparser`` example ``c_json.py``.
"""

import functools
import re
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

from pycparser.c_ast import Node as ASTNode
from pycparser.plyparser import Coord

from .lexer import TokenCoord

__all__ = [
    "ast_to_dict",
    "dict_to_ast",
    "JSONNode",
]

T = TypeVar('T')
MaybeList = Union[T, List[T]]
JSONNode = Dict[str, Any]

RE_CHILD_ARRAY = re.compile(r'(.*)\[(.*)\]')
RE_INTERNAL_ATTR = re.compile('__.*__')

AVAILABLE_NODES: Dict[str, Type[ASTNode]] = {klass.__name__: klass for klass in ASTNode.__subclasses__()}
NODE_TYPE_ATTRIBUTE = "_nodetype"
CHILDREN_ATTRIBUTE = "children"
TOKEN_POS_ATTRIBUTE = "coord"


@functools.lru_cache()
def child_attrs_of(klass: Type[ASTNode]):
    r"""Given a Node class, get a set of child attrs.
    Memoized to avoid highly repetitive string manipulation
    """
    non_child_attrs = set(klass.attr_names)
    all_attrs = set([i for i in klass.__slots__ if not RE_INTERNAL_ATTR.match(i)])
    all_attrs -= {"coord"}
    return all_attrs - non_child_attrs


def ast_to_dict(root: ASTNode, tokens: Optional[List[TokenCoord]] = None) -> JSONNode:
    r"""Recursively convert an AST into dictionary representation.

    :param root: The AST to convert.
    :param tokens: A list of lexed token coordinates. If specified, will replace node position (``coord``) with the
        index in the lexed token list.
    """
    if tokens is not None:
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
        result[NODE_TYPE_ATTRIBUTE] = klass.__name__

        # Local node attributes
        for attr in klass.attr_names:
            result[attr] = getattr(node, attr)

        # Coord object
        if (tokens is not None and node.coord is not None and
                node.coord.line > 0):  # some nodes have coordinates of (0, 1), which is invalid
            coord: Coord = node.coord
            pos = find_token(coord.line, coord.column)
            result[TOKEN_POS_ATTRIBUTE] = pos
        else:
            result[TOKEN_POS_ATTRIBUTE] = None

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
        result[CHILDREN_ATTRIBUTE] = children

        return result

    ast_json = traverse(root)
    return ast_json


def dict_to_ast(node_dict: JSONNode) -> ASTNode:
    r"""Recursively build an AST from dictionary representation. Coordinate information is discarded.
    """
    class_name = node_dict[NODE_TYPE_ATTRIBUTE]
    klass = AVAILABLE_NODES[class_name]

    # Create a new dict containing the key-value pairs which we can pass to node constructors.
    kwargs = {'coord': None}
    children: Dict[str, MaybeList[JSONNode]] = node_dict[CHILDREN_ATTRIBUTE]
    for name, child in children.items():
        if isinstance(child, list):
            kwargs[name] = [dict_to_ast(item) for item in child]
        else:
            kwargs[name] = dict_to_ast(child) if child is not None else None

    for key, value in node_dict.items():
        if key in [NODE_TYPE_ATTRIBUTE, CHILDREN_ATTRIBUTE, TOKEN_POS_ATTRIBUTE]:
            continue
        kwargs[key] = value  # must be primitive attributes

    return klass(**kwargs)
