"""
Utilities for serialization and deserialization of ``pycparser`` ASTs.
Adapted from ``pycparser`` example ``c_json.py``.
"""

import functools
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union

from pycparser.c_ast import Node as ASTNode
from pycparser.plyparser import Coord

from .lexer import Token

__all__ = [
    "ast_to_dict",
    "dict_to_ast",
    "get_ast_class",
    "visit_dict",
    "JSONNode",
    "NODE_TYPE_ATTR",
    "CHILDREN_ATTR",
    "TOKEN_POS_ATTR",
]

T = TypeVar('T')
K = TypeVar('K')
MaybeList = Union[T, List[T]]
JSONNode = Dict[str, Any]

RE_CHILD_ARRAY = re.compile(r'(.*)\[(.*)\]')
RE_INTERNAL_ATTR = re.compile('__.*__')

AVAILABLE_NODES: Dict[str, Type[ASTNode]] = {klass.__name__: klass for klass in ASTNode.__subclasses__()}
NODE_TYPE_ATTR = "_t"
CHILDREN_ATTR = "_c"
TOKEN_POS_ATTR = "_p"


@functools.lru_cache()
def child_attrs_of(klass: Type[ASTNode]):
    r"""Given a Node class, get a set of child attrs.
    Memoized to avoid highly repetitive string manipulation
    """
    non_child_attrs = set(klass.attr_names)
    all_attrs = set([i for i in klass.__slots__ if not RE_INTERNAL_ATTR.match(i)])
    all_attrs -= {"coord"}
    return all_attrs - non_child_attrs


def find_first(arr: List[T], cond_fn: Callable[[T], bool]) -> int:
    r"""Use binary search to find the index of the first element in ``arr`` such that ``cond_fn`` returns ``True``."""
    l, r = 0, len(arr)
    while l < r:
        mid = (l + r) >> 1
        if not cond_fn(arr[mid]):
            l = mid + 1
        else:
            r = mid
    return l


def ast_to_dict(root: ASTNode, tokens: Optional[List[Token]] = None) -> JSONNode:
    r"""Recursively convert an AST into dictionary representation.

    :param root: The AST to convert.
    :param tokens: A list of lexed token coordinates. If specified, will replace node position (``coord``) with the
        index in the lexed token list.
    """
    if tokens is not None:
        tokens_ = tokens  # so that the `find_token` function type-checks
        line_range: Dict[int, Tuple[int, int]] = {}

    def find_token(line: int, column: int) -> Optional[int]:
        if line not in line_range:
            l = find_first(tokens_, lambda tok: line <= tok.line)
            r = find_first(tokens_, lambda tok: line < tok.line)
            line_range[line] = l, r
        else:
            l, r = line_range[line]
        ret = find_first(tokens_[l:r], lambda tok: column < tok.column) + l - 1
        if ret < 0:
            # In rare cases `ret` where `l == 0` and the first code token has `column > 1`, the coordinates of the root
            # node might still have `column == 1`, which results in `ret == -1`.
            return None
        return ret

    def traverse(node: ASTNode, depth: int = 0) -> JSONNode:
        klass = node.__class__

        result = {}

        # Node type
        result[NODE_TYPE_ATTR] = klass.__name__

        # Local node attributes
        for attr in klass.attr_names:
            result[attr] = getattr(node, attr)

        # Token position
        if tokens is not None:
            if node.coord is not None and node.coord.line > 0:  # some nodes have invalid coordinate (0, 1)
                coord: Coord = node.coord
                pos = find_token(coord.line, coord.column)
                result[TOKEN_POS_ATTR] = pos
            else:
                result[TOKEN_POS_ATTR] = None

        # node_name = (" " * (2 * depth) + klass.__name__).ljust(35)
        # if node.coord is not None:
        #     coord: Coord = node.coord
        #     pos = result['coord']
        #     print(node_name, coord.line, coord.column, pos, (tokens[pos] if pos else None), sep='\t')
        # else:
        #     print(node_name)

        # Children nodes
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
        # Missing children are filled with `None` values in the dictionary.
        for child_attr in child_attrs_of(klass):
            if child_attr not in children:
                children[child_attr] = None
        result[CHILDREN_ATTR] = children

        return result

    ast_json = traverse(root)
    return ast_json


def visit_dict(visit_fn: Callable[[JSONNode, List[T]], T], node_dict: JSONNode) -> T:
    # visit_fn: (node, [children_result]) -> result
    children_result: List[T] = []
    for name, child in node_dict[CHILDREN_ATTR].items():
        if isinstance(child, list):
            children_result.extend(visit_dict(visit_fn, item) for item in child)
        elif child is not None:
            children_result.append(visit_dict(visit_fn, child))
    return visit_fn(node_dict, children_result)


def get_ast_class(name: str) -> Type[ASTNode]:
    return AVAILABLE_NODES[name]


def dict_to_ast(node_dict: JSONNode) -> ASTNode:
    r"""Recursively build an AST from dictionary representation. Coordinate information is discarded.
    """
    class_name = node_dict[NODE_TYPE_ATTR]
    klass = get_ast_class(class_name)

    # Create a new dict containing the key-value pairs which we can pass to node constructors.
    kwargs: Dict[str, Any] = {'coord': None}
    children: Dict[str, MaybeList[JSONNode]] = node_dict[CHILDREN_ATTR]
    for name, child in children.items():
        if isinstance(child, list):
            kwargs[name] = [dict_to_ast(item) for item in child]
        else:
            kwargs[name] = dict_to_ast(child) if child is not None else None

    for key, value in node_dict.items():
        if key in [NODE_TYPE_ATTR, CHILDREN_ATTR, TOKEN_POS_ATTR]:
            continue
        kwargs[key] = value  # must be primitive attributes

    return klass(**kwargs)
