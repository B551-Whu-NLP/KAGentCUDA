"""
C 代码循环提取模块
使用 tree-sitter 提取 C 代码中的所有循环 (for, while, do-while)
"""

from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional

from tree_sitter import Language, Parser
from data_module.parser_libraries import grammar_library

_parser: Optional[Parser] = None


def get_parser() -> Parser:
    """获取或创建 C 解析器"""
    global _parser
    if _parser is not None:
        return _parser
    lang = Language(grammar_library('c'), "c")
    p = Parser()
    p.set_language(lang)
    _parser = p
    return _parser


@dataclass
class LoopInfo:
    """循环信息"""
    loop_type: str  # for_loop / while_loop / do_while_loop
    start_line: int
    end_line: int
    start_byte: int
    end_byte: int
    source_code: str
    parent_function: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


_LOOP_TYPE_MAP = {
    'for_statement': 'for_loop',
    'while_statement': 'while_loop',
    'do_statement': 'do_while_loop',
}


def _get_function_name(func_def_node) -> Optional[str]:
    """从 function_definition 节点提取函数名"""
    declarator = func_def_node.child_by_field_name('declarator')
    while declarator is not None and declarator.type != 'function_declarator':
        inner = declarator.child_by_field_name('declarator')
        if inner is None or inner is declarator:
            break
        declarator = inner
    if declarator is None:
        return None

    for child in declarator.children:
        if child.type == 'identifier':
            text = child.text
            return text.decode('utf-8', errors='replace') if isinstance(text, bytes) else str(text)
        if child.type in ('parenthesized_declarator', 'pointer_declarator'):
            for sub in child.children:
                if sub.type == 'identifier':
                    text = sub.text
                    return text.decode('utf-8', errors='replace') if isinstance(text, bytes) else str(text)
    return None


def _find_parent_function(node) -> Optional[str]:
    """向上查找包裹当前节点的函数定义名"""
    current = node.parent if node else None
    while current is not None:
        if current.type == 'function_definition':
            return _get_function_name(current)
        current = current.parent
    return None


def _walk_loops(node, results: List[LoopInfo]):
    if node is None:
        return
    if node.type in _LOOP_TYPE_MAP:
        text = node.text
        source_code = text.decode('utf-8', errors='replace') if isinstance(text, bytes) else str(text)
        results.append(LoopInfo(
            loop_type=_LOOP_TYPE_MAP[node.type],
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            start_byte=node.start_byte,
            end_byte=node.end_byte,
            source_code=source_code,
            parent_function=_find_parent_function(node),
        ))
    for child in node.children:
        _walk_loops(child, results)


def extract_loops(file_path: str | Path) -> List[LoopInfo]:
    """从 C 文件中提取所有循环"""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, 'rb') as f:
        source_bytes = f.read()
    tree = get_parser().parse(source_bytes)
    results: List[LoopInfo] = []
    _walk_loops(tree.root_node, results)
    return results


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        loops = extract_loops(sys.argv[1])
        print(f"Found {len(loops)} loops:")
        for loop in loops:
            print(f"  [{loop.loop_type}] L{loop.start_line}-{loop.end_line} in {loop.parent_function}")
