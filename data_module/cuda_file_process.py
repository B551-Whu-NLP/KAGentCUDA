"""
CUDA 代码 Kernel 提取模块
使用 tree-sitter 提取 CUDA 代码中的所有 kernel 函数
"""

import re
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Optional

from tree_sitter import Language, Parser
from data_module.parser_libraries import grammar_library

_parser: Optional[Parser] = None


def get_parser() -> Parser:
    """获取或创建 CUDA 解析器"""
    global _parser
    if _parser is not None:
        return _parser
    lang = Language(grammar_library('cuda'), "cuda")
    p = Parser()
    p.set_language(lang)
    _parser = p
    return _parser


@dataclass
class KernelInfo:
    """Kernel 信息"""
    kernel_name: str
    start_line: int
    end_line: int
    start_byte: int
    end_byte: int
    source_code: str
    parameters: List[str] = field(default_factory=list)
    grid_dim: Optional[str] = None
    block_dim: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


def _find_kernel_launches(content: str) -> dict:
    """查找文件中所有 kernel 启动调用 (<<<...>>>)"""
    launches = {}
    pattern = r'(\w+)\s*<<<\s*([^,]+?)\s*,\s*([^>]+?)\s*>>>\s*\('
    for match in re.finditer(pattern, content):
        kernel_name = match.group(1)
        launches[kernel_name] = {
            'grid': match.group(2).strip(),
            'block': match.group(3).strip(),
        }
    return launches


def _get_function_name(declarator) -> Optional[str]:
    """从 function_declarator/嵌套声明子里挖出 identifier"""
    while declarator is not None and declarator.type != 'function_declarator':
        inner = declarator.child_by_field_name('declarator')
        if inner is None or inner is declarator:
            return None
        declarator = inner
    if declarator is None:
        return None
    for child in declarator.children:
        if child.type == 'identifier':
            t = child.text
            return t.decode('utf-8', errors='replace') if isinstance(t, bytes) else str(t)
        if child.type in ('parenthesized_declarator', 'pointer_declarator'):
            for sub in child.children:
                if sub.type == 'identifier':
                    t = sub.text
                    return t.decode('utf-8', errors='replace') if isinstance(t, bytes) else str(t)
    return None


def _is_cuda_kernel(func_def_node) -> bool:
    """判断 function_definition 是否带 __global__ / __device__"""
    for child in func_def_node.children:
        text = child.text
        text = text.decode('utf-8', errors='replace') if isinstance(text, bytes) else str(text)
        if '__global__' in text or '__device__' in text:
            return True
        # 只看声明前缀，碰到 declarator 就停
        if child.type in ('function_declarator', 'compound_statement'):
            break
    return False


def _extract_params(func_def_node) -> List[str]:
    """从 function_definition 里提取参数列表"""
    declarator = func_def_node.child_by_field_name('declarator')
    while declarator is not None and declarator.type != 'function_declarator':
        inner = declarator.child_by_field_name('declarator')
        if inner is None:
            return []
        declarator = inner
    if declarator is None:
        return []
    params = []
    for child in declarator.children:
        if child.type == 'parameter_list':
            for p in child.children:
                if p.type == 'parameter_declaration':
                    t = p.text
                    t = t.decode('utf-8', errors='replace') if isinstance(t, bytes) else str(t)
                    params.append(t.strip())
    return params


def extract_kernels(file_path: str | Path) -> List[KernelInfo]:
    """从 CUDA 文件中提取所有 __global__ / __device__ 函数"""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, 'rb') as f:
        source_bytes = f.read()
    content = source_bytes.decode('utf-8', errors='replace')

    tree = get_parser().parse(source_bytes)
    launches = _find_kernel_launches(content)
    results: List[KernelInfo] = []

    def walk(node):
        if node is None:
            return
        if node.type == 'function_definition' and _is_cuda_kernel(node):
            func_name = _get_function_name(node.child_by_field_name('declarator'))
            if func_name:
                text = node.text
                source_code = text.decode('utf-8', errors='replace') if isinstance(text, bytes) else str(text)
                launch = launches.get(func_name, {})
                results.append(KernelInfo(
                    kernel_name=func_name,
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    start_byte=node.start_byte,
                    end_byte=node.end_byte,
                    source_code=source_code,
                    parameters=_extract_params(node),
                    grid_dim=launch.get('grid'),
                    block_dim=launch.get('block'),
                ))
        for child in node.children:
            walk(child)

    walk(tree.root_node)
    results.sort(key=lambda x: x.start_line)
    return results


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        kernels = extract_kernels(sys.argv[1])
        print(f"Found {len(kernels)} kernels:")
        for k in kernels:
            print(f"  [{k.kernel_name}] L{k.start_line}-{k.end_line}")
            if k.grid_dim:
                print(f"    Grid: {k.grid_dim}, Block: {k.block_dim}")
