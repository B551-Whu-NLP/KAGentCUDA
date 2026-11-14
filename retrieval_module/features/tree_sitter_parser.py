# ============================================================
# tree-sitter 解析器封装（兼容 C 和 CUDA parser）
# 基于 AST 的循环/内核结构分析：数组访问模式、读写集合、
# FMA 识别（排除纯线程索引计算）、CUDA 二维展平连续访问等。
# ============================================================

import re
from typing import Dict, Set, Tuple
from collections import defaultdict
from tree_sitter import Language, Parser
from data_module.parser_libraries import grammar_library


class AstVisitorBase:
    """AST访问器基类，包含所有分析逻辑"""

    def __init__(self, code: str, parser: Parser):
        self.code = code
        self.parser = parser
        self.tree = parser.parse(bytes(code, 'utf8'))
        self.root_node = self.tree.root_node

    def get_node_text(self, node) -> str:
        return self.code[node.start_byte:node.end_byte]

    # ======================================================
    # 兼容性工具
    # ======================================================
    def _is_pure_index_math(self, node, compute_arrays: Set[str]) -> bool:
        """
        判断是否是纯粹的索引/线程计算（不包含任何业务数组参与的计算）。
        这是防止 CUDA 线程计算污染业务 FMA 的核心过滤器。
        """
        text = self.get_node_text(node)
        
        # 如果表达式里包含任何一个计算数组的名字，说明这是业务逻辑，不能过滤！
        for arr in compute_arrays:
            if re.search(rf'\b{re.escape(arr)}\b', text):
                return False 
                
        # 如果没有业务数组，且包含了线程/区块关键字，那就是纯索引计算
        return any(kw in text for kw in ('blockIdx', 'threadIdx', 'blockDim', 'gridDim'))
        
    def _get_subscript_parts(self, node):
        array_node = node.child_by_field_name('array')
        index_node = node.child_by_field_name('index')

        # 格式1：两个字段都有
        if array_node and index_node:
            return array_node, index_node

        # 格式2：只有index字段，array是第一个非括号子节点
        if index_node:
            for child in node.children:
                if child.type not in ('[', ']') and child != index_node:
                    return child, index_node
            return None, index_node

        # 格式3：CUDA parser，identifier + subscript_argument_list
        array_node = None
        index_node = None
        for child in node.children:
            if child.type == 'identifier' and array_node is None:
                array_node = child
            elif child.type == 'subscript_argument_list':
                for sub in child.children:
                    if sub.type not in ('[', ']', ','):
                        index_node = sub
                        break
        return array_node, index_node

    def _is_inside_subscript(self, node) -> bool:
        """检查节点是否在数组下标表达式内部"""
        parent = node.parent
        while parent is not None:
            if parent.type == 'subscript_expression':
                return True
            parent = parent.parent
        return False

    # ======================================================
    # 计算数组提取
    # ======================================================
    
    def get_compute_arrays(self) -> Set[str]:
        """
        获取计算数组：all_arrays减去明显的索引数组
        """
        _, all_arrays = self.analyze_read_write()

        if not all_arrays:
            return set()

        # 排除明显的索引/指针数组
        INDEX_HINTS = ('ptr', 'idx', 'index', 'offset', 'row', 'col', 'ind',
                    'map', 'list', 'table', 'keys', 'hash')
        compute = {
            name for name in all_arrays
            if not any(hint in name.lower() for hint in INDEX_HINTS)
        }

        # 过滤后为空则退回all_arrays
        return compute if compute else all_arrays

    # ======================================================
    # 算术操作符计数
    # ======================================================

    def count_arithmetic_operations(self, exclude_subscripts: bool = True) -> int:
        count = 0
        compute_arrays = self.get_compute_arrays()

        def visit(node):
            nonlocal count
            # 如果是纯索引计算，直接跳过，不计入业务算术
            if self._is_pure_index_math(node, compute_arrays):
                return
                
            if node.type == 'binary_expression':
                op_node = node.child_by_field_name('operator')
                if op_node and self.get_node_text(op_node) in ('+', '-', '*', '/'):
                    if not exclude_subscripts or not self._is_inside_subscript(node):
                        count += 1
            for child in node.children:
                visit(child)

        visit(self.root_node)
        return count

    # ======================================================
    # FMA 检测
    # ======================================================

    def count_fma_operations(self) -> int:
        count = 0
        compute_arrays = self.get_compute_arrays()

        def visit(node):
            nonlocal count
            
            # 如果是纯索引计算，直接跳过，保护业务FMA
            if self._is_pure_index_math(node, compute_arrays):
                return
                
            # 原有：binary_expression 里的 +/-
            if node.type == 'binary_expression':
                op_node = node.child_by_field_name('operator')
                if op_node and self.get_node_text(op_node) in ('+', '-'):
                    if not self._is_inside_subscript(node):
                        for side in ['left', 'right']:
                            child = node.child_by_field_name(side)
                            if child and child.type == 'binary_expression':
                                child_op = child.child_by_field_name('operator')
                                if child_op and self.get_node_text(child_op) == '*':
                                    if not self._is_inside_subscript(child):
                                        count += 1

            # 新增：+= / -= 里的乘法（sum += A[k] * B[k]）
            elif node.type == 'assignment_expression':
                op_node = node.child_by_field_name('operator')
                if op_node and self.get_node_text(op_node) in ('+=', '-='):
                    right = node.child_by_field_name('right')
                    if right and right.type == 'binary_expression':
                        right_op = right.child_by_field_name('operator')
                        if right_op and self.get_node_text(right_op) == '*':
                            if not self._is_inside_subscript(right):
                                count += 1

            for child in node.children:
                visit(child)

        visit(self.root_node)
        return count

    # ======================================================
    # 内存访问计数
    # ======================================================

    def count_memory_accesses(self) -> int:
        count = 0
        def visit(node):
            nonlocal count
            if node.type == 'subscript_expression':
                count += 1
            for child in node.children:
                visit(child)
        visit(self.root_node)
        return count

    # ======================================================
    # 数组访问模式分析
    # ======================================================

    def analyze_array_accesses(
        self,
        index_vars: Set[str],
        compute_arrays: Set[str] = None
    ) -> Dict:
        result = {
            'total': 0,
            'indirect': 0,
            'consecutive': 0,
            'offset': 0,
            'strided': 0,
            'stencil_offsets': set()
        }

        def visit(node):
            if node.type == 'subscript_expression':
                result['total'] += 1
                array_node, index_node = self._get_subscript_parts(node)

                arr_name = (
                    self.get_node_text(array_node)
                    if array_node and array_node.type == 'identifier'
                    else ''
                )

                if index_node:
                    idx_text = self.get_node_text(index_node).strip()
                    idx_clean = idx_text.replace(' ', '')

                    # 间接访问 a[b[i]]
                    if index_node.type == 'subscript_expression':
                        result['indirect'] += 1

                    # ====== 连续访问与空间局部性对齐 (兼容 CUDA 展平) ======
                    is_consecutive = False
                    # 1. 精确匹配
                    if idx_text in index_vars:
                        is_consecutive = True
                    # 2. CUDA典型合并访问
                    elif 'threadIdx.x' in idx_text and ('*threadIdx.x' not in idx_clean and 'threadIdx.x*' not in idx_clean):
                        is_consecutive = True
                    # 3. 兼容展平的2D访问 (row * N + col)
                    else:
                        for var in index_vars:
                            if re.search(rf'\b{re.escape(var)}\b', idx_text):
                                if re.search(rf'\+\s*{re.escape(var)}\b', idx_text) or re.search(rf'\b{re.escape(var)}\s*\+', idx_text):
                                    is_consecutive = True
                                break
                    
                    if is_consecutive:
                        result['consecutive'] += 1

                    # 偏移访问 a[i+c] / a[i-c]
                    for var in index_vars:
                        # 性能优化：前置包含判断
                        if var in idx_clean:
                            match = re.search(rf'\b{re.escape(var)}\s*([+-])\s*(\d+|num_\d+)', idx_text)
                            if match:
                                result['offset'] += 1
                                if compute_arrays is None or arr_name in compute_arrays:
                                    try:
                                        offset_str = match.group(2)
                                        if offset_str.startswith('num_'):
                                            offset = 1  # 混淆后无法知道具体值，用1代替表示有偏移
                                        else:
                                            offset = int(offset_str)
                                        if match.group(1) == '-':
                                            offset = -offset
                                        result['stencil_offsets'].add(offset)
                                    except ValueError:
                                        pass
                                break

                    # 跨步访问 a[i*N] / a[N*i]
                    for var in index_vars:
                        if var in idx_clean:
                            if (re.search(rf'\b{re.escape(var)}\s*\*', idx_text) or
                                    re.search(rf'\*\s*{re.escape(var)}\b', idx_text)):
                                result['strided'] += 1
                                break

            for child in node.children:
                visit(child)

        visit(self.root_node)
        return result

    # ======================================================
    # 读写分析（迭代式遍历，避免深层递归）
    # ======================================================

    def analyze_read_write(self) -> Tuple[Set[str], Set[str]]:
        write_arrays = set()
        all_arrays   = set()

        def visit(node):
            if node.type == 'subscript_expression':
                array_node, _ = self._get_subscript_parts(node)
                if array_node and array_node.type == 'identifier':
                    arr_name = self.get_node_text(array_node)
                    all_arrays.add(arr_name)

                    parent = node.parent
                    while parent:
                        if parent.type == 'assignment_expression':
                            left = parent.child_by_field_name('left')
                            # 性能优化：用字节位置比较替代递归
                            if left and left.start_byte <= node.start_byte and left.end_byte >= node.end_byte:
                                write_arrays.add(arr_name)
                                break
                        parent = parent.parent

            for child in node.children:
                visit(child)

        visit(self.root_node)
        return write_arrays, all_arrays

    # ======================================================
    # self_product_diversity
    # ======================================================

    def analyze_self_product_diversity(self, compute_arrays: Set[str]) -> bool:
        array_subs = defaultdict(set)

        def visit(node):
            if node.type == 'subscript_expression':
                array_node, index_node = self._get_subscript_parts(node)
                if array_node and index_node:
                    name = self.get_node_text(array_node)
                    if name in compute_arrays:
                        idx = self.get_node_text(index_node).strip()
                        array_subs[name].add(idx)
            for child in node.children:
                visit(child)

        visit(self.root_node)
        return any(len(subs) > 1 for subs in array_subs.values())

    # ======================================================
    # 规约操作
    # ======================================================

    def count_reduction_operations(self) -> int:
        count = 0

        def visit(node):
            nonlocal count
            if node.type == 'assignment_expression':
                left = node.child_by_field_name('left')
                op_node = node.child_by_field_name('operator')
                
                if not left or not op_node:
                    return

                # 场景 1: 使用 +=, -=, *=, /=
                if self.get_node_text(op_node) in ('+=', '-=', '*=', '/='):
                    if left.type != 'subscript_expression':
                        count += 1
                        
                # 场景 2: 使用 a = a + b
                elif self.get_node_text(op_node) == '=':
                    right = node.child_by_field_name('right')
                    if right and right.type == 'binary_expression':
                        right_op = right.child_by_field_name('operator')
                        if right_op and self.get_node_text(right_op) in ('+', '-', '*', '/'):
                            rl = right.child_by_field_name('left')
                            rr = right.child_by_field_name('right')
                            left_text = self.get_node_text(left)
                            # 如果等号左边的变量出现在了等号右边的算术运算中
                            if (rl and self.get_node_text(rl) == left_text) or \
                               (rr and self.get_node_text(rr) == left_text):
                                if left.type != 'subscript_expression':
                                    count += 1
                                    
            for child in node.children:
                visit(child)

        visit(self.root_node)
        return count


# ============================================================
# C代码 AST访问器
# ============================================================

class CAstVisitor(AstVisitorBase):
    """C代码 AST访问器，使用 tree-sitter C parser"""

    def __init__(self, code: str):
        lang_c = Language(grammar_library('c'), 'c')
        parser = Parser()
        parser.set_language(lang_c)
        super().__init__(code, parser)


# ============================================================
# CUDA代码 AST访问器
# ============================================================

class CudaAstVisitor(AstVisitorBase):
    """CUDA代码 AST访问器，使用 tree-sitter CUDA parser"""

    def __init__(self, code: str):
        lang_cuda = Language(grammar_library('cuda'), 'cuda')
        parser = Parser()
        parser.set_language(lang_cuda)
        super().__init__(code, parser)