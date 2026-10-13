# ============================================================
# C侧循环特征提取器 - 基于 tree-sitter AST
# ============================================================

import re
import numpy as np
from retrieval_module.features.tree_sitter_parser import CAstVisitor
from retrieval_module.features.normalization import C_FEATURE_ORDER, NORM_FUNCS


class CLoopFeatureExtractor:
    """
    C侧循环特征提取器 - 基于 tree-sitter AST
    """

    def __init__(self, source_code: str):
        self.code = source_code
        self.feats = {}
        self.ast   = CAstVisitor(source_code)

    def extract(self) -> 'CLoopFeatureExtractor':
        self._get_loop_vars()
        self._extract_access_patterns()
        self._extract_computation()
        return self

    def to_vector(self) -> np.ndarray:
        vec = np.array(
            [NORM_FUNCS[f](self.feats.get(f, 0.0)) for f in C_FEATURE_ORDER],
            dtype=np.float32
        )
        return np.clip(vec, 0.0, 1.0)

    # ----------------------------------------------------------
    # Step 1: 提取循环变量
    # ----------------------------------------------------------
    def _get_loop_vars(self):
        self.loop_vars = set()

        def visit(node):
            if node.type == 'for_statement':
                init = node.child_by_field_name('initializer')
                if init:
                    for child in init.children:
                        if child.type == 'declaration':
                            for sub in child.children:
                                if sub.type == 'init_declarator':
                                    id_node = sub.child_by_field_name('declarator')
                                    if id_node and id_node.type == 'identifier':
                                        self.loop_vars.add(
                                            self.ast.get_node_text(id_node)
                                        )
                        elif child.type == 'assignment_expression':
                            left = child.child_by_field_name('left')
                            if left and left.type == 'identifier':
                                self.loop_vars.add(self.ast.get_node_text(left))
                        elif child.type == 'comma_expression':
                            for sub in child.children:
                                if sub.type == 'assignment_expression':
                                    left = sub.child_by_field_name('left')
                                    if left and left.type == 'identifier':
                                        self.loop_vars.add(
                                            self.ast.get_node_text(left)
                                        )
            for child in node.children:
                visit(child)

        visit(self.ast.root_node)

        if not self.loop_vars:
            for m in re.finditer(
                r'for\s*\(\s*(?:\w+\s+)?(\w+)\s*[=<>!]', self.code
            ):
                var = m.group(1)
                if var not in ('int', 'long', 'size_t', 'unsigned'):
                    self.loop_vars.add(var)

    # ----------------------------------------------------------
    # Step 2: 访问模式特征
    # ----------------------------------------------------------
    def _extract_access_patterns(self):
        loop_vars = self.loop_vars
        compute_arrays = self.ast.get_compute_arrays()
        access = self.ast.analyze_array_accesses(loop_vars, compute_arrays)
        total  = max(1, access['total'])

        self.feats['indirect_access_ratio']    = access['indirect']    / total
        self.feats['consecutive_access_ratio'] = access['consecutive'] / total
        self.feats['offset_access_ratio']      = access['offset']      / total
        self.feats['strided_access_ratio']     = access['strided']     / total

        stencil_offsets = access['stencil_offsets']
        self.feats['has_stencil_pattern']         = len(stencil_offsets) > 0
        self.feats['stencil_neighbor_count_norm'] = len(stencil_offsets)

        self.feats['self_product_diversity'] = (
            self.ast.analyze_self_product_diversity(compute_arrays)
        )

        write_arrays, all_arrays = self.ast.analyze_read_write()
        total_arrays = max(1, len(all_arrays))
        self.feats['read_write_ratio'] = len(write_arrays) / total_arrays

        self.feats['spatial_locality_score'] = (
            (access['consecutive'] + access['offset']) / total
        )

        self.feats['reduction_count_norm'] = self.ast.count_reduction_operations()

    # ----------------------------------------------------------
    # Step 3: 计算操作特征
    # ----------------------------------------------------------
    def _extract_computation(self):
        # 必须排除下标计算，由 _is_pure_index_math 保驾护航
        arith = self.ast.count_arithmetic_operations(exclude_subscripts=True)
        mem   = max(1, self.ast.count_memory_accesses())
        self.feats['compute_intensity'] = arith / mem
        fma = self.ast.count_fma_operations()
        self.feats['fma_ratio'] = fma / max(1, arith)