# ============================================================
# CUDA侧Kernel特征提取器 - 基于 tree-sitter CUDA AST
# ============================================================

import re
import numpy as np
from retrieval_module.features.tree_sitter_parser import CudaAstVisitor
from retrieval_module.features.normalization import CUDA_FEATURE_ORDER, NORM_FUNCS


class CudaKernelFeatureExtractor:
    """
    CUDA侧Kernel特征提取器
    """

    def __init__(self, source_code: str):
        self.code  = source_code
        self.feats = {}
        self.ast   = CudaAstVisitor(source_code)

    def extract(self) -> 'CudaKernelFeatureExtractor':
        self._get_thread_vars()
        self._extract_access_patterns()
        self._extract_computation()
        return self

    def to_vector(self) -> np.ndarray:
        vec = np.array(
            [NORM_FUNCS[f](self.feats.get(f, 0.0)) for f in CUDA_FEATURE_ORDER],
            dtype=np.float32
        )
        return np.clip(vec, 0.0, 1.0)

    # ----------------------------------------------------------
    # Step 1: 推断线程索引变量
    # ----------------------------------------------------------
    def _get_thread_vars(self):
        self.thread_vars = set()

        def visit(node):
            if node.type == 'init_declarator':
                value = node.child_by_field_name('value')
                if value:
                    val_text = self.ast.get_node_text(value)
                    if 'threadIdx' in val_text or 'blockIdx' in val_text:
                        declarator = node.child_by_field_name('declarator')
                        if declarator and declarator.type == 'identifier':
                            self.thread_vars.add(
                                self.ast.get_node_text(declarator)
                            )

            elif node.type == 'assignment_expression':
                right = node.child_by_field_name('right')
                if right:
                    right_text = self.ast.get_node_text(right)
                    if 'threadIdx' in right_text or 'blockIdx' in right_text:
                        left = node.child_by_field_name('left')
                        if left and left.type == 'identifier':
                            self.thread_vars.add(self.ast.get_node_text(left))

            elif node.type == 'field_expression':
                obj = node.child_by_field_name('argument')
                if obj and self.ast.get_node_text(obj) == 'threadIdx':
                    field = node.child_by_field_name('field')
                    if field:
                        self.thread_vars.add(
                            f"threadIdx.{self.ast.get_node_text(field)}"
                        )

            for child in node.children:
                visit(child)

        visit(self.ast.root_node)
        for m in re.finditer(r'\b(?:int|uint|long|size_t|unsigned)\s+(tid|idx|idy|idz|gid|gidx|x|y|z|row|col)\b', self.code):
            self.thread_vars.add(m.group(1))

    # ----------------------------------------------------------
    # Step 2: 访问模式特征
    # ----------------------------------------------------------
    def _extract_access_patterns(self):
        thread_vars = self.thread_vars
        compute_arrays = self.ast.get_compute_arrays()
        access = self.ast.analyze_array_accesses(thread_vars, compute_arrays)
        total  = max(1, access['total'])

        self.feats['indirect_access_ratio']  = access['indirect']    / total
        self.feats['coalesced_access_ratio'] = access['consecutive'] / total
        self.feats['offset_access_ratio']    = access['offset']      / total
        self.feats['strided_access_ratio']   = access['strided']     / total

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

        # 规约操作（注释掉了原子操作等正则匹配，统一和C侧对齐基于 += 的 AST 统计）
        reduction = self.ast.count_reduction_operations()
        # reduction += len(re.findall(r'\b(?:atomicAdd|atomicSub|atomicMin|atomicMax|atomicExch)\s*\(', self.code))
        # reduction += len(re.findall(r'\b__(?:shfl_down|shfl_xor|reduce)_sync\s*\(', self.code))
        self.feats['reduction_count_norm'] = reduction

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