"""
CUDA代码静态分析器 - 分析代码结构和最佳实践
"""

import re
from pathlib import Path
from typing import Optional, List, Tuple, Set

import logging

from ..core.models import CodeMetrics

logger = logging.getLogger(__name__)


class CudaStaticAnalyzer:
    """CUDA代码静态分析器"""
    
    def __init__(self):
        # 正则表达式模式
        self.patterns = {
            'kernel': re.compile(r'__global__\s+[\w\s]+\([^)]*\)\s*\{'),
            'shared': re.compile(r'__shared__\s+'),
            'syncthreads': re.compile(r'__syncthreads\s*\(\s*\)'),
            'kernel_call': re.compile(r'<<<\s*([^,]+)\s*,\s*([^,>]+)(?:\s*,\s*([^>]+))?\s*>>>'),
            'boundary_check': re.compile(r'if\s*\([^)]*(?:tid|thread|idx|i\s*<)[^)]*\)'),
            'cuda_malloc': re.compile(r'cudaMalloc\s*\('),
            'cuda_memcpy': re.compile(r'cudaMemcpy\s*\('),
            'error_check': re.compile(r'(?:cudaSuccess|cudaError|if.*cuda.*==)'),
            'comment_line': re.compile(r'^\s*//'),
            'comment_block_start': re.compile(r'^\s*/\*'),
            'comment_block_end': re.compile(r'\*/'),
        }
        
    def analyze(self, source_path: Path) -> CodeMetrics:
        """分析CUDA代码，返回指标"""
        with open(source_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            
        source = ''.join(lines)
        
        # 统计代码行数
        total_lines = len(lines)
        code_lines = 0
        comment_lines = 0
        blank_lines = 0
        
        in_block_comment = False
        
        for line in lines:
            stripped = line.strip()
            
            if not stripped:
                blank_lines += 1
                continue
                
            # 检查块注释
            if in_block_comment:
                comment_lines += 1
                if self.patterns['comment_block_end'].search(line):
                    in_block_comment = False
                continue
                
            if self.patterns['comment_block_start'].search(line):
                comment_lines += 1
                in_block_comment = True
                if self.patterns['comment_block_end'].search(line):
                    in_block_comment = False
                continue
                
            # 检查行注释
            if self.patterns['comment_line'].match(line):
                comment_lines += 1
                continue
                
            code_lines += 1
            
        # Kernel数量
        kernels = self.patterns['kernel'].findall(source)
        kernel_count = len(kernels)
        
        # Shared memory使用
        shared_mems = self.patterns['shared'].findall(source)
        shared_memory_count = len(shared_mems)
        
        # 边界检查
        has_boundary_check = bool(self.patterns['boundary_check'].search(source))
        
        # CUDA错误检查
        has_malloc = bool(self.patterns['cuda_malloc'].search(source))
        has_memcpy = bool(self.patterns['cuda_memcpy'].search(source))
        has_error_check = bool(self.patterns['error_check'].search(source))
        
        # 提取Grid/Block配置
        block_size = None
        grid_size = None
        
        kernel_calls = self.patterns['kernel_call'].findall(source)
        if kernel_calls:
            # 取第一个kernel调用的配置
            grid_str, block_str, _ = kernel_calls[0]
            grid_size = self._parse_dimension(grid_str.strip())
            block_size = self._parse_dimension(block_str.strip())
            
        return CodeMetrics(
            total_lines=total_lines,
            code_lines=code_lines,
            comment_lines=comment_lines,
            blank_lines=blank_lines,
            kernel_count=kernel_count,
            shared_memory_count=shared_memory_count,
            has_boundary_check=has_boundary_check,
            has_error_check=has_error_check if (has_malloc or has_memcpy) else None,
            block_size=block_size,
            grid_size=grid_size,
        )
        
    def _parse_dimension(self, dim_str: str) -> Optional[tuple]:
        """解析维度配置，如 128, dim3(64, 64)"""
        # 简单的数字
        if dim_str.isdigit():
            return (int(dim_str),)
            
        # dim3(x, y, z) 格式
        dim3_match = re.match(r'dim3\s*\(\s*([^,]+)\s*(?:,\s*([^,]+)\s*)?(?:,\s*([^)]+))?\s*\)', dim_str)
        if dim3_match:
            dims = []
            for d in dim3_match.groups():
                if d and d.strip().isdigit():
                    dims.append(int(d.strip()))
            return tuple(dims) if dims else None
            
        return None
        
    def check_best_practices(self, source_path: Path) -> dict:
        """检查CUDA最佳实践，返回详细的检查结果"""
        metrics = self.analyze(source_path)
        results = {
            'warnings': [],
            'suggestions': [],
        }
        
        # 检查block size是否是32的倍数
        if metrics.block_size:
            if metrics.block_size[0] % 32 != 0:
                results['warnings'].append(
                    f"Block size {metrics.block_size[0]} 不是32的倍数，"
                    f"可能导致warp利用率低下"
                )
                
        # 检查是否有边界检查
        if metrics.kernel_count > 0 and not metrics.has_boundary_check:
            results['warnings'].append(
                "Kernel中没有发现边界检查 (if (tid < N))，"
                "可能导致线程越界访问"
            )
            
        # 检查是否使用shared memory
        if metrics.kernel_count > 0 and metrics.shared_memory_count == 0:
            results['suggestions'].append(
                "没有使用shared memory，考虑是否可以通过"
                "共享内存优化全局内存访问"
            )
            
        # 检查是否有错误检查
        if metrics.has_error_check is False:
            results['warnings'].append(
                "使用了cudaMalloc/cudaMemcpy但没有错误检查，"
                "建议检查CUDA API返回值"
            )
            
        return results
