"""
正确性验证器 - 对比生成代码的输出与ground-truth
"""

import re
import struct
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List

import logging

from ..core.models import CorrectnessResult, CorrectnessStatus

logger = logging.getLogger(__name__)


class OutputVerifier:
    """输出验证器"""
    
    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance
        
    def verify(
        self,
        test_output_path: Path,
        baseline_output_path: Path,
    ) -> CorrectnessResult:
        """
        验证两个输出文件的一致性
        """
        if not test_output_path.exists():
            return CorrectnessResult(
                status=CorrectnessStatus.INCORRECT,
                error_message=f"测试输出文件不存在: {test_output_path}",
            )
            
        if not baseline_output_path.exists():
            return CorrectnessResult(
                status=CorrectnessStatus.SKIPPED,
                error_message=f"基准输出文件不存在: {baseline_output_path}",
            )
            
        # 尝试多种方式解析和对比
        try:
            return self._verify_numerical(test_output_path, baseline_output_path)
        except Exception as e:
            logger.warning(f"数值对比失败，尝试字节对比: {e}")
            return self._verify_bytes(test_output_path, baseline_output_path)
            
    def _verify_numerical(
        self,
        test_path: Path,
        baseline_path: Path,
    ) -> CorrectnessResult:
        """数值对比：解析数字并计算误差
        
        使用双标准验证：
        1. 相对误差 > 1e-5 的元素比例 < 1%
        2. 整体 RMSE < 1e-3
        """
        
        test_numbers = self._extract_numbers(test_path)
        baseline_numbers = self._extract_numbers(baseline_path)
        
        if len(test_numbers) == 0 or len(baseline_numbers) == 0:
            # 没有提取到数字，退回字节对比
            return self._verify_bytes(test_path, baseline_path)

        comparison = (f"OutputVerifier numerical comparison: actual={test_path}, baseline={baseline_path}; "
                      f"parsed elements: actual={len(test_numbers)}, baseline={len(baseline_numbers)}")
        if len(test_numbers) != len(baseline_numbers):
            logger.warning(f"输出长度不一致: test={len(test_numbers)}, baseline={len(baseline_numbers)}")
            return CorrectnessResult(
                status=CorrectnessStatus.INCORRECT,
                max_absolute_error=float('inf'),
                mean_absolute_error=float('inf'),
                rmse=float('inf'),
                comparison_log=comparison + '; numeric element counts differ',
                output_path=str(test_path),
                baseline_path=str(baseline_path),
                tolerance=self.tolerance,
            )

        # 转换为numpy数组计算
        test_arr = np.array(test_numbers, dtype=np.float64)
        baseline_arr = np.array(baseline_numbers, dtype=np.float64)
        
        actual_nonfinite = bool(np.any(~np.isfinite(test_arr)))
        baseline_nonfinite = bool(np.any(~np.isfinite(baseline_arr)))
        if actual_nonfinite or baseline_nonfinite:
            return CorrectnessResult(
                status=CorrectnessStatus.INCORRECT,
                actual_nonfinite=actual_nonfinite,
                baseline_nonfinite=baseline_nonfinite,
                comparison_log=(comparison + f'; nonfinite values: actual={actual_nonfinite}, '
                                f'baseline={baseline_nonfinite}'),
                output_path=str(test_path), baseline_path=str(baseline_path),
            )
        abs_errors = np.abs(test_arr - baseline_arr)
        max_abs_error = float(np.max(abs_errors))
        mean_abs_error = float(np.mean(abs_errors))
        p99_abs_error = float(np.percentile(abs_errors, 99))
        worst_index = int(np.argmax(abs_errors))
        rmse = float(np.sqrt(np.mean((test_arr - baseline_arr) ** 2)))
        
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_errors = abs_errors / np.maximum(np.abs(baseline_arr), np.finfo(np.float64).eps)
            max_rel_error = float(np.max(rel_errors))
        
        # 双标准验证
        rel_error_threshold = 1e-5
        high_rel_error_count = np.sum(rel_errors > rel_error_threshold)
        high_rel_error_ratio = high_rel_error_count / len(baseline_arr)
        
        rmse_threshold = 1e-3
        
        condition1 = high_rel_error_ratio < 0.01  # < 1%
        condition2 = rmse < rmse_threshold
        
        # 判断状态（优先使用双标准）
        if np.array_equal(test_arr, baseline_arr):
            status = CorrectnessStatus.EXACT_MATCH
        elif np.all(np.isfinite(test_arr)) and np.all(np.isfinite(baseline_arr)) and condition1 and condition2:
            status = CorrectnessStatus.WITHIN_TOLERANCE
        else:
            status = CorrectnessStatus.INCORRECT
        
        logger.info(f"正确性验证结果: {status}")
        logger.info(f"  - 双标准: 大误差比例={high_rel_error_ratio*100:.4f}% (阈值1%), RMSE={rmse:.2e} (阈值{rmse_threshold:.0e})")
        logger.info(f"  - 最大误差: {max_abs_error:.2e}, 平均误差: {mean_abs_error:.2e}")
        
        # 如果验证失败，展示有误差的数据对比
        if status == CorrectnessStatus.INCORRECT:
            # 找出有误差的索引
            error_indices = np.where(abs_errors > 0)[0]
            # 最多显示10个有误差的数据
            sample_indices = error_indices[:10]
            
            if len(sample_indices) == 0:
                logger.info(f"\n  ┌─────────────────────────────────────────────────────────────┐")
                logger.info(f"  │                    未找到有误差的数据                         │")
                logger.info(f"  └─────────────────────────────────────────────────────────────┘")
            else:
                logger.info(f"\n  ┌─────────────────────────────────────────────────────────────┐")
                logger.info(f"  │                有误差的数据对比 (共{len(error_indices)}个)               │")
                logger.info(f"  ├─────────┬─────────────────┬─────────────────┬─────────────┤")
                logger.info(f"  │  索引    │    Gold 值      │    Actual 值    │   绝对误差   │")
                logger.info(f"  ├─────────┼─────────────────┼─────────────────┼─────────────┤")
                
                for i in sample_indices:
                    gold_val = baseline_arr[i]
                    actual_val = test_arr[i]
                    error = abs_errors[i]
                    logger.info(f"  │   {i:4d}   │   {gold_val:13.6f}   │   {actual_val:13.6f}   │  {error:9.4e}  │")
                
                logger.info(f"  └─────────┴─────────────────┴─────────────────┴─────────────┘")
            
        comparison_lines = [
            comparison,
            f'status={status.value}; RMSE={rmse:.9g} (threshold < {rmse_threshold:.9g}); '
            f'fraction(relative error > {rel_error_threshold:.9g})={high_rel_error_ratio:.9g} (threshold < 0.01)',
            f'max absolute error={max_abs_error:.9g}; p99 absolute error={p99_abs_error:.9g}',
            f'worst flat index={worst_index}: baseline={baseline_arr[worst_index]:.17g}, '
            f'actual={test_arr[worst_index]:.17g}',
        ]
        if status == CorrectnessStatus.INCORRECT:
            comparison_lines.extend(
                f'flat index={int(i)}: baseline={baseline_arr[i]:.17g}, actual={test_arr[i]:.17g}, '
                f'absolute error={abs_errors[i]:.9g}'
                for i in np.argsort(abs_errors)[-5:][::-1] if abs_errors[i] > 0
            )
        return CorrectnessResult(
            status=status,
            comparison_log='\n'.join(comparison_lines),
            max_absolute_error=max_abs_error,
            mean_absolute_error=mean_abs_error,
            p99_absolute_error=p99_abs_error,
            worst_index=worst_index,
            worst_actual=float(test_arr[worst_index]),
            worst_expected=float(baseline_arr[worst_index]),
            rmse=rmse,
            max_relative_error=max_rel_error,
            output_path=str(test_path),
            baseline_path=str(baseline_path),
            tolerance=self.tolerance,
            high_rel_error_ratio=high_rel_error_ratio,
        )
        
    def _verify_bytes(
        self,
        test_path: Path,
        baseline_path: Path,
    ) -> CorrectnessResult:
        """字节级对比"""
        
        with open(test_path, 'rb') as f:
            test_content = f.read()
        with open(baseline_path, 'rb') as f:
            baseline_content = f.read()
            
        identical = test_content == baseline_content
        return CorrectnessResult(
            status=CorrectnessStatus.EXACT_MATCH if identical else CorrectnessStatus.INCORRECT,
            max_absolute_error=0.0 if identical else float('inf'),
            mean_absolute_error=0.0 if identical else float('inf'),
            rmse=0.0 if identical else float('inf'),
            comparison_log=(f'OutputVerifier byte comparison: actual={test_path} ({len(test_content)} bytes), '
                            f'baseline={baseline_path} ({len(baseline_content)} bytes); '
                            f'bytes_equal={identical}; numerical comparison unavailable'),
            output_path=str(test_path),
            baseline_path=str(baseline_path),
            tolerance=self.tolerance,
        )
            
    def _extract_numbers(self, file_path: Path) -> List[float]:
        """从文件中提取所有数字"""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
        # 匹配浮点数和整数
        numbers = []
        # 支持: 123, 123.45, .45, 123., 1e10, -123.45e-6
        pattern = r'(?<![\w.])(?:[-+]?(?:nan|inf(?:inity)?)(?!\w)|[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)'
        
        for match in re.finditer(pattern, content, flags=re.IGNORECASE):
            try:
                num = float(match.group())
                numbers.append(num)
            except ValueError:
                continue
                
        return numbers
