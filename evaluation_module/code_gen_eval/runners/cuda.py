"""
CUDA运行器 - 负责编译和运行CUDA代码
"""

import re
import os
import time
from pathlib import Path
from typing import Optional, Tuple, List

import logging

from .base import BaseRunner
from ..core.models import CompilationResult, RunResult, CompilationStatus, RunStatus

logger = logging.getLogger(__name__)


class CudaRunner(BaseRunner):
    """CUDA代码编译运行器"""
    
    def __init__(
        self,
        workdir: Path,
        nvcc_path: str = "nvcc",
        arch: str = "sm_80",
        cxx_flags: Optional[List[str]] = None,
        ld_flags: Optional[List[str]] = None,
        timeout: int = 120,
    ):
        super().__init__(workdir, timeout)
        self.nvcc_path = nvcc_path
        self.arch = arch
        self.cxx_flags = cxx_flags or ["-O2", "-std=c++17"]
        self.ld_flags = ld_flags or ["-lm"]
        
        # 检测CUDA环境
        self._check_cuda_env()
        
    def _check_cuda_env(self):
        """检查CUDA环境是否可用"""
        ret, out, err, _ = self.run_command([self.nvcc_path, "--version"])
        if ret != 0:
            logger.warning(f"CUDA编译器不可用: {self.nvcc_path}")
            self.cuda_available = False
        else:
            self.cuda_available = True
            # 提取版本信息
            version_match = re.search(r"release (\d+\.\d+)", out)
            if version_match:
                logger.info(f"检测到CUDA版本: {version_match.group(1)}")
        
    def compile(
        self,
        source_path: Path,
        output_name: Optional[str] = None,
    ) -> CompilationResult:
        """
        编译CUDA代码
        返回: CompilationResult
        """
        if not self.cuda_available:
            return CompilationResult(
                status=CompilationStatus.FAILED,
                error_message="CUDA编译器不可用",
            )
            
        source_path = Path(source_path)
        if not source_path.exists():
            return CompilationResult(
                status=CompilationStatus.FAILED,
                error_message=f"源文件不存在: {source_path}",
            )
            
        if output_name is None:
            output_name = source_path.stem + "_cuda"
            
        output_path = self.workdir / output_name
        
        # 构建编译命令
        cmd = [
            self.nvcc_path,
            f"-arch={self.arch}",
            *self.cxx_flags,
            str(source_path),
            "-o", str(output_path),
            *self.ld_flags,
        ]
        
        logger.info(f"编译CUDA代码: {source_path.name}")
        ret, stdout, stderr, elapsed = self.run_command(cmd)
        
        # 保存编译日志
        build_log = stdout + "\n" + stderr
        build_log_path = self.workdir / f"{output_name}_build.log"
        with open(build_log_path, 'w', encoding='utf-8') as f:
            f.write(build_log)
            
        # 分析警告
        warnings = re.findall(r"warning\s*:([^\n]+)", stderr, re.IGNORECASE)
        warning_types = []
        for w in warnings:
            # 提取警告类型
            type_match = re.search(r"warning\s*#?(\d+)", w) or re.search(r"([A-Z]+_WARNING)", w)
            if type_match:
                warning_types.append(type_match.group(1))
        
        result = CompilationResult(
            status=CompilationStatus.SUCCESS if ret == 0 else CompilationStatus.FAILED,
            return_code=ret,
            build_time_ms=elapsed * 1000,
            binary_size_bytes=output_path.stat().st_size if output_path.exists() else 0,
            warning_count=len(warnings),
            warning_types=sorted(set(warning_types)),
            error_message=None if ret == 0 else self._parse_compile_error(stderr),
            build_log=str(build_log_path),
            binary_path=str(output_path) if output_path.exists() else None,
        )
        
        if result.status == CompilationStatus.SUCCESS:
            logger.info(f"编译成功! 耗时: {elapsed:.2f}s, 二进制大小: {result.binary_size_bytes / 1024:.1f}KB")
        else:
            logger.warning(f"编译失败: {result.error_message}")
            
        return result
        
    def _parse_compile_error(self, stderr: str) -> str:
        """解析编译错误，提取核心错误信息"""
        # 找到第一个error
        error_match = re.search(r"error\s*:([^\n]+)", stderr, re.IGNORECASE)
        if error_match:
            return error_match.group(1).strip()
            
        # 如果没有找到，返回前两行
        lines = [l.strip() for l in stderr.strip().split("\n") if l.strip()]
        return " | ".join(lines[:2]) if lines else "Unknown compilation error"
        
    def run(
        self,
        binary_path: Path,
        args: Optional[List[str]] = None,
        num_runs: int = 1,
        warmup_runs: int = 0,
        capture_output: bool = True,
    ) -> RunResult:
        """
        运行CUDA二进制文件
        返回: RunResult
        """
        binary_path = Path(binary_path)
        if not binary_path.exists():
            return RunResult(
                status=RunStatus.FAILED,
                error_message=f"二进制文件不存在: {binary_path}",
            )
            
        args = args or []
        runtimes: List[float] = []
        full_output = []
        
        # 预热运行
        for i in range(warmup_runs):
            logger.debug(f"预热运行 {i+1}/{warmup_runs}")
            ret, stdout, stderr, _ = self.run_command(
                [str(binary_path), *args],
                timeout=self.timeout,
            )
            if ret != 0:
                logger.warning(f"预热运行失败，跳过性能测试")
                return RunResult(
                    status=self._classify_run_error(ret, stderr),
                    return_code=ret,
                    error_message=self._parse_run_error(stderr),
                    num_runs=0,
                )
        
        # 正式运行
        logger.info(f"运行测试，共 {num_runs} 次")
        run_status = RunStatus.SUCCESS
        final_error = None
        
        for i in range(num_runs):
            ret, stdout, stderr, elapsed = self.run_command(
                [str(binary_path), *args],
                timeout=self.timeout,
            )
            
            full_output.append(f"=== Run {i+1} ===")
            full_output.append(stdout)
            full_output.append(stderr)
            
            if ret != 0:
                run_status = self._classify_run_error(ret, stderr)
                final_error = self._parse_run_error(stderr)
                logger.warning(f"运行 {i+1} 失败: {run_status}")
                break
                
            # 尝试从输出提取运行时间
            runtime_ms = self._extract_runtime(stdout + "\n" + stderr)
            if runtime_ms is not None:
                runtimes.append(runtime_ms)
            else:
                # 如果无法从输出提取，使用wall time
                runtimes.append(elapsed * 1000)
                
        # 保存运行日志
        run_log_path = self.workdir / f"{binary_path.stem}_run.log"
        with open(run_log_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(full_output))
            
        # 计算统计值
        import numpy as np
        result = RunResult(
            status=run_status,
            return_code=ret if run_status != RunStatus.SUCCESS else 0,
            runtime_ms=runtimes,
            runtime_mean_ms=float(np.mean(runtimes)) if runtimes else None,
            runtime_median_ms=float(np.median(runtimes)) if runtimes else None,
            runtime_std_ms=float(np.std(runtimes)) if len(runtimes) > 1 else None,
            error_message=final_error,
            run_log=str(run_log_path),
            num_runs=len(runtimes),
        )
        
        if result.status == RunStatus.SUCCESS and runtimes:
            logger.info(f"运行成功! 平均耗时: {result.runtime_mean_ms:.2f}ms, "
                       f"中位数: {result.runtime_median_ms:.2f}ms")
            
        return result
        
    def _classify_run_error(self, return_code: int, stderr: str) -> RunStatus:
        """分类运行错误类型"""
        if return_code == -1 and "TIMEOUT" in stderr:
            return RunStatus.TIMEOUT
        if return_code == -11:  # SIGSEGV
            return RunStatus.SEGFAULT
        if return_code == -4:   # SIGILL
            return RunStatus.ILLEGAL_INSTRUCTION
            
        # 检查输出中的错误信息
        lower_stderr = stderr.lower()
        if "segmentation fault" in lower_stderr:
            return RunStatus.SEGFAULT
        if "illegal instruction" in lower_stderr:
            return RunStatus.ILLEGAL_INSTRUCTION
            
        return RunStatus.FAILED
        
    def _parse_run_error(self, stderr: str) -> str:
        """解析运行错误信息"""
        # 查找CUDA错误
        cuda_error = re.search(r"cudaError\w*[:\s]+([^\n]+)", stderr, re.IGNORECASE)
        if cuda_error:
            return cuda_error.group(1).strip()
            
        lines = [l.strip() for l in stderr.strip().split("\n") if l.strip()]
        return " | ".join(lines[:2]) if lines else "Unknown runtime error"
        
    def _extract_runtime(self, output: str) -> Optional[float]:
        """
        从程序输出中提取运行时间（毫秒）
        支持多种格式:
          - Time: 123.45(ms)
          - Time = 123.45 ms
          - runtime: 0.1234 s
          - 只有数字的一行 (PolyBench风格)
        """
        patterns = [
            r"[Tt]ime[:\s=]+([0-9]+\.?[0-9]*)\s*\(?ms\)?",
            r"[Tt]ime[:\s=]+([0-9]+\.?[0-9]*)\s*milli",
            r"[Rr]untime[:\s=]+([0-9]+\.?[0-9]*)\s*ms",
            r"elapsed[:\s=]+([0-9]+\.?[0-9]*)\s*ms",
        ]
        
        for pat in patterns:
            m = re.search(pat, output)
            if m:
                return float(m.group(1))
                
        # 检查秒单位
        patterns_seconds = [
            r"[Tt]ime[:\s=]+([0-9]+\.?[0-9]*)\s*s\b",
            r"[Tt]ime\s+in\s+seconds\s*=\s*([0-9]+\.?[0-9]*)",
            r"[Rr]untime[:\s=]+([0-9]+\.?[0-9]*)\s*s",
        ]
        
        for pat in patterns_seconds:
            m = re.search(pat, output)
            if m:
                return float(m.group(1)) * 1000
                
        # PolyBench风格: 只有一行数字
        for line in output.split("\n"):
            line = line.strip()
            if re.match(r"^[0-9]+\.?[0-9]*$", line):
                val = float(line)
                # 如果值大于100，可能是ms；否则是s
                if val > 100:
                    return val
                else:
                    return val * 1000
                    
        return None
