"""
基于Makefile的CUDA运行器 - 兼容现有数据集的构建系统

采用简洁的目录组织：
eval_workdir/
├── libs/                      ← 全局共享，一次复制，500MB
│   ├── headers/
│   ├── utils/
│   └── input_datas/
│
├── type-1/                    ← type 目录，每个 type 一份 Makefile + run.sh
│   ├── Makefile               ← 只复制一次
│   ├── run.sh                 ← 只复制一次
│   ├── c_1.c                  ← 任务源代码
│   ├── c_1/                   ← 任务专属目录
│   │   └── bin/               ← 编译产物
│   │   └── output/            ← 运行输出
│   ├── c_2.c                  ← 任务2源代码
│   └── c_2/
│
├── type-2/                    ← 其他 type 同理
│   └── ...
└── ...
"""

import re
import os
import shutil
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import logging

from .base import BaseRunner
from ..core.models import CompilationResult, RunResult, CompilationStatus, RunStatus

logger = logging.getLogger(__name__)


class MakefileCudaRunner(BaseRunner):
    """
    使用数据集自带的Makefile + run.sh 的运行器
    
    核心优化：
    1. libs 放在工作目录根目录，全局共享，只需复制一次
    2. 每个 type 只需要复制一次 Makefile 和 run.sh
    3. 不需要软链接，不需要每个任务复制文件
    4. 390个任务只需复制：500MB(libs) + 390×10KB = 504MB
    """
    
    def __init__(
        self,
        makefile_dir: Path,
        workdir: Path,
        plat: str = "cuda",
        dataset_size: str = "SMALL",
        num_threads: int = 4,
        timeout: int = 300,
        common_libs_dir: Optional[Path] = None,  # 保持向后兼容，但不再使用
        task_type: Optional[str] = None,  # 新增：任务类型
    ):
        super().__init__(workdir, timeout)
        self.makefile_dir = Path(makefile_dir)  # 数据集的 type 目录
        self.plat = plat
        self.dataset_size = dataset_size
        self.num_threads = num_threads
        self.task_type = task_type
        
        # 验证 Makefile 存在
        if not (self.makefile_dir / "Makefile").exists():
            logger.warning(f"Makefile不存在: {self.makefile_dir}/Makefile")
            
        # 验证 run.sh 存在
        if not (self.makefile_dir / "run.sh").exists():
            logger.warning(f"run.sh不存在: {self.makefile_dir}/run.sh")
        
    def prepare_workdir(self, source_path: Optional[Path] = None, target_name: Optional[str] = None):
        """
        准备工作目录 - 按 type 组织，libs 在上级目录共享
        
        注意：
        - libs 已经在 workdir/libs/（由 pipeline 初始化时复制一次）
        - Makefile 和 run.sh 在 type 目录初始化时已经复制
        - 这里只需要复制源文件即可
        """
        # 确保工作目录存在
        self.workdir.mkdir(parents=True, exist_ok=True)
        
        # 创建任务专属目录
        if target_name:
            task_dir = self.workdir / target_name
            task_dir.mkdir(exist_ok=True)
            
            # 复制源文件（在 type 目录下）
            if source_path and target_name:
                # 根据 plat 选择后缀
                if self.plat == 'cpu':
                    ext = '.c'
                else:
                    ext = '.cu'
                    
                dest_file = self.workdir / f"{target_name}{ext}"
                shutil.copy(source_path, dest_file)
                # logger.info(f"源文件已复制: {source_path.name} -> {dest_file.name}")
            
    def compile_with_make(
        self,
        target: str,
        source_path: Optional[Path] = None,
    ) -> CompilationResult:
        """
        使用Makefile编译
        
        Args:
            target: make目标名称（如 c_1, 2mm, etc.）
            source_path: 可选的源文件路径（替换默认源码）
        """
        # logger.info(f"编译任务: {target} (plat={self.plat})")
        
        # 准备工作目录
        # self.prepare_workdir(source_path, target)
        
        # 设置环境变量
        env = {}
        
        # 对于 type-5，需要设置 DATASET 参数和 NPB_DEFFILE 环境变量
        if self.task_type == "type-5":
            # type-5 的 Makefile 默认目标是 all，不需要传递 target
            cmd = ["make", f"PLAT={self.plat}", f"DATASET={self.dataset_size}"]
            # 设置 NPB_DEFFILE 环境变量，指向 type 目录下的 make.def（通过软链接）
            # self.workdir 是任务目录，如 eval_workdir/type-5/BT
            # libs 软链接在 type-5 目录下，所以需要上一级
            make_def_path = self.workdir.parent / 'libs' / 'utils' / 'make.def'
            env['NPB_DEFFILE'] = str(make_def_path)
        else:
            # 其他 type 需要传递 target
            cmd = ["make", target, f"PLAT={self.plat}"]
        
        ret, stdout, stderr, elapsed = self.run_command(
            cmd,
            cwd=self.workdir,  # 在任务目录下执行 make
            timeout=self.timeout,
            env=env,
        )
        
        # logger.info(f"编译命令输出:\n{stdout}")
        
        # 分析编译输出
        build_log = stdout + "\n" + stderr
        
        # 查找二进制文件
        bin_path = None
        
        # 对于 type-5，二进制文件格式是 bin/bt_CLASS_PLAT（小写）
        # 对于其他 type，格式是 bin/target_PLAT
        if self.task_type == "type-5":
            # 尝试查找 type-5 格式的二进制文件
            dataset_map = {
                "MINI": "S", "SMALL": "W", "MEDIUM": "A",
                "LARGE": "B", "EXTRALARGE": "C",
            }
            cls = dataset_map.get(self.dataset_size.upper(), "W")
            benchmark = target.lower()  # 转小写
            
            # 先尝试完整格式
            candidates = [
                self.workdir / "bin" / f"{benchmark}_{cls}_{self.plat}",
                self.workdir / "bin" / f"{benchmark}_{self.plat}",
                self.workdir / "bin" / f"{benchmark}",
            ]
            for candidate in candidates:
                if candidate.exists():
                    bin_path = candidate
                    break
        else:
            # 其他 type 的格式
            candidates = [
                self.workdir / "bin" / f"{target}_{self.plat}",
                self.workdir / f"{target}_{self.plat}",
            ]
            for candidate in candidates:
                if candidate.exists():
                    bin_path = candidate
                    break
        # 如果没找到，尝试搜索 bin 目录下的所有文件
        if not bin_path and (self.workdir / "bin").exists():
            for f in (self.workdir / "bin").iterdir():
                if f.is_file() and os.access(f, os.X_OK):
                    bin_path = f
                    break
        
                
        # 统计警告
        warnings = re.findall(r"warning\s*:([^\n]+)", stderr, re.IGNORECASE)
        warning_types = []
        for w in warnings:
            type_match = re.search(r"warning\s*#?(\d+)", w) or re.search(r"([A-Z]+_WARNING)", w)
            if type_match:
                warning_types.append(type_match.group(1))
        
        # 判断编译状态
        if ret != 0:
            error_msg = self._parse_compile_error(stderr)
            logger.warning(f"编译失败: {error_msg}")
            return CompilationResult(
                status=CompilationStatus.FAILED,
                return_code=ret,
                build_time_ms=elapsed * 1000,
                binary_size_bytes=0,
                warning_count=len(warnings),
                warning_types=sorted(set(warning_types)),
                error_message=error_msg,
                build_log=build_log,
                binary_path=None,
            )
            
        binary_size = bin_path.stat().st_size if bin_path else 0
        
        # logger.info(f"编译成功: 耗时={elapsed:.2f}s, 大小={binary_size/1024:.1f}KB")
        
        return CompilationResult(
            status=CompilationStatus.SUCCESS,
            return_code=0,
            build_time_ms=elapsed * 1000,
            binary_size_bytes=binary_size,
            warning_count=len(warnings),
            warning_types=sorted(set(warning_types)),
            error_message=None,
            build_log=build_log,
            binary_path=str(bin_path) if bin_path else None,
        )
        
    def run_with_runsh(
        self,
        target: str,
        num_runs: int = 3,
        warmup_runs: int = 1,
        enable_dump_output: bool = True,
    ) -> RunResult:
        """
        使用run.sh运行
        
        Args:
            target: 运行目标名称
            num_runs: 正式运行次数
            warmup_runs: 预热运行次数
            enable_dump_output: 是否dump输出到output目录
        """
        # logger.info(f"运行任务: {target} (plat={self.plat}, runs={num_runs})")
        
        runtimes = []
        full_output = []
        
        # 预热运行
        for i in range(warmup_runs):
            logger.debug(f"预热运行 {i+1}/{warmup_runs}")
            ret, stdout, stderr, _ = self._run_once(target, enable_dump_output=False)
            if ret != 0:
                error_msg = self._parse_run_error(stderr)
                logger.warning(f"预热运行失败 (返回码 {ret}): {error_msg}")
                logger.warning(f"标准输出:\n{stdout}")
                logger.warning(f"错误输出:\n{stderr}")
                return RunResult(
                    status=self._classify_run_error(ret, stderr),
                    return_code=ret,
                    runtime_ms=runtimes,
                    runtime_mean_ms=None,
                    runtime_median_ms=None,
                    runtime_std_ms=None,
                    error_message=error_msg,
                    run_log=None,
                    num_runs=0,
                )
                
        # 正式运行
        for i in range(num_runs):
            ret, stdout, stderr, elapsed = self._run_once(
                target, 
                enable_dump_output=enable_dump_output and (i == num_runs - 1)  # 只在最后一次dump
            )
            
            run_log = f"=== Run {i+1} ===\n{stdout}\n{stderr}"
            full_output.append(run_log)
            
            if ret != 0:
                status = self._classify_run_error(ret, stderr)
                error = self._parse_run_error(stderr)
                logger.warning(f"运行 {i+1} 失败: {status}")
                return RunResult(
                    status=status,
                    return_code=ret,
                    runtime_ms=runtimes,
                    runtime_mean_ms=None,
                    runtime_median_ms=None,
                    runtime_std_ms=None,
                    error_message=error,
                    run_log="\n".join(full_output),
                    num_runs=len(runtimes),
                )
                
            # 提取运行时间
            runtime_ms = self._extract_runtime(stdout + "\n" + stderr)
            if runtime_ms is not None:
                runtimes.append(runtime_ms)
            else:
                # 无法从输出提取，使用 elapsed 时间
                runtimes.append(elapsed * 1000)
                
        # 计算统计值
        import numpy as np
        mean_ms = float(np.mean(runtimes)) if runtimes else None
        median_ms = float(np.median(runtimes)) if runtimes else None
        std_ms = float(np.std(runtimes)) if len(runtimes) > 1 else None
        
        # logger.info(f"运行成功: {len(runtimes)}次, "
        #            f"平均={mean_ms:.1f}ms, "
        #            f"中位数={median_ms:.1f}ms, "
        #            f"标准差={std_ms:.1f}ms" if std_ms else "")
        
        return RunResult(
            status=RunStatus.SUCCESS,
            return_code=0,
            runtime_ms=runtimes,
            runtime_mean_ms=mean_ms,
            runtime_median_ms=median_ms,
            runtime_std_ms=std_ms,
            error_message=None,
            run_log="\n".join(full_output),
            num_runs=len(runtimes),
        )
        
    def _run_once(
        self, 
        target: str, 
        enable_dump_output: bool = True
    ) -> Tuple[int, str, str, float]:
        """
        执行一次运行
        
        不同type的run.sh参数格式不同
        """
        # 使用显式传入的 task_type，或者从 workdir 路径推断
        type_name = self.task_type
        if not type_name:
            # 尝试从 workdir 路径推断（兼容旧代码）
            type_name = self.workdir.name
            logger.warning(f"task_type 为空，从 workdir.name 推断: {type_name}")
        
        # logger.info(f"执行 _run_once: type_name={type_name}, target={target}, plat={self.plat}")
        
        if type_name == "type-1":
            cmd = ["bash", "run.sh", target, self.plat]
            if enable_dump_output:
                cmd.append("--verbose")
                
        elif type_name == "type-3":
            cmd = ["bash", "run.sh", "-b", target, "-p", self.plat]
            if enable_dump_output:
                cmd.append("-v")
                
        elif type_name == "type-5":
            dataset_map = {
                "MINI": "S", "SMALL": "W", "MEDIUM": "A",
                "LARGE": "B", "EXTRALARGE": "C",
            }
            cls = dataset_map.get(self.dataset_size.upper(), "W")
            cmd = ["bash", "run.sh", "-c", cls, "-p", self.plat]
            if enable_dump_output:
                cmd.append("-v")
                
        else:  # type-2, type-4
            cmd = [
                "bash", "run.sh",
                target,
                "-d", self.dataset_size,
                "-p", self.plat,
                "-t", str(self.num_threads),
            ]
            if enable_dump_output:
                cmd.append("-v")
            
        # logger.info(f"执行命令: {' '.join(cmd)} (cwd={self.workdir})")
        return self.run_command(cmd, cwd=self.workdir, timeout=self.timeout)
        
    def get_output_file(self) -> Optional[Path]:
        """获取生成的output文件路径"""
        # 搜索工作目录下所有可能的 txt 输出文件
        patterns = [
            "output.txt",
            "*_output.txt",
            f"{self.plat}_*.txt",
        ]
        
        # 对于 type-5，bt.cpp 会生成 bt_out.txt
        if self.task_type == "type-5":
            patterns.append("bt_out.txt")
            patterns.append("*_out.txt")
        
        for pattern in patterns:
            for f in self.workdir.glob(pattern):
                if f.is_file():
                    return f
                    
        return None
        
    def _parse_compile_error(self, stderr: str) -> str:
        """解析编译错误"""
        error_match = re.search(r"error\s*:([^\n]+)", stderr, re.IGNORECASE)
        if error_match:
            return error_match.group(1).strip()
        lines = [l.strip() for l in stderr.strip().split("\n") if l.strip()]
        return " | ".join(lines[:2]) if lines else "Unknown compilation error"
        
    def _classify_run_error(self, return_code: int, stderr: str) -> RunStatus:
        """分类运行错误类型"""
        if return_code == -1 and "TIMEOUT" in stderr:
            return RunStatus.TIMEOUT
        if return_code == -11:
            return RunStatus.SEGFAULT
        if return_code == -4:
            return RunStatus.ILLEGAL_INSTRUCTION
            
        lower_stderr = stderr.lower()
        if "segmentation fault" in lower_stderr:
            return RunStatus.SEGFAULT
        if "illegal instruction" in lower_stderr:
            return RunStatus.ILLEGAL_INSTRUCTION
            
        return RunStatus.FAILED
        
    def _parse_run_error(self, stderr: str) -> str:
        """解析运行错误"""
        cuda_error = re.search(r"cudaError\w*[:\s]+([^\n]+)", stderr, re.IGNORECASE)
        if cuda_error:
            return cuda_error.group(1).strip()
        lines = [l.strip() for l in stderr.strip().split("\n") if l.strip()]
        return " | ".join(lines[:2]) if lines else "Unknown runtime error"
        
    def _extract_runtime(self, output: str) -> Optional[float]:
        """从输出提取运行时间（毫秒）"""
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
                
        # 秒单位
        patterns_seconds = [
            r"[Tt]ime[:\s=]+([0-9]+\.?[0-9]*)\s*s\b",
            r"[Tt]ime\s+in\s+seconds\s*=\s*([0-9]+\.?[0-9]*)",
        ]
        
        for pat in patterns_seconds:
            m = re.search(pat, output)
            if m:
                return float(m.group(1)) * 1000
                
        # PolyBench: 只有一行数字
        for line in output.split("\n"):
            line = line.strip()
            if re.match(r"^[0-9]+\.?[0-9]*$", line):
                val = float(line)
                # 如果值大于100，可能是毫秒，否则是秒
                return val if val > 100 else val * 1000
                    
        return None
