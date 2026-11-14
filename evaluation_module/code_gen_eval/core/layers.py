
"""
L1-L4 层级评估系统 - 统一的评估接口
"""
import csv
import io
import os
import re
import time
import shutil
import subprocess
import statistics
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
from abc import ABC, abstractmethod
import logging
from .timing import validate_timing_report
logger = logging.getLogger(__name__)


class LayerType(Enum):
    L1 = "L1"  # 编译检查
    L2 = "L2"  # 运行时安全检查
    L3 = "L3"  # 功能正确性检查
    L4 = "L4"  # 性能检查


class LayerResult:
    """单个层级的评估结果"""
    def __init__(self, layer: LayerType):
        self.layer = layer
        self.passed: bool = False
        self.errors: List[Dict[str, Any]] = []
        self.metrics: Dict[str, Any] = {}
        self.raw_log: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer": self.layer.value,
            "passed": self.passed,
            "errors": self.errors,
            "metrics": self.metrics,
            "raw_log": self.raw_log
        }

    def build_optimization_prompt(self, source_code: str, task_type: str, task_id: str) -> str:
        """生成优化提示词"""
        layer_name = self.layer.value

        if self.layer == LayerType.L1:
            signal = f"编译失败\n"
            for e in self.errors:
                signal += f"- {e.get('type', 'compile_error')} 第{e.get('line', '?')}行: {e.get('message', '')}\n"
            return f"""你是 CUDA 编程专家。生成的 CUDA 程序编译失败，请修复代码。

诊断信号 ({layer_name}: 编译失败):
{signal}
当前代码:
```cuda
{source_code}
```

请返回完整修复后的 CUDA 代码，放在 ```cuda 代码块中。"""

        elif self.layer == LayerType.L2:
            signal = f"运行时安全违规\n"
            for v in self.errors:
                signal += (f"- {v.get('type', 'violation')} x{v.get('count', 1)}: "
                           f"line={v.get('line')}, thread={v.get('thread')}, "
                           f"block={v.get('block')}, address={v.get('address')}, "
                           f"size={v.get('size')} {v.get('message', '')}\n")
            return f"""你是 CUDA 编程专家。生成的 CUDA 程序存在运行时安全问题，请修复。

诊断信号 ({layer_name}: 运行时安全):
{signal}
当前代码:
```cuda
{source_code}
```

请返回完整修复后的 CUDA 代码，放在 ```cuda 代码块中。"""

        elif self.layer == LayerType.L3:
            m = self.metrics or {}
            num_signal = []
            if m.get("rmse") is not None:
                num_signal.append(f"- RMSE={m['rmse']:.4e} (阈值 1e-3)")
            if m.get("high_rel_error_ratio") is not None:
                num_signal.append(
                    f"- 相对误差>1e-5 的元素比例={m['high_rel_error_ratio']*100:.4f}% (阈值 1%)"
                )
            if m.get("max_absolute_error") is not None:
                num_signal.append(f"- 最大绝对误差={m['max_absolute_error']:.4e}")
            if m.get('p99_absolute_error') is not None:
                num_signal.append(f"- p99 绝对误差={m['p99_absolute_error']:.4e}")
            if m.get('worst_index') is not None:
                num_signal.append(f"- 最差元素索引={m['worst_index']}, CPU={m['worst_expected']}, CUDA={m['worst_actual']}")
            num_signal.append(f"- NaN/Inf: CPU={m.get('baseline_nonfinite', False)}, CUDA={m.get('actual_nonfinite', False)}")
            num_signal_str = "\n".join(num_signal) if num_signal else "输出与 CPU 基准不匹配。"
            return f"""你是 CUDA 编程专家。生成的 CUDA 程序输出结果不正确，请修复。

诊断信号 ({layer_name}: 功能正确性):
{num_signal_str}

当前代码:
```cuda
{source_code}
```

请返回完整修复后的 CUDA 代码，放在 ```cuda 代码块中。"""

        else:  # L4
            m = self.metrics or {}
            nsys = m.get("nsys") or {}
            perf_lines = []
            for field in ('kernel_time_ms', 'h2d_time_ms', 'd2h_time_ms',
                          'memcpy_kernel_ratio', 'has_overlap'):
                value = nsys.get(field)
                display = 'unknown（未知）' if value is None else str(value)
                perf_lines.append(f"- {field}={display}")
            perf_signal = "\n".join(perf_lines)
            return f"""你是 CUDA 性能优化专家。请优化以下 CUDA 代码以获得更好的性能。

诊断信号 ({layer_name}: 性能优化):
{perf_signal}

当前代码:
```cuda
{source_code}
```

请返回完整优化后的 CUDA 代码，放在 ```cuda 代码块中。"""


class BaseLayerEvaluator(ABC):
    """层级评估器的基类"""

    def __init__(self, workdir: Path, config: Dict[str, Any]):
        self.workdir = workdir
        self.config = config
        self.workdir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def evaluate(self, source_path: Path, task_type: str, task_id: str) -> LayerResult:
        pass


class L1Evaluator(BaseLayerEvaluator):
    """L1: 编译检查"""

    def evaluate(self, source_path: Path, task_type: str, task_id: str) -> LayerResult:
        result = LayerResult(LayerType.L1)

        logger.info("L1: 编译检查")

        # 准备包含路径
        dataset_root = Path(self.config.get("dataset_root", "./code_datas"))
        include_path = dataset_root / "libs" / "headers"

        bin_dir = self.workdir / "bin"
        bin_dir.mkdir(parents=True, exist_ok=True)
        bin_path = bin_dir / f"{task_id}_cuda"

        cmd = [
            "nvcc",
            str(source_path.resolve()),
            "-o", str(bin_path.resolve()),
            "-lm"
        ]
        if include_path.exists():
            cmd.extend(["-I", str(include_path.resolve())])

        try:
            proc = subprocess.run(
                cmd,
                cwd=self.workdir,
                capture_output=True,
                text=True,
                timeout=self.config.get("timeout", 300)
            )
            result.raw_log = proc.stdout + "\n" + proc.stderr

            # 解析错误
            errors = []
            pattern = re.compile(
                r"(?P<file>[^:\s\(]+)\((?P<line>\d+)\):\s+error:\s+(?P<msg>.+)"
            )
            for match in pattern.finditer(result.raw_log):
                errors.append({
                    "line": int(match.group("line")),
                    "type": match.group('msg').split(':', 1)[0].strip(),
                    "message": match.group("msg").strip()
                })

            # 去重
            seen = set()
            unique_errors = []
            for e in errors:
                key = (e["line"], e["message"])
                if key not in seen:
                    seen.add(key)
                    unique_errors.append(e)

            result.errors = unique_errors
            result.passed = proc.returncode == 0
            result.metrics["error_count"] = len(unique_errors)
            result.metrics["binary_path"] = str(bin_path) if result.passed else None

        except subprocess.TimeoutExpired:
            result.passed = False
            result.errors.append({"message": "编译超时"})
        except Exception as e:
            result.passed = False
            result.errors.append({"message": f"编译异常: {str(e)}"})

        return result


class L2Evaluator(BaseLayerEvaluator):
    """L2: 运行时安全检查"""

    def evaluate(self, source_path: Path, task_type: str, task_id: str) -> LayerResult:
        result = LayerResult(LayerType.L2)
        logger.info("L2: 运行时安全检查")

        # 编译调试版本
        dataset_root = Path(self.config.get("dataset_root", "./code_datas"))
        include_path = dataset_root / "libs" / "headers"

        bin_dir = self.workdir / "bin"
        bin_dir.mkdir(parents=True, exist_ok=True)
        bin_path = bin_dir / f"{task_id}_cuda_debug"

        cmd = [
            "nvcc",
            str(source_path.resolve()),
            "-o", str(bin_path.resolve()),
            "-g", "-G", "-lm"
        ]
        if include_path.exists():
            cmd.extend(["-I", str(include_path.resolve())])

        try:
            proc = subprocess.run(
                cmd,
                cwd=self.workdir,
                capture_output=True,
                text=True,
                timeout=self.config.get("timeout", 300)
            )
            if proc.returncode != 0:
                result.passed = False
                result.errors.append({"message": "调试版本编译失败"})
                result.raw_log = proc.stdout + "\n" + proc.stderr
                return result

            logs = []
            failed_tools = []
            for tool in ('memcheck', 'racecheck', 'initcheck', 'synccheck'):
                proc = subprocess.run(
                    ['compute-sanitizer', '--tool', tool, '--error-exitcode', '1',
                     str(bin_path.resolve())], cwd=self.workdir, capture_output=True,
                    text=True, timeout=self.config.get('timeout', 300)
                )
                logs.append(f'[{tool}]\n{proc.stdout}\n{proc.stderr}')
                if proc.returncode != 0:
                    failed_tools.append(tool)
            result.raw_log = '\n'.join(logs)

            violations = []
            # 解析越界访问
            oob_pattern = re.compile(
                r"Invalid __global__ (?P<rw>read|write) of size (?P<sz>\d+) bytes.*?"
                r"at (?P<addr>0x[0-9a-fA-F]+) in [^:\n]+:(?P<line>\d+):",
                re.DOTALL
            )
            for m in oob_pattern.finditer(result.raw_log):
                context = result.raw_log[m.end():].split('\n', 1)[0]
                coords = re.search(r'thread\s*\(([^)]+)\).*?block\s*\(([^)]+)\)',
                                   context, re.IGNORECASE | re.DOTALL)
                violations.append({
                    "type": f"out_of_bounds_{m.group('rw')}",
                    "line": int(m.group("line")),
                    "size": int(m.group("sz")),
                    "address": m.group("addr"),
                    "thread": coords.group(1) if coords else None,
                    "block": coords.group(2) if coords else None,
                })

            # 解析竞态条件
            race_pattern = re.compile(
                r"Warning: Race reported between.*?"
                r"in (?P<path1>[^:]+):(?P<line1>\d+):.*?"
                r"in (?P<path2>[^:]+):(?P<line2>\d+):",
                re.DOTALL
            )
            for m in race_pattern.finditer(result.raw_log):
                violations.append({
                    "type": "race_condition",
                    "write_line": int(m.group("line1")),
                    "read_line": int(m.group("line2"))
                })

            grouped = {}
            for violation in violations:
                kind = violation['type']
                if kind not in grouped:
                    grouped[kind] = {**violation, 'count': 0}
                grouped[kind]['count'] += 1
            result.errors = list(grouped.values()) + [
                {'type': tool, 'message': f'{tool} failed (unparsed or runtime failure)', 'count': 1}
                for tool in failed_tools if not any(log.startswith(f'[{tool}]') and
                                                      any(word in log for word in ('Invalid __global__', 'Race reported'))
                                                      for log in logs)
            ]
            result.passed = not failed_tools and not violations
            result.metrics['violation_count'] = len(violations)
            result.metrics['failed_tools'] = failed_tools

        except Exception as e:
            result.passed = False
            result.errors.append({"message": f"运行时安全检查异常: {str(e)}"})

        return result


class L3Evaluator(BaseLayerEvaluator):
    """L3: 功能正确性检查

    使用项目自带的 Makefile + run.sh 进行编译和运行
    1) 使用 MakefileCudaRunner 编译（复用现有构建系统）
    2) 使用 run.sh 运行（自动处理 --verbose 参数）
    3) 使用 OutputVerifier 与 outputs_gold/{type}/{task_id}_output.txt 对比
       采用双标准: high_rel_error_ratio < 1% AND rmse < 1e-3
    """

    def evaluate(self, source_path: Path, task_type: str, task_id: str) -> LayerResult:
        result = LayerResult(LayerType.L3)
        logger.info("L3: 功能正确性检查")

        try:
            outputs_gold = Path(self.config.get("outputs_gold", "./outputs_gold_LARGE"))
            gold_file = outputs_gold / task_type / f"{task_id}_output.txt"
            if not gold_file.exists():
                result.passed = False
                result.errors.append({"message": f"Gold 输出文件不存在: {gold_file}"})
                result.metrics["gold_file"] = None
                return result

            if task_type not in {f'type-{i}' for i in range(1, 6)} or not re.fullmatch(r'[A-Za-z0-9_-]+', task_id):
                raise ValueError('Invalid benchmark type or task identifier')
            dataset_root = Path(self.config.get("dataset_root", "./code_datas")).resolve()
            task_dir = self.workdir / task_type / task_id
            task_dir.mkdir(parents=True, exist_ok=True)
            libs = self.workdir / 'libs'
            dataset_libs = dataset_root / 'libs'
            if not dataset_libs.is_dir():
                raise FileNotFoundError(f'Benchmark libraries missing: {dataset_libs}')
            if libs.is_symlink():
                if libs.resolve() != dataset_libs:
                    raise ValueError('L3 libs link points to another corpus')
            elif libs.exists():
                raise ValueError('L3 libs path must be an isolated link to the selected dataset')
            else:
                libs.symlink_to(dataset_libs, target_is_directory=True)

            target_source = task_dir / f"{task_id}.cu"
            if Path(source_path).resolve() != target_source.resolve():
                shutil.copy(source_path, target_source)
            makefile_src = dataset_root / task_type / 'Makefile'
            runsh_src = dataset_root / task_type / 'run.sh'
            if not makefile_src.is_file() or not runsh_src.is_file():
                raise FileNotFoundError(f'Missing benchmark Makefile or run.sh for {task_type}')
            shutil.copy(makefile_src, task_dir / 'Makefile')
            shutil.copy(runsh_src, task_dir / 'run.sh')
            os.chmod(task_dir / 'run.sh', 0o755)

            from ..runners.makefile_runner import MakefileCudaRunner

            runner = MakefileCudaRunner(
                makefile_dir=task_dir,
                workdir=task_dir,
                plat="cuda",
                dataset_size=self.config.get("dataset_size", "SMALL"),
                num_threads=self.config.get("num_threads", 4),
                timeout=self.config.get("timeout", 300),
                task_type=task_type,
            )

            # 编译
            from ..core.models import CompilationStatus
            compile_result = runner.compile_with_make(target=task_id)
            if compile_result.status != CompilationStatus.SUCCESS:
                result.passed = False
                result.errors.append({
                    "message": f"编译失败: {compile_result.error_message}"
                })
                result.raw_log = compile_result.build_log or ""
                return result

            # 运行（enable_dump_output=True 会自动添加 --verbose）
            from ..core.models import RunStatus
            run_result = runner.run_with_runsh(
                target=task_id,
                num_runs=1,
                warmup_runs=0,
                enable_dump_output=True,
            )

            if run_result.status != RunStatus.SUCCESS:
                result.passed = False
                result.errors.append({
                    "message": f"运行失败: {run_result.error_message}"
                })
                result.raw_log = run_result.run_log or ""
                return result

            # 6. 查找实际输出文件
            actual_file = runner.get_output_file()
            if not actual_file:
                # 如果没找到，尝试查找 run.sh 生成的输出文件
                actual_file = task_dir / f"{task_id}_output.txt"

            if not actual_file or not actual_file.exists():
                result.passed = False
                result.errors.append({"message": "未找到输出文件"})
                return result

            # 7. 使用 OutputVerifier 进行双标准对比
            from ..verifiers.output import OutputVerifier
            from .models import CorrectnessStatus

            verifier = OutputVerifier(tolerance=self.config.get("tolerance", 1e-5))
            cr = verifier.verify(test_output_path=actual_file, baseline_output_path=gold_file)
            result.raw_log = cr.comparison_log or ''

            # 8. 写入 metrics
            result.metrics.update({
                "gold_file": str(gold_file),
                "actual_file": str(actual_file),
                "status": cr.status.value if hasattr(cr.status, "value") else str(cr.status),
                "max_absolute_error": cr.max_absolute_error,
                "mean_absolute_error": cr.mean_absolute_error,
                "rmse": cr.rmse,
                "max_relative_error": cr.max_relative_error,
                "high_rel_error_ratio": cr.high_rel_error_ratio,
                "p99_absolute_error": cr.p99_absolute_error,
                "worst_index": cr.worst_index,
                "worst_actual": cr.worst_actual,
                "worst_expected": cr.worst_expected,
                "actual_nonfinite": cr.actual_nonfinite,
                "baseline_nonfinite": cr.baseline_nonfinite,
            })

            # passed: EXACT_MATCH 或 WITHIN_TOLERANCE (双标准已在 verifier 里判定)
            result.passed = cr.status in (
                CorrectnessStatus.EXACT_MATCH,
                CorrectnessStatus.WITHIN_TOLERANCE,
            )

            if not result.passed:
                result.errors.append({
                    "message": f"输出不正确 (status={result.metrics['status']})",
                    "rmse": cr.rmse,
                    "high_rel_error_ratio": cr.high_rel_error_ratio,
                })

        except Exception as e:
            result.passed = False
            result.errors.append({"message": f"正确性检查异常: {str(e)}"})

        return result


class L4Evaluator(BaseLayerEvaluator):
    """L4: 性能检查

    1) 编译 CUDA 与 C baseline (若未就绪)
    2) Only the exact-source synthetic adapter can supply instrumented samples.
    3) Synthetic speedup is illustrative, not validated benchmark evidence; arbitrary adapters fail closed.
    Process times are proxy-only; collect structured nsys diagnostics only after a synthetic timing failure.
    """

    WARMUP = 3
    RUNS = 3
    
    def _compile_cuda_release(self, source_path: Path, task_id: str) -> Optional[Path]:
        dataset_root = Path(self.config.get("dataset_root", "./code_datas"))
        include_path = dataset_root / "libs" / "headers"
        bin_dir = self.workdir / "bin"
        bin_dir.mkdir(parents=True, exist_ok=True)
        bin_path = bin_dir / f"{task_id}_cuda"
        if bin_path.exists() and not self.config.get('synthetic_vector_add_timing'):
            return bin_path
        cmd = ["nvcc", str(source_path.resolve()), "-o", str(bin_path.resolve()), "-O3", "-arch=sm_89", "-lm"]
        if include_path.exists():
            cmd.extend(["-I", str(include_path.resolve())])
        try:
            proc = subprocess.run(cmd, cwd=self.workdir, capture_output=True, text=True,
                                  timeout=self.config.get("timeout", 300))
            return bin_path if proc.returncode == 0 else None
        except Exception:
            return None

    def _compile_c_baseline(self, task_type: str, task_id: str) -> Optional[Path]:
        dataset_root = Path(self.config.get("dataset_root", "./code_datas"))
        c_source = Path(self.config.get('synthetic_cpu_source', '')) if self.config.get('synthetic_vector_add_timing') else (
            dataset_root / task_type / task_id / "c" / f"{task_id}.c")
        if not c_source.exists() and not self.config.get('synthetic_vector_add_timing'):
            c_source = dataset_root / task_type / task_id / "c" / f"c_{task_id.split('_')[-1]}.c"
        if not c_source.is_file():
            logger.warning(f"L4: 找不到 C baseline 源文件 ({task_id})")
            return None

        include_path = dataset_root / "libs" / "headers"
        bin_dir = self.workdir / "bin"
        bin_dir.mkdir(parents=True, exist_ok=True)
        bin_path = bin_dir / f"{task_id}_c"
        if bin_path.exists() and not self.config.get('synthetic_vector_add_timing'):
            return bin_path
        cmd = ["gcc", "-O3", str(c_source.resolve()), "-o", str(bin_path.resolve()), "-lm"]
        if include_path.exists():
            cmd.extend(["-I", str(include_path.resolve())])
        try:
            proc = subprocess.run(cmd, cwd=self.workdir, capture_output=True, text=True,
                                  timeout=self.config.get("timeout", 300))
            if proc.returncode != 0:
                logger.warning(f"L4: C baseline 编译失败: {proc.stderr[-300:]}")
                return None
            return bin_path
        except Exception as e:
            logger.warning(f"L4: C baseline 编译异常: {e}")
            return None

    def _parse_self_reported_time(self, stdout: str) -> Optional[float]:
        """尝试从程序输出里解析自报时间，支持常见格式。"""
        patterns = [
            r'[Tt]ime[:\s]+([0-9.]+)\s*ms',
            r'[Ee]lapsed[:\s]+([0-9.]+)\s*ms',
            r'[Dd]uration[:\s]+([0-9.]+)\s*ms',
            r'[Tt]otal[:\s]+([0-9.]+)\s*ms',
            r'([0-9.]+)\s*ms',  # 兜底：第一个 ms 数字
        ]
        for pat in patterns:
            m = re.search(pat, stdout)
            if m:
                try:
                    val = float(m.group(1))
                    if val > 0:
                        return val
                except ValueError:
                    continue
        return None

    def _time_binary_ms(
        self, binary: Path, kind: str = "cuda", force_wall: bool = False
    ) -> Tuple[Optional[float], str]:
        """运行 WARMUP + RUNS 次。
        优先使用程序自报时间，不可用时降级到 wall-clock。
        force_wall=True 时强制使用 wall-clock（用于来源不一致时重测）。
 
        Returns:
            (ms, source)  source: 'self_reported' | 'wall_clock' | 'timeout' | 'error'
        """
        times_wall: List[float] = []
        times_self: List[float] = []
        run_timeout = self.config.get("run_timeout", 120)
 
        for _ in range(self.WARMUP + self.RUNS):
            t0 = time.perf_counter()
            try:
                proc = subprocess.run(
                    [str(binary.resolve())], cwd=self.workdir,
                    capture_output=True, text=True, timeout=run_timeout
                )
            except subprocess.TimeoutExpired:
                return None, "timeout"
            if proc.returncode != 0:
                return None, "error"
 
            times_wall.append((time.perf_counter() - t0) * 1000.0)
            t = self._parse_self_reported_time(proc.stdout)
            if t is not None:
                times_self.append(t)
 
        kept_wall = times_wall[self.WARMUP:]
        kept_self = (
            []
            if force_wall
            else (times_self[self.WARMUP:] if len(times_self) == self.WARMUP + self.RUNS else [])
        )
 
        if kept_self:
            ms = statistics.median(kept_self)
            logger.info(f"L4: [{kind}] 自报时间 = {ms:.2f}ms (wall={statistics.median(kept_wall):.2f}ms)")
            return ms, "self_reported"
        else:
            ms = statistics.median(kept_wall)
            logger.info(f"L4: [{kind}] wall-clock = {ms:.2f}ms")
            return ms, "wall_clock"

    @staticmethod
    def _parse_nsys_csv(text: str) -> Optional[List[Tuple[str, float]]]:
        columns = None
        rows = []
        scales = {'ns': 1e-6, 'us': 1e-3, 'µs': 1e-3, 'ms': 1.0, 's': 1e3}
        try:
            for row in csv.reader(io.StringIO(text), strict=True):
                cells = [cell.strip().lstrip('\ufeff') for cell in row]
                if columns is None:
                    total = next((
                        (i, re.fullmatch(r'Total\s+Time\s*\((ns|us|µs|ms|s)\)', cell, re.I))
                        for i, cell in enumerate(cells)
                        if re.fullmatch(r'Total\s+Time\s*\((ns|us|µs|ms|s)\)', cell, re.I)
                    ), None)
                    name = next((i for i, cell in enumerate(cells)
                                 if cell.lower() in ('name', 'operation')), None)
                    if total is not None and name is not None:
                        columns = (len(cells), total[0], name, scales[total[1].group(1).lower()])
                    continue
                if not cells or not any(cells):
                    if rows:
                        break
                    continue
                width, time_index, name_index, scale = columns
                if len(cells) != width:
                    return None
                value = float(cells[time_index].replace(',', '')) * scale
                if not cells[name_index] or not 0 <= value < float('inf'):
                    return None
                rows.append((cells[name_index], value))
        except (csv.Error, ValueError):
            return None
        return rows or None

    def _run_nsys(self, binary: Path) -> Dict[str, Any]:
        """Collect device-time summaries; overlap requires timeline evidence."""
        diagnostics = dict.fromkeys(('kernel_time_ms', 'h2d_time_ms', 'd2h_time_ms',
                                     'memcpy_kernel_ratio', 'has_overlap'))
        raw_stats = []
        try:
            report = self.workdir / 'nsys_report'
            profile = subprocess.run(
                ['nsys', 'profile', '--output', str(report),
                 '--force-overwrite', 'true', str(binary.resolve())],
                cwd=self.workdir, capture_output=True, text=True,
                timeout=self.config.get('timeout', 300),
            )
            if profile.returncode != 0:
                return diagnostics
            for kind, reports in (
                ('kernel', ('cuda_gpu_kern_sum', 'gpukernsum')),
                ('memcpy', ('cuda_gpu_mem_time_sum', 'gpumemtimesum')),
            ):
                rows = None
                for report_name in reports:
                    stats = subprocess.run(
                        ['nsys', 'stats', '--format', 'csv', '--report', report_name,
                         str(report) + '.nsys-rep'],
                        cwd=self.workdir, capture_output=True, text=True,
                        timeout=self.config.get('timeout', 300),
                    )
                    if stats.returncode == 0 and stats.stdout.strip():
                        raw_stats.append(f'[{report_name}]\n{stats.stdout}')
                        rows = self._parse_nsys_csv(stats.stdout)
                        if rows is not None:
                            break
                if rows is None:
                    continue
                if kind == 'kernel':
                    total = sum(value for _, value in rows)
                    if total < float('inf'):
                        diagnostics['kernel_time_ms'] = total
                else:
                    transfers = {'h2d_time_ms': [], 'd2h_time_ms': []}
                    unknown_direction = False
                    for name, value in rows:
                        direction = re.sub(r'[^a-z0-9]', '', name.lower())
                        if 'htod' in direction or 'hosttodevice' in direction:
                            transfers['h2d_time_ms'].append(value)
                        elif 'dtoh' in direction or 'devicetohost' in direction:
                            transfers['d2h_time_ms'].append(value)
                        elif not any(label in direction for label in
                                     ('dtod', 'devicetodevice', 'htoh', 'hosttohost', 'memset')):
                            unknown_direction = True
                    if not unknown_direction:
                        for field, values in transfers.items():
                            if values and sum(values) < float('inf'):
                                diagnostics[field] = sum(values)
            kernel = diagnostics['kernel_time_ms']
            h2d, d2h = diagnostics['h2d_time_ms'], diagnostics['d2h_time_ms']
            if kernel is not None and kernel > 0 and h2d is not None and d2h is not None:
                ratio = (h2d + d2h) / kernel
                if ratio < float('inf'):
                    diagnostics['memcpy_kernel_ratio'] = ratio
            return diagnostics
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return diagnostics
        finally:
            diagnostics['raw_stats'] = '\n'.join(raw_stats)

    
    def evaluate(self, source_path: Path, task_type: str, task_id: str) -> LayerResult:
        result = LayerResult(LayerType.L4)
        logger.info("L4: 性能检查")
 
        try:
            from .synthetic_timing import SyntheticVectorAddTiming
            timing = self.config.get('timing_adapter')
            synthetic = self.config.get('synthetic_vector_add_timing', False)
            if timing is not None and synthetic:
                raise ValueError('Caller-provided timing_adapter is not allowed with synthetic timing')
            if timing is not None:
                raise ValueError('Untrusted external timing_adapter cannot establish validated speedup')
            if synthetic:
                timing = SyntheticVectorAddTiming(
                    source_path,
                    self.config['synthetic_cpu_source'],
                    self.workdir / 'instrumented',
                    self.config.get('timeout', 300),
                )
                timing.validate_sources()
            proxy = self.config.get('allow_proxy_timing', False)
            result.metrics.update({'timing_protocol': 'unverified', 'speedup': None})
            cuda_bin = self._compile_cuda_release(source_path, task_id)
            if cuda_bin is None:
                result.errors.append({'message': 'L4: CUDA release compilation failed'})
                return result
            c_bin = self._compile_c_baseline(task_type, task_id)
            if c_bin is None:
                result.errors.append({'message': 'L4: CPU baseline unavailable; speedup cannot be established'})
                result.metrics.update({'cpu_ms': None, 'baseline_available': False})
                return result
            if timing is None and not proxy:
                result.errors.append({'message': 'L4: instrumented H2D+kernel+D2H cudaEvent and comparable CPU timing unavailable'})
                return result
            if timing is not None:
                report = timing(cuda_bin, c_bin, self.WARMUP, self.RUNS)
                cuda_ms, cpu_ms = validate_timing_report(report, self.WARMUP, self.RUNS)
                result.metrics['timing_provenance'] = report['provenance']
                cuda_src, cpu_src = 'cudaEvent', 'monotonic_host'
            else:
                cuda_ms, cuda_src = self._time_binary_ms(cuda_bin, kind="cuda")
                cpu_ms,  cpu_src  = self._time_binary_ms(c_bin,    kind="c")
 
            if cuda_ms is None or cpu_ms is None:
                result.passed = False
                result.errors.append({
                    "message": "L4: 计时失败 (CUDA 或 CPU 运行异常)",
                    "cuda_ms": cuda_ms, "cpu_ms": cpu_ms,
                })
                return result
 
            # 来源不一致（一个自报、一个 wall-clock）→ 统一降级到 wall-clock 重测
            if timing is None and cuda_src != cpu_src:
                logger.warning(
                    f"L4: 计时来源不一致 (cuda={cuda_src}, cpu={cpu_src})，"
                    f"统一降级到 wall-clock 重测"
                )
                cuda_ms, cuda_src = self._time_binary_ms(cuda_bin, kind="cuda", force_wall=True)
                cpu_ms,  cpu_src  = self._time_binary_ms(c_bin,    kind="c",    force_wall=True)
 
                if cuda_ms is None or cpu_ms is None:
                    result.passed = False
                    result.errors.append({"message": "L4: wall-clock 重测失败"})
                    return result
 
            if not all(isinstance(value, (int, float)) and 0 < value < float('inf')
                       for value in (cuda_ms, cpu_ms)):
                result.errors.append({'message': 'L4: invalid nonpositive or nonfinite timing'})
                return result
            speedup = cpu_ms / cuda_ms
            result.metrics.update({
                'cuda_ms': cuda_ms, 'cpu_ms': cpu_ms,
                'timing_protocol': 'synthetic_vector_add_only' if timing else 'proxy_unvalidated',
                'baseline_available': True, 'warmup': self.WARMUP, 'runs': self.RUNS,
                'timing_source_cuda': cuda_src, 'timing_source_cpu': cpu_src,
            })
            if timing is None:
                result.metrics['proxy_speedup'] = speedup
                result.errors.append({'message': 'L4: timing protocol not validated; recorded as proxy_speedup'})
                return result
            result.metrics['speedup'] = speedup
            result.passed = speedup > 1.0
            if not result.passed:
                result.errors.append({'message': f'L4: speedup={speedup:.6g} <= 1.0'})
                result.metrics['nsys'] = self._run_nsys(cuda_bin)
                result.raw_log = result.metrics['nsys'].get('raw_stats', '')
                if not result.raw_log:
                    result.errors.append({'message': 'L4: nsys diagnostics unavailable or unvalidated'})

        except Exception as e:
            result.passed = False
            result.metrics.update({'timing_protocol': 'unverified', 'speedup': None})
            result.errors.append({'message': f'L4 timing/diagnostic failure: {e}'})

        return result


def create_layer_evaluator(layer: LayerType, workdir: Path, config: Dict[str, Any]) -> BaseLayerEvaluator:
    """创建层级评估器"""
    evaluators = {
        LayerType.L1: L1Evaluator,
        LayerType.L2: L2Evaluator,
        LayerType.L3: L3Evaluator,
        LayerType.L4: L4Evaluator,
    }
    return evaluators[layer](workdir, config)

