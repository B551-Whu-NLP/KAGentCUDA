"""
评估结果数据模型 - 结构化存储所有评估指标
"""

from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class CompilationStatus(str, Enum):
    """编译状态"""
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class RunStatus(str, Enum):
    """运行状态"""
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    SEGFAULT = "segfault"
    ILLEGAL_INSTRUCTION = "illegal_instruction"
    SKIPPED = "skipped"


class CorrectnessStatus(str, Enum):
    """正确性状态"""
    EXACT_MATCH = "exact_match"
    WITHIN_TOLERANCE = "within_tolerance"
    INCORRECT = "incorrect"
    SKIPPED = "skipped"


class CompilationResult(BaseModel):
    """编译阶段结果"""
    status: CompilationStatus = CompilationStatus.SKIPPED
    return_code: Optional[int] = None
    build_time_ms: float = 0.0
    binary_size_bytes: int = 0
    warning_count: int = 0
    warning_types: List[str] = Field(default_factory=list)
    error_message: Optional[str] = None
    build_log: Optional[str] = None
    binary_path: Optional[str] = None


class RunResult(BaseModel):
    """运行阶段结果"""
    status: RunStatus = RunStatus.SKIPPED
    return_code: Optional[int] = None
    runtime_ms: List[float] = Field(default_factory=list)  # 多次运行的时间
    runtime_mean_ms: Optional[float] = None
    runtime_median_ms: Optional[float] = None
    runtime_std_ms: Optional[float] = None
    peak_memory_mb: Optional[float] = None
    error_message: Optional[str] = None
    run_log: Optional[str] = None
    num_runs: int = 0


class CorrectnessResult(BaseModel):
    """正确性验证结果"""
    status: CorrectnessStatus = CorrectnessStatus.SKIPPED
    max_absolute_error: Optional[float] = None
    mean_absolute_error: Optional[float] = None
    rmse: Optional[float] = None
    max_relative_error: Optional[float] = None
    high_rel_error_ratio: Optional[float] = None  # 相对误差>1e-5的元素比例
    p99_absolute_error: Optional[float] = None
    worst_index: Optional[int] = None
    worst_actual: Optional[float] = None
    worst_expected: Optional[float] = None
    actual_nonfinite: bool = False
    baseline_nonfinite: bool = False
    comparison_log: Optional[str] = None
    output_path: Optional[str] = None
    baseline_path: Optional[str] = None
    tolerance: float = 1e-6


class PerformanceResult(BaseModel):
    """性能指标结果"""
    speedup_vs_cpu: Optional[float] = None  # 相对于原始C代码
    speedup_vs_reference: Optional[float] = None  # 相对于ground-truth CUDA
    performance_achievement_rate: Optional[float] = None  # 性能达成率 %
    cpu_runtime_ms: Optional[float] = None
    reference_cuda_runtime_ms: Optional[float] = None


class CodeMetrics(BaseModel):
    """代码静态指标"""
    total_lines: int = 0
    code_lines: int = 0  # 去掉空行和注释
    comment_lines: int = 0
    blank_lines: int = 0
    kernel_count: int = 0
    shared_memory_count: int = 0
    has_boundary_check: Optional[bool] = None
    has_error_check: Optional[bool] = None
    block_size: Optional[tuple] = None
    grid_size: Optional[tuple] = None


class TaskEvaluationResult(BaseModel):
    """单个任务的完整评估结果"""
    task_id: str
    task_type: str  # type-1, type-2, etc.
    source_cuda_path: str
    success: bool = False
    
    # 各阶段结果
    compilation: CompilationResult
    run: RunResult
    correctness: CorrectnessResult
    performance: PerformanceResult
    code_metrics: CodeMetrics
    
    # 元信息
    start_time: datetime
    end_time: Optional[datetime] = None
    total_duration_ms: float = 0.0
    evaluation_errors: List[str] = Field(default_factory=list)
    
    # 产出物路径
    artifacts: Dict[str, str] = Field(default_factory=dict)


class SummaryStatistics(BaseModel):
    """整体评估汇总统计"""
    total_tasks: int = 0
    compilation_success_count: int = 0
    compilation_success_rate: float = 0.0
    
    run_success_count: int = 0
    run_success_rate: float = 0.0
    
    correctness_exact_match_count: int = 0
    correctness_within_tolerance_count: int = 0
    correctness_total_pass_count: int = 0
    correctness_rate: float = 0.0
    
    mean_speedup_vs_cpu: float = 0.0
    median_speedup_vs_cpu: float = 0.0
    max_speedup_vs_cpu: float = 0.0
    min_speedup_vs_cpu: float = 0.0
    
    mean_speedup_vs_reference: float = 0.0
    median_speedup_vs_reference: float = 0.0
    
    mean_build_time_ms: float = 0.0
    mean_runtime_ms: float = 0.0
    
    # 失败分类统计
    compilation_failures: Dict[str, int] = Field(default_factory=dict)
    run_failures: Dict[str, int] = Field(default_factory=dict)
    correctness_failures: Dict[str, int] = Field(default_factory=dict)


class EvaluationReport(BaseModel):
    """完整的评估报告"""
    model_name: str
    config: Dict[str, Any]
    summary: SummaryStatistics
    results: List[TaskEvaluationResult]
    start_time: datetime
    end_time: Optional[datetime] = None
    total_duration_ms: float = 0.0
