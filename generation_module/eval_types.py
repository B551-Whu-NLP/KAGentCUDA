"""
评估结果数据类 + 迭代优化器（封装 L1-L4 评估调用）。
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class EvaluationResult:
    """评估结果"""
    layer: str
    passed: bool
    passed_count: int = 0
    total_count: int = 4
    errors: List[str] = field(default_factory=list)
    metrics: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'layer': self.layer,
            'passed': self.passed,
            'passed_count': self.passed_count,
            'total_count': self.total_count,
            'errors': self.errors,
            'metrics': self.metrics
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'EvaluationResult':
        return cls(
            layer=data.get('layer', 'L0'),
            passed=data.get('passed', False),
            passed_count=data.get('passed_count', 0),
            total_count=data.get('total_count', 4),
            errors=data.get('errors', []),
            metrics=data.get('metrics', {})
        )


class IterativeOptimizer:
    """迭代优化器 - 集成评估和代码优化"""

    def __init__(self, max_iterations: int = 3):
        self.max_iterations = max_iterations

    def evaluate_layers(
        self,
        source_path: Path,
        task_type: str,
        task_id: str,
        model_name: str = "default",
        seed: int = 42,
        workspace_root: Path = Path("./optimization_workspace"),
        iteration: int = 1,
    ) -> EvaluationResult:
        """运行 L1-L4 评估"""
        try:
            from evaluation_module.code_gen_eval.core import FullOptimizationPipeline

            config = {
                "dataset_root": "./code_datas",
                "outputs_gold": "./outputs_gold_LARGE",
                "dataset_size": "LARGE",
                "timeout": 300,
            }

            # 每次迭代独立 output_dir，保留各次的 pipeline_result.json / optimization_prompt.txt
            # 结构: optimization_workspace/<model>/seed<seed>/<type>_<task_id>/iter<N>/
            output_dir = (
                workspace_root
                / model_name
                / f"seed{seed}"
                / f"{task_type}_{task_id}"
                / f"iter{iteration}"
            )

            pipeline = FullOptimizationPipeline(config=config)
            result = pipeline.run(
                task_type=task_type,
                task_id=task_id,
                source_path=source_path,
                output_dir=output_dir,
            )

            # 从 final_status 直接解析通过层数，避免短路执行导致计数偏低
            # final_status: "passed_all" | "failed_L1" | "failed_L2" | "failed_L3" | "failed_L4"
            final_status = result.get("final_status", "")
            if final_status == "passed_all":
                passed_count = 4
            elif final_status.startswith("failed_L"):
                try:
                    passed_count = int(final_status[-1]) - 1
                except ValueError:
                    passed_count = 0
            else:
                passed_count = 0

            # 收集失败层的错误信息，正确提取 dict 里的 message 字段
            errors = []
            layers_data = result.get("layers", {})
            for layer_name in ["L1", "L2", "L3", "L4"]:
                layer_info = layers_data.get(layer_name, {})
                if not layer_info.get("passed", False):
                    layer_errors = layer_info.get("errors", [])
                    if layer_errors:
                        msgs = ", ".join(
                            e.get("message", str(e)) if isinstance(e, dict) else str(e)
                            for e in layer_errors
                        )
                        errors.append(f"{layer_name}: {msgs}")

            return EvaluationResult(
                layer=f"L{passed_count}",
                passed=passed_count == 4,
                passed_count=passed_count,
                total_count=4,
                errors=errors,
                metrics=result,  # 完整 pipeline 结果，含 optimization_prompt
            )

        except Exception as e:
            return EvaluationResult(
                layer="L0",
                passed=False,
                passed_count=0,
                errors=[f"评估异常: {str(e)}"],
            )

    @staticmethod
    def score(result: EvaluationResult) -> float:
        status = result.metrics.get('final_status', '')
        if status == 'passed_all' or status == 'failed_L4':
            speedup = result.metrics.get('layers', {}).get('L4', {}).get('metrics', {}).get('speedup')
            if not isinstance(speedup, (float, int)) or not (0 <= speedup < float('inf')):
                return 3.0 if status == 'failed_L4' else 0.0
            if status == 'passed_all' and speedup <= 1:
                return 3.0 + speedup
            return (4.0 if status == 'passed_all' else 3.0) + speedup
        return float(result.passed_count)

    def is_better(self, new_result: EvaluationResult, old_result: EvaluationResult) -> bool:
        return self.score(new_result) > self.score(old_result)
