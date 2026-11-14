"""
完整的 L1-L4 优化流水线 - 统一的入口
"""
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import logging

from .layers import (
    LayerType, LayerResult, create_layer_evaluator
)

logger = logging.getLogger(__name__)


class FullOptimizationPipeline:
    """
    完整的 L1-L4 优化流水线

    功能:
    1. 顺序执行 L1-L4 评估
    2. 在失败层级生成优化提示词
    3. 保存完整的评估结果
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        workdir_base: Optional[Path] = None,
        verbose: bool = False,          # 控制是否打印摘要，迭代器调用时传 False
    ):
        self.config = config or self._default_config()
        self.workdir_base = workdir_base or Path("./optimization_workspace")
        self.verbose = verbose

    @staticmethod
    def _default_config() -> Dict[str, Any]:
        return {
            "dataset_root": "./code_datas",
            "outputs_gold": "./outputs_gold_LARGE",
            "timeout": 300,
            "num_runs": 3,
            "warmup_runs": 1
        }

    def run(
        self,
        task_type: str,
        task_id: str,
        source_path: Path,
        output_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        if output_dir is None:
            output_dir = self.workdir_base / f"{task_type}_{task_id}"
        output_dir.mkdir(parents=True, exist_ok=True)

        prepared_source = self._prepare_source(source_path, output_dir, task_type, task_id)
        source_code = prepared_source.read_text(encoding="utf-8", errors="ignore")

        logger.info(f"开始优化流水线: {task_type}/{task_id}")

        results = {
            "task_type": task_type,
            "task_id": task_id,
            "source_path": str(prepared_source),
            "layers": {},
            "first_failed_layer": None,
            "optimization_prompt": None,
            "final_status": "running",
            "start_time": datetime.now().isoformat()
        }

        l1_result = self._evaluate_layer(LayerType.L1, prepared_source, task_type, task_id, output_dir)
        results["layers"]["L1"] = l1_result.to_dict()
        if not l1_result.passed:
            results["first_failed_layer"] = "L1"
            results["optimization_prompt"] = l1_result.build_optimization_prompt(source_code, task_type, task_id)
            results["final_status"] = "failed_L1"
            return self._finalize_results(results, output_dir)

        l2_result = self._evaluate_layer(LayerType.L2, prepared_source, task_type, task_id, output_dir)
        results["layers"]["L2"] = l2_result.to_dict()
        if not l2_result.passed:
            results["first_failed_layer"] = "L2"
            results["optimization_prompt"] = l2_result.build_optimization_prompt(source_code, task_type, task_id)
            results["final_status"] = "failed_L2"
            return self._finalize_results(results, output_dir)

        l3_result = self._evaluate_layer(LayerType.L3, prepared_source, task_type, task_id, output_dir)
        results["layers"]["L3"] = l3_result.to_dict()
        if not l3_result.passed:
            results["first_failed_layer"] = "L3"
            results["optimization_prompt"] = l3_result.build_optimization_prompt(source_code, task_type, task_id)
            results["final_status"] = "failed_L3"
            return self._finalize_results(results, output_dir)

        l4_result = self._evaluate_layer(LayerType.L4, prepared_source, task_type, task_id, output_dir)
        results["layers"]["L4"] = l4_result.to_dict()
        if l4_result.passed:
            results["final_status"] = "passed_all"
        else:
            results["first_failed_layer"] = "L4"
            results["optimization_prompt"] = l4_result.build_optimization_prompt(source_code, task_type, task_id)
            results["final_status"] = "failed_L4"

        return self._finalize_results(results, output_dir)

    def _prepare_source(self, source_path, output_dir, task_type, task_id):
        shutil.copy2(source_path, output_dir / "original.cu")
        task_dir = output_dir / task_type / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        target_path = task_dir / f"{task_id}.cu"
        shutil.copy2(source_path, target_path)
        return target_path

    def _evaluate_layer(self, layer, source_path, task_type, task_id, output_dir):
        layer_workdir = output_dir / layer.value.lower()
        evaluator = create_layer_evaluator(layer, layer_workdir, self.config)
        return evaluator.evaluate(source_path, task_type, task_id)

    def _finalize_results(self, results, output_dir):
        results["end_time"] = datetime.now().isoformat()

        with open(output_dir / "pipeline_result.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        if results.get("optimization_prompt"):
            with open(output_dir / "optimization_prompt.txt", "w", encoding="utf-8") as f:
                f.write(results["optimization_prompt"])

        if self.verbose:
            self._print_summary(results)

        return results

    def _print_summary(self, results):
        print("\n" + "="*60)
        print("最终结果")
        print("="*60)
        for layer_name, layer_result in results.get("layers", {}).items():
            status = "✓ 通过" if layer_result.get("passed", False) else "✗ 失败"
            print(f"  {layer_name}: {status}")
        print(f"\n状态: {results.get('final_status', 'unknown')}")
        if results.get("first_failed_layer"):
            print(f"第一个失败层级: {results['first_failed_layer']}")
        print("="*60)