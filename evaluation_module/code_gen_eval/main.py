#!/usr/bin/env python3
"""
统一的命令行入口
"""
import argparse
import sys
import json
from pathlib import Path
from typing import Optional

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import logging
from rich.console import Console
from rich.logging import RichHandler

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("code_gen_eval")
console = Console()


DEFAULT_MODEL = "qwen3-coder-next"
DEFAULT_SEED = 42
CUDA_GEN_ROOT = Path("./cuda_gen_results/cuda_codes")
WORKSPACE_ROOT = Path("./optimization_workspace")


def resolve_source_path(args) -> Path:
    """根据 type/task/model/seed 推导 CUDA 源文件路径；显式 --source 优先。"""
    if getattr(args, "source", None):
        return Path(args.source).resolve()

    candidate = (
        CUDA_GEN_ROOT
        / args.model
        / f"seed{args.seed}"
        / args.type
        / f"{args.type}_{args.task}.cu"
    )
    resolved = candidate.resolve()
    if not resolved.exists():
        raise FileNotFoundError(
            f"自动推导的 CUDA 源文件不存在: {resolved}\n"
            f"请检查 --model / --seed / type / task 是否正确，或用 --source 显式指定。"
        )
    return resolved


def resolve_output_dir(args) -> Optional[Path]:
    """根据 type/task/model/seed 推导输出目录；显式 -o 优先。"""
    if getattr(args, "output_dir", None):
        return Path(args.output_dir).resolve()

    # 默认: optimization_workspace/<model>/seed<seed>/<type>_<task>
    return (WORKSPACE_ROOT / args.model / f"seed{args.seed}" / f"{args.type}_{args.task}").resolve()


def run_full_pipeline(args):
    """运行完整的 L1-L4 优化流水线"""
    from evaluation_module.code_gen_eval.core import FullOptimizationPipeline

    source_path = resolve_source_path(args)
    output_dir = resolve_output_dir(args)

    console.print(f"[bold]源文件:[/bold] {source_path}")
    console.print(f"[bold]输出目录:[/bold] {output_dir}")

    config = {
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "outputs_gold": str(Path(args.outputs_gold).resolve()),
        "timeout": args.timeout
    }

    pipeline = FullOptimizationPipeline(
        config=config,
        workdir_base=Path(args.workdir).resolve() if args.workdir else None
    )

    result = pipeline.run(
        task_type=args.type,
        task_id=args.task,
        source_path=source_path,
        output_dir=output_dir
    )

    return result


def run_single_layer(args):
    """运行单个层级评估"""
    from evaluation_module.code_gen_eval.core import (
        LayerType, create_layer_evaluator
    )

    layer_map = {
        "l1": LayerType.L1,
        "l2": LayerType.L2,
        "l3": LayerType.L3,
        "l4": LayerType.L4
    }

    source_path = resolve_source_path(args)

    config = {
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "outputs_gold": str(Path(args.outputs_gold).resolve()),
        "timeout": args.timeout
    }

    layer = layer_map[args.layer.lower()]
    workdir = Path(args.workdir).resolve() if args.workdir else Path(f"./{args.layer}_workdir")
    evaluator = create_layer_evaluator(layer, workdir, config)

    result = evaluator.evaluate(
        source_path=source_path,
        task_type=args.type,
        task_id=args.task
    )

    console.print(f"\n[bold]{args.layer.upper()} 结果:[/bold]")
    console.print(f"  通过: [green]{'✓' if result.passed else '✗'}[/green]")
    console.print(f"  错误数: {len(result.errors)}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        console.print(f"  结果已保存到: {args.output}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="CodeGenEval - 统一的代码生成评估工具"
    )

    # 子命令
    subparsers = parser.add_subparsers(title="子命令", dest="command")

    # 完整流水线
    pipeline_parser = subparsers.add_parser(
        "pipeline",
        help="运行完整的 L1-L4 优化流水线",
        description=(
            "示例:\n"
            "  python -m evaluation_module.code_gen_eval.main pipeline type-1 c_236\n"
            "  python -m evaluation_module.code_gen_eval.main pipeline type-2 c_119 --model claude --seed 7\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    pipeline_parser.add_argument("type", help="任务类型 (type-1, type-2, ...)")
    pipeline_parser.add_argument("task", help="任务 ID (例如 c_236)")
    pipeline_parser.add_argument("--model", "-m", default=DEFAULT_MODEL,
                                 help=f"生成模型名 (默认 {DEFAULT_MODEL})")
    pipeline_parser.add_argument("--seed", default=DEFAULT_SEED,
                                 help=f"随机种子 (默认 {DEFAULT_SEED})")
    pipeline_parser.add_argument("--source", "-s",
                                 help="显式指定 CUDA 源文件路径 (覆盖自动推导)")
    pipeline_parser.add_argument("--output-dir", "-o",
                                 help="显式指定输出目录 (覆盖自动推导)")
    pipeline_parser.add_argument("--workdir", help="工作目录基础路径")
    pipeline_parser.add_argument("--dataset-root", default="./code_datas", help="数据集根目录")
    pipeline_parser.add_argument("--outputs-gold", default="./outputs_gold_LARGE", help="Gold 输出目录")
    pipeline_parser.add_argument("--timeout", type=int, default=300, help="超时时间 (秒)")

    # 单个层级
    layer_parser = subparsers.add_parser(
        "layer",
        help="运行单个层级评估 (L1/L2/L3/L4)",
        description=(
            "示例:\n"
            "  python -m evaluation_module.code_gen_eval.main layer -l l1 type-1 c_236\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    layer_parser.add_argument("--layer", "-l", required=True, choices=["l1", "l2", "l3", "l4"], help="层级")
    layer_parser.add_argument("type", help="任务类型")
    layer_parser.add_argument("task", help="任务 ID")
    layer_parser.add_argument("--model", "-m", default=DEFAULT_MODEL,
                              help=f"生成模型名 (默认 {DEFAULT_MODEL})")
    layer_parser.add_argument("--seed", default=DEFAULT_SEED,
                              help=f"随机种子 (默认 {DEFAULT_SEED})")
    layer_parser.add_argument("--source", "-s", help="显式指定 CUDA 源文件路径")
    layer_parser.add_argument("--output", "-o", help="结果输出文件 (JSON)")
    layer_parser.add_argument("--workdir", help="工作目录")
    layer_parser.add_argument("--dataset-root", default="./code_datas", help="数据集根目录")
    layer_parser.add_argument("--outputs-gold", default="./outputs_gold_LARGE", help="Gold 输出目录")
    layer_parser.add_argument("--timeout", type=int, default=300, help="超时时间 (秒)")

    args = parser.parse_args()

    if args.command == "pipeline":
        run_full_pipeline(args)
    elif args.command == "layer":
        run_single_layer(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

