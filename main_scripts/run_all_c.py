#!/usr/bin/env python3
"""
批量运行配对语料中所有 C baseline，生成 gold output。

这是项目的第一步：准备数据后先运行此脚本，
生成 outputs_gold_<DATASET>/ 目录，供后续 L3 正确性验证使用。

使用示例：
  # 生成 LARGE 规模的 gold output（推荐，与评估默认规模一致）
  python main_scripts/run_all_c.py --dataset LARGE

  # 只跑 type-1 和 type-3
  python main_scripts/run_all_c.py --dataset LARGE --types type-1 type-3

  # 跳过已生成的任务（断点续跑）
  python main_scripts/run_all_c.py --dataset LARGE --skip-completed
"""

import os
import sys
import json
import shutil
import time
import argparse
from pathlib import Path
from typing import List, Dict

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation_module.code_gen_eval.run_single_task import copy_build_and_run

DATASET_ROOT = PROJECT_ROOT / "code_datas"

try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TaskProgressColumn, TimeRemainingColumn
    from rich.table import Table
    RICH = True
except ImportError:
    RICH = False


# ----------------------------------------------------------
# 数据集扫描
# ----------------------------------------------------------

def scan_tasks(dataset_root: Path, types: List[str]) -> List[Dict]:
    """扫描 code_datas 目录，返回所有 C 任务列表。"""
    tasks = []
    for task_type in types:
        type_dir = dataset_root / task_type
        if not type_dir.exists():
            print(f"警告: {type_dir} 不存在，跳过")
            continue
        for task_dir in sorted(type_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            task_id = task_dir.name
            c_dir = task_dir / "c"
            c_files = list(c_dir.glob("*.c")) + list(c_dir.glob("*.cpp")) if c_dir.exists() else []
            if not c_files:
                continue
            tasks.append({
                "task_type": task_type,
                "task_id":   task_id,
                "c_path":    str(c_files[0]),
            })
    return tasks


# ----------------------------------------------------------
# Gold output 管理
# ----------------------------------------------------------

def gold_output_exists(task_type: str, task_id: str, dataset: str) -> bool:
    """检查该任务的 gold output 是否已存在。"""
    gold_dir = PROJECT_ROOT / f"outputs_gold_{dataset}" / task_type
    if not gold_dir.exists():
        return False
    if task_type == "type-5":
        return any(gold_dir.glob(f"{task_id.lower()}*out.txt"))
    return any(gold_dir.glob(f"*{task_id}*.txt"))


def save_gold_output(workdir: Path, task_type: str, task_id: str, dataset: str) -> int:
    """把任务工作目录里的 txt 输出复制到 gold 目录，返回保存文件数。

    run_single_task 的输出文件落在 workdir/type-X/ 层（与 Makefile 同级），
    文件名格式通常是 <task_id>_output.txt 或 <task_id>_out.txt。
    """
    gold_dir = PROJECT_ROOT / f"outputs_gold_{dataset}" / task_type
    gold_dir.mkdir(parents=True, exist_ok=True)

    # 输出文件在 type 层（Makefile 同级），按 task_id 前缀精确匹配
    type_dir = workdir / task_type
    output_files: List[Path] = []
    for pattern in [f"{task_id}_output.txt", f"{task_id}_out.txt",
                    f"{task_id.lower()}_output.txt", f"{task_id.lower()}_out.txt",
                    f"*{task_id}*.txt"]:
        output_files = list(type_dir.glob(pattern))
        if output_files:
            break

    saved = 0
    for src in output_files:
        dst = gold_dir / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
            saved += 1
    return saved


# ----------------------------------------------------------
# 批量运行
# ----------------------------------------------------------

def run_all(args, tasks: List[Dict], workdir: Path) -> Dict:
    stats = {"total": len(tasks), "success": 0, "skipped": 0,
             "compile_fail": 0, "run_fail": 0, "no_output": 0}
    detail_log = []
    start = time.time()

    def _run_one(task: Dict) -> str:
        """跑单个任务，返回状态字符串。"""
        task_type = task["task_type"]
        task_id   = task["task_id"]

        if args.skip_completed and gold_output_exists(task_type, task_id, args.dataset):
            stats["skipped"] += 1
            return "skipped"

        try:
            result = copy_build_and_run(
                type_name    = task_type,
                task_id      = task_id,
                lang         = "c",
                work_dir     = workdir,
                dataset_size = args.dataset,
                num_runs     = 1,
                warmup       = 0,
                verbose      = True,
            )
        except Exception as e:
            stats["compile_fail"] += 1
            return f"error: {e}"

        status = result.get("status", "")
        if status == "compile_failed":
            stats["compile_fail"] += 1
            return f"compile_fail: {result.get('compile_stderr', '')[-200:]}"
        if status in ("run_failed", "compile_success_no_binary"):
            stats["run_fail"] += 1
            runs = result.get("runs", [])
            err = runs[-1].get("stderr", "")[-200:] if runs else "no runs"
            return f"run_fail: {err}"

        # 保存 gold output（输出文件在 workdir/type-1/ 层）
        saved = save_gold_output(workdir, task_type, task_id, args.dataset)
        if saved == 0:
            stats["no_output"] += 1
            return "no_output"

        stats["success"] += 1
        return "ok"

    if RICH:
        console = Console()
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(), TaskProgressColumn(), TimeRemainingColumn(),
            console=console,
        ) as progress:
            bar = progress.add_task("运行 C baseline...", total=len(tasks))
            for task in tasks:
                desc = f"{task['task_type']}/{task['task_id']}"
                progress.update(bar, advance=1, description=desc)
                status = _run_one(task)
                detail_log.append({"task": desc, "status": status})
    else:
        for i, task in enumerate(tasks, 1):
            desc = f"{task['task_type']}/{task['task_id']}"
            print(f"[{i}/{len(tasks)}] {desc}", end="  ", flush=True)
            status = _run_one(task)
            print(status)
            detail_log.append({"task": desc, "status": status})

    stats["elapsed_s"] = round(time.time() - start, 1)
    return stats, detail_log


# ----------------------------------------------------------
# 打印汇总
# ----------------------------------------------------------

def print_summary(stats: Dict, gold_dir: Path):
    total = stats["total"]
    print("\n" + "=" * 60)
    print("评估完成")
    print("=" * 60)
    rows = [
        ("总任务",     total,                   ""),
        ("✅ 成功",    stats["success"],         f"{stats['success']/total*100:.1f}%"),
        ("⏭ 跳过",    stats["skipped"],         f"{stats['skipped']/total*100:.1f}%"),
        ("❌ 编译失败", stats["compile_fail"],   f"{stats['compile_fail']/total*100:.1f}%"),
        ("❌ 运行失败", stats["run_fail"],       f"{stats['run_fail']/total*100:.1f}%"),
        ("⚠ 无输出",  stats["no_output"],       f"{stats['no_output']/total*100:.1f}%"),
        ("耗时",       f"{stats['elapsed_s']}s", ""),
    ]
    for label, val, pct in rows:
        print(f"  {label:<12} {str(val):<8} {pct}")
    print(f"\nGold output 已保存到: {gold_dir.resolve()}")
    print("=" * 60)


# ----------------------------------------------------------
# 入口
# ----------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="批量生成配对语料 C baseline 的 gold output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--types", nargs="+",
                        default=["type-1", "type-2", "type-3", "type-4", "type-5"])
    parser.add_argument("--dataset", "-d", default="LARGE",
                        choices=["MINI", "SMALL", "MEDIUM", "LARGE", "EXTRALARGE"])
    parser.add_argument("--workdir", default="./eval_workdir")
    parser.add_argument("--skip-completed", action="store_true",
                        help="跳过已有 gold output 的任务")
    args = parser.parse_args()

    workdir  = Path(args.workdir)
    gold_dir = PROJECT_ROOT / f"outputs_gold_{args.dataset}"
    workdir.mkdir(parents=True, exist_ok=True)

    # 扫描任务
    tasks = scan_tasks(DATASET_ROOT, args.types)
    if not tasks:
        print("没有找到任何任务，请检查 code_datas 目录")
        sys.exit(1)

    # 统计
    type_counts = {}
    for t in tasks:
        type_counts[t["task_type"]] = type_counts.get(t["task_type"], 0) + 1

    print("=" * 60)
    print(f"C Baseline — gold output 生成")
    print("=" * 60)
    print(f"数据集规模 : {args.dataset}")
    print(f"工作目录   : {workdir.resolve()}")
    print(f"输出目录   : {gold_dir.resolve()}")
    print(f"断点续跑   : {'开启' if args.skip_completed else '关闭'}")
    print("\n任务分布:")
    for t, c in sorted(type_counts.items()):
        print(f"  {t}: {c}")
    print(f"  总计: {len(tasks)}")
    print("\n按 Enter 开始，Ctrl+C 退出...")
    try:
        input()
    except KeyboardInterrupt:
        print("\n已取消")
        sys.exit(0)

    # 运行
    stats, detail_log = run_all(args, tasks, workdir)

    # 保存日志
    log_path = gold_dir / "run_log.json"
    gold_dir.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({"stats": stats, "details": detail_log}, f, indent=2, ensure_ascii=False)

    print_summary(stats, gold_dir)
    print(f"详细日志: {log_path.resolve()}")


if __name__ == "__main__":
    main()