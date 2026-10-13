#!/usr/bin/env python3
"""
单任务编译运行脚本

功能：
  1. 编译运行指定 type 和 task_id 的 C 和 CUDA 代码
  2. 输出运行日志和结果文件
  3. 支持只运行 C 或 CUDA

使用示例：
  python run_single_task.py --type type-3 --task 2mm --cuda      # 只运行 CUDA
  python run_single_task.py --type type-5 --task BT --c         # 只运行 C
  python run_single_task.py --type type-3 --task 2mm --both    # 同时运行 C 和 CUDA
"""

import argparse
import sys
import os
from pathlib import Path
import shutil
import subprocess
import json

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# 数据集路径（优先使用项目根目录下的 code_datas）
DATASET_ROOT = Path(__file__).parent.parent.parent / "code_datas"


def run_command(cmd, cwd=None, timeout=300):
    """运行命令并返回结果"""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=True,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Timeout after {} seconds".format(timeout)


def find_source_file(type_path: Path, task_id: str, lang: str) -> Path:
    """查找源文件路径"""
    task_dir = type_path / task_id

    if not task_dir.exists():
        raise FileNotFoundError(f"Task directory not found: {task_dir}")

    lang_dir = task_dir / lang

    if lang_dir.exists():
        # 查找 .c, .cpp, .cu 文件
        exts = ["*.c", "*.cpp", "*.cu"] if lang == "c" else ["*.cu"]
        for ext in exts:
            files = list(lang_dir.glob(ext))
            if files:
                return files[0]

    # 如果没有 lang 子目录，直接在 task_dir 查找
    exts = ["*.c", "*.cpp"] if lang == "c" else ["*.cu"]
    for ext in exts:
        files = list(task_dir.glob(ext))
        if files:
            return files[0]

    raise FileNotFoundError(f"No {lang} source file found for {task_id} in {task_dir}")


def copy_build_and_run(
    type_name: str,
    task_id: str,
    lang: str,
    work_dir: Path,
    dataset_size: str = "SMALL",
    num_runs: int = 3,
    warmup: int = 1,
    verbose: bool = False,
    info_log: bool = False,
):
    """
    编译并运行单个任务

    Args:
        type_name: 类型名称 (type-1, type-3, type-5)
        task_id: 任务 ID (如 2mm, BT, etc.)
        lang: 'c' 或 'cuda'
        work_dir: 工作目录
        dataset_size: 数据集大小 (MINI, SMALL, MEDIUM, LARGE)
        num_runs: 运行次数
        warmup: 预热次数
        verbose: 是否输出结果文件
        info_log: 是否打印信息日志

    Returns:
        运行结果字典
    """
    # lang 用于目录查找（c/ cuda/），plat 用于传给 Makefile（cpu cuda）
    plat = "cpu" if lang == "c" else lang   # ← 加这一行

    type_path = DATASET_ROOT / type_name

    if not type_path.exists():
        raise FileNotFoundError(f"Type directory not found: {type_path}")

    # 查找源文件
    source_file = find_source_file(type_path, task_id, lang)
    if info_log:
        print(f"[INFO] Found source file: {source_file}")

    # 创建工作目录
    task_work_dir = work_dir / type_name / task_id
    task_work_dir.mkdir(parents=True, exist_ok=True)

    # 复制 Makefile 和 run.sh (从 type 目录)
    for f in ["Makefile", "run.sh"]:
        src = type_path / f
        if src.exists():
            dst = task_work_dir.parent / f  # 放在 type 目录下
            if not dst.exists():
                shutil.copy2(src, dst)

    # 复制 libs (如果不存在)
    libs_dst = work_dir / "libs"
    if not libs_dst.exists():
        libs_src = DATASET_ROOT / "libs"
        if libs_src.exists():
            shutil.copytree(libs_src, libs_dst)

    # 复制源文件
    ext = source_file.suffix
    dst_source = task_work_dir / f"{task_id}{ext}"
    shutil.copy2(source_file, dst_source)

    # 关键修复：同时在 Makefile 同级目录放一份源文件
    type_dir = task_work_dir.parent
    dst_source_type = type_dir / f"{task_id}{ext}"
    if not dst_source_type.exists():
        shutil.copy2(source_file, dst_source_type)
    else:
        # 如果文件存在，只有内容不同时才复制
        import filecmp
        if not filecmp.cmp(source_file, dst_source_type):
            shutil.copy2(source_file, dst_source_type)

    # 编译
    if info_log:
        print(f"[INFO] Compiling {lang}...")

    if type_name == "type-5":
        # type-5 (NAS) 使用 Makefile 编译
        makefile_path = task_work_dir.parent / "Makefile"
        if makefile_path.exists():
            # 设置 NPB_DEFFILE 环境变量
            env = os.environ.copy()
            make_def = work_dir / "libs" / "utils" / "make.def"
            if make_def.exists():
                env["NPB_DEFFILE"] = str(make_def)

            compile_cmd = f"make PLAT={plat} DATASET={dataset_size}"
            rc, stdout, stderr = run_command(compile_cmd, cwd=task_work_dir.parent, timeout=300)
        else:
            # 直接使用 nvcc/gcc
            if lang == "cuda":
                compile_cmd = f"nvcc {dst_source} -o {task_work_dir / task_id.lower()}_{plat} -lm"
            else:
                compile_cmd = f"gcc {dst_source} -o {task_work_dir / task_id.lower()}_{plat} -lm"
            rc, stdout, stderr = run_command(compile_cmd, cwd=task_work_dir, timeout=300)
    else:
        # type-1, type-2, type-3, type-4 使用 Makefile
        makefile_path = task_work_dir.parent / "Makefile"
        if makefile_path.exists():
            # 关键修复：临时重命名任务目录，避免和 make 目标名冲突
            import os
            task_dir_temp = task_work_dir.parent / f"_{task_id}_temp"
            dir_was_renamed = False
            if task_work_dir.exists():
                task_work_dir.rename(task_dir_temp)
                dir_was_renamed = True

            try:
                compile_cmd = f"make {task_id} PLAT={plat}"
                rc, stdout, stderr = run_command(compile_cmd, cwd=task_work_dir.parent, timeout=300)
            finally:
                # 恢复目录名
                if dir_was_renamed and task_dir_temp.exists():
                    task_dir_temp.rename(task_work_dir)
        else:
            # 直接使用 nvcc/gcc
            if lang == "cuda":
                compile_cmd = f"nvcc {dst_source} -o {task_work_dir / f'{task_id}_{lang}'} -lm"
            else:
                compile_cmd = f"gcc {dst_source} -o {task_work_dir / f'{task_id}_{lang}'} -lm"
            rc, stdout, stderr = run_command(compile_cmd, cwd=task_work_dir, timeout=300)

    if rc != 0:
        print(f"[ERROR] Compilation failed!")
        if verbose:
            print(stdout)
            print(stderr)
        return {
            "status": "compile_failed",
            "compile_returncode": rc,
            "compile_stdout": stdout,
            "compile_stderr": stderr,
        }

    if info_log:
        print("[INFO] Compilation successful!")

    # 查找二进制文件
    bin_files = []
    bin_dir = task_work_dir.parent / "bin"
    if bin_dir.exists():
        bin_files = list(bin_dir.glob(f"{task_id.lower()}*"))
        if not bin_files:
            bin_files = list(bin_dir.glob(f"*{lang}*"))

    if not bin_files:
        # 尝试在任务目录查找
        bin_files = list(task_work_dir.glob(f"{task_id}*"))
        if not bin_files:
            bin_files = list(task_work_dir.glob(f"*{lang}*"))

    if not bin_files:
        print("[WARNING] No binary file found, but compilation succeeded")
        return {
            "status": "compile_success_no_binary",
            "compile_returncode": rc,
            "compile_stdout": stdout,
            "compile_stderr": stderr,
        }

    binary = bin_files[0]
    if info_log:
        print(f"[INFO] Binary: {binary}")

    # 运行
    if info_log:
        print(f"[INFO] Running {lang} ({num_runs} runs after {warmup} warmups...")

    run_results = []
    output_files = []

    for i in range(warmup + num_runs):
        is_warmup = i < warmup

        if type_name == "type-5":
            # type-5 使用 run.sh
            run_sh = task_work_dir.parent / "run.sh"
            if run_sh.exists():
                run_cmd = f"bash run.sh -c {dataset_size} -p {plat}"
                # 添加 verbose 参数以生成输出文件
                if verbose:
                    run_cmd += " -v"
                rc, stdout, stderr = run_command(run_cmd, cwd=task_work_dir.parent, timeout=300)
            else:
                rc, stdout, stderr = run_command(f"./{binary.name}", cwd=binary.parent, timeout=300)
        else:
            # 其他 type 使用 run.sh
            run_sh = task_work_dir.parent / "run.sh"
            if run_sh.exists():
                if type_name == "type-3":
                    run_cmd = f"bash run.sh -b {task_id} -p {plat}"
                    # 添加 verbose 参数以生成输出文件
                    if verbose:
                        run_cmd += " -v"
                else:
                    run_cmd = f"bash run.sh {task_id} {plat}"
                    # 添加 verbose 参数以生成输出文件
                    if verbose:
                        run_cmd += " --verbose"
                rc, stdout, stderr = run_command(run_cmd, cwd=task_work_dir.parent, timeout=300)
            else:
                rc, stdout, stderr = run_command(f"./{binary.name}", cwd=binary.parent, timeout=300)

        if not is_warmup:
            run_results.append({
                "run": len(run_results) + 1,
                "returncode": rc,
                "stdout": stdout,
                "stderr": stderr,
            })

            if info_log and rc == 0:
                print(f"  Run {len(run_results)}: SUCCESS")
            elif rc != 0:
                print(f"  Run {len(run_results)}: FAILED (rc={rc})")
                if verbose:
                    print(f"    stdout: {stdout[:200]}...")

    # 查找输出文件
    output_patterns = [
        "*out.txt",
        "*output.txt",
        "*_output.txt",  # 如 c_119_output.txt
        "output_*.txt",
        "*.bin",
        "*.dat",
    ]

    for pattern in output_patterns:
        found = list(task_work_dir.glob(pattern)) + \
                list(task_work_dir.parent.glob(pattern)) + \
                list(work_dir.glob(pattern))
        output_files.extend([str(f) for f in found])

    # 保存结果
    result = {
        "type": type_name,
        "task_id": task_id,
        "lang": lang,
        "status": "success" if all(r["returncode"] == 0 for r in run_results) else "run_failed",
        "source_file": str(source_file),
        "binary": str(binary),
        "work_dir": str(task_work_dir),
        "compile": {
            "returncode": 0,
            "stdout": stdout,
            "stderr": stderr,
        },
        "runs": run_results,
        "output_files": output_files,
    }

    # 保存 JSON 结果
    result_file = task_work_dir / f"result_{lang}.json"
    with open(result_file, "w") as f:
        json.dump(result, f, indent=2)

    if info_log:
        print(f"[INFO] Results saved to: {result_file}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="编译运行单个任务",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--type", "-t",
        required=True,
        choices=["type-1", "type-2", "type-3", "type-4", "type-5"],
        help="任务类型",
    )

    parser.add_argument(
        "--task",
        required=True,
        help="任务 ID (如 2mm, BT, jacobi-1d)",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--c", action="store_true", help="只运行 C")
    group.add_argument("--cuda", action="store_true", help="只运行 CUDA")
    group.add_argument("--both", action="store_true", help="同时运行 C 和 CUDA")

    parser.add_argument(
        "--dataset-size",
        default="LARGE",
        choices=["MINI", "SMALL", "MEDIUM", "LARGE", "EXTRALARGE"],
        help="数据集大小 (默认: SMALL)",
    )

    parser.add_argument(
        "--num-runs",
        type=int,
        default=1,
        help="运行次数 (默认: 1)",
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="预热次数 (默认: 1)",
    )

    parser.add_argument(
        "--work-dir",
        default="./work_single",
        help="工作目录 (默认: ./work_single)",
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="打印详细输出",
    )

    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"任务: {args.type} / {args.task}")
    print(f"模式: {'C + CUDA' if args.both else ('C' if args.c else 'CUDA')}")
    print(f"数据集: {args.dataset_size}")
    print("=" * 70)

    results = {}

    if args.c or args.both:
        print("\n" + "=" * 70)
        print("运行 C 版本")
        print("=" * 70)
        results["c"] = copy_build_and_run(
            type_name=args.type,
            task_id=args.task,
            lang="c",
            work_dir=work_dir,
            dataset_size=args.dataset_size,
            num_runs=args.num_runs,
            warmup=args.warmup,
            verbose=args.verbose,
        )

    if args.cuda or args.both:
        print("\n" + "=" * 70)
        print("运行 CUDA 版本")
        print("=" * 70)
        results["cuda"] = copy_build_and_run(
            type_name=args.type,
            task_id=args.task,
            lang="cuda",
            work_dir=work_dir,
            dataset_size=args.dataset_size,
            num_runs=args.num_runs,
            warmup=args.warmup,
            verbose=args.verbose,
        )

    # 保存汇总结果
    summary_file = work_dir / f"summary_{args.type}_{args.task}.json"
    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("汇总结果")
    print("=" * 70)

    for lang, result in results.items():
        print(f"{lang.upper()}: {result['status']}")
        if result["status"] == "success":
            print(f"  运行次数: {len(result['runs'])}")
            print(f"  输出文件: {len(result['output_files'])} 个")

    print(f"\n汇总结果已保存: {summary_file}")


if __name__ == "__main__":
    main()
