#!/usr/bin/env python3
"""
跨语言检索完整流程批处理脚本
支持一键执行：retrieval -> check -> top-k -> ndcg

使用示例：
    # 使用默认参数执行完整流程（同数据集配对检索）
    python run_full_pipeline.py --method procrustes
    
    # 自定义 alpha/beta，保存结果
    python run_full_pipeline.py --method procrustes --alpha 0.3 --beta 0.7 --obfuscate
    
    # 跨数据集检索模式
    python run_full_pipeline.py --method procrustes --task cuda_only
    
    # 只执行部分步骤
    python run_full_pipeline.py --method procrustes --skip-check
    
    # GraphCodeBERT 配对检索
    python run_full_pipeline.py --method graphcodebert --task paired
"""

import argparse
import subprocess
import sys
import json
from datetime import datetime
from pathlib import Path


# 确保工作目录是项目根目录 (脚本所在目录的上一级)
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
if Path.cwd() != PROJECT_ROOT:
    import os
    os.chdir(PROJECT_ROOT)
    sys.path.insert(0, str(PROJECT_ROOT))


# 方法 -> 脚本文件名映射
METHOD_MAP = {
    'procrustes':    'paired4eval_procrustes_retrieval.py',
    'graphcodebert': 'paired4eval_graphcodebert_retrieval.py',
    'codebert':      'paired4eval_codebert_retrieval.py',
    'bm25':         'bm25-based-method.py'
}


def run_command(cmd: str, description: str, dry_run: bool = False) -> bool:
    """执行命令并返回是否成功"""
    print(f"\n{'='*80}")
    print(f"▶️  {description}")
    print(f"$ {cmd}")
    print(f"{'='*80}")

    if dry_run:
        print("⏭️  Dry-run 模式，跳过执行")
        return True

    try:
        result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
        success = result.returncode == 0
        print(f"\n{'✅' if success else '❌'} {description} {'完成' if success else '失败'}")
        return success
    except Exception as e:
        print(f"❌ 执行失败: {e}")
        return False


def build_retrieval_cmd(args) -> str:
    """构建检索命令"""
    filename = METHOD_MAP.get(args.method, f"{args.method}-based-method.py")
    base = f"python retrieval_module/ret_methods/{filename}"
    if args.method == 'bm25':
        import shlex
        return (f'{shlex.quote(sys.executable)} retrieval_module/ret_methods/{filename}'
                f' --parashift_dir {shlex.quote(args.parashift_dir)}'
                f' --output_dir {shlex.quote(args.output_dir)} --seed {args.seed}'
                + (' --obfuscate' if args.obfuscate else '')
                + (f' --loop_review {shlex.quote(args.loop_review)}' if args.loop_review else ''))
    params = []
    # Procrustes 特有参数
    if args.method == 'procrustes':
        if args.alpha is not None:
            params.append(f"--alpha {args.alpha}")
        if args.beta is not None:
            params.append(f"--beta {args.beta}")

    # 通用参数
    if args.obfuscate:
        params.append("--obfuscate")
    if args.seed is not None:
        params.append(f"--seed {args.seed}")
    if args.parashift_dir:
        params.append(f"--parashift_dir {args.parashift_dir}")
    if args.cuda_only_dir:
        params.append(f"--cuda_only_dir {args.cuda_only_dir}")
    if args.output_dir:
        params.append(f"--output_dir {args.output_dir}")

    # cuda_only 任务特有参数
    if args.task == 'cuda_only':
        if args.num_loops is not None:
            params.append(f"--num_loops {args.num_loops}")
        if args.loop_ids_file:
            params.append(f"--loop_ids_file {args.loop_ids_file}")

    return f"{base} {' '.join(params)}"


def build_check_cmd(args) -> str:
    """构建结果检查命令"""
    cmd = f"python evaluation_module/code_ret_eval/check_valid_res.py --method {args.method}"
    if args.alpha is not None and args.beta is not None:
        cmd += f" --alpha {args.alpha} --beta {args.beta}"
    if args.obfuscate:
        cmd += " --obfuscate"
    if args.method == 'bm25':
        import shlex
        cmd += f" --data-dir {shlex.quote(args.parashift_dir)}"
    return cmd


def build_topk_cmd(args) -> str:
    """构建 Top-K 评估命令"""
    cmd = f"python evaluation_module/code_ret_eval/top-k.py --method {args.method}"
    if args.alpha is not None and args.beta is not None:
        cmd += f" --alpha {args.alpha} --beta {args.beta}"
    if args.obfuscate:
        cmd += " --obfuscate"
    if args.method == 'bm25':
        import shlex
        cmd += f" --data-dir {shlex.quote(args.parashift_dir)}"
    return cmd


def build_ndcg_cmd(args) -> str:
    """构建 NDCG 评估命令"""
    cmd = f"python evaluation_module/code_ret_eval/ndcg.py --method {args.method}"
    if args.alpha is not None and args.beta is not None:
        cmd += f" --alpha {args.alpha} --beta {args.beta}"
    if args.obfuscate:
        cmd += " --obfuscate"
    if args.method == 'bm25':
        import shlex
        cmd += f" --data-dir {shlex.quote(args.parashift_dir)}"
    return cmd


def save_results(args, results: dict, output_path: str):
    """保存执行结果"""
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "method": args.method,
            "task": args.task,
            "alpha": args.alpha,
            "beta": args.beta,
            "obfuscate": args.obfuscate,
            "parashift_dir": args.parashift_dir,
        },
        "steps": results
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n📄 执行记录已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='跨语言检索完整流程批处理脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # 方法 & 任务
    parser.add_argument(
        '--method', '-m',
        type=str, default='procrustes',
        choices=['procrustes', 'codebert', 'graphcodebert', 'ast', 'tfidf', 'bm25'],
        help='检索方法 (默认: procrustes)'
    )
    parser.add_argument(
        '--task',
        type=str, default='paired',
        choices=['paired', 'cuda_only'],
        help='检索任务: paired=有答案评估, cuda_only=实际检索 (默认: paired)'
    )

    # Procrustes 权重参数
    parser.add_argument('--alpha', type=float, default=None, help='结构特征权重')
    parser.add_argument('--beta',  type=float, default=None, help='矮人特征权重')

    # 数据目录
    parser.add_argument('--parashift_dir',  type=str, default='./code_datas',  help='配对语料数据目录')
    parser.add_argument('--cuda_only_dir',  type=str, default='./CUDA-only',  help='CUDA-only 数据目录')
    parser.add_argument('--output_dir',     type=str, default='./res_rag',    help='结果输出目录')

    # cuda_only 任务参数
    parser.add_argument('--num_loops',      type=int, default=None, help='随机选取 loop 数（cuda_only）')
    parser.add_argument('--loop_ids_file',  type=str, default=None, help='指定 loop IDs 的 JSON 文件（cuda_only）')

    # 其他
    parser.add_argument('--seed',      type=int,  default=42)
    parser.add_argument('--obfuscate', action='store_true', help='启用代码混淆')
    parser.add_argument('--loop_review', help='Source-bound paired loop review (BM25 only)')

    # 流程控制
    parser.add_argument('--skip-retrieval', action='store_true', help='跳过检索步骤')
    parser.add_argument('--skip-check',     action='store_true', help='跳过结果检查')
    parser.add_argument('--skip-topk',      action='store_true', help='跳过 Top-K 评估')
    parser.add_argument('--skip-ndcg',      action='store_true', help='跳过 NDCG 评估')

    # 输出
    parser.add_argument('--save-results', action='store_true', help='保存执行记录到 JSON')
    parser.add_argument('--output', type=str, default='./logs/pipeline_results.json', help='记录输出路径')
    parser.add_argument('--dry-run', action='store_true', help='只打印命令，不实际执行')

    args = parser.parse_args()
    if args.method == 'bm25':
        if args.task != 'paired':
            parser.error('BM25 basic utility supports only --task paired; use ret_gen_iter.py for standalone retrieval')
        if Path(args.output_dir) != Path('./res_rag'):
            parser.error('BM25 pipeline scorers require --output_dir ./res_rag')
        print('Basic paired BM25 utility run')

    # 打印开始信息
    print("\n" + "🚀" * 40)
    print("🚀 跨语言检索完整流程批处理")
    print("🚀" * 40)
    print(f"\n📋 配置:")
    print(f"   方法         : {args.method}")
    print(f"   任务         : {args.task}")
    if args.alpha is not None:
        print(f"   alpha        : {args.alpha}")
    if args.beta is not None:
        print(f"   beta         : {args.beta}")
    print(f"   混淆         : {'是' if args.obfuscate else '否'}")
    print(f"   配对语料目录: {args.parashift_dir}")
    if args.task == 'cuda_only':
        print(f"   CUDA-only 目录: {args.cuda_only_dir}")

    results = {}
    start_time = datetime.now()

    # 步骤 1: 检索
    if not args.skip_retrieval:
        cmd = build_retrieval_cmd(args)
        results['retrieval'] = run_command(cmd, "1/4 执行检索", dry_run=args.dry_run)

    # 步骤 2: 结果检查
    if not args.skip_check:
        cmd = build_check_cmd(args)
        results['check'] = run_command(cmd, "2/4 检查结果有效性", dry_run=args.dry_run)

    # 步骤 3: Top-K 评估
    if not args.skip_topk:
        cmd = build_topk_cmd(args)
        results['topk'] = run_command(cmd, "3/4 Top-K 准确率评估", dry_run=args.dry_run)

    # 步骤 4: NDCG 评估
    if not args.skip_ndcg:
        cmd = build_ndcg_cmd(args)
        results['ndcg'] = run_command(cmd, "4/4 NDCG 评估", dry_run=args.dry_run)

    # 总结
    elapsed = (datetime.now() - start_time).total_seconds()
    print("\n" + "✅" * 40)
    print("✅ 完整流程执行完成")
    print("✅" * 40)
    print(f"\n⏱️  总耗时: {elapsed:.1f} 秒")
    print(f"\n📊 执行状态:")
    for step, success in results.items():
        status = "✅ 成功" if success else "❌ 失败"
        print(f"   {step:12s}: {status}")

    if args.save_results:
        save_results(args, results, args.output)

    print("\n🎉 全部完成！")

    if results and not all(results.values()):
        sys.exit(1)


if __name__ == '__main__':
    main()