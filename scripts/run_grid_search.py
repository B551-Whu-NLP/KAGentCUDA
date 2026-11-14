import argparse
import subprocess
import re
import pandas as pd
import sys
from pathlib import Path


def run_command(cmd: str, description: str, capture=False):
    """运行终端命令"""
    print(f"\n{'='*80}")
    print(f"▶️  {description}")
    print(f"$ {cmd}")
    print(f"{'='*80}")
    if capture:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout
    else:
        result = subprocess.run(cmd, shell=True)
        return None

METHOD_MAP = {
    'procrustes':    'paired4eval_procrustes_retrieval.py',
    'graphcodebert': 'paired4eval_graphcodebert_retrieval.py',
    'codebert':      'paired4eval_codebert_retrieval.py'
}

def get_result_filename(alpha, beta, obfuscate, seed=42):
    """根据参数生成结果文件名"""
    obf_tag = "_obf" if obfuscate else ""
    base = f"procrustes_paired_seed{seed}_a{alpha}_b{beta}{obf_tag}"
    result_file = Path(f'res_rag/procrustes_output/{base}.csv')
    return result_file


def main():
    parser = argparse.ArgumentParser(description="网格搜索 - 自动调参")
    parser.add_argument(
        "--method", "-m", type=str, default='procrustes',
        choices=['procrustes', 'codebert', 'graphcodebert'],
        help="检索方法"
    )
    parser.add_argument(
        "--parashift_dir", "-p", type=str, default='./code_datas',
        help="配对语料数据目录"
    )
    parser.add_argument(
        "--skip-retrieve", action="store_true",
        help="跳过检索步骤，直接用已有的 CSV 文件评测 (节省时间)"
    )
    parser.add_argument(
        "--obfuscate", "-o", action="store_true",
        help="是否对代码进行混淆"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="随机种子"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("🚀 开始自动网格搜索 (Grid Search)")
    if args.skip_retrieve:
        print("  ⏭️  模式: 跳过检索，直接使用已有结果评测")
    print("=" * 60)

    # 1. 定义搜索空间：保证 alpha + beta = 1.0，步长 0.1
    alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # 用于收集结果
    leaderboard = []

    for alpha in alphas:
        beta = round(1.0 - alpha, 1)
        print(f"\n[{alpha}/{beta}] ===========================================")

        # 将参数转为字符串，匹配 argparse 格式
        a_str = str(alpha)
        b_str = str(beta)

        # 检查结果文件是否存在
        result_file = get_result_filename(a_str, b_str, args.obfuscate, args.seed)
        file_exists = Path(result_file).exists()

        # 步骤 1: 运行检索生成特征并匹配（如果需要）
        if args.skip_retrieve and file_exists:
            print(f"  ⏭️  跳过检索，结果文件已存在: {result_file}")
        else:
            if args.skip_retrieve and not file_exists:
                print(f"  ⚠️  虽然指定了 --skip-retrieve，但 {result_file} 不存在，需要重新检索")

            filename = METHOD_MAP.get(args.method, f"{args.method}-based-method.py")
            cmd_retrieve = f"python retrieval_module/ret_methods/{filename} --alpha {a_str} --beta {b_str}"
            if args.obfuscate:
                cmd_retrieve += " --obfuscate"
            if args.seed is not None:
                cmd_retrieve += f" --seed {args.seed}"
            if args.parashift_dir:
                cmd_retrieve += f" --parashift_dir {args.parashift_dir}"
            run_command(cmd_retrieve, "执行检索", capture=False)

        # 步骤 2: 检查有效性
        cmd_check = f"python evaluation_module/code_ret_eval/check_valid_res.py --method {args.method} --alpha {a_str} --beta {b_str}"
        if args.obfuscate:
            cmd_check += " --obfuscate"
        run_command(cmd_check, "检查结果有效性", capture=False)

        # 步骤 3: 运行 Top-K 评测并捕获输出
        cmd_topk = f"python evaluation_module/code_ret_eval/top-k.py --method {args.method} --alpha {a_str} --beta {b_str}"
        if args.obfuscate:
            cmd_topk += " --obfuscate"
        topk_output = run_command(cmd_topk, "Top-K 准确率评估", capture=True)

        # 步骤 4: 用正则表达式从输出中"榨取"核心指标
        try:
            top1 = re.search(r'Top-1\s+:\s+([\d\.]+)%', topk_output).group(1)
            top5 = re.search(r'Top-5\s+:\s+([\d\.]+)%', topk_output).group(1)
            top10 = re.search(r'Top-10\s+:\s+([\d\.]+)%', topk_output).group(1)
            top50 = re.search(r'Top-50\s+:\s+([\d\.]+)%', topk_output).group(1)
            mrr = re.search(r'MRR\s+:\s+([\d\.]+)', topk_output).group(1)
            median = re.search(r'中位数排名\s+:\s+([\d\.]+)', topk_output).group(1)

            leaderboard.append({
                "Alpha (AST)": alpha,
                "Beta (Dwarf)": beta,
                "Top-1 (%)": float(top1),
                "Top-5 (%)": float(top5),
                "Top-10 (%)": float(top10),
                "Top-50 (%)": float(top50),
                "MRR": float(mrr),
                "Median Rank": float(median)
            })
            print(f"✅ 记录成功: Top-1={top1}%, Top-5={top5}%, Top-50={top50}%")

        except Exception as e:
            print(f"❌ 解析输出失败 (Alpha={alpha}, Beta={beta})! 报错: {e}")
            print(f"原始输出:\n{topk_output}")

    # =======================================================
    # 打印最终排行榜
    # =======================================================
    print("\n\n" + "=" * 80)
    print("🏆 网格搜索最终排行榜 (按 Top-1 降序排列)")
    print("=" * 80)

    if leaderboard:
        df = pd.DataFrame(leaderboard)
        # 按 Top-1 命中率从高到低排序
        df = df.sort_values(by="Top-1 (%)", ascending=False).reset_index(drop=True)
        print(df.to_string(index=True))

        # 保存为 CSV 方便后续做图表
        df.to_csv("grid_search_results.csv", index=False)
        print("\n💾 完整结果已保存至: grid_search_results.csv")
    else:
        print("没有收集到有效数据。")


if __name__ == "__main__":
    main()
