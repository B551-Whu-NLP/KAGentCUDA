# ============================================================
# evaluation/top_k.py
# 计算文件级 Top-K 准确率
# 输入: res_rag/xxx_output/xxx-checked.csv
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import argparse

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data_module.data_loader import DataLoader


def format_param(p):
    if isinstance(p, float) and p.is_integer():
        return int(p)
    return p


def get_result_file(method, alpha, beta, obfuscate, seed=42, similarity_metric='cosine'):
    """
    根据方法和参数返回 checked CSV 路径。
    文件名与各检索脚本的 _save_results 保持一致。
    """
    obf_tag   = '_obf' if obfuscate else ''
    alpha_fmt = format_param(alpha)
    beta_fmt  = format_param(beta)

    if method == 'procrustes':
        base = f"procrustes_paired_seed{seed}_a{alpha_fmt}_b{beta_fmt}{obf_tag}"
        return Path(f'res_rag/procrustes_output/{base}-checked.csv')

    elif method == 'graphcodebert':
        sort_key = 'cosine_similarity' if similarity_metric in ['cosine', 'both'] else 'euclidean_distance'
        base = f"graphcodebert_paired_{sort_key}_seed{seed}{obf_tag}"
        return Path(f'res_rag/graphcodebert_output/{base}-checked.csv')

    elif method == 'codebert':
        sort_key = 'cosine_similarity' if similarity_metric in ['cosine', 'both'] else 'euclidean_distance'
        base = f"codebert_paired_{sort_key}_seed{seed}{obf_tag}"
        return Path(f'res_rag/codebert_output/{base}-checked.csv')

    elif method == 'tfidf':
        return Path(f'res_rag/tfidf_output/retrieval_results_tfidf{obf_tag}-checked.csv')

    elif method == 'bm25':
        return Path(f'res_rag/bm25_output/retrieval_results_bm25{obf_tag}-checked.csv')

    elif method == 'ast':
        return Path(f'res_rag/ast_path_output/retrieval_results_ast_path{obf_tag}-checked.csv')

    else:
        raise ValueError(f"Unknown method: {method}")


def parse_args():
    parser = argparse.ArgumentParser(description='计算文件级 Top-K 准确率')
    parser.add_argument('--method', required=True,
                        choices=['procrustes', 'codebert', 'tfidf', 'bm25', 'ast', 'graphcodebert'],
                        help='检索方法')
    parser.add_argument('--data-dir', default='code_datas', help='数据目录')
    parser.add_argument('--obfuscate', action='store_true', help='是否使用混淆数据')
    parser.add_argument('--alpha', type=float, default=0.3, help='结构特征权重（procrustes）')
    parser.add_argument('--beta',  type=float, default=0.7, help='矮人特征权重（procrustes）')
    parser.add_argument('--seed',  type=int,   default=42,  help='随机种子')
    parser.add_argument('--similarity_metric', default='cosine',
                        choices=['cosine', 'euclidean', 'both'],
                        help='相似度类型（codebert/graphcodebert）')
    return parser.parse_args()


TOP_K_LIST = [1, 3, 5, 10, 20, 50]


# ============================================================
def main():
    args = parse_args()

    DATA_DIR    = Path(args.data_dir)
    RESULT_FILE = get_result_file(
        args.method, args.alpha, args.beta,
        args.obfuscate, args.seed, args.similarity_metric
    )

    if not RESULT_FILE.exists():
        print(f"❌ 结果文件不存在: {RESULT_FILE}")
        sys.exit(1)

    # 加载 id -> filename 映射
    loader = DataLoader(DATA_DIR)
    if not loader.load_cache():
        loader.load()

    loop_to_file = {c['id']: c['filename'] for c in loader.c_codes}
    all_c_files  = set(c['filename'] for c in loader.c_codes)
    n_projects   = len(all_c_files)

    # 加载 checked 结果
    df = pd.read_csv(RESULT_FILE)

    # Step1: 每个 loop 的最佳命中 rank
    loop_best_rank = {}
    for loop_idx, group in df.groupby('loop_idx'):
        hits = group[group['hit'] == True]
        loop_best_rank[loop_idx] = int(hits['rank'].min()) if len(hits) > 0 else 999999

    # Step2: 每个文件的最佳 rank（取所有 loop 里最小的）
    file_best_rank = {}
    for loop_idx, rank in loop_best_rank.items():
        filename = loop_to_file.get(loop_idx)
        if filename is None:
            continue
        if filename not in file_best_rank:
            file_best_rank[filename] = rank
        else:
            file_best_rank[filename] = min(file_best_rank[filename], rank)

    for f in all_c_files:
        if f not in file_best_rank:
            file_best_rank[f] = 999999

    ranks = list(file_best_rank.values())
    n     = len(ranks)

    # Step3: 计算 Top-K
    print(f"{'='*50}")
    print(f"文件级 Top-K 准确率  (文件数={n})")
    print(f"方法: {args.method}  seed={args.seed}"
          + (f"  a={args.alpha} b={args.beta}" if args.method == 'procrustes' else "")
          + (f"  obfuscate" if args.obfuscate else ""))
    print(f"{'='*50}")
    for k in TOP_K_LIST:
        hits = sum(1 for r in ranks if r <= k)
        print(f"  Top-{k:<3}: {hits/n*100:.2f}%  ({hits}/{n})")

    mrr = np.mean([1.0 / r for r in ranks])
    print(f"\n  MRR       : {mrr:.4f}")
    print(f"  平均排名   : {np.mean(ranks):.1f}")
    print(f"  中位数排名 : {np.median(ranks):.1f}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()