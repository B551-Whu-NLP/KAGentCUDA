# ============================================================
# evaluation/ndcg.py
# 计算 NDCG@K
# 输入: res_rag/xxx-checked.csv
# 运行: python evaluation/ndcg.py
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


NDCG_K_LIST = [1, 3, 5, 10, 20, 50]


# ============================================================
def main():
    args = parse_args()

    DATA_DIR    = Path(args.data_dir)
    RESULT_FILE = get_result_file(
        args.method, args.alpha, args.beta,
        args.obfuscate, args.seed, args.similarity_metric
    )
    
    loader = DataLoader(DATA_DIR)
    if not loader.load_cache():
        loader.load()

    loop_to_project  = {c['id']: c['project'] for c in loader.c_codes}
    all_projects     = set(c['project'] for c in loader.c_codes)
    n_projects       = len(all_projects)

    df = pd.read_csv(RESULT_FILE)

    # Step1: 每个loop的最佳命中rank
    loop_best_rank = {}
    for loop_idx, group in df.groupby('loop_idx'):
        hits = group[group['hit'] == True]
        if len(hits) > 0:
            loop_best_rank[loop_idx] = int(hits['rank'].min())
        else:
            loop_best_rank[loop_idx] = 999999

    # Step2: 每个project取最小rank
    project_best_rank = {}
    for loop_idx, rank in loop_best_rank.items():
        project = loop_to_project.get(loop_idx)
        if project is None:
            continue
        if project not in project_best_rank:
            project_best_rank[project] = rank
        else:
            project_best_rank[project] = min(project_best_rank[project], rank)

    for p in all_projects:
        if p not in project_best_rank:
            project_best_rank[p] = 999999

    ranks = list(project_best_rank.values())
    n     = len(ranks)
    idcg  = 1.0 / np.log2(2)  # 理想排第1的DCG

    # Step3: 计算NDCG@K
    print(f"{'='*50}")
    print(f"NDCG@K  (项目数={n})")
    print(f"{'='*50}")
    for k in NDCG_K_LIST:
        ndcg_scores = []
        for r in ranks:
            if r <= k:
                dcg = 1.0 / np.log2(r + 1)
            else:
                dcg = 0.0
            ndcg_scores.append(dcg / idcg)
        ndcg = np.mean(ndcg_scores)
        print(f"  NDCG@{k:<3}: {ndcg:.4f}")
    print(f"{'='*50}")

if __name__ == '__main__':
    main()