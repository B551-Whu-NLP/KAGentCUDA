# ============================================================
# check_valid_res.py
# 验证检索结果，判断文件级命中
# 输入: res_rag/xxx_output/xxx.csv (loop_idx, kernel_idx, similarity, rank)
# 输出: 带 hit 列的新 CSV，供 evaluation 模块计算指标
# ============================================================

import pandas as pd
from pathlib import Path
import sys
import argparse

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data_module.data_loader import DataLoader


def format_param(p):
    if isinstance(p, float) and p.is_integer():
        return int(p)
    return p


def get_file_paths(method, alpha, beta, obfuscate, seed=42, similarity_metric='cosine'):
    """
    根据方法和参数返回 (result_file, output_file)。
    文件名与各检索脚本的 _save_results 保持一致。
    """
    obf_tag   = '_obf' if obfuscate else ''
    alpha_fmt = format_param(alpha)
    beta_fmt  = format_param(beta)

    if method == 'procrustes':
        base = f"procrustes_paired_seed{seed}_a{alpha_fmt}_b{beta_fmt}{obf_tag}"
        result_file = Path(f'res_rag/procrustes_output/{base}.csv')
        output_file = Path(f'res_rag/procrustes_output/{base}-checked.csv')

    elif method == 'graphcodebert':
        sort_key = 'cosine_similarity' if similarity_metric in ['cosine', 'both'] else 'euclidean_distance'
        base = f"graphcodebert_paired_{sort_key}_seed{seed}{obf_tag}"
        result_file = Path(f'res_rag/graphcodebert_output/{base}.csv')
        output_file = Path(f'res_rag/graphcodebert_output/{base}-checked.csv')

    elif method == 'codebert':
        sort_key = 'cosine_similarity' if similarity_metric in ['cosine', 'both'] else 'euclidean_distance'
        base = f"codebert_paired_{sort_key}_seed{seed}{obf_tag}"
        result_file = Path(f'res_rag/codebert_output/{base}.csv')
        output_file = Path(f'res_rag/codebert_output/{base}-checked.csv')

    elif method == 'tfidf':
        result_file = Path(f'res_rag/tfidf_output/retrieval_results_tfidf{obf_tag}.csv')
        output_file = Path(f'res_rag/tfidf_output/retrieval_results_tfidf{obf_tag}-checked.csv')

    elif method == 'bm25':
        result_file = Path(f'res_rag/bm25_output/retrieval_results_bm25{obf_tag}.csv')
        output_file = Path(f'res_rag/bm25_output/retrieval_results_bm25{obf_tag}-checked.csv')

    elif method == 'ast':
        result_file = Path(f'res_rag/ast_path_output/retrieval_results_ast_path{obf_tag}.csv')
        output_file = Path(f'res_rag/ast_path_output/retrieval_results_ast_path{obf_tag}-checked.csv')

    else:
        raise ValueError(f"Unknown method: {method}")

    return result_file, output_file


def parse_args():
    parser = argparse.ArgumentParser(description='验证检索结果，判断文件级命中')
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


# ============================================================
# 主逻辑
# ============================================================
def main():
    args = parse_args()

    DATA_DIR = Path(args.data_dir)
    RESULT_FILE, OUTPUT_FILE = get_file_paths(
        args.method, args.alpha, args.beta, args.obfuscate, args.seed, args.similarity_metric
    )

    if not RESULT_FILE.exists():
        print(f"❌ 检索结果文件不存在: {RESULT_FILE}")
        sys.exit(1)

    loader = DataLoader(DATA_DIR)
    if not loader.load_cache():
        loader.load()

    loop_to_pair_kernels = {
        c['id']: set(c['pair_kernel_ids'])
        for c in loader.c_codes
    }
    print(f"loop->pair_kernels 映射: {len(loop_to_pair_kernels)} 个")

    df = pd.read_csv(RESULT_FILE)
    print(f"检索结果行数: {len(df)}")

    def is_hit(row):
        pair_kernels = loop_to_pair_kernels.get(int(row['loop_idx']), set())
        return int(row['kernel_idx']) in pair_kernels

    df['hit'] = df.apply(is_hit, axis=1)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"结果已保存: {OUTPUT_FILE}")
    print(f"命中行数: {df['hit'].sum()} / {len(df)}")


if __name__ == '__main__':
    main()