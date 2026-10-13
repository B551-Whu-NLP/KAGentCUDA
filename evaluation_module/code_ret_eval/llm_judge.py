# ============================================================
# evaluation/llm_judge.py
# 用LLM对检索结果进行质量评估
# 输入: res_rag/xxx-checked.csv + C文件代码 + CUDA kernel代码
# 输出: res_rag/xxx-llm-judged.json
# ============================================================
# 配置
# ============================================================
MODEL_NAME   = 'qwen3-coder-next'
ORG          = 'ali'
TOP_K        = 20       # 每个C文件取Top-20候选
MAX_WORKERS  = 8
# ============================================================

import json
import sys
import threading
import argparse
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from tqdm import tqdm
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from data_module.data_loader import DataLoader
from llm_utils import OpenAILLM


def format_param(p):
    if isinstance(p, float) and p.is_integer():
        return int(p)
    return p


def get_result_file(method, alpha, beta, obfuscate):
    obf_tag = '_obf' if obfuscate else ''
    alpha_fmt = format_param(alpha)
    beta_fmt = format_param(beta)
    if method == 'codebert':
        return Path(f'res_rag/codebert_output/retrieval_results_both{obf_tag}-checked.csv')
    elif method == 'manual':
        return Path(f'res_rag/procrustes_output/procrustes-v1{obf_tag}_a{alpha_fmt}_b{beta_fmt}-checked.csv')
    elif method == 'tfidf':
        return Path(f'res_rag/tfidf_output/retrieval_results_tfidf{obf_tag}-checked.csv')
    elif method == 'bm25':
        return Path(f'res_rag/bm25_output/retrieval_results_bm25{obf_tag}-checked.csv')
    else:
        raise ValueError(f"Unknown method: {method}")


SYSTEM_PROMPT = """
你是一位GPU并行计算专家，精通C/C++和CUDA编程。你的任务是评估：给定一个C语言循环代码，
对检索到的Top-5 CUDA Kernel候选进行质量评估。

【评估维度（每个维度 1-5 分，5分最好）】：
1. 🔢 计算相关性 (compute_correlation)
   - 1分：计算模式完全不同（比如C是循环累加，kernel是矩阵乘法）
   - 3分：有部分重叠的计算模式，但核心操作不同
   - 5分：计算模式高度匹配（都是规约、都是Stencil、都是矩阵乘法等）
2. 🚀 并行参考价值 (parallel_value)
   - 1分：完全无法提供任何并行化参考
   - 3分：有部分可借鉴的并行思路，但需要大量改写
   - 5分：可以直接参考或少量修改就能作为该C代码的并行实现
3. 📝 语义相似性 (semantic_similarity)
   - 1分：做的是完全不同的事情
   - 3分：解决类似问题但实现方式差异大
   - 5分：从语义上看是同一个算法/同一类问题

【输出要求】
严格输出JSON格式，不要额外解释：
{
  "overall_rank": [1, 2, 3, 4, 5],
  "scores": {
    "kernel_1": {"compute_correlation": X, "parallel_value": X, "semantic_similarity": X, "reason": "简短理由"},
    "kernel_2": {...},
    ...
  }
}

【注意】
- 先读完所有5个kernel再打分和排序
- 打分要客观且有区分度，不要都给4-5分
- 重点关注：内存访问模式、计算操作类型、循环结构特征
- 忽略变量名，只看计算逻辑
"""


def build_user_prompt(c_code: str, kernels: list) -> str:
    """构建用户prompt"""
    prompt = f"【C_LOOP】\n```c\n{c_code[:3000]}\n```\n\n"
    for i, k in enumerate(kernels, 1):
        prompt += f"【KERNEL_{i}】（来自{k['filename']}，函数名:{k['kernel_name']}）\n"
        prompt += f"```cuda\n{k['code'][:2000]}\n```\n\n"
    return prompt


def judge_single_file(llm, system_prompt, c_file_info: dict) -> dict:
    """对单个C文件的检索结果进行评估"""
    user_prompt = build_user_prompt(
        c_file_info['c_code'],
        c_file_info['top_k_kernels']
    )
    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user',   'content': user_prompt},
    ]
    try:
        response, input_tokens, output_tokens, cost = llm.chat(
            messages, stream=False
        )
        response = response.strip()
        if response.startswith('```json'):
            response = response[7:].rstrip('`').strip()
        elif response.startswith('```'):
            response = response[3:].rstrip('`').strip()

        result = json.loads(response)
        result['success']       = True
        result['input_tokens']  = input_tokens
        result['output_tokens'] = output_tokens
        result['cost']          = cost
        return result

    except Exception as e:
        return {'success': False, 'error': str(e)}


# ----------------------------------------------------------------
# 线程本地LLM
# ----------------------------------------------------------------
_local = threading.local()

def get_llm():
    if not hasattr(_local, 'llm'):
        _local.llm = OpenAILLM(
            model_name=MODEL_NAME,
            org=ORG,
            log_path='./logs',
            verbose=False,
            temperature=0.1,
            top_p=0.95,
        )
    return _local.llm


def parse_args():
    parser = argparse.ArgumentParser(description='LLM评估检索结果质量')
    parser.add_argument('--method', required=True, choices=['codebert', 'manual', 'tfidf', 'bm25'],
                        help='检索方法: codebert, manual, tfidf, bm25')
    parser.add_argument('--data-dir', default='code_datas', help='数据目录')
    parser.add_argument('--obfuscate', action='store_true', help='是否使用混淆数据')
    parser.add_argument('--alpha', type=float, default=0, help='manual方法的结构特征权重')
    parser.add_argument('--beta', type=float, default=1, help='manual方法的矮人特征权重')
    parser.add_argument('--top-k', type=int, default=5, help='每个文件取Top-K候选')
    return parser.parse_args()


def main():
    args = parse_args()
    
    global TOP_K
    TOP_K = args.top_k
    
    RESULT_FILE = get_result_file(args.method, args.alpha, args.beta, args.obfuscate)
    
    method_tag = f"{args.method}"
    if args.method == 'manual':
        method_tag += f"_a{args.alpha}_b{args.beta}"
    if args.obfuscate:
        method_tag += '_obf'
    
    OUTPUT_FILE = RESULT_FILE.parent / f'{RESULT_FILE.stem.replace("-checked", "")}-llm-judged.json'
    
    print(f"{'='*60}")
    print(f"方法: {method_tag}")
    print(f"输入文件: {RESULT_FILE}")
    print(f"输出文件: {OUTPUT_FILE}")
    print(f"Top-K: {TOP_K}")
    print(f"{'='*60}")

    # 1. 加载数据
    loader = DataLoader(Path(args.data_dir), obfuscate_code=args.obfuscate)
    if not loader.load_cache():
        loader.load()

    # 建立索引
    id_to_c    = {c['id']: c for c in loader.c_codes}
    id_to_k    = {k['id']: k for k in loader.cuda_codes}
    file_to_loops = defaultdict(list)
    for c in loader.c_codes:
        file_to_loops[c['filename']].append(c)

    # 2. 加载检索结果
    df = pd.read_csv(RESULT_FILE)
    print(f"检索结果行数: {len(df)}")

    # 3. 对每个C文件，取所有loop的最高ranked Top-K kernel
    #    以文件为粒度，取文件内所有loop的前TOP_K个不重复kernel
    all_c_files = sorted(set(c['filename'] for c in loader.c_codes))

    file_tasks = []
    for filename in all_c_files:
        loops = file_to_loops[filename]
        loop_ids = [l['id'] for l in loops]

        # 取这些loop的所有检索结果，按rank排序，去重取Top-K
        file_df = df[df['loop_idx'].isin(loop_ids)].copy()
        if file_df.empty:
            continue

        # 文件级别：对每个kernel取最好的rank（跨loop）
        top_k_per_loop = set()
        for loop_id in loop_ids:
            loop_df = file_df[file_df['loop_idx'] == loop_id]
            top_kids = loop_df.nsmallest(TOP_K, 'rank')['kernel_idx'].tolist()
            top_k_per_loop.update(top_kids)

        # 对并集里的kernel，取跨loop最小rank，重新排序
        best_rank = file_df[file_df['kernel_idx'].isin(top_k_per_loop)]\
            .groupby('kernel_idx')['rank'].min().reset_index()\
            .sort_values('rank').head(TOP_K)

        top_k_kernels = []
        for _, row in best_rank.iterrows():
            kid = int(row['kernel_idx'])
            if kid in id_to_k:
                k = id_to_k[kid]
                top_k_kernels.append({
                    'kernel_idx': kid,
                    'rank':       int(row['rank']),
                    'filename':   k['filename'],
                    'kernel_name': k['kernel_name'],
                    'code':       k['code'],
                    'is_correct': bool(file_df[
                        (file_df['kernel_idx'] == kid) &
                        (file_df['hit'] == True)
                    ].any().any()) if 'hit' in file_df.columns else False,
                })

        if not top_k_kernels:
            continue

        # C文件代码：拼接所有loop的代码
        c_code = '\n\n'.join(
            f"// loop_{i+1} (line {l['start_line']}-{l['end_line']})\n{l['code']}"
            for i, l in enumerate(loops)
        )

        file_tasks.append({
            'filename':      filename,
            'c_code':        c_code,
            'top_k_kernels': top_k_kernels,
            'has_hit':       any(k['is_correct'] for k in top_k_kernels),
        })

    print(f"待评估C文件数: {len(file_tasks)}")
    print(f"其中有精确命中的: {sum(1 for t in file_tasks if t['has_hit'])}")

    # 4. 断点
    checkpoint_file = Path(OUTPUT_FILE).with_suffix('.checkpoint.json')
    results         = []
    processed       = set()

    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            ckpt = json.load(f)
        results   = ckpt.get('results', [])
        processed = {r['filename'] for r in results if r.get('success')}
        print(f"断点恢复: 已处理 {len(processed)} 个文件")

    remaining = [t for t in file_tasks if t['filename'] not in processed]
    print(f"剩余待处理: {len(remaining)} 个\n")

    lock      = threading.Lock()
    save_interval = 20

    def process_task(task):
        llm    = get_llm()
        result = judge_single_file(llm, SYSTEM_PROMPT, task)
        result['filename'] = task['filename']
        result['has_hit']  = task['has_hit']
        result['top_k_kernel_ids'] = [
            k['kernel_idx'] for k in task['top_k_kernels']
        ]
        result['correct_kernel_rank'] = next(
            (i+1 for i, k in enumerate(task['top_k_kernels']) if k['is_correct']),
            None
        )
        return result

    completed_count = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_task, task): task
            for task in remaining
        }
        with tqdm(total=len(remaining), desc=MODEL_NAME, unit='file') as pbar:
            for future in as_completed(futures):
                result = future.result()
                with lock:
                    results.append(result)
                    completed_count += 1
                    if completed_count % save_interval == 0:
                        with open(checkpoint_file, 'w') as f:
                            json.dump({'results': results}, f)
                pbar.update(1)

    # 5. 统计并保存
    success = [r for r in results if r.get('success')]
    fail    = [r for r in results if not r.get('success')]
    total_cost = sum(r.get('cost', 0) for r in success)

    # 统计平均分
    scores_agg = defaultdict(lambda: defaultdict(list))
    for r in success:
        for kname, kscores in r.get('scores', {}).items():
            for metric in ['compute_correlation', 'parallel_value', 'semantic_similarity']:
                v = kscores.get(metric)
                if v is not None:
                    scores_agg['all'][metric].append(v)
                    # Top-1单独统计
                    rank = r.get('overall_rank', [])
                    if rank and kname == f"kernel_{rank[0]}":
                        scores_agg['top1'][metric].append(v)

    print(f"\n{'='*55}")
    print(f"LLM Judge 评估结果  方法: {method_tag}")
    print(f"{'='*55}")
    print(f"成功: {len(success)}/{len(results)}")
    print(f"费用: ${total_cost:.4f}")
    print(f"\n平均分（所有Top-{TOP_K}候选）:")
    for metric in ['compute_correlation', 'parallel_value', 'semantic_similarity']:
        vals = scores_agg['all'][metric]
        if vals:
            print(f"  {metric}: {sum(vals)/len(vals):.3f}")
    print(f"\n平均分（LLM认为的Top-1）:")
    for metric in ['compute_correlation', 'parallel_value', 'semantic_similarity']:
        vals = scores_agg['top1'][metric]
        if vals:
            print(f"  {metric}: {sum(vals)/len(vals):.3f}")
    print(f"{'='*55}")

    output = {
        'method':        method_tag,
        'model':         MODEL_NAME,
        'timestamp':     datetime.now().isoformat(),
        'total_files':   len(results),
        'success_count': len(success),
        'fail_count':    len(fail),
        'total_cost':    total_cost,
        'results':       results,
    }

    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    if checkpoint_file.exists():
        checkpoint_file.unlink()

    print(f"\n结果保存到: {OUTPUT_FILE}")


if __name__ == '__main__':
    main()