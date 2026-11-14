# ============================================================
# scripts/crosslang_llm_judge.py
# 跨语言检索结果 LLM 评估: C loop (配对语料) -> CUDA kernel (CUDA-only)
# 输入: crosslang 检索结果 CSV + 完整 C 文件代码 + CUDA kernel代码
# 输出: res_rag/crosslang_codebert_output/xxx-llm-judged.json
# ============================================================
# 配置
# ============================================================
MODEL_NAME   = 'qwen3-coder-next'
ORG          = 'ali'
TOP_K        = 5        # 每个 C loop 取 Top-5 候选
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
from data_module.cuda_only_data_loader import CUDAOnlyDataLoader
from llm_utils import OpenAILLM


SYSTEM_PROMPT = """
你是一位GPU并行计算专家，精通C/C++和CUDA编程。你的任务是评估：给定一个完整的C语言文件代码
（其中包含待评估的C loop），对检索到的Top-5 CUDA Kernel候选进行质量评估。

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
- 先读完整个C文件和所有5个kernel再打分和排序
- 打分要客观且有区分度，不要都给4-5分
- 重点关注：内存访问模式、计算操作类型、循环结构特征
- 忽略变量名，只看计算逻辑
"""


def build_user_prompt(c_file_full_code: str, target_loop_info: dict, kernels: list) -> str:
    """构建用户prompt
    Args:
        c_file_full_code: 完整的C文件代码
        target_loop_info: 目标loop的信息
        kernels: Top-5 kernels列表
    """
    prompt = f"【完整C文件】\n```c\n{c_file_full_code[:5000]}\n```\n\n"
    prompt += f"【目标评估 Loop】（位于 {target_loop_info['filename']}，行 {target_loop_info['start_line']}-{target_loop_info['end_line']}）\n"
    prompt += f"Loop类型: {target_loop_info['loop_type']}\n"
    prompt += f"```c\n{target_loop_info['code'][:1500]}\n```\n\n"
    prompt += "=" * 60 + "\n\n"
    
    for i, k in enumerate(kernels, 1):
        prompt += f"【KERNEL_{i}】（文件:{k['filename']}，函数名:{k['kernel_name']}）\n"
        prompt += f"```cuda\n{k['code'][:2000]}\n```\n\n"
    return prompt


def judge_single_loop(llm, system_prompt, loop_task: dict) -> dict:
    """对单个 C loop 的 Top-5 检索结果进行评估"""
    user_prompt = build_user_prompt(
        loop_task['c_file_full_code'],
        loop_task['loop_info'],
        loop_task['top_k_kernels']
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
    parser = argparse.ArgumentParser(description='跨语言检索结果 LLM 评估')
    parser.add_argument('--input-csv', default=None, 
                        help='检索结果CSV文件路径 (默认: crosslang_cosine_seed42.csv)')
    parser.add_argument('--parashift-dir', default='code_datas',
                        help='配对语料目录 (C loops来源)')
    parser.add_argument('--cuda-only-dir', default='CUDA-only', 
                        help='CUDA-only数据集目录 (kernels来源)')
    parser.add_argument('--seed', type=int, default=42, 
                        help='随机种子 (用于定位结果文件)')
    parser.add_argument('--top-k', type=int, default=5, 
                        help='每个loop取Top-K候选')
    parser.add_argument('--max-loops', type=int, default=None, 
                        help='最多处理N个loop (用于快速测试)')
    return parser.parse_args()


def load_c_file_content(filepath: Path) -> str:
    """加载完整的C文件内容"""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            return f.read()
    except Exception as e:
        print(f"[WARN] 无法读取C文件 {filepath}: {e}")
        return ""


def main():
    args = parse_args()
    
    global TOP_K
    TOP_K = args.top_k
    
    # 默认输入文件
    if args.input_csv is None:
        args.input_csv = f'res_rag/crosslang_codebert_output/crosslang_retrieval_results_cosine_seed{args.seed}.csv'
    
    INPUT_FILE  = Path(args.input_csv)
    OUTPUT_FILE = INPUT_FILE.parent / f'{INPUT_FILE.stem}-llm-judged.json'
    
    print(f"{'='*60}")
    print(f"跨语言检索 LLM Judge")
    print(f"模型: {MODEL_NAME}")
    print(f"输入文件: {INPUT_FILE}")
    print(f"输出文件: {OUTPUT_FILE}")
    print(f"配对语料目录: {args.parashift_dir}")
    print(f"CUDA-only目录: {args.cuda_only_dir}")
    print(f"Top-K: {TOP_K}")
    print(f"{'='*60}")

    # 1. 加载数据
    print("\n[1/5] 加载配对语料 C loops 数据...")
    c_loader = DataLoader(Path(args.parashift_dir), obfuscate_code=False)
    if not c_loader.load_cache():
        c_loader.load()
    print(f"  加载了 {len(c_loader.c_codes)} 个 C loops")
    unique_c_files = sorted(set(c['filename'] for c in c_loader.c_codes))
    print(f"  包含 {len(unique_c_files)} 个 C 文件")
    
    print("\n[2/5] 加载 CUDA-only kernels 数据...")
    k_loader = CUDAOnlyDataLoader(Path(args.cuda_only_dir))
    if not k_loader.load_cache():
        k_loader.load()
    print(f"  加载了 {len(k_loader.cuda_codes)} 个 CUDA kernels")

    # 建立索引
    loop_id_to_loop  = {c['id']: c for c in c_loader.c_codes}
    kernel_id_to_k   = {k['id']: k for k in k_loader.cuda_codes}
    
    # C文件名到文件路径的映射
    filename_to_cfile = {}
    base_dir = Path(c_loader.base_path)
    
    # 配对语料目录结构多样，递归查找所有 C/C++ 文件
    available_files = {}
    for f in base_dir.rglob('*.c'):
        available_files[f.name] = f
    for f in base_dir.rglob('*.cpp'):
        available_files[f.name] = f
    
    # 为每个唯一文件名找对应的文件路径
    for filename in unique_c_files:
        if filename in available_files:
            filename_to_cfile[filename] = available_files[filename]
        else:
            # 尝试匹配后缀
            stem = Path(filename).stem
            for ext in ['.c', '.cpp']:
                candidate = stem + ext
                if candidate in available_files:
                    filename_to_cfile[filename] = available_files[candidate]
                    break
    
    print(f"  找到了 {len(filename_to_cfile)}/{len(unique_c_files)} 个 C 文件的完整路径")

    # 2. 加载检索结果
    print(f"\n[3/5] 加载检索结果...")
    df = pd.read_csv(INPUT_FILE)
    print(f"  检索结果行数: {len(df)}")
    
    unique_loops = sorted(df['loop_idx'].unique())
    print(f"  唯一查询 loops: {len(unique_loops)}")

    # 3. 为每个loop准备任务
    print(f"\n[4/5] 准备评估任务...")
    loop_tasks = []
    
    for loop_id in unique_loops:
        if loop_id not in loop_id_to_loop:
            print(f"  [WARN] loop_id={loop_id} 不在数据中，跳过")
            continue
        
        loop_info = loop_id_to_loop[loop_id]
        
        # 取该loop的Top-K检索结果
        loop_df = df[df['loop_idx'] == loop_id].nsmallest(TOP_K, 'rank')
        
        top_k_kernels = []
        for _, row in loop_df.iterrows():
            kid = int(row['kernel_idx'])
            if kid in kernel_id_to_k:
                k = kernel_id_to_k[kid]
                top_k_kernels.append({
                    'kernel_idx': int(kid),  # 确保是 Python int
                    'rank':       int(row['rank']),
                    'filename':   k['filename'],
                    'kernel_name': k['kernel_name'],
                    'code':       k['code'],
                    'similarity': float(row.get('cosine_similarity', 0)),
                })
        
        if not top_k_kernels:
            continue
            
        # 加载完整的C文件代码
        c_filename = loop_info['filename']
        c_filepath = filename_to_cfile.get(c_filename)
        c_file_full_code = ""
        if c_filepath:
            c_file_full_code = load_c_file_content(c_filepath)
        
        # 如果文件读取失败，用loop所在的代码上下文
        if not c_file_full_code.strip():
            print(f"  [WARN] 无法读取C文件 {c_filename}，使用loop代码替代")
            c_file_full_code = f"// 文件 {c_filename} (无法读取完整文件，仅显示loop)\n" + loop_info['code']

        loop_tasks.append({
            'loop_id':         loop_id,
            'loop_info':       loop_info,
            'c_filename':      c_filename,
            'c_file_full_code': c_file_full_code,
            'top_k_kernels':   top_k_kernels,
        })
        
        if args.max_loops and len(loop_tasks) >= args.max_loops:
            print(f"  达到 --max-loops={args.max_loops} 限制，停止加载")
            break

    print(f"  准备好 {len(loop_tasks)} 个 loop 评估任务")

    # 4. LLM 评估
    print(f"\n[5/5] 开始LLM评估...")
    
    checkpoint_file = Path(OUTPUT_FILE).with_suffix('.checkpoint.json')
    results         = []
    processed       = set()

    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            ckpt = json.load(f)
        results   = ckpt.get('results', [])
        processed = {r['loop_id'] for r in results if r.get('success')}
        print(f"  断点恢复: 已处理 {len(processed)} 个 loops")

    remaining = [t for t in loop_tasks if t['loop_id'] not in processed]
    print(f"  剩余待处理: {len(remaining)} 个 loops\n")

    lock      = threading.Lock()
    save_interval = 5  # 保存间隔小一点，因为loop少

    def convert_numpy_types(obj):
        """递归转换 numpy 类型为 Python 原生类型"""
        if isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(v) for v in obj]
        elif hasattr(obj, 'item'):  # numpy scalar
            return obj.item()
        else:
            return obj

    def process_task(task):
        llm    = get_llm()
        result = judge_single_loop(llm, SYSTEM_PROMPT, task)
        result['loop_id']    = int(task['loop_id'])  # 确保是 Python int
        result['c_filename'] = task['c_filename']
        result['kernel_count'] = int(len(task['top_k_kernels']))
        result['top_k_kernel_ids'] = [
            int(k['kernel_idx']) for k in task['top_k_kernels']
        ]
        # 确保所有数值都是 JSON 可序列化的
        result = convert_numpy_types(result)
        return result

    completed_count = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_task, task): task
            for task in remaining
        }
        with tqdm(total=len(remaining), desc=MODEL_NAME, unit='loop') as pbar:
            for future in as_completed(futures):
                result = future.result()
                with lock:
                    results.append(result)
                    completed_count += 1
                    if completed_count % save_interval == 0:
                        with open(checkpoint_file, 'w') as f:
                            json.dump({'results': convert_numpy_types(results)}, f)
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
    print(f"跨语言检索 LLM Judge 评估结果")
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
        'method':        'crosslang_codebert',
        'model':         MODEL_NAME,
        'seed':          args.seed,
        'timestamp':     datetime.now().isoformat(),
        'total_loops':   len(results),
        'success_count': len(success),
        'fail_count':    len(fail),
        'total_cost':    total_cost,
        'results':       results,
    }

    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(convert_numpy_types(output), f, indent=2, ensure_ascii=False)

    if checkpoint_file.exists():
        checkpoint_file.unlink()

    print(f"\n结果保存到: {OUTPUT_FILE}")


if __name__ == '__main__':
    main()