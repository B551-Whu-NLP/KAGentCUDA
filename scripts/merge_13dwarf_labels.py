# ============================================================
# scripts/merge_13dwarf_labels.py
# 合并多个模型的13矮人分类结果
# 输入: <labels_dir>/label_loops 与 <labels_dir>/label_kernels 下的 JSON 文件
# 输出: <labels_dir>/merged_loops.json 与 <labels_dir>/merged_kernels.json
# labels_dir 由下方 LABELS_DIR 配置
# ============================================================

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

# ============================================================
# 配置
# ============================================================
LABELS_DIR  = Path('dwarf_labels')
LOOP_DIR    = LABELS_DIR / 'label_loops'
KERNEL_DIR  = LABELS_DIR / 'label_kernels'
LOOP_OUTPUT = LABELS_DIR / 'merged_loops.json'
KERN_OUTPUT = LABELS_DIR / 'merged_kernels.json'

DWARF_NAMES = [
    'dense_linear_algebra', 'sparse_linear_algebra', 'spectral_methods',
    'n_body_methods', 'structured_grids', 'unstructured_grids',
    'map_reduce', 'combinational_logic', 'graph_traversal',
    'dynamic_programming', 'backtrack_branch_bound',
    'graphical_models', 'finite_state_machines',
]

# ============================================================
def load_results(loop_dir: Path, kernel_dir: Path):
    loop_files   = []
    kernel_files = []

    for f in sorted(loop_dir.glob('*.json')):
        with open(f, 'r', encoding='utf-8') as fp:
            data = json.load(fp)
        loop_files.append(data)
        print(f"Loop文件: {f.name} ({data['total_loops']}个, 模型:{data['model_name']})")

    for f in sorted(kernel_dir.glob('*.json')):
        with open(f, 'r', encoding='utf-8') as fp:
            data = json.load(fp)
        kernel_files.append(data)
        print(f"Kernel文件: {f.name} ({data['total_kernels']}个, 模型:{data['model_name']})")

    return loop_files, kernel_files


def merge_by_confidence_avg(files: list, id_field: str) -> dict:
    """
    按confidence_avg合并多个模型的结果
    返回: {unique_key: {dwarf: avg_score, ...}}
    """
    # 收集每个unique_key的所有模型得分
    all_scores = defaultdict(lambda: defaultdict(list))
    meta       = {}  # unique_key -> 元数据

    for data in files:
        for result in data['results']:
            if not result.get('success', False):
                continue
            key = result['unique_key']
            for dwarf in DWARF_NAMES:
                score = result.get(dwarf, 0.0)
                all_scores[key][dwarf].append(score)
            # 保存元数据（只保存一次）
            if key not in meta:
                meta[key] = {
                    k: result[k] for k in result
                    if k not in DWARF_NAMES + ['matched_patterns', 'pattern_count',
                                                'reasoning', 'input_tokens',
                                                'output_tokens', 'cost']
                }

    # 计算平均值并归一化
    merged = {}
    for key, scores in all_scores.items():
        # 平均置信度
        avg = {d: np.mean(scores[d]) for d in DWARF_NAMES}

        # 归一化：低于0.1的置0，然后L2归一化
        vec = np.array([avg[d] for d in DWARF_NAMES], dtype=np.float32)
        vec[vec < 0.1] = 0.0
        norm = np.linalg.norm(vec)
        if norm > 1e-10:
            vec = vec / norm

        merged[key] = {
            'dwarf_vector': vec.tolist(),
            'dwarf_scores': avg,
            **meta.get(key, {})
        }

    return merged


def main():
    print("加载分类结果...")
    loop_files, kernel_files = load_results(LOOP_DIR, KERNEL_DIR)

    print(f"\nLoop文件数: {len(loop_files)}")
    print(f"Kernel文件数: {len(kernel_files)}")

    # 合并loop结果
    if loop_files:
        print("\n合并Loop结果...")
        merged_loops = merge_by_confidence_avg(loop_files, 'unique_key')
        print(f"合并后Loop数: {len(merged_loops)}")

        with open(LOOP_OUTPUT, 'w', encoding='utf-8') as f:
            json.dump(merged_loops, f, indent=2, ensure_ascii=False)
        print(f"保存到: {LOOP_OUTPUT}")

    # 合并kernel结果
    if kernel_files:
        print("\n合并Kernel结果...")
        merged_kernels = merge_by_confidence_avg(kernel_files, 'unique_key')
        print(f"合并后Kernel数: {len(merged_kernels)}")

        with open(KERN_OUTPUT, 'w', encoding='utf-8') as f:
            json.dump(merged_kernels, f, indent=2, ensure_ascii=False)
        print(f"保存到: {KERN_OUTPUT}")

    # 验证：看一个样例
    if loop_files:
        sample_key = list(merged_loops.keys())[0]
        sample = merged_loops[sample_key]
        print(f"\n样例 ({sample_key}):")
        print(f"  file_id: {sample.get('file_id')}")
        print(f"  loop_index: {sample.get('loop_index')}")
        print(f"  dwarf_vector: {[f'{v:.3f}' for v in sample['dwarf_vector']]}")
        top3 = sorted(sample['dwarf_scores'].items(),
                      key=lambda x: x[1], reverse=True)[:3]
        print(f"  Top3: {top3}")


if __name__ == '__main__':
    main()