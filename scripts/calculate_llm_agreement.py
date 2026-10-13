# ============================================================
# scripts/calculate_llm_agreement.py
# 计算三个 LLM 标注者之间的一致性：Krippendorff's alpha, Cohen's kappa, 分歧率
# ============================================================
import json
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import krippendorff

# 13个dwarf模式
PATTERNS = [
    'dense_linear_algebra',
    'sparse_linear_algebra',
    'spectral_methods',
    'n_body_methods',
    'structured_grids',
    'unstructured_grids',
    'map_reduce',
    'combinational_logic',
    'graph_traversal',
    'dynamic_programming',
    'backtrack_branch_bound',
    'graphical_models',
    'finite_state_machines',
]

LABEL_DIR = Path('dwarf_labels/label_kernels')


def load_annotations(filepath: Path) -> dict:
    """加载单个标注者的结果"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    annotations = {}
    for r in data['results']:
        if not r.get('success'):
            continue
        key = r['unique_key']
        # 存储匹配到的模式集合
        matched = set(r.get('matched_patterns', []))
        # 存储置信度分数
        scores = {p: r.get(p, 0) for p in PATTERNS}
        # 存储top-1模式（置信度最高的）
        top_pattern = max(PATTERNS, key=lambda p: scores[p])
        
        annotations[key] = {
            'matched_patterns': matched,
            'scores': scores,
            'top_pattern': top_pattern,
            'top_score': scores[top_pattern],
        }
    return annotations


def compute_cohen_kappa(annotator1: dict, annotator2: dict, common_keys: set) -> float:
    """计算两个标注者之间的 Cohen's kappa（基于top-1模式分类）"""
    # 构建混淆矩阵
    labels = list(PATTERNS)
    label_to_idx = {l: i for i, l in enumerate(labels)}
    n = len(labels)
    
    # 混淆矩阵
    conf_matrix = np.zeros((n, n))
    for key in common_keys:
        l1 = label_to_idx[annotator1[key]['top_pattern']]
        l2 = label_to_idx[annotator2[key]['top_pattern']]
        conf_matrix[l1, l2] += 1
    
    # 计算观测一致性
    total = len(common_keys)
    observed_agreement = np.trace(conf_matrix) / total
    
    # 计算期望一致性（随机）
    row_sums = conf_matrix.sum(axis=1)
    col_sums = conf_matrix.sum(axis=0)
    expected_agreement = np.sum((row_sums / total) * (col_sums / total))
    
    kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement)
    return kappa


def compute_krippendorff_alpha(annotators: list, common_keys: set) -> float:
    """计算 Krippendorff's alpha（基于top-1模式的名义数据）"""
    # 构建标注矩阵: annotators x items
    labels = list(PATTERNS)
    label_to_idx = {l: i for i, l in enumerate(labels)}
    
    n_annotators = len(annotators)
    n_items = len(common_keys)
    key_list = list(common_keys)
    
    reliability_data = np.zeros((n_annotators, n_items))
    for a_idx, annotator in enumerate(annotators):
        for i, key in enumerate(key_list):
            reliability_data[a_idx, i] = label_to_idx[annotator[key]['top_pattern']]
    
    # Krippendorff's alpha for nominal data
    alpha = krippendorff.alpha(reliability_data=reliability_data, level_of_measurement='nominal')
    return alpha


def compute_disagreement_rate(annotator1: dict, annotator2: dict, annotator3: dict, common_keys: set) -> dict:
    """计算分歧率"""
    total = len(common_keys)
    
    # Top-1模式完全一致
    perfect_agree_top1 = 0
    # 至少两个一致
    at_least_two_agree_top1 = 0
    # 完全不一致
    all_disagree_top1 = 0
    
    # 模式集合的Jaccard相似度
    jaccard_similarities = []
    
    # 分数的Pearson相关系数
    pearson_correlations_12 = []
    pearson_correlations_13 = []
    pearson_correlations_23 = []
    
    for key in common_keys:
        a1_top = annotator1[key]['top_pattern']
        a2_top = annotator2[key]['top_pattern']
        a3_top = annotator3[key]['top_pattern']
        
        # Top-1一致性
        if a1_top == a2_top == a3_top:
            perfect_agree_top1 += 1
            at_least_two_agree_top1 += 1
        elif a1_top == a2_top or a1_top == a3_top or a2_top == a3_top:
            at_least_two_agree_top1 += 1
        else:
            all_disagree_top1 += 1
        
        # Jaccard相似度（对匹配的模式集合）
        s1 = annotator1[key]['matched_patterns']
        s2 = annotator2[key]['matched_patterns']
        s3 = annotator3[key]['matched_patterns']
        union = s1 | s2 | s3
        intersection = s1 & s2 & s3
        if union:
            jaccard = len(intersection) / len(union)
        else:
            jaccard = 1.0
        jaccard_similarities.append(jaccard)
        
        # 分数向量的相关系数
        scores1 = [annotator1[key]['scores'][p] for p in PATTERNS]
        scores2 = [annotator2[key]['scores'][p] for p in PATTERNS]
        scores3 = [annotator3[key]['scores'][p] for p in PATTERNS]
        
        if np.std(scores1) > 0 and np.std(scores2) > 0:
            pearson_correlations_12.append(np.corrcoef(scores1, scores2)[0, 1])
        if np.std(scores1) > 0 and np.std(scores3) > 0:
            pearson_correlations_13.append(np.corrcoef(scores1, scores3)[0, 1])
        if np.std(scores2) > 0 and np.std(scores3) > 0:
            pearson_correlations_23.append(np.corrcoef(scores2, scores3)[0, 1])
    
    return {
        'total': total,
        'perfect_agree_top1': perfect_agree_top1,
        'perfect_agree_top1_pct': perfect_agree_top1 / total * 100,
        'at_least_two_agree_top1': at_least_two_agree_top1,
        'at_least_two_agree_top1_pct': at_least_two_agree_top1 / total * 100,
        'all_disagree_top1': all_disagree_top1,
        'all_disagree_top1_pct': all_disagree_top1 / total * 100,
        'avg_jaccard_similarity': np.mean(jaccard_similarities),
        'avg_pearson_12': np.mean(pearson_correlations_12),
        'avg_pearson_13': np.mean(pearson_correlations_13),
        'avg_pearson_23': np.mean(pearson_correlations_23),
    }


def analyze_pattern_distribution(annotator: dict, name: str) -> dict:
    """分析单个标注者的模式分布"""
    pattern_counts = Counter()
    for key, data in annotator.items():
        pattern_counts[data['top_pattern']] += 1
    return pattern_counts


def main():
    print("=" * 70)
    print("三个 LLM 标注者的一致性分析")
    print("=" * 70)
    
    # 加载三个标注者的结果
    files = {
        'Qwen3-Coder-Next': LABEL_DIR / 'qwen3-coder-next_20260513_112414.json',
        'GPT-4.1-Mini': LABEL_DIR / 'gpt-4.1-mini_20260513_113416.json',
        'DeepSeek-V4-Pro': LABEL_DIR / 'deepseek-v4-pro_20260513_124040.json',
    }
    
    annotators = {}
    for name, filepath in files.items():
        if filepath.exists():
            annotators[name] = load_annotations(filepath)
            print(f"\n{name}: {len(annotators[name])} 个标注")
        else:
            print(f"\n[WARN] 文件不存在: {filepath}")
    
    if len(annotators) < 2:
        print("\n[ERROR] 至少需要2个标注者才能计算一致性")
        return
    
    # 找到共同标注的kernel
    common_keys = set.intersection(*[set(a.keys()) for a in annotators.values()])
    print(f"\n共同标注的 kernel 数量: {len(common_keys)}")
    
    # 列出标注者对
    annotator_names = list(annotators.keys())
    annotator_list = [annotators[n] for n in annotator_names]
    
    print("\n" + "=" * 70)
    print("1. Cohen's Kappa (Top-1 模式分类一致性)")
    print("=" * 70)
    for i in range(len(annotator_names)):
        for j in range(i + 1, len(annotator_names)):
            kappa = compute_cohen_kappa(
                annotator_list[i], annotator_list[j], common_keys
            )
            print(f"  {annotator_names[i]} vs {annotator_names[j]}: {kappa:.4f}")
    
    print("\n" + "=" * 70)
    print("2. Krippendorff's Alpha (三个标注者的整体一致性)")
    print("=" * 70)
    if len(annotator_list) >= 3:
        alpha = compute_krippendorff_alpha(annotator_list, common_keys)
        print(f"  Krippendorff's alpha: {alpha:.4f}")
    else:
        print("  [SKIP] 需要至少3个标注者")
    
    print("\n" + "=" * 70)
    print("3. 分歧率统计")
    print("=" * 70)
    if len(annotator_list) >= 3:
        stats = compute_disagreement_rate(*annotator_list, common_keys)
        print(f"  总样本数: {stats['total']}")
        print(f"  三者完全一致 (Top-1): {stats['perfect_agree_top1']} ({stats['perfect_agree_top1_pct']:.1f}%)")
        print(f"  至少两者一致 (Top-1): {stats['at_least_two_agree_top1']} ({stats['at_least_two_agree_top1_pct']:.1f}%)")
        print(f"  三者完全不一致 (Top-1): {stats['all_disagree_top1']} ({stats['all_disagree_top1_pct']:.1f}%)")
        print(f"\n  平均 Jaccard 相似度 (匹配模式集合): {stats['avg_jaccard_similarity']:.4f}")
        print(f"\n  平均 Pearson 相关系数 (置信度分数向量):")
        print(f"    {annotator_names[0]} vs {annotator_names[1]}: {stats['avg_pearson_12']:.4f}")
        print(f"    {annotator_names[0]} vs {annotator_names[2]}: {stats['avg_pearson_13']:.4f}")
        print(f"    {annotator_names[1]} vs {annotator_names[2]}: {stats['avg_pearson_23']:.4f}")
    
    print("\n" + "=" * 70)
    print("4. 各标注者的 Top-1 模式分布")
    print("=" * 70)
    for name in annotator_names:
        dist = analyze_pattern_distribution(annotators[name], name)
        print(f"\n{name}:")
        for pattern, count in dist.most_common():
            pct = count / len(annotators[name]) * 100
            bar = '█' * int(pct / 100 * 40)
            print(f"  {pattern:<25} {bar:<40} {count:>3} ({pct:>5.1f}%)")
    
    # 保存详细结果到JSON
    result = {
        'common_kernels': len(common_keys),
        'annotators': annotator_names,
        'cohens_kappa': {},
        'krippendorffs_alpha': alpha if len(annotator_list) >= 3 else None,
        'disagreement_stats': stats if len(annotator_list) >= 3 else None,
        'pattern_distributions': {
            name: dict(analyze_pattern_distribution(annotators[name], name))
            for name in annotator_names
        },
    }
    
    for i in range(len(annotator_names)):
        for j in range(i + 1, len(annotator_names)):
            kappa = compute_cohen_kappa(
                annotator_list[i], annotator_list[j], common_keys
            )
            result['cohens_kappa'][f"{annotator_names[i]}_vs_{annotator_names[j]}"] = kappa
    
    output_file = 'dwarf_labels/llm_agreement_analysis.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'=' * 70}")
    print(f"详细结果已保存到: {output_file}")
    print("=" * 70)


if __name__ == '__main__':
    main()
