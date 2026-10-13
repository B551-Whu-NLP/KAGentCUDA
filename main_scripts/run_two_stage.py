#!/usr/bin/env python3
"""
批量两阶段检索脚本
遍历指定文件夹下所有 .c 文件，逐个执行两阶段检索，汇总结果。

查询文件须位于 code_datas/type-*/<held-out project>/c/ 目录。
使用示例：
    python main_scripts/run_two_stage.py --input_dir ./code_datas/type-1/<held-out-project>/c --heldout_projects /path/to/test_projects.json --cuda_only_dir /path/to/CUDA-only
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

SCRIPT_DIR   = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from retrieval_module.ret_methods.two_stage_retrieval import TwoStageRetrieval


def parse_args():
    parser = argparse.ArgumentParser(description="批量两阶段 C → CUDA 检索")
    parser.add_argument('--input_dir',      required=True,
                        help='包含 .c 查询文件的目录')
    parser.add_argument('--parashift_dir',  default='./code_datas',
                        help='配对语料数据目录（默认: ./code_datas）')
    parser.add_argument('--cuda_only_dir',  default='./CUDA-only',
                        help='CUDA-only 数据目录（默认: ./CUDA-only）')
    parser.add_argument('--output_dir',     default='./res_rag/two_stage_output',
                        help='结果输出目录（默认: ./res_rag/two_stage_output）')
    parser.add_argument('--alpha', '-a',    type=float, default=0.4,
                        help='结构特征权重（默认: 0.4）')
    parser.add_argument('--beta',  '-b',    type=float, default=0.6,
                        help='矮人特征权重（默认: 0.6）')
    parser.add_argument('--seed',           type=int,   default=42)
    parser.add_argument('--heldout_projects', required=True, help='JSON file containing the 109 held-out projects')
    parser.add_argument('--no_obfuscate',   action='store_true')
    parser.add_argument('--model_name',     default='qwen3-coder-next', help='LLM 模型名称')
    parser.add_argument('--org',            default='ali',              help='LLM 平台')
    parser.add_argument('--no_llm',         action='store_true',        help='跳过 LLM 标注（调试用）')
    parser.add_argument('--loop_review', help='Source-bound JSON review for query and paired loops')
    return parser.parse_args()


def main():
    args = parse_args()

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # 找所有 .c 文件
    c_files = sorted(input_dir.glob('*.c')) + sorted(input_dir.glob('*.cpp'))
    c_files = sorted(c_files)  # 统一按文件名排序
    if not c_files:
        print(f"❌ 未找到任何 .c 文件: {input_dir}")
        sys.exit(1)

    print(f"\n找到 {len(c_files)} 个 C 文件:")
    for f in c_files:
        print(f"  {f.name}")

    heldout = json.loads(Path(args.heldout_projects).read_text(encoding='utf-8'))['projects']
    samples = json.loads((PROJECT_ROOT / 'selected_samples' / 'selected_samples.json').read_text(encoding='utf-8'))
    selected_projects = {item['project_name'] for item in samples}
    if len(heldout) != 109 or set(heldout) != selected_projects or len(selected_projects) != 109:
        raise ValueError('Held-out split must contain exactly the 109 selected sample projects')
    retriever = TwoStageRetrieval(
        parashift_dir  = args.parashift_dir,
        cuda_only_dir  = args.cuda_only_dir,
        alpha          = args.alpha,
        beta           = args.beta,
        seed           = args.seed,
        obfuscate_code = not args.no_obfuscate,
        model_name     = args.model_name,
        org            = args.org,
        use_llm        = not args.no_llm,
        heldout_projects = heldout,
        loop_review = args.loop_review,
    )
    retriever.prepare()

    # 逐文件检索
    summary = []
    failed  = []

    for i, c_file in enumerate(c_files, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(c_files)}] {c_file.name}")
        print(f"{'='*60}")
        try:
            result = retriever.run(str(c_file), str(output_dir))
            stage1, stage2 = result['stage1'], result['stage2']

            summary.append({
                "file":           c_file.name,
                "status":         "ok",
                "stage1_top1":    stage1[0]["c_filename"] if stage1 else "",
                "stage1_score":   stage1[0]["score"]      if stage1 else 0.0,
                "stage2_top1":    stage2[0]["kernel_name"] if stage2 else "",
                "stage2_score":   stage2[0]["score"]       if stage2 else 0.0,
            })

        except Exception as e:
            print(f"  ❌ 失败: {e}")
            failed.append({"file": c_file.name, "error": str(e)})
            summary.append({"file": c_file.name, "status": "failed", "error": str(e)})

    # 汇总报告
    print(f"\n{'='*60}")
    print(f"批量检索完成  {len(c_files)} 个文件")
    print(f"{'='*60}")
    print(f"{'文件':<30} {'状态':<8} {'Stage1 Top1':<35} {'Stage2 Top1'}")
    print("-" * 100)
    for row in summary:
        if row["status"] == "ok":
            s1 = Path(row["stage1_top1"]).name if row["stage1_top1"] else "-"
            print(f"  {row['file']:<28} {'✅':<8} {s1:<35} {row['stage2_top1']}")
        else:
            print(f"  {row['file']:<28} {'❌':<8} {row.get('error','')}")

    # 保存汇总 JSON
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n汇总已保存: {summary_path}")

    if failed:
        print(f"\n⚠️  {len(failed)} 个文件失败:")
        for f in failed:
            print(f"  {f['file']}: {f['error']}")
        sys.exit(1)


if __name__ == '__main__':
    main()