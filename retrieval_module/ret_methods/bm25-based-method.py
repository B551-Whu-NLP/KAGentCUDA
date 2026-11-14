import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data_module.data_loader import DataLoader, RANKING_REPRESENTATION
from retrieval_module.ret_methods.bm25_ranking import BM25, validate_paired_snippets
from retrieval_module.ret_methods.loop_exclusions import LoopExclusions, reviewed_paired_loops


def main():
    parser = argparse.ArgumentParser(description='Basic paired BM25 retrieval utility')
    parser.add_argument('--parashift_dir', default='./code_datas')
    parser.add_argument('--output_dir', default='./res_rag')
    parser.add_argument('--obfuscate', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cuda_only_dir', default='./CUDA-only')
    parser.add_argument('--loop_review')
    args = parser.parse_args()
    loader = DataLoader(args.parashift_dir, obfuscate_code=False)
    if not loader.load_cache(Path(__file__).resolve().parents[2] / '.data_cache'):
        raise FileNotFoundError('Prepared paired corpus cache required')
    validate_paired_snippets(loader.c_codes, args.parashift_dir, 'c')
    validate_paired_snippets(loader.cuda_codes, args.parashift_dir, 'cuda')
    excluded = []
    if args.loop_review:
        review = LoopExclusions(args.loop_review)
        review.validate_sources('paired', args.parashift_dir)
        loader.c_codes, excluded = reviewed_paired_loops(loader.c_codes, review, args.parashift_dir)
    index = BM25([entry['code'] for entry in loader.cuda_codes], args.obfuscate)
    root = Path(args.output_dir) / 'bm25_output'
    root.mkdir(parents=True, exist_ok=True)
    tag = '_obf' if args.obfuscate else ''
    with (root / f'retrieval_results_bm25{tag}.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['loop_idx', 'kernel_idx', 'similarity', 'rank'])
        for loop in loader.c_codes:
            scores = index.scores(loop['code'])
            for rank, position in enumerate(sorted(range(len(scores)), key=lambda i: (-scores[i], i)), 1):
                writer.writerow([loop['id'], loader.cuda_codes[position]['id'], scores[position], rank])
    config = dict(method='bm25', k1=1.5, b=0.75, seed=args.seed,
                  ranking_representation=RANKING_REPRESENTATION if args.obfuscate else 'raw',
                  protocol='basic paired utility',
                  loop_review='reviewed' if args.loop_review else 'not_reviewed',
                  excluded_paired_loops=len(excluded))
    (root / f'config_bm25{tag}.json').write_text(json.dumps(config, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
