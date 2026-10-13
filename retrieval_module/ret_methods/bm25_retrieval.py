import csv
import json
from pathlib import Path

from generation_module.reference_sources import paired_programs
from retrieval_module.ret_methods.bm25_ranking import BM25, validate_paired_snippets
from retrieval_module.ret_methods.two_stage_retrieval import TwoStageRetrieval


class BM25Retrieval(TwoStageRetrieval):
    def __init__(self, parashift_dir, cuda_only_dir, heldout_projects,
                 obfuscate_code=True, seed=42, loop_review=None):
        super().__init__(parashift_dir, cuda_only_dir, heldout_projects=heldout_projects,
                         obfuscate_code=obfuscate_code, seed=seed,
                         loop_review=loop_review, use_llm=False)
        self.method = 'bm25' if obfuscate_code else 'raw-bm25'

    def prepare(self):
        self._parashift_loader = self._load_parashift()
        validate_paired_snippets(self._parashift_loader.c_codes, self.parashift_dir, 'c')
        validate_paired_snippets(self._parashift_loader.cuda_codes, self.parashift_dir, 'cuda')
        self._paired_sources = paired_programs(self._parashift_loader, self.parashift_dir)
        self._cuda_only_loader = self._load_cuda_only()
        self._c_index = BM25([entry['code'] for entry in self._parashift_loader.c_codes],
                             self.obfuscate_code)
        self._cuda_index = BM25([entry['code'] for entry in self._cuda_only_loader.cuda_codes],
                                self.obfuscate_code)
        self._c_vecs = self._cuda_only_vecs = None

    def _stage1(self, query_loops, parashift_loader, c_vecs):
        scores = self._c_index.max_scores([loop['source_code'] for loop in query_loops])
        file_scores = {}
        for entry, score in zip(parashift_loader.c_codes, scores):
            key = (entry['project'], entry['filename'])
            if key in self._paired_sources:
                file_scores[key] = max(file_scores.get(key, float('-inf')), score)
        results = []
        for key, score in sorted(file_scores.items(), key=lambda item: (-item[1], item[0]))[:1]:
            source = self._paired_sources[key]
            results.append({
                'project': key[0], 'c_filename': key[1], 'score': score,
                'c_code': source['c_code'], 'c_path': source['c_path'],
                'pair_kernels': [{'filename': Path(source['cuda_path']).name,
                                  'code': source['cuda_code'], 'source_path': source['cuda_path']}],
            })
        return results

    def _stage2(self, query_loops, cuda_only_loader, cuda_only_vecs):
        scores = self._cuda_index.max_scores([loop['source_code'] for loop in query_loops])
        results, seen_files = [], set()
        indices = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        for index in indices:
            entry = cuda_only_loader.cuda_codes[index]
            if entry['filename'] in seen_files:
                continue
            seen_files.add(entry['filename'])
            results.append({
                'rank': len(results) + 1, 'kernel_id': entry['id'],
                'kernel_name': entry.get('kernel_name', ''), 'filename': entry['filename'],
                'repo_name': entry.get('repo_name', ''), 'score': scores[index],
                'code': self._standalone_sources[entry['filename']],
                'source_path': str(Path(self.cuda_only_dir) / 'db_cuda_codes' / entry['filename']),
            })
            if len(results) == 2:
                break
        return results

    def run(self, c_file, output_dir):
        if self._parashift_loader is None:
            self.prepare()
        return super().run(c_file, output_dir)

    def _save(self, result, output_dir, c_file):
        result['config'].pop('alpha', None)
        result['config'].pop('beta', None)
        result['config'].update(method=self.method, k1=1.5, b=0.75,
                                score_aggregation='max over query loops and candidate loops/kernels')
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        base = f'{self.method}_{Path(c_file).stem}'
        (root / f'{base}.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
        with (root / f'{base}_stage2.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(['rank', 'kernel_id', 'kernel_name', 'filename', 'repo_name', 'score'])
            for entry in result['stage2']:
                writer.writerow([entry[key] for key in
                                 ('rank', 'kernel_id', 'kernel_name', 'filename', 'repo_name', 'score')])
