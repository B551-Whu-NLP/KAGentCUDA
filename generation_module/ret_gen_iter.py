"""
C到CUDA代码翻译工具 - 迭代优化版本
集成 L1/L2/L3/L4 评估 + 两阶段检索，支持代码迭代优化
"""

import re
import sys
import json
import argparse
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import contextlib
import io

sys.path.insert(0, str(Path(__file__).parent.parent))
from llm_utils import OpenAILLM
from data_module.data_loader import DataLoader

from generation_module.eval_types import EvaluationResult, IterativeOptimizer
from generation_module.reference_sources import paired_programs
from generation_module.prompts_gen_iter import (
    build_initial_translation_prompt,
    build_optimization_prompt,
)


def _collect_retrieval_examples(retrieval_result: dict) -> tuple:
    pairs = retrieval_result.get('stage1', [])
    standalone = retrieval_result.get('stage2', [])
    if len(pairs) != 1 or len(standalone) != 2:
        raise ValueError('Retrieval requires exactly 1 paired program and 2 standalone CUDA files')
    pair = pairs[0]
    if len(pair.get('pair_kernels', [])) != 1:
        raise ValueError('Paired program has no unique CUDA source')
    c_code = pair.get('c_code', '')
    cuda_code = pair['pair_kernels'][0].get('code', '')
    codes = [item.get('code', '') for item in standalone]
    if not all(code.strip() for code in [c_code, cuda_code, *codes]):
        raise ValueError('Missing original source text for retrieved example')
    if len(set(codes)) != 2:
        raise ValueError('Standalone examples must be distinct source files')
    return [{'c_code': c_code, 'cuda_code': cuda_code}], codes


class CToCudaIterativeTranslator:
    """C到CUDA代码翻译器 - 带迭代优化 + 两阶段检索"""

    def __init__(
        self,
        model_name="qwen3-coder-next",
        org="ali",
        input_dir=None,
        max_retries=3,
        seed=None,
        max_samples=1,
        parashift_dir=None,
        max_iterations=3,
        retriever=None,          # 外部传入的 TwoStageRetrieval 实例（初始化一次复用）
        retrieval_output_dir=None,  # 检索结果保存目录
    ):
        self.model_name    = model_name
        self.org           = org
        self.max_retries   = max_retries
        self.seed          = seed
        self.max_samples   = max_samples
        self.max_iterations = max_iterations
        self.retriever     = retriever
        self.retrieval_output_dir = Path(retrieval_output_dir) if retrieval_output_dir \
                                    else Path("./retrieval_results")

        self.input_dir     = Path(input_dir)     if input_dir     else Path("./selected_samples")
        self.parashift_dir = Path(parashift_dir) if parashift_dir else Path("./code_datas")
        self.output_dir    = Path("./cuda_gen_results")
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.workspace_dir  = Path("./optimization_workspace")
        retrieval_method = getattr(retriever, 'method', 'ours')
        if retrieval_method != 'ours':
            self.output_dir /= retrieval_method
            self.checkpoint_dir = self.output_dir / 'checkpoints'
            self.workspace_dir /= retrieval_method
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.retrieval_output_dir.mkdir(parents=True, exist_ok=True)

        self.llm = OpenAILLM(
            model_name=model_name,
            org=org,
            log_path="./logs",
            verbose=False,
            temperature=0.3,
            top_p=0.95,
        )

        self.checkpoint_file = (
            self.checkpoint_dir / f"{model_name}_iterative_seed{seed}_checkpoint.json"
            if seed is not None
            else self.checkpoint_dir / f"{model_name}_iterative_checkpoint.json"
        )

        self._fallback_pairs = self._load_fallback_pairs() if retriever is None else []
        self.optimizer = IterativeOptimizer(max_iterations=max_iterations)

    def _load_fallback_pairs(self) -> List[dict]:
        loader = DataLoader(self.parashift_dir, obfuscate_code=False)
        if not loader.load_cache():
            loader.load()
        pairs = list(paired_programs(loader, self.parashift_dir).values())
        if len(pairs) < 3:
            raise ValueError('Random 3-shot baseline requires at least 3 complete paired programs')
        return pairs

    def _random_pairs(self, task_id: str) -> List[dict]:
        eligible = [p for p in self._fallback_pairs if p['project'] != task_id]
        if len(eligible) < 3:
            raise ValueError('Random 3-shot baseline requires 3 non-query paired programs')
        return random.sample(eligible, 3)

    def load_samples(self) -> List[Dict]:
        json_file = self.input_dir / "selected_samples.json"
        if not json_file.exists():
            print(f"错误: 未找到样本列表文件 {json_file}")
            return []

        with open(json_file, 'r', encoding='utf-8') as f:
            sample_list = json.load(f)

        codes_dir = self.input_dir / "codes"
        samples = []

        for item in sample_list:
            unique_key = item.get('id', 'unknown')
            task_id    = item.get('project_name', '')
            task_type  = item.get('type', 'unknown')

            if not task_id:
                parts = unique_key.split('_', 1)
                task_id = parts[1] if len(parts) == 2 else unique_key

            code_path = None
            raw = item.get('code_path', '')
            if raw and Path(raw).exists():
                code_path = str(raw)

            if not code_path:
                source_dir = self.parashift_dir / task_type / task_id / 'c'
                candidates = list(source_dir.glob('*.c')) + list(source_dir.glob('*.cpp'))
                if len(candidates) == 1:
                    code_path = str(candidates[0])
            if not code_path:
                for name in [unique_key, task_id]:
                    for ext in ['.c', '.cpp']:
                        p = codes_dir / f"{name}{ext}"
                        if p.exists():
                            code_path = str(p)
                            break
                    if code_path:
                        break

            if not code_path:
                code_path = str(codes_dir / f"{unique_key}.c")

            samples.append({
                'unique_key': unique_key,
                'id':         unique_key,
                'task_id':    task_id,
                'type':       task_type,
                'c_file':     item.get('c_file', ''),
                'cuda_file':  item.get('cuda_file', ''),
                'code_path':  code_path,
            })

        print(f"从 {json_file} 加载了 {len(samples)} 个样本")
        return samples

    def extract_code_with_retry(self, file_path: str, max_retries: int = 3) -> tuple:
        for attempt in range(max_retries):
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    code = f.read()
                if code and len(code.strip()) > 0:
                    return code, None
                print(f"   警告: 文件内容为空 (尝试 {attempt+1}/{max_retries})")
            except Exception as e:
                print(f"   警告: 读取失败: {e} (尝试 {attempt+1}/{max_retries})")
        return "", f"代码提取失败，已重试 {max_retries} 次"

    # ----------------------------------------------------------
    # 检索
    # ----------------------------------------------------------

    def _retrieve(self, code_path: str, task_id: str) -> tuple:
        if self.retriever is None:
            return self._random_pairs(task_id), []

        print('   检索中...', end='  ', flush=True)
        output_dir = str(self.retrieval_output_dir / task_id)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            result = self.retriever.run(c_file=code_path, output_dir=output_dir)
        pairs, kernels = _collect_retrieval_examples(result)
        print(f'获得 {len(pairs)} 个完整 C/CUDA 配对和 {len(kernels)} 个独立 CUDA 源文件')
        return pairs, kernels

    # ----------------------------------------------------------
    # 文件保存
    # ----------------------------------------------------------

    def _save_cuda_file(
        self,
        cuda_code: str,
        task_type: str,
        task_id: str,
        iteration: int,
    ) -> Path:
        type_dir = self.output_dir / "cuda_codes" / self.model_name / f"seed{self.seed}" / task_type
        type_dir.mkdir(parents=True, exist_ok=True)

        iter_dir = type_dir / "iter"
        iter_dir.mkdir(exist_ok=True)
        with open(iter_dir / f"{task_type}_{task_id}_iter{iteration}.cu", "w", encoding="utf-8") as f:
            f.write(cuda_code)

        latest = type_dir / f"{task_type}_{task_id}.cu"
        with open(latest, "w", encoding="utf-8") as f:
            f.write(cuda_code)
        return latest

    def _save_best_code(self, code: str, task_type: str, task_id: str):
        latest = (self.output_dir / 'cuda_codes' / self.model_name / f'seed{self.seed}' /
                  task_type / f'{task_type}_{task_id}.cu')
        latest.write_text(code, encoding='utf-8')

    # ----------------------------------------------------------
    # 代码生成
    # ----------------------------------------------------------

    @staticmethod
    def _extract_cuda_from_response(response: str) -> str:
        match = re.search(r'```(?:cuda|CUDA|cu)\s*\n(.*?)```', response, re.DOTALL)
        return match.group(1).strip() if match else ''

    @staticmethod
    def _validate_cuda_code(code: str) -> bool:
        if not code or len(code.strip()) == 0:
            return False
        s = code.strip()

        return bool(re.search(r'\b(?:int|void)\s+main\s*\(', s) and
                    '__global__' in s and '```' not in s)

    def generate_cuda(
        self,
        c_code: str,
        task_type: str,
        task_id: str,
        reference_kernels: Optional[List[str]] = None,
        paired_examples: Optional[List[dict]] = None,
        previous_code: Optional[str] = None,
        previous_result: Optional[EvaluationResult] = None,
        response_budget: Optional[int] = None,
    ) -> Dict:
        last_error = None
        responses = 0
        if not c_code.strip():
            raise ValueError('Input C program is empty')

        for attempt in range(min(self.max_retries, response_budget if response_budget is not None else self.max_retries)):
            try:
                if previous_result is not None and previous_code:
                    # 迭代优化：只用 pipeline_opt_prompt，不传 reference
                    pipeline_opt_prompt = previous_result.metrics.get("optimization_prompt", "")
                    system_prompt, user_prompt = build_optimization_prompt(
                        cuda_code=previous_code,
                        c_code=c_code,
                        pipeline_opt_prompt=pipeline_opt_prompt,
                        errors=previous_result.errors,
                        is_first_pass=False,
                    )
                else:
                    system_prompt, user_prompt = build_initial_translation_prompt(
                        c_code=c_code,
                        task_type=task_type,
                        reference_kernels=reference_kernels or [],
                        paired_examples=paired_examples or [],
                    )

                response, input_tokens, output_tokens, cost = self.llm.chat(
                    [{"role": "system", "content": system_prompt},
                     {"role": "user",   "content": user_prompt}],
                    stream=False,
                    max_tokens=8192,
                )
                responses += 1

                cuda_code = self._extract_cuda_from_response(response)

                if not self._validate_cuda_code(cuda_code):
                    last_error = f"无效CUDA代码 (尝试 {attempt+1}/{self.max_retries})"
                    print(f"   警告: {last_error}")
                    continue

                if len(cuda_code) > 50:
                    return {
                        'cuda_code': cuda_code, 'success': True,
                        'input_tokens': input_tokens,
                        'output_tokens': output_tokens, 'cost': cost,
                        'responses': responses,
                    }

                last_error = f"代码太短 (尝试 {attempt+1}/{self.max_retries})"
                print(f"   警告: {last_error}")

            except Exception as e:
                last_error = f"生成失败: {e} (尝试 {attempt+1}/{self.max_retries})"
                print(f"   警告: {last_error}")

        return {'cuda_code': '', 'success': False, 'responses': responses,
                'error': last_error or f"代码生成失败，已重试 {self.max_retries} 次"}

    # ----------------------------------------------------------
    # 迭代优化主流程
    # ----------------------------------------------------------

    def iterative_optimize(
        self,
        c_code: str,
        task_type: str,
        task_id: str,
        code_path: str,          # 用于检索的 C 文件路径
    ) -> Dict:
        print(f"   迭代优化: {task_type}/{task_id} (最多 {self.max_iterations} 次)")

        paired_examples, reference_kernels = self._retrieve(code_path, task_id)

        best_code   = ""
        best_result = EvaluationResult(layer="L0", passed=False, passed_count=0)
        iteration_codes = []
        responses = 0

        for i in range(1, self.max_iterations + 1):
            if responses >= self.max_iterations:
                break
            print(f"   iter {i}/{self.max_iterations}", end="  ", flush=True)

            if i == 1:
                # 首次：用检索结果作为 reference
                gen_result = self.generate_cuda(
                    c_code, task_type, task_id,
                    reference_kernels=reference_kernels,
                    paired_examples=paired_examples,
                    response_budget=self.max_iterations - responses,
                )
            else:
                # 后续迭代：只用 pipeline_opt_prompt，不传 reference
                gen_result = self.generate_cuda(
                    c_code, task_type, task_id,
                    previous_code=best_code,
                    previous_result=best_result,
                    response_budget=self.max_iterations - responses,
                )

            responses += gen_result.get('responses', 0)
            if not gen_result['success'] or not gen_result.get('cuda_code'):
                print('生成失败，响应预算停止')
                break

            cuda_code = gen_result['cuda_code']
            cu_file   = self._save_cuda_file(cuda_code, task_type, task_id, iteration=i)

            eval_result = self.optimizer.evaluate_layers(
                cu_file, task_type, task_id,
                model_name     = self.model_name,
                seed           = self.seed,
                workspace_root = self.workspace_dir,
                iteration      = i,
            )

            # 紧凑日志
            layers_data  = eval_result.metrics.get("layers", {})
            layer_icons  = "".join(
                "✓" if layers_data.get(l, {}).get("passed") else "✗"
                for l in ["L1", "L2", "L3", "L4"]
            )
            final_status = eval_result.metrics.get("final_status", "")
            speedup      = layers_data.get("L4", {}).get("metrics", {}).get("speedup")
            speedup_str  = f"  speedup={speedup:.2f}x" if speedup else ""
            print(f"[{layer_icons}] {final_status}{speedup_str}")

            iteration_codes.append({
                'iteration': i, 'code_length': len(cuda_code),
                'result': eval_result.to_dict(),
                'score': self.optimizer.score(eval_result), 'responses': responses,
            })

            if eval_result.passed:
                return {'success': True, 'final_code': cuda_code,
                        'iterations': i, 'responses': responses,
                        'best_result': eval_result.to_dict(),
                        'all_iterations': iteration_codes}

            if self.optimizer.is_better(eval_result, best_result):
                best_code   = cuda_code
                best_result = eval_result
            elif i > 1:
                print(f"         → 未提升，保留 {best_result.layer}")
            if best_code:
                self._save_best_code(best_code, task_type, task_id)

        return {
            'success': best_result.passed,
            'final_code': best_code,
            'iterations': len(iteration_codes), 'responses': responses,
            'best_result': best_result.to_dict() if best_code else None,
            'all_iterations': iteration_codes,
        }

    # ----------------------------------------------------------
    # 断点 & 批量
    # ----------------------------------------------------------

    def _load_checkpoint(self):
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict) and "results" in data:
                        return data
            except Exception as e:
                print(f"加载checkpoint失败: {e}")
        return None

    def _save_checkpoint(self, results: list, total_files: int):
        for r in results:
            opt = r.get('optimization_result', {})
            if isinstance(opt.get('best_result'), EvaluationResult):
                opt['best_result'] = opt['best_result'].to_dict()
            for it in opt.get('all_iterations', []):
                if isinstance(it.get('result'), EvaluationResult):
                    it['result'] = it['result'].to_dict()

        with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump({
                "results": results,
                "total_files": total_files,
                "processed_count": len(results),
                "model_name": self.model_name,
                "timestamp": datetime.now().isoformat(),
            }, f, indent=2, ensure_ascii=False)

    def _print_stats(self, results: list):
        print("\n" + "=" * 60)
        print("迭代优化统计")
        print("=" * 60)
        success = [r for r in results if r.get("optimization_result", {}).get("success")]
        print(f"总计: {len(results)}  全部通过: {len(success)}  未通过: {len(results)-len(success)}")
        if success:
            type_stats: Dict[str, int] = {}
            for r in success:
                t = r.get('type', 'unknown')
                type_stats[t] = type_stats.get(t, 0) + 1
            print("\n成功按类型分布:")
            for t, cnt in sorted(type_stats.items()):
                print(f"   {t}: {cnt}")

    def batch_optimize(self):
        all_samples = self.load_samples()
        if not all_samples:
            return

        print(f"\n模型: {self.model_name}  seed: {self.seed}  最大迭代: {self.max_iterations}")
        print(f"输入: {self.input_dir}  输出: {self.output_dir}")
        print(f"检索: {'开启 (' + type(self.retriever).__name__ + ')' if self.retriever else '关闭（随机 fallback）'}")

        checkpoint = self._load_checkpoint()
        results: list = []
        processed_keys: set = set()

        if checkpoint:
            results = checkpoint["results"]
            processed_keys = {r["unique_key"] for r in results}
            print(f"检测到断点，已处理 {len(results)} 个，继续从第 {len(results)+1} 个开始")

        print("-" * 60)

        remaining = [s for s in all_samples if s["unique_key"] not in processed_keys]
        total = len(all_samples)
        done  = 0

        for i, sample in enumerate(remaining, len(results) + 1):
            if self.max_samples and done >= self.max_samples:
                break
            done += 1

            unique_key = sample['unique_key']
            task_id    = sample['task_id']
            task_type  = sample['type']
            code_path  = sample['code_path']

            print(f"\n[{i}/{total}] {task_type} :: {task_id}")

            code, error = self.extract_code_with_retry(code_path)
            if error:
                print(f"   [FAIL] {error}")
                results.append({
                    'unique_key': unique_key, 'id': unique_key,
                    'task_id': task_id, 'type': task_type,
                    'optimization_result': {'success': False, 'error': error},
                })
                continue

            opt_result = self.iterative_optimize(code, task_type, task_id, code_path)

            if opt_result.get("success"):
                print(f"   [OK] {opt_result.get('iterations', 0)} 次迭代")
            else:
                best = opt_result.get('best_result')
                if best:
                    print(f"   [PARTIAL] {best.get('layer','L0')} ({best.get('passed_count',0)}/4)")
                else:
                    print(f"   [FAIL] {opt_result.get('error','未知错误')}")

            results.append({
                'unique_key': unique_key, 'id': unique_key,
                'task_id': task_id, 'type': task_type,
                'optimization_result': opt_result,
            })

            if i % 5 == 0:
                self._save_checkpoint(results, total)
                print(f"   checkpoint 已保存（{i} 个）")

        if results:
            self._save_checkpoint(results, total)

        self._print_stats(results)
        return results


def main():
    parser = argparse.ArgumentParser(description="C到CUDA代码翻译工具 - 迭代优化版本")
    parser.add_argument("--model",               default="qwen3-coder-next")
    parser.add_argument("--org",                 default="ali")
    parser.add_argument("--input_dir",           default="./selected_samples")
    parser.add_argument("--parashift_dir",       default="./code_datas")
    parser.add_argument("--cuda_only_dir",       default="./CUDA-only")
    parser.add_argument("--max_retries",         type=int,   default=3)
    parser.add_argument("--seed",                type=int,   default=42)
    parser.add_argument("--max_samples",         type=int,   default=1)
    parser.add_argument("--max_iterations",      type=int,   default=3)
    parser.add_argument("--retrieval_method", choices=['ours', 'bm25', 'raw-bm25'], default='ours')
    parser.add_argument("--retrieval_alpha",     type=float, default=0.4)
    parser.add_argument("--retrieval_beta",      type=float, default=0.6)
    parser.add_argument("--no_retrieval",        action="store_true", help="关闭检索，使用随机 reference")
    parser.add_argument("--retrieval_output_dir", default="./retrieval_results")
    parser.add_argument("--heldout_projects", help="JSON file containing the held-out projects array")
    parser.add_argument("--loop_review", help="Source-bound JSON review for query and paired loops")
    args = parser.parse_args()

    # 初始化检索器（一次，复用）
    retriever = None
    if not args.no_retrieval:
        if not args.heldout_projects:
            parser.error('--heldout_projects is required for retrieval')
        heldout = json.loads(Path(args.heldout_projects).read_text(encoding='utf-8'))['projects']
        selected = json.loads((Path(args.input_dir) / 'selected_samples.json').read_text(encoding='utf-8'))
        selected_projects = {item['project_name'] for item in selected}
        if len(heldout) != 109 or set(heldout) != selected_projects or len(selected_projects) != 109:
            raise ValueError('Held-out split must contain exactly the 109 selected sample projects')
        retrieval_args = dict(parashift_dir=args.parashift_dir,
                              cuda_only_dir=args.cuda_only_dir, seed=args.seed,
                              heldout_projects=heldout, loop_review=args.loop_review)
        if args.retrieval_method == 'ours':
            from retrieval_module.ret_methods.two_stage_retrieval import TwoStageRetrieval
            retriever = TwoStageRetrieval(
                **retrieval_args, alpha=args.retrieval_alpha, beta=args.retrieval_beta,
                model_name=args.model, org=args.org, use_llm=True)
        else:
            from retrieval_module.ret_methods.bm25_retrieval import BM25Retrieval
            retriever = BM25Retrieval(
                **retrieval_args, obfuscate_code=args.retrieval_method == 'bm25')
        retriever.prepare()

    print(f"[INFO] 模型: {args.model}  平台: {args.org}  Seed: {args.seed}  最大迭代: {args.max_iterations}")
    print(f"[INFO] 检索: {'关闭' if args.no_retrieval else args.retrieval_method}")

    CToCudaIterativeTranslator(
        model_name           = args.model,
        org                  = args.org,
        parashift_dir        = args.parashift_dir,
        input_dir            = args.input_dir,
        max_retries          = args.max_retries,
        seed                 = args.seed,
        max_samples          = args.max_samples,
        max_iterations       = args.max_iterations,
        retriever            = retriever,
        retrieval_output_dir = args.retrieval_output_dir,
    ).batch_optimize()


if __name__ == "__main__":
    main()