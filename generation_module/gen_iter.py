"""Zero-shot Tool-Feedback baseline: raw tool logs and latest candidate only."""

import re
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from llm_utils import OpenAILLM
from generation_module.eval_types import IterativeOptimizer
from generation_module.prompts_gen_iter import build_initial_translation_prompt


class CToCudaIterativeTranslator:
    """C到CUDA代码翻译器 - 带迭代优化"""

    def __init__(
        self,
        model_name="qwen3-coder-next",
        org="ali",
        input_dir=None,
        max_retries=3,
        seed=None,
        max_samples=1,
        parashift_dir=None,
        max_iterations=3
    ):
        self.model_name = model_name
        self.org = org
        self.max_retries = max_retries
        self.seed = seed
        self.max_samples = max_samples
        self.max_iterations = max_iterations

        self.input_dir    = Path(input_dir)    if input_dir    else Path("./selected_samples")
        self.parashift_dir = Path(parashift_dir) if parashift_dir else Path("./code_datas")
        self.output_dir   = Path("./cuda_gen_results")
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.workspace_dir  = Path("./optimization_workspace")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

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

        self.optimizer = IterativeOptimizer(max_iterations=max_iterations)

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
            unique_key = item.get('id', 'unknown')       # "type-1_c_236"，用于去重
            task_id    = item.get('project_name', '')    # "c_236"，传给 pipeline/run.sh
            task_type  = item.get('type', 'unknown')     # "type-1"

            if not task_id:
                parts = unique_key.split('_', 1)
                task_id = parts[1] if len(parts) == 2 else unique_key

            # 按优先级查找 code_path
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
                code_path = str(codes_dir / f"{unique_key}.c")  # 占位，extract 时会报错

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
                print(f"   警告: 文件内容为空 (尝试 {attempt + 1}/{max_retries})")
            except Exception as e:
                print(f"   警告: 读取失败: {e} (尝试 {attempt + 1}/{max_retries})")
        return "", f"代码提取失败，已重试 {max_retries} 次"

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
        """
        目录结构：
          cuda_codes/<model>/seed<seed>/<type>/
            <type>_<task_id>.cu               ← 最新候选（始终覆盖）
            ret-iter/<type>_<task_id>_iter<N>.cu  ← 每次迭代存档
        """
        type_dir = self.output_dir / "cuda_codes" / self.model_name / f"seed{self.seed}" / task_type
        type_dir.mkdir(parents=True, exist_ok=True)

        iter_dir = type_dir / "ret-iter"
        iter_dir.mkdir(exist_ok=True)
        with open(iter_dir / f"{task_type}_{task_id}_iter{iteration}.cu", "w", encoding="utf-8") as f:
            f.write(cuda_code)

        latest = type_dir / f"{task_type}_{task_id}.cu"
        with open(latest, "w", encoding="utf-8") as f:
            f.write(cuda_code)

        return latest

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
        return bool(re.search(r'\b(?:int|void)\s+main\s*\(', code) and
                    '__global__' in code and '```' not in code)

    def generate_cuda(
        self,
        c_code: str,
        task_type: str,
        task_id: str,
        previous_code: Optional[str] = None,
        previous_result: Optional[dict] = None,
        response_budget: Optional[int] = None,
    ) -> Dict:
        last_error = None
        responses = 0
        if previous_result is not None and previous_code:
            raw_logs = previous_result.get('raw_tool_logs', '')
            if not raw_logs.strip():
                return {'cuda_code': '', 'success': False, 'responses': 0,
                        'error': 'Tool-Feedback requires raw verification output'}
            system_prompt = 'You are a CUDA programming expert. Return a complete corrected CUDA program.'
            user_prompt = (f'Original C program:\n```c\n{c_code}\n```\n'
                           f'Latest CUDA candidate:\n```cuda\n{previous_code}\n```\n'
                           f'Raw verification tool output:\n{raw_logs}\n'
                           'Correct the latest candidate. Return complete CUDA in a ```cuda code block.')
        else:
            system_prompt, user_prompt = build_initial_translation_prompt(
                c_code=c_code, task_type=task_type,
                reference_kernels=[], paired_examples=[],
            )
        for attempt in range(min(self.max_retries, response_budget if response_budget is not None else self.max_retries)):
            try:
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

    def iterative_optimize(self, c_code: str, task_type: str, task_id: str) -> Dict:
        print(f"   迭代优化: {task_type}/{task_id} (最多 {self.max_iterations} 次)")

        latest_code = ""
        latest_result = None
        iteration_codes = []
        responses = 0
        stop_reason = None

        for i in range(1, self.max_iterations + 1):
            if responses >= self.max_iterations:
                break
            print(f"   iter {i}/{self.max_iterations}", end="  ", flush=True)

            gen_result = self.generate_cuda(
                c_code, task_type, task_id,
                previous_code   = latest_code if i > 1 else None,
                previous_result = latest_result if i > 1 else None,
                response_budget = self.max_iterations - responses,
            )
            responses += gen_result.get('responses', 0)
            if not gen_result['success'] or not gen_result.get('cuda_code'):
                stop_reason = gen_result.get('error', 'Generation failed')
                print(f'生成失败，停止轨迹: {stop_reason}')
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

            layers_data = eval_result.metrics.get('layers', {})
            failed_layer = eval_result.metrics.get('first_failed_layer')
            if not failed_layer:
                failed_layer = next((name for name in ('L1', 'L2', 'L3', 'L4')
                                     if name in layers_data and not layers_data[name].get('passed')), None)
            raw_logs = layers_data.get(failed_layer, {}).get('raw_log', '') if failed_layer else ''
            latest_code = cuda_code
            latest_result = {'raw_tool_logs': raw_logs,
                             'evaluation': eval_result.to_dict()}
            iteration_codes.append({
                'iteration': i, 'code_length': len(cuda_code),
                'result': eval_result.to_dict(), 'responses': responses,
            })
            if eval_result.passed:
                break
            if not raw_logs.strip() and responses < self.max_iterations:
                stop_reason = f'{failed_layer or "unknown layer"} failed without raw verification output'
                print(f'{stop_reason}; stop Tool-Feedback trajectory')
                break

        return {
            'success': bool(latest_result and latest_result['evaluation']['passed']),
            'final_code': latest_code,
            'iterations': len(iteration_codes), 'responses': responses,
            'latest_result': latest_result['evaluation'] if latest_result else None,
            'all_iterations': iteration_codes,
            'stop_reason': stop_reason,
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

        checkpoint = self._load_checkpoint()
        results: list = []
        processed_keys: set = set()

        if checkpoint:
            results = checkpoint["results"]
            processed_keys = {r["unique_key"] for r in results}
            print(f"检测到断点，已处理 {len(results)} 个，继续从第 {len(results)+1} 个开始")

        print("-" * 60)

        remaining   = [s for s in all_samples if s["unique_key"] not in processed_keys]
        total       = len(all_samples)
        done        = 0

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

            opt_result = self.iterative_optimize(code, task_type, task_id)

            if opt_result.get("success"):
                print(f"   [OK] {opt_result.get('iterations', 0)} 次迭代")
            else:
                latest = opt_result.get('latest_result')
                if latest:
                    print(f"   [PARTIAL] {latest.get('layer','L0')} ({latest.get('passed_count',0)}/4)")
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
    parser.add_argument("--model",          default="qwen3-coder-next")
    parser.add_argument("--org",            default="ali")
    parser.add_argument("--input_dir",      default="./selected_samples")
    parser.add_argument("--parashift_dir",  default="./code_datas")
    parser.add_argument("--max_retries",    type=int, default=3)
    parser.add_argument("--seed",           type=int, default=42)
    parser.add_argument("--max_samples",    type=int, default=1)
    parser.add_argument("--max_iterations", type=int, default=3)
    args = parser.parse_args()

    print(f"[INFO] 模型: {args.model}  平台: {args.org}  Seed: {args.seed}  最大迭代: {args.max_iterations}")

    CToCudaIterativeTranslator(
        model_name     = args.model,
        org            = args.org,
        parashift_dir  = args.parashift_dir,
        input_dir      = args.input_dir,
        max_retries    = args.max_retries,
        seed           = args.seed,
        max_samples    = args.max_samples,
        max_iterations = args.max_iterations,
    ).batch_optimize()


if __name__ == "__main__":
    main()