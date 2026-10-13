"""
C到CUDA代码翻译工具
数据来源: ./selected_samples/
支持重试机制：提取失败最多重试3次
参考代码从 data_loader 中随机选取
"""

import os
import sys
import json
import argparse
import random
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from llm_utils import OpenAILLM
from data_module.data_loader import DataLoader
from generation_module.reference_sources import paired_programs
from generation_module.prompts_gen_iter import build_initial_translation_prompt


class CToCudaTranslator:
    def __init__(self, model_name="qwen3-coder-next", org="ali", input_dir=None, max_retries=3, seed=None, max_samples=1, parashift_dir=None):
        self.model_name = model_name
        self.org = org
        self.max_retries = max_retries
        self.seed = seed
        self.max_samples = max_samples
        
        if input_dir:
            self.input_dir = Path(input_dir)
        else:
            self.input_dir = Path("./selected_samples")
        
        # 配对语料目录，用于加载参考 kernel
        if parashift_dir:
            self.parashift_dir = Path(parashift_dir)
        else:
            self.parashift_dir = Path("./code_datas")
        
        self.output_dir = Path("./cuda_gen_results")
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.llm = OpenAILLM(
            model_name=model_name,
            org=org,
            log_path="./logs",
            verbose=False,
            temperature=0.3,
            top_p=0.95
        )
        
        # checkpoint文件名包含seed（如果提供）
        if seed is not None:
            self.checkpoint_file = self.checkpoint_dir / f"{model_name}_cuda_gen_seed{seed}_checkpoint.json"
        else:
            self.checkpoint_file = self.checkpoint_dir / f"{model_name}_cuda_gen_checkpoint.json"
        
        loader = DataLoader(self.parashift_dir, obfuscate_code=False)
        if not loader.load_cache():
            loader.load()
        self.reference_pairs = list(paired_programs(loader, self.parashift_dir).values())
        if len(self.reference_pairs) < 3:
            raise ValueError('Random 3-shot requires 3 complete C/CUDA programs')
    
    def load_samples(self):
        """从 selected_samples.json 加载样本列表"""
        samples = []
        
        json_file = self.input_dir / "selected_samples.json"
        if not json_file.exists():
            print(f"错误: 未找到样本列表文件 {json_file}")
            return samples
        
        codes_dir = self.input_dir / "codes"
        
        with open(json_file, 'r', encoding='utf-8') as f:
            sample_list = json.load(f)
        
        for item in sample_list:
            sample_id = item.get('id', item.get('project_name', 'unknown'))
            
            source_dir = self.parashift_dir / item.get('type', '') / item.get('project_name', '') / 'c'
            candidates = list(source_dir.glob('*.c')) + list(source_dir.glob('*.cpp'))
            code_path = candidates[0] if len(candidates) == 1 else codes_dir / f"{sample_id}.c"
            if not code_path.exists():
                code_path = codes_dir / f"{sample_id}.cpp"
            
            samples.append({
                'id': sample_id,
                'type': item.get('type', 'unknown'),
                'c_file': item.get('c_file', ''),
                'cuda_file': item.get('cuda_file', ''),
                'code_path': str(code_path),
                'unique_key': sample_id
            })
        
        print(f"从 {json_file} 加载了 {len(samples)} 个样本")
        return samples
    
    def extract_code_with_retry(self, file_path: str, max_retries: int = 3) -> tuple:
        """
        带重试的代码提取
        
        Returns:
            (code: str, error: str or None)
        """
        for attempt in range(max_retries):
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    code = f.read()
                
                if code and len(code.strip()) > 0:
                    return code, None
                else:
                    error = f"文件内容为空 (尝试 {attempt + 1}/{max_retries})"
                    print(f"   警告: {error}")
                    
            except Exception as e:
                error = f"读取失败: {e} (尝试 {attempt + 1}/{max_retries})"
                print(f"   警告: {error}")
        
        return "", f"代码提取失败，已重试 {max_retries} 次"
    
    def generate_cuda_code(self, c_code: str, file_id: str, type_name: str = "") -> dict:
        """将C代码翻译为CUDA代码"""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                eligible = [p for p in self.reference_pairs if p['project'] != file_id.split('_', 1)[-1]]
                if len(eligible) < 3:
                    raise ValueError('Not enough non-query paired examples')
                system_prompt, user_prompt = build_initial_translation_prompt(
                    c_code=c_code, task_type=type_name, reference_kernels=[],
                    paired_examples=random.sample(eligible, 3),
                )
                
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                
                response, input_tokens, output_tokens, cost = self.llm.chat(messages, stream=False, max_tokens=8192)
                
                import re
                match = re.search(r'```(?:cuda|CUDA|cu)\s*\n(.*?)```', response, re.DOTALL)
                cuda_code = match.group(1).strip() if match else ''
                
                if cuda_code and len(cuda_code) > 50:
                    return {
                        'cuda_code': cuda_code,
                        'success': True,
                        'input_tokens': input_tokens,
                        'output_tokens': output_tokens,
                        'cost': cost
                    }
                else:
                    last_error = f"生成的CUDA代码太短或为空 (尝试 {attempt + 1}/{self.max_retries})"
                    print(f"   警告: {last_error}")
                    
            except Exception as e:
                last_error = f"生成失败: {e} (尝试 {attempt + 1}/{self.max_retries})"
                print(f"   警告: {last_error}")
        
        return {
            'cuda_code': '',
            'success': False,
            'error': last_error or f"代码生成失败，已重试 {self.max_retries} 次"
        }
    
    def _get_code_extension(self, code: str) -> str:
        """根据代码内容判断文件类型"""
        if 'std::vector' in code or 'std::cout' in code or 'namespace std' in code:
            return 'cpp'
        return 'c'
    
    def _load_checkpoint(self):
        """加载断点"""
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
        """保存断点"""
        checkpoint_data = {
            "results": results,
            "total_files": total_files,
            "processed_count": len(results),
            "model_name": self.model_name,
            "timestamp": datetime.now().isoformat()
        }
        with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
    
    def _save_results_with_timestamp(self, results: list):
        """按时间戳保存结果（支持seed区分）"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 文件名包含seed（如果提供）
        if self.seed is not None:
            filename = f"{self.model_name}_cuda_gen_seed{self.seed}_{timestamp}.json"
        else:
            filename = f"{self.model_name}_cuda_gen_{timestamp}.json"
        filepath = self.output_dir / filename
        
        result_data = {
            "model_name": self.model_name,
            "org": self.org,
            "timestamp": datetime.now().isoformat(),
            "total_files": len(results),
            "success_count": sum(1 for r in results if r["success"]),
            "fail_count": sum(1 for r in results if not r["success"]),
            "results": results
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        
        # 保存单独的 .cu 文件
        self._save_cuda_files(results)
        
        self._update_global_index(filepath)
        
        return filepath
    
    def _save_cuda_files(self, results: list):
        """将生成的CUDA代码保存为单独的 .cu 文件（按模型+seed+类型分开）"""
        cuda_codes_dir = self.output_dir / "cuda_codes"
        cuda_codes_dir.mkdir(exist_ok=True)
        
        # 按模型名称创建子目录
        model_dir = cuda_codes_dir / self.model_name
        model_dir.mkdir(exist_ok=True)
        
        # 按seed创建子目录（如果提供）
        if self.seed is not None:
            run_dir = model_dir / f"seed{self.seed}"
        else:
            run_dir = model_dir / "default"
        run_dir.mkdir(exist_ok=True)
        
        for result in results:
            if result.get("success") and result.get("cuda_code"):
                sample_id = result.get("id", result.get("unique_key", "unknown"))
                type_name = result.get("type", "unknown")
                
                # 按类型创建子目录
                type_dir = run_dir / type_name
                type_dir.mkdir(exist_ok=True)
                
                cu_file = type_dir / f"{sample_id}.cu"
                try:
                    with open(cu_file, 'w', encoding='utf-8') as f:
                        f.write(result["cuda_code"])
                except Exception as e:
                    print(f"警告: 保存 {cu_file} 失败: {e}")
        
        print(f"\nCUDA代码文件已保存到: {run_dir} (按类型分开)")
    
    def _update_global_index(self, result_file: Path):
        """更新全局索引"""
        index_file = self.output_dir / "index.json"
        
        if index_file.exists():
            with open(index_file, 'r', encoding='utf-8') as f:
                index_data = json.load(f)
        else:
            index_data = {
                "total_runs": 0,
                "model_runs": {},
                "history": []
            }
        
        result_info = {
            "file": result_file.name,
            "model_name": self.model_name,
            "timestamp": datetime.now().isoformat(),
            "path": str(result_file.relative_to(self.output_dir))
        }
        
        index_data["total_runs"] += 1
        index_data["history"].append(result_info)
        
        if self.model_name not in index_data["model_runs"]:
            index_data["model_runs"][self.model_name] = []
        index_data["model_runs"][self.model_name].append(result_info)
        
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, indent=2, ensure_ascii=False)
    
    def batch_generate(self):
        """批量翻译所有样本"""
        all_samples = self.load_samples()
        if not all_samples:
            return
        
        print(f"\n模型: {self.model_name}")
        print(f"输入目录: {self.input_dir}")
        print(f"输出目录: {self.output_dir}")
        print(f"最大重试次数: {self.max_retries}")
        
        checkpoint = self._load_checkpoint()
        results = []
        processed_keys = set()
        
        if checkpoint:
            results = checkpoint["results"]
            processed_keys = {r["unique_key"] for r in results if r.get("success")}
            print(f"检测到断点，已处理 {len(results)} 个文件，继续从第 {len(results)+1} 个开始")
        
        print("-" * 60)
        
        remaining_samples = [s for s in all_samples if s["unique_key"] not in processed_keys]
        total_samples = len(all_samples)
        
        done_samples = 0
        for i, sample in enumerate(remaining_samples, len(results) + 1):
            if self.max_samples is not None and self.max_samples > 0 and done_samples >= self.max_samples:
                break
            done_samples += 1
            
            sample_id = sample['id']
            type_name = sample['type']
            code_path = sample['code_path']
            
            print(f"[{i}/{total_samples}] {type_name} :: {sample_id}")
            
            code, error = self.extract_code_with_retry(code_path, max_retries=3)
            
            if error:
                print(f"   [FAIL] 代码提取失败: {error}")
                results.append({
                    'unique_key': sample['unique_key'],
                    'id': sample_id,
                    'type': type_name,
                    'c_file': sample.get('c_file', ''),
                    'code_path': code_path,
                    'cuda_code': '',
                    'success': False,
                    'error': error
                })
                continue
            
            gen_result = self.generate_cuda_code(code, sample_id, type_name)
            gen_result.update({
                'unique_key': sample['unique_key'],
                'id': sample_id,
                'type': type_name,
                'c_file': sample.get('c_file', ''),
                'code_path': code_path
            })
            
            if gen_result["success"]:
                print(f"   [OK] 生成成功 (tokens: {gen_result['output_tokens']})")
            else:
                print(f"   [FAIL] {gen_result.get('error', '未知错误')}")
            
            results.append(gen_result)
            
            if i % 5 == 0:
                self._save_checkpoint(results, total_samples)
                print(f"   已保存checkpoint，累计处理 {i} 个文件")
        
        if results:
            self._save_checkpoint(results, total_samples)
            result_file = self._save_results_with_timestamp(results)
            print(f"\n结果已保存到: {result_file}")
        
        self._print_stats(results)
        
        return results
    
    def _print_stats(self, results: list):
        """打印统计信息"""
        print("\n" + "=" * 60)
        print("代码生成统计")
        print("=" * 60)
        
        success = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]
        
        print(f"总计: {len(results)} 个文件")
        print(f"成功: {len(success)} 个")
        print(f"失败: {len(failed)} 个")
        
        if success:
            type_stats = {}
            for r in success:
                t = r.get('type', 'unknown')
                type_stats[t] = type_stats.get(t, 0) + 1
            
            print("\n成功按类型分布:")
            for t, cnt in sorted(type_stats.items()):
                print(f"   {t}: {cnt}")
        
        total_cost = sum(r.get("cost", 0) for r in success)
        total_input_tokens = sum(r.get("input_tokens", 0) for r in success)
        total_output_tokens = sum(r.get("output_tokens", 0) for r in success)
        
        print(f"\nLLM统计:")
        print(f"   总输入tokens: {total_input_tokens}")
        print(f"   总输出tokens: {total_output_tokens}")
        print(f"   总费用: ${total_cost:.6f}")


def main():
    parser = argparse.ArgumentParser(description="将C代码翻译为CUDA代码")
    parser.add_argument("--model", default="qwen3-coder-next", help="LLM模型名称")
    parser.add_argument("--org", default="ali", help="组织/平台名称")
    parser.add_argument("--input_dir", default="./selected_samples", help="输入目录（包含selected_samples.json）")
    parser.add_argument("--parashift_dir", default="./code_datas", help="配对程序源文件目录")
    parser.add_argument("--max_retries", type=int, default=3, help="最大重试次数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，用于区分不同批次的结果，避免覆盖")
    parser.add_argument("--max_samples", type=int, default=1, help="最大处理样本数")
    
    args = parser.parse_args()
    
    translator = CToCudaTranslator(
        model_name=args.model,
        org=args.org,
        parashift_dir=args.parashift_dir,
        input_dir=args.input_dir,
        max_retries=args.max_retries,
        seed=args.seed,
        max_samples=args.max_samples
    )
    
    print(f"[INFO] C到CUDA代码翻译工具")
    print(f"[INFO] 模型: {args.model}")
    print(f"[INFO] 平台: {args.org}")
    if args.seed is not None:
        print(f"[INFO] Seed: {args.seed}")
    
    translator.batch_generate()


if __name__ == "__main__":
    main()
