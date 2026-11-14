"""
对多仓库提取的CUDA kernel进行循环模式标注
支持按模型名称和时间戳保存结果，便于对比分析
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from llm_utils import OpenAILLM

from core_algorithm.feature_extraction.classify_13_dwarf_prompt import (
    Dwarf13CudaPrompt,
)


class MultiRepoKernelClassifier:
    def __init__(self, model_name="qwen3-coder-next", org="ali", input_dir=None):
        self.model_name = model_name
        self.org = org
        
        if input_dir:
            self.input_dir = Path(input_dir)
        else:
            self.input_dir = Path("./code_datas")
        
        self.output_dir = Path("./label_par_results")
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.llm = OpenAILLM(
            model_name=model_name,
            org=org,
            log_path="./logs",
            verbose=False,
            temperature=0.1,
            top_p=0.95
        )
        
        self.cuda_system_prompt = Dwarf13CudaPrompt.get_system_prompt()
        self.checkpoint_file = self.checkpoint_dir / f"{model_name}_checkpoint.json"
    
    def load_all_kernels(self):
        """从所有仓库加载kernel"""
        kernels = []
        
        if not self.input_dir.exists():
            print(f"错误: 输入目录不存在 {self.input_dir}")
            return kernels
        
        repo_dirs = [d for d in self.input_dir.iterdir() if d.is_dir()]
        print(f"找到 {len(repo_dirs)} 个仓库")
        
        for repo_dir in repo_dirs:
            index_file = repo_dir / "index.json"
            if not index_file.exists():
                continue
            
            try:
                with open(index_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                repo_name = repo_dir.name
                github_url = data.get("github_url", "")
                
                for file_entry in data.get("files", []):
                    kernel_file = repo_dir / file_entry.get("kernel_file")
                    if kernel_file.exists():
                        with open(kernel_file, 'r', encoding='utf-8') as f:
                            kernel_data = json.load(f)
                            
                            for kernel in kernel_data.get("kernels", []):
                                kernels.append({
                                    "repo_name": repo_name,
                                    "github_url": github_url,
                                    "source_file": kernel_data.get("source_file", ""),
                                    "folder_name": kernel_data.get("folder_name", ""),
                                    "kernel_name": kernel.get("name", "unknown"),
                                    "full_code": kernel.get("full_code", ""),
                                    "unique_key": f"{repo_name}_{kernel_data.get('source_file', '')}_{kernel.get('name', 'unknown')}"
                                })
            except Exception as e:
                print(f"读取仓库 {repo_dir.name} 失败: {e}")
        
        print(f"总共加载 {len(kernels)} 个kernel")
        return kernels
    
    def classify_single_kernel(self, kernel: dict) -> dict:
        """分类单个kernel"""
        try:
            kernel_code = kernel["full_code"]
            kernel_name = kernel["kernel_name"]
            
            if len(kernel_code) > 3500:
                kernel_code = kernel_code[:3500] + "\n// ... [代码截断]"
            
            user_prompt = Dwarf13CudaPrompt.get_user_prompt(kernel_code)
            
            messages = [
                {"role": "system", "content": self.cuda_system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            response, input_tokens, output_tokens, cost = self.llm.chat(messages, stream=False)
            
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:-3].strip()
            elif response.startswith("```"):
                response = response[3:-3].strip()
            
            result = json.loads(response)
            
            patterns = []
            for key, value in result.items():
                if isinstance(value, float) and 0.0 <= value <= 1.0:
                    if value >= 0.3:
                        patterns.append(key)
            
            result["matched_patterns"] = patterns
            result["pattern_count"] = len(patterns)
            
            result.update({
                "kernel_name": kernel_name,
                "repo_name": kernel["repo_name"],
                "github_url": kernel["github_url"],
                "source_file": kernel["source_file"],
                "folder_name": kernel["folder_name"],
                "unique_key": kernel["unique_key"],
                "success": True,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost": cost
            })
            
            return result
            
        except Exception as e:
            return {
                "kernel_name": kernel["kernel_name"],
                "repo_name": kernel["repo_name"],
                "github_url": kernel["github_url"],
                "source_file": kernel["source_file"],
                "folder_name": kernel["folder_name"],
                "unique_key": kernel["unique_key"],
                "success": False,
                "error": str(e)
            }
    
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
    
    def _save_checkpoint(self, results: list, total_kernels: int):
        """保存断点"""
        checkpoint_data = {
            "results": results,
            "total_kernels": total_kernels,
            "processed_count": len(results),
            "model_name": self.model_name,
            "timestamp": datetime.now().isoformat()
        }
        with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
    
    def _save_results_with_timestamp(self, results: list):
        """按时间戳保存结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.model_name}_{timestamp}.json"
        filepath = self.output_dir / filename
        
        result_data = {
            "model_name": self.model_name,
            "org": self.org,
            "timestamp": datetime.now().isoformat(),
            "total_kernels": len(results),
            "success_count": sum(1 for r in results if r["success"]),
            "fail_count": sum(1 for r in results if not r["success"]),
            "results": results
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        
        self._update_global_index(filepath)
        
        return filepath
    
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
        
        result_data = {
            "file": result_file.name,
            "model_name": self.model_name,
            "timestamp": datetime.now().isoformat(),
            "path": str(result_file.relative_to(self.output_dir))
        }
        
        index_data["total_runs"] += 1
        index_data["history"].append(result_data)
        
        if self.model_name not in index_data["model_runs"]:
            index_data["model_runs"][self.model_name] = []
        index_data["model_runs"][self.model_name].append(result_data)
        
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, indent=2, ensure_ascii=False)
    
    def batch_classify(self):
        """批量分类所有kernel"""
        all_kernels = self.load_all_kernels()
        if not all_kernels:
            return
        
        print(f"\n模型: {self.model_name}")
        print(f"输入目录: {self.input_dir}")
        print(f"输出目录: {self.output_dir}")
        
        checkpoint = self._load_checkpoint()
        results = []
        processed_keys = set()
        
        if checkpoint:
            results = checkpoint["results"]
            processed_keys = {r["unique_key"] for r in results if r.get("success")}
            print(f"检测到断点，已处理 {len(results)} 个kernel，继续从第 {len(results)+1} 个开始")
        
        print("-" * 60)
        
        remaining_kernels = [k for k in all_kernels if k["unique_key"] not in processed_keys]
        total_kernels = len(all_kernels)
        
        for i, kernel in enumerate(remaining_kernels, len(results) + 1):
            repo_name = kernel["repo_name"]
            kernel_name = kernel["kernel_name"]
            source_file = kernel["source_file"]
            
            print(f"[{i}/{total_kernels}] {repo_name} :: {source_file} :: {kernel_name}")
            
            result = self.classify_single_kernel(kernel)
            
            if result["success"]:
                matched = result.get("matched_patterns", [])
                count = result.get("pattern_count", 0)
                patterns_str = ", ".join(matched)
                print(f"   [OK] 匹配{count}个模式: {patterns_str}")
            else:
                print(f"   [FAIL] 失败: {result.get('error', '未知错误')}")
            
            results.append(result)
            
            if i % 10 == 0:
                self._save_checkpoint(results, total_kernels)
                print(f"   已保存checkpoint，累计处理 {i} 个kernel")
        
        if results:
            self._save_checkpoint(results, total_kernels)
            result_file = self._save_results_with_timestamp(results)
            print(f"\n结果已保存到: {result_file}")
        
        self._print_stats(results)
        
        return results
    
    def _print_stats(self, results: list):
        """打印统计信息"""
        print("\n" + "=" * 60)
        print("分类统计")
        print("=" * 60)
        
        success = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]
        
        print(f"总计: {len(results)} 个kernel")
        print(f"成功: {len(success)} 个")
        print(f"失败: {len(failed)} 个")
        
        if success:
            pattern_counts = {}
            for r in success:
                for p in r.get("matched_patterns", []):
                    pattern_counts[p] = pattern_counts.get(p, 0) + 1
            
            print("\n最常见的模式Top5：")
            for p, cnt in sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"   {p}: {cnt}次")
        
        total_cost = sum(r.get("cost", 0) for r in success)
        total_input_tokens = sum(r.get("input_tokens", 0) for r in success)
        total_output_tokens = sum(r.get("output_tokens", 0) for r in success)
        
        print(f"\nLLM统计:")
        print(f"   总输入tokens: {total_input_tokens}")
        print(f"   总输出tokens: {total_output_tokens}")
        print(f"   总费用: ${total_cost:.6f}")


def main():
    parser = argparse.ArgumentParser(description="对多仓库kernel进行循环模式标注")
    parser.add_argument("--model", default="qwen3-coder-next", help="LLM模型名称")
    parser.add_argument("--org", default="ali", help="组织/平台名称")
    parser.add_argument("--input_dir", default=None, help="输入目录（多仓库提取结果）")
    
    args = parser.parse_args()
    
    classifier = MultiRepoKernelClassifier(
        model_name=args.model,
        org=args.org,
        input_dir=args.input_dir
    )
    
    print(f"🔬 Kernel循环模式标注工具")
    print(f"模型: {args.model}")
    print(f"平台: {args.org}")
    
    classifier.batch_classify()


if __name__ == "__main__":
    main()
