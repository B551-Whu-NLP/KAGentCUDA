"""
对多仓库提取的C代码循环进行模式标注
支持按模型名称和时间戳保存结果，便于对比分析

参考: label_kernels_via_llm.py
数据来源: code_datas/type-*/c_*/c/c_*.c
"""

import os
import sys
import json
import argparse
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))
from llm_utils import OpenAILLM

from core_algorithm.feature_extraction.classify_13_dwarf_prompt import (
    Dwarf13CPrompt,
)


class CLoopClassifier:
    def __init__(self, model_name="qwen3-coder-next", org="ali", input_dir=None):
        self.model_name = model_name
        self.org = org
        
        if input_dir:
            self.input_dir = Path(input_dir)
        else:
            self.input_dir = Path("./code_datas/")
        
        self.output_dir = Path("./label_loop_results")
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
        
        self.c_system_prompt = Dwarf13CPrompt.get_system_prompt()
        self.checkpoint_file = self.checkpoint_dir / f"{model_name}_loop_checkpoint.json"
    
    def extract_loops_from_c_code(self, code: str) -> List[Dict]:
        """
        从C代码中提取所有for循环
        
        Returns:
            List[Dict]: 每个循环包含 {'loop_code': str, 'loop_index': int, 'start_line': int}
        """
        loops = []
        lines = code.split('\n')
        
        # 简单的循环提取：匹配for关键字，找到对应的{}块
        for_pattern = r'^\s*for\s*\('
        
        loop_stack = []
        current_loop_lines = []
        in_loop = False
        brace_count = 0
        
        for line_idx, line in enumerate(lines, 1):
            # 检查是否是for循环开始
            if re.match(for_pattern, line):
                if in_loop:
                    # 嵌套循环，压栈
                    loop_stack.append((current_loop_lines, brace_count))
                current_loop_lines = [line]
                in_loop = True
                brace_count = line.count('{') - line.count('}')
                
            elif in_loop:
                current_loop_lines.append(line)
                brace_count += line.count('{') - line.count('}')
                
                # 检查循环是否结束（括号匹配）
                if brace_count <= 0 and loop_stack:
                    # 弹出嵌套
                    saved = loop_stack.pop()
                    current_loop_lines = saved[0] + current_loop_lines
                    brace_count = saved[1]
                elif brace_count <= 0 and not loop_stack:
                    # 循环结束
                    loop_code = '\n'.join(current_loop_lines)
                    loops.append({
                        'loop_code': loop_code.strip(),
                        'loop_index': len(loops),
                        'start_line': line_idx - len(current_loop_lines) + 1
                    })
                    in_loop = False
                    current_loop_lines = []
        
        return loops
    
    def load_all_c_files(self):
        """从所有type-*目录加载所有C代码文件"""
        files_data = []
        
        if not self.input_dir.exists():
            print(f"错误: 输入目录不存在 {self.input_dir}")
            return files_data
        
        # 找到所有 type-* 目录
        type_dirs = [d for d in self.input_dir.iterdir() if d.is_dir() and d.name.startswith('type-')]
        print(f"找到 {len(type_dirs)} 个type目录: {[d.name for d in type_dirs]}")
        
        total_c_dirs = 0
        
        for type_dir in type_dirs:
            # 在每个type目录下找 c_* 子目录
            c_subdirs = [d for d in type_dir.iterdir() if d.is_dir() and d.name.startswith('c_')]
            total_c_dirs += len(c_subdirs)
            
            for subdir in c_subdirs:
                c_dir = subdir / 'c'
                if not c_dir.exists():
                    continue
                
                c_files = list(c_dir.glob('*.c'))
                if not c_files:
                    continue
                
                c_file = c_files[0]
                file_id = subdir.name
                
                try:
                    with open(c_file, 'r', encoding='utf-8', errors='ignore') as f:
                        code = f.read()
                    
                    metadata_file = subdir / 'metadata.json'
                    metadata = {}
                    if metadata_file.exists():
                        with open(metadata_file, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                    
                    files_data.append({
                        'file_id': file_id,
                        'file_path': str(c_file),
                        'type': type_dir.name,
                        'code': code,
                        'metadata': metadata
                    })
                    
                except Exception as e:
                    print(f"读取文件 {c_file} 失败: {e}")
        
        print(f"总共找到 {total_c_dirs} 个C代码目录")
        print(f"总共加载 {len(files_data)} 个C文件")
        return files_data
    
    def classify_single_loop(self, loop: Dict, file_info: Dict) -> dict:
        """分类单个循环"""
        try:
            loop_code = loop["loop_code"]
            loop_index = loop["loop_index"]
            start_line = loop["start_line"]
            file_id = file_info["file_id"]
            
            if len(loop_code) > 3500:
                loop_code = loop_code[:3500] + "\n// ... [代码截断]"
            
            user_prompt = Dwarf13CPrompt.get_user_prompt(loop_code)
            
            messages = [
                {"role": "system", "content": self.c_system_prompt},
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
                "loop_index": loop_index,
                "start_line": start_line,
                "file_id": file_id,
                "file_path": file_info["file_path"],
                "loop_code": loop_code,
                "unique_key": f"{file_id}_loop_{loop_index}",
                "success": True,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost": cost
            })
            
            return result
            
        except Exception as e:
            return {
                "loop_index": loop.get("loop_index", -1),
                "start_line": loop.get("start_line", -1),
                "file_id": file_info.get("file_id", "unknown"),
                "file_path": file_info.get("file_path", "unknown"),
                "unique_key": f"{file_info.get('file_id', 'unknown')}_loop_{loop.get('loop_index', -1)}",
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
    
    def _save_checkpoint(self, results: list, total_loops: int):
        """保存断点"""
        checkpoint_data = {
            "results": results,
            "total_loops": total_loops,
            "processed_count": len(results),
            "model_name": self.model_name,
            "timestamp": datetime.now().isoformat()
        }
        with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
    
    def _save_results_with_timestamp(self, results: list):
        """按时间戳保存结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.model_name}_loops_{timestamp}.json"
        filepath = self.output_dir / filename
        
        result_data = {
            "model_name": self.model_name,
            "org": self.org,
            "timestamp": datetime.now().isoformat(),
            "total_loops": len(results),
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
        """批量分类所有循环"""
        all_files = self.load_all_c_files()
        if not all_files:
            return
        
        print(f"\n模型: {self.model_name}")
        print(f"输入目录: {self.input_dir}")
        print(f"输出目录: {self.output_dir}")
        
        # 收集所有循环
        all_loops = []
        for file_info in all_files:
            loops = self.extract_loops_from_c_code(file_info['code'])
            for loop in loops:
                loop['unique_key'] = f"{file_info['file_id']}_loop_{loop['loop_index']}"
                all_loops.append({
                    'loop': loop,
                    'file_info': file_info
                })
        
        print(f"总共找到 {len(all_loops)} 个循环")
        
        checkpoint = self._load_checkpoint()
        results = []
        processed_keys = set()
        
        if checkpoint:
            results = checkpoint["results"]
            processed_keys = {r["unique_key"] for r in results if r.get("success")}
            print(f"检测到断点，已处理 {len(results)} 个循环，继续从第 {len(results)+1} 个开始")
        
        print("-" * 60)
        
        remaining_loops = [item for item in all_loops if item['loop']['unique_key'] not in processed_keys]
        total_loops = len(all_loops)
        
        for i, item in enumerate(remaining_loops, len(results) + 1):
            loop = item['loop']
            file_info = item['file_info']
            
            file_id = file_info['file_id']
            loop_index = loop['loop_index']
            start_line = loop['start_line']
            
            print(f"[{i}/{total_loops}] {file_id} :: loop_{loop_index} :: line {start_line}")
            
            result = self.classify_single_loop(loop, file_info)
            
            if result["success"]:
                matched = result.get("matched_patterns", [])
                count = result.get("pattern_count", 0)
                patterns_str = ", ".join(matched)
                print(f"   [OK] 匹配{count}个模式: {patterns_str}")
            else:
                print(f"   [FAIL] 失败: {result.get('error', '未知错误')}")
            
            results.append(result)
            
            if i % 10 == 0:
                self._save_checkpoint(results, total_loops)
                print(f"   已保存checkpoint，累计处理 {i} 个循环")
        
        if results:
            self._save_checkpoint(results, total_loops)
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
        
        print(f"总计: {len(results)} 个循环")
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
    parser = argparse.ArgumentParser(description="对C代码循环进行模式标注")
    parser.add_argument("--model", default="qwen3-coder-next", help="LLM模型名称")
    parser.add_argument("--org", default="ali", help="组织/平台名称")
    parser.add_argument("--input_dir", default=None, help="输入目录（配对语料根目录，包含所有type-*）")
    
    args = parser.parse_args()
    
    classifier = CLoopClassifier(
        model_name=args.model,
        org=args.org,
        input_dir=args.input_dir
    )
    
    print(f"🔬 C代码循环模式标注工具")
    print(f"模型: {args.model}")
    print(f"平台: {args.org}")
    
    classifier.batch_classify()


if __name__ == "__main__":
    main()
