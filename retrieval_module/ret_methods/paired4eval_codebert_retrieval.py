# ============================================================
# retrieval_module/codebert_retrieval.py
#
# CodeBERT 跨语言检索 - 统一入口
#
# 包含三个类：
#   CodeBERTRetrieval  - 核心：模型加载 + embedding + 相似度计算（共享）
#   PairedEvaluation   - 任务1：C-CUDA pair，有答案，用于评估
#   CUDAOnlyRetrieval  - 任务2：C → CUDA-only，无答案，实际检索
# ============================================================

import os
import csv
import json
import random
import argparse
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from transformers import RobertaModel, RobertaTokenizer
from scipy.spatial.distance import cosine, euclidean
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from data_module.data_loader import DataLoader
from data_module.cuda_only_data_loader import CUDAOnlyDataLoader


# ============================================================
# 工具函数
# ============================================================
def set_seed(seed: int = 42):
    """设置随机种子确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================================
# 核心类：模型加载 + embedding + 相似度计算
# ============================================================
class CodeBERTRetrieval:
    """
    CodeBERT 检索核心。
    负责：
      - 加载模型和 tokenizer
      - 生成代码 embedding
      - 计算相似度（cosine / euclidean / both）
      - 通用检索接口（排序 + 保存结果）
    """

    def __init__(
        self,
        model_path: str = "./models/codebert",
        similarity_metric: str = "cosine",
        device: Optional[str] = None,
        seed: int = 42,
        obfuscate_code: bool = False
    ):
        set_seed(seed)
        self.seed = seed
        self.obfuscate_code = obfuscate_code
        self.similarity_metric = similarity_metric.lower()

        if self.similarity_metric not in ["cosine", "euclidean", "both"]:
            raise ValueError("similarity_metric must be 'cosine', 'euclidean', or 'both'")

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if not Path(model_path).exists():
            raise ValueError(f"Model path not found: {model_path}")

        self.model_path = model_path
        self.tokenizer = RobertaTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = RobertaModel.from_pretrained(model_path, local_files_only=True)
        self.model.to(self.device)
        self.model.eval()

        print(f"{'='*60}")
        print(f"CodeBERT 跨语言检索")
        print(f"  Model path        : {model_path}")
        print(f"  Device            : {self.device}")
        print(f"  Similarity metric : {similarity_metric}")
        print(f"  Seed              : {seed}")
        print(f"  Obfuscate code    : {obfuscate_code}")
        print(f"{'='*60}\n")

    # ----------------------------------------------------------
    # Embedding
    # ----------------------------------------------------------
    def _get_embedding(self, code: str) -> np.ndarray:
        if not code.strip():
            return np.zeros(self.model.config.hidden_size)

        inputs = self.tokenizer(
            code,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        return outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()

    # ----------------------------------------------------------
    # 相似度计算
    # ----------------------------------------------------------
    def _calculate_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> Dict[str, float]:
        results = {}
        if self.similarity_metric in ["cosine", "both"]:
            results["cosine_similarity"] = float(1 - cosine(emb1, emb2))
        if self.similarity_metric in ["euclidean", "both"]:
            results["euclidean_distance"] = float(euclidean(emb1, emb2))
        return results

    def _sort_key(self) -> str:
        """返回排序用的 key 字符串"""
        if self.similarity_metric in ["cosine", "both"]:
            return "cosine_similarity"
        return "euclidean_distance"

    def _sort_rankings(self, rankings: list) -> list:
        key = self._sort_key()
        reverse = key == "cosine_similarity"
        return sorted(rankings, key=lambda x: x["similarity"][key], reverse=reverse)

    # ----------------------------------------------------------
    # 通用检索
    # ----------------------------------------------------------
    def retrieval(
        self,
        query_loops: List[Dict[str, Any]],
        corpus_kernels: List[Dict[str, Any]],
        output_dir: str,
        output_tag: str = "",
        save_json: bool = False,
    ) -> Dict[str, Any]:
        """
        通用检索接口：对每个 query loop 在 corpus_kernels 中排序。

        Args:
            query_loops    : 查询的 C loops（含 source_code / id 等字段）
            corpus_kernels : 检索目标 CUDA kernels
            output_dir     : 输出目录
            output_tag     : 输出文件名标识（区分 paired / cuda_only）
            save_json      : 是否保存完整 JSON（默认关闭）
        """
        # 生成 kernel embeddings
        print("生成 kernel embeddings...")
        kernel_embeddings = []
        for kernel in tqdm(corpus_kernels, desc="Kernels", unit="kernel"):
            kernel_embeddings.append({
                "kernel": kernel,
                "embedding": self._get_embedding(kernel.get("source_code", ""))
            })

        # 生成 loop embeddings
        print("\n生成 loop embeddings...")
        loop_embeddings = []
        for loop in tqdm(query_loops, desc="Loops", unit="loop"):
            loop_embeddings.append({
                "loop": loop,
                "embedding": self._get_embedding(loop.get("source_code", ""))
            })

        # 计算相似度并排序
        results = {
            "config": {
                "model_path": self.model_path,
                "similarity_metric": self.similarity_metric,
                "device": self.device,
                "seed": self.seed,
                "obfuscate_code": self.obfuscate_code,
                "num_kernels": len(corpus_kernels),
                "num_loops": len(query_loops)
            },
            "retrieval_results": []
        }

        print("\n计算相似度并排序...")
        for le in tqdm(loop_embeddings, desc="Retrieving", unit="loop"):
            loop = le["loop"]
            rankings = []
            for ke in kernel_embeddings:
                sim = self._calculate_similarity(le["embedding"], ke["embedding"])
                rankings.append({
                    "kernel_idx": ke["kernel"]["id"],
                    "kernel_name": ke["kernel"].get("kernel_name", ""),
                    "kernel_filename": ke["kernel"].get("filename", ""),
                    "similarity": sim
                })

            rankings = self._sort_rankings(rankings)
            for rank, entry in enumerate(rankings, 1):
                entry["rank"] = rank

            results["retrieval_results"].append({
                "loop_idx": loop["id"],
                "loop_info": {
                    "project": loop.get("project_name", ""),
                    "filename": loop.get("filename", ""),
                    "loop_type": loop.get("loop_type", ""),
                },
                "rankings": rankings
            })

        self._save_results(results, query_loops, output_dir, output_tag, save_json)
        return results

    def _save_results(self, results, query_loops, output_dir, output_tag, save_json: bool = False):
        """保存 CSV（以及可选的 JSON / query_loops.json）"""
        actual_dir = os.path.join(output_dir, "codebert_output")
        os.makedirs(actual_dir, exist_ok=True)
        obf_tag = '_obf' if self.obfuscate_code else ''
        sort_key_str = self._sort_key()
        base = f"codebert_{output_tag}_{sort_key_str}_seed{self.seed}{obf_tag}"

        # CSV
        csv_path = os.path.join(actual_dir, f"{base}.csv")
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                "loop_idx", "loop_project", "loop_filename", "loop_type",
                "rank", "kernel_idx", "kernel_name", "kernel_filename",
                sort_key_str
            ])
            for lr in results["retrieval_results"]:
                li = lr["loop_info"]
                for entry in lr["rankings"]:
                    sim_val = entry["similarity"].get(sort_key_str)
                    writer.writerow([
                        lr["loop_idx"], li["project"], li["filename"], li["loop_type"],
                        entry["rank"], entry["kernel_idx"],
                        entry["kernel_name"], entry["kernel_filename"],
                        f"{sim_val:.6f}"
                    ])
        print(f"CSV 已保存: {csv_path}")

        # JSON（默认关闭，数据量大时很慢；需要 debug 时用 --save_json 开启）
        if save_json:
            json_path = os.path.join(actual_dir, f"{base}.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"JSON 已保存: {json_path}")

        # query loops 信息
        samples_path = os.path.join(actual_dir, f"query_loops_seed{self.seed}.json")
        loop_samples = [
            {
                "id": loop["id"],
                "project": loop.get("project_name", ""),
                "filename": loop.get("filename", ""),
                "loop_type": loop.get("loop_type", ""),
                "code_preview": loop["source_code"][:200] + "..."
            }
            for loop in query_loops
        ]
        with open(samples_path, 'w', encoding='utf-8') as f:
            json.dump(loop_samples, f, indent=2, ensure_ascii=False)
        print(f"Query loops 已保存: {samples_path}")


# ============================================================
# 任务1：有答案 —— C-CUDA pair 评估
# ============================================================
class PairedEvaluation:
    """
    任务1：C-CUDA pair 数据集（有答案），用于评估检索性能。

    数据来源：配对 C–CUDA 语料（C loops + 配对的 CUDA kernels）
    流程：加载全量数据 → 全量 C loops 检索全量 CUDA kernels → 保存结果
    """

    def __init__(self, retrieval: CodeBERTRetrieval, data_dir: str):
        self.retrieval = retrieval
        self.data_dir = data_dir

    def _load_data(self) -> tuple:
        loader = DataLoader(self.data_dir, obfuscate_code=self.retrieval.obfuscate_code)
        if not loader.load_cache():
            loader.load()
            loader.save_cache()

        query_loops = [
            {
                "id": c['id'],
                "project_name": c['project'],
                "filename": c['filename'],
                "source_code": c['code'],
                "loop_type": c['loop_type'],
                "parent_function": c['parent_function'],
                "start_line": c['start_line'],
                "end_line": c['end_line'],
                "pair_kernel_ids": c['pair_kernel_ids'],
            }
            for c in loader.c_codes
        ]

        corpus_kernels = [
            {
                "id": k['id'],
                "project_name": k['project'],
                "filename": k['filename'],
                "source_code": k['code'],
                "kernel_name": k['kernel_name'],
                "start_line": k['start_line'],
                "end_line": k['end_line'],
                "pair_loop_ids": k['pair_loop_ids'],
            }
            for k in loader.cuda_codes
        ]

        print(f"Loaded {len(corpus_kernels)} kernels, {len(query_loops)} loops from paired corpus")
        return query_loops, corpus_kernels

    def run(self, output_dir: str, save_json: bool = False) -> Dict[str, Any]:
        query_loops, corpus_kernels = self._load_data()
        return self.retrieval.retrieval(
            query_loops, corpus_kernels, output_dir,
            output_tag="paired", save_json=save_json
        )


# ============================================================
# 命令行入口
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CodeBERT 跨语言检索")
    parser.add_argument('--model_path', default='./models/codebert')
    parser.add_argument('--parashift_dir', default='./code_datas')
    parser.add_argument('--cuda_only_dir', default='./CUDA-only')
    parser.add_argument('--output_dir', default='./res_rag')
    parser.add_argument('--similarity_metric', default='cosine',
                        choices=['cosine', 'euclidean', 'both'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--obfuscate', action='store_true')
    parser.add_argument('--num_loops', type=int, default=10,
                        help='随机选取 loop 数（cuda_only 任务）')
    parser.add_argument('--loop_ids_file', default=None,
                        help='指定 loop IDs 的 JSON 文件（cuda_only 任务）')
    parser.add_argument('--save_json', action='store_true',
                        help='保存完整 JSON 结果（默认关闭）')
    args = parser.parse_args()

    core = CodeBERTRetrieval(
        model_path=args.model_path,
        similarity_metric=args.similarity_metric,
        seed=args.seed,
        obfuscate_code=args.obfuscate
    )

    
    task = PairedEvaluation(core, args.parashift_dir)
    task.run(args.output_dir, save_json=args.save_json)