# ============================================================
# retrieval_module/procrustes_retrieval.py
#
# Procrustes 跨语言检索 - 统一入口
#
# 包含三个类：
#   ProcrustesRetrieval   - 核心：训练 + 检索逻辑（共享）
#   PairedEvaluation      - 任务1：C-CUDA pair，有答案，用于评估
#   CUDAOnlyRetrieval     - 任务2：C → CUDA-only，无答案，实际检索
#
# 特征: tree-sitter AST 结构特征
# Procrustes 训练: Z-score 标准化
# ============================================================

import os
import csv
import json
import pickle
import random
import argparse
import hashlib
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from data_module.data_loader import DataLoader, RANKING_REPRESENTATION, obfuscate_ranking_code
from data_module.cuda_only_data_loader import CUDAOnlyDataLoader
from retrieval_module.features.c_loop_feature_class import CLoopFeatureExtractor
from retrieval_module.features.cuda_kernel_feature_class import CudaKernelFeatureExtractor

# ============================================================
# 常量
# ============================================================
FEATURE_DIM = 12
FEATURE_NAMES = [
    'indirect', 'coalesced_consecutive', 'offset',
    'stencil', 'stencil_nb', 'self_prod',
    'fma', 'compute_int', 'rw_ratio',
    'strided', 'spatial_locality', 'reduction'
]


# ============================================================
# 工具函数
# ============================================================
def format_param(p):
    """格式化参数：浮点数的整数形式转为整数"""
    if isinstance(p, float) and p.is_integer():
        return int(p)
    return p


def set_seed(seed: int = 42):
    """设置随机种子确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)


# ============================================================
# 特征缓存
# ============================================================
def _feature_cache_path(data_dir: str, side: str, obfuscate: bool) -> Path:
    """
    缓存文件路径。
    side: 'c' 或 'cuda'
    obfuscate: 混淆与否使用不同缓存
    """
    obf_tag = '_obf' if obfuscate else ''
    return Path(data_dir) / f".feature_cache_{side}{obf_tag}.pkl"


def load_feature_cache(data_dir: str, side: str, obfuscate: bool) -> dict:
    """加载特征缓存，返回 {id: vector} dict，不存在则返回空 dict"""
    path = _feature_cache_path(data_dir, side, obfuscate)
    if path.exists():
        with open(path, 'rb') as f:
            cache = pickle.load(f)
        print(f"  特征缓存已加载: {path} ({len(cache)} 条)")
        return cache
    return {}


def save_feature_cache(data_dir: str, side: str, obfuscate: bool, cache: dict):
    """保存特征缓存到磁盘"""
    path = _feature_cache_path(data_dir, side, obfuscate)
    with open(path, 'wb') as f:
        pickle.dump(cache, f)
    print(f"  特征缓存已保存: {path} ({len(cache)} 条)")


def extract_ranking_vecs(codes: list, data_dir: str, side: str) -> dict:
    """Versioned source-bound ranking vectors, separate from historical feature caches."""
    if side not in ('c', 'cuda'):
        raise ValueError(f'Unknown ranking feature side: {side}')
    path = Path(data_dir) / f'.feature_cache_{side}_{RANKING_REPRESENTATION}.pkl'
    cache = {}
    if path.exists():
        with path.open('rb') as f:
            stored = pickle.load(f)
        if isinstance(stored, dict) and stored.get('version') == RANKING_REPRESENTATION:
            entries = stored.get('entries')
            if isinstance(entries, dict):
                cache = entries
    extractor = CLoopFeatureExtractor if side == 'c' else CudaKernelFeatureExtractor
    vectors = {}
    changed = False
    for item in tqdm(codes, desc=f'{side} ranking features'):
        code = item['code']
        digest = hashlib.sha256(code.encode('utf-8')).hexdigest()
        entry = cache.get(item['id'])
        if not isinstance(entry, tuple) or len(entry) != 2 or entry[0] != digest:
            entry = (digest, extractor(obfuscate_ranking_code(code)).extract().to_vector())
            cache[item['id']] = entry
            changed = True
        vectors[item['id']] = entry[1]
    if changed:
        with path.open('wb') as f:
            pickle.dump({'version': RANKING_REPRESENTATION, 'entries': cache}, f)
    return vectors


def extract_c_vecs(codes: list, data_dir: str, obfuscate: bool) -> dict:
    """
    提取 C loop 结构特征向量，自动读写缓存。

    Args:
        codes    : loader.c_codes 列表，每项含 'id' 和 'code'
        data_dir : 数据目录（缓存写在这里）
        obfuscate: 是否混淆

    Returns:
        {id: np.ndarray}
    """
    cache = load_feature_cache(data_dir, 'c', obfuscate)
    missing = [c for c in codes if c['id'] not in cache]

    if missing:
        print(f"  提取 C 特征（缓存缺失 {len(missing)}/{len(codes)} 条）...")
        for c in tqdm(missing, desc="C loops", unit="loop"):
            cache[c['id']] = CLoopFeatureExtractor(c['code']).extract().to_vector()
        save_feature_cache(data_dir, 'c', obfuscate, cache)
    else:
        print(f"  C 特征全部命中缓存 ({len(codes)} 条)")

    return cache


def extract_cuda_vecs(codes: list, data_dir: str, obfuscate: bool) -> dict:
    """
    提取 CUDA kernel 结构特征向量，自动读写缓存。

    Args:
        codes    : loader.cuda_codes 列表，每项含 'id' 和 'code'
        data_dir : 数据目录（缓存写在这里）
        obfuscate: 是否混淆

    Returns:
        {id: np.ndarray}
    """
    cache = load_feature_cache(data_dir, 'cuda', obfuscate)
    missing = [k for k in codes if k['id'] not in cache]

    if missing:
        print(f"  提取 CUDA 特征（缓存缺失 {len(missing)}/{len(codes)} 条）...")
        for k in tqdm(missing, desc="CUDA kernels", unit="kernel"):
            cache[k['id']] = CudaKernelFeatureExtractor(k['code']).extract().to_vector()
        save_feature_cache(data_dir, 'cuda', obfuscate, cache)
    else:
        print(f"  CUDA 特征全部命中缓存 ({len(codes)} 条)")

    return cache


# ============================================================
# 核心类：Procrustes 训练 + 检索
# ============================================================
class ProcrustesRetrieval:
    """
    Procrustes 跨语言检索核心。
    负责：
      - 用配对数据（C-CUDA pair）训练 Procrustes 变换矩阵（Z-score 标准化版）
      - 提供通用的相似度计算和检索接口
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        seed: int = 42,
        obfuscate_code: bool = False
    ):
        set_seed(seed)
        self.alpha = alpha
        self.beta = beta
        self.seed = seed
        self.obfuscate_code = obfuscate_code

        # 训练后填充
        self.mu_C = None
        self.std_C = None
        self.mu_K = None
        self.std_K = None
        self.W = None

        print(f"{'='*60}")
        print(f"Procrustes 跨语言检索")
        print(f"  Alpha (结构特征权重) : {alpha}")
        print(f"  Beta  (矮人特征权重) : {beta}")
        print(f"  Seed                 : {seed}")
        print(f"  Obfuscate code       : {obfuscate_code}")
        print(f"{'='*60}\n")

    # ----------------------------------------------------------
    # 训练
    # ----------------------------------------------------------
    def train_procrustes(self, V_C: np.ndarray, V_K: np.ndarray):
        """
        用配对数据训练 Procrustes 变换矩阵（Z-score 标准化版）。

        步骤：
          1. 对 C 侧和 CUDA 侧分别做 Z-score 标准化
          2. 计算跨协方差矩阵并 SVD 求正交变换 W
          3. 保存均值、标准差、W，供后续检索使用
        """
        print("\n训练 Procrustes W 矩阵（Z-score 标准化版）...")

        mu_C = V_C.mean(axis=0)
        mu_K = V_K.mean(axis=0)

        std_C = V_C.std(axis=0)
        std_K = V_K.std(axis=0)
        std_C[std_C < 1e-8] = 1.0
        std_K[std_K < 1e-8] = 1.0

        V_C_norm = (V_C - mu_C) / std_C
        V_K_norm = (V_K - mu_K) / std_K

        M = V_K_norm.T @ V_C_norm
        U, S, Vt = np.linalg.svd(M)
        W = U @ Vt

        residual = np.linalg.norm(
            V_K_norm @ W - V_C_norm, 'fro'
        ) / max(1e-10, np.linalg.norm(V_C_norm, 'fro'))
        print(f"残差: {residual:.4f}")

        self.mu_C = mu_C
        self.std_C = std_C
        self.mu_K = mu_K
        self.std_K = std_K
        self.W = W

    # ----------------------------------------------------------
    # 相似度计算
    # ----------------------------------------------------------
    def _map_kernel_vec(self, k_vec: np.ndarray) -> np.ndarray:
        """将 CUDA kernel 结构特征向量映射到 C 空间（Z-score + Procrustes）"""
        return ((k_vec - self.mu_K) / self.std_K) @ self.W

    def _map_loop_vec(self, c_vec: np.ndarray) -> np.ndarray:
        """对 C loop 结构特征向量做 Z-score 标准化（与映射后的 kernel 同坐标系）"""
        return (c_vec - self.mu_C) / self.std_C

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        """L2 归一化余弦相似度"""
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-10 or nb < 1e-10:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def _dwarf_sim(self, loop_dw, kernel_dw) -> float:
        """矮人特征余弦相似度"""
        a = np.array(loop_dw, dtype=np.float32)
        b = np.array(kernel_dw, dtype=np.float32)
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-10 or nb < 1e-10:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def _calculate_similarity(
        self,
        loop_mapped: np.ndarray,
        kernel_mapped: np.ndarray,
        loop_dw,
        kernel_dw
    ) -> dict:
        """
        加权融合相似度：
          overall = alpha * 结构特征余弦相似度 + beta * 矮人特征余弦相似度
        """
        struct_sim = self._cosine_sim(loop_mapped, kernel_mapped)
        dw_sim = self._dwarf_sim(loop_dw, kernel_dw)
        overall = self.alpha * struct_sim + self.beta * dw_sim
        return {
            "structure_similarity": struct_sim,
            "dwarf_similarity": dw_sim,
            "overall_similarity": overall
        }

    # ----------------------------------------------------------
    # 通用检索
    # ----------------------------------------------------------
    def retrieval(
        self,
        query_loops: list,
        corpus_kernels: list,
        output_dir: str,
        output_tag: str = "",
        c_vecs: dict = None,
        cuda_vecs: dict = None,
        save_json: bool = False,
    ) -> dict:
        """
        通用检索接口：对每个 query loop 在 corpus_kernels 中排序。

        Args:
            query_loops    : 查询的 C loops（含 source_code / dwarf_vector / id 等字段）
            corpus_kernels : 检索目标 CUDA kernels
            output_dir     : 输出目录
            output_tag     : 输出文件名标识（区分 paired / cuda_only）
            c_vecs         : 预计算的 C 特征向量 {id: vector}，None 则现场提取
            cuda_vecs      : 预计算的 CUDA 特征向量 {id: vector}，None 则现场提取
        """
        assert self.W is not None, "请先调用 train_procrustes() 训练模型"

        # 预计算 kernel 映射向量
        print("\n预计算 CUDA kernel 特征向量...")
        kernel_features = []
        for k in tqdm(corpus_kernels, desc="Kernels", unit="kernel"):
            raw_vec = cuda_vecs[k['id']] if (cuda_vecs and k['id'] in cuda_vecs) \
                else CudaKernelFeatureExtractor(k['source_code']).extract().to_vector()
            kernel_features.append({
                "kernel": k,
                "mapped_vec": self._map_kernel_vec(raw_vec),
                "dwarf_vec": k.get('dwarf_vector', [])
            })

        # 预计算 loop 映射向量
        print("\n预计算 C loop 特征向量...")
        loop_features = []
        for loop in tqdm(query_loops, desc="Loops", unit="loop"):
            raw_vec = c_vecs[loop['id']] if (c_vecs and loop['id'] in c_vecs) \
                else CLoopFeatureExtractor(loop['source_code']).extract().to_vector()
            loop_features.append({
                "loop": loop,
                "mapped_vec": self._map_loop_vec(raw_vec),
                "dwarf_vec": loop.get('dwarf_vector', [])
            })

        # 计算相似度并排序
        results = {
            "config": {
                "method": "procrustes",
                "alpha": self.alpha,
                "beta": self.beta,
                "seed": self.seed,
                "obfuscate_code": self.obfuscate_code,
                "num_kernels": len(corpus_kernels),
                "num_loops": len(query_loops)
            },
            "retrieval_results": []
        }

        print("\n计算相似度并排序...")
        for lf in tqdm(loop_features, desc="Retrieving", unit="loop"):
            loop = lf["loop"]
            rankings = []
            for kf in kernel_features:
                sim = self._calculate_similarity(
                    lf["mapped_vec"], kf["mapped_vec"],
                    lf["dwarf_vec"], kf["dwarf_vec"]
                )
                rankings.append({
                    "kernel_idx": kf["kernel"]["id"],
                    "kernel_name": kf["kernel"].get("kernel_name", ""),
                    "kernel_filename": kf["kernel"].get("filename", ""),
                    "similarity": sim
                })

            rankings.sort(key=lambda x: x["similarity"]["overall_similarity"], reverse=True)
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

        # 保存结果
        self._save_results(results, query_loops, output_dir, output_tag, save_json=save_json)
        return results

    def _save_results(self, results, query_loops, output_dir, output_tag, save_json: bool = False):
        """保存 CSV（以及可选的 JSON / query_loops.json）"""
        actual_dir = os.path.join(output_dir, "procrustes_output")
        os.makedirs(actual_dir, exist_ok=True)
        alpha_fmt = format_param(self.alpha)
        beta_fmt = format_param(self.beta)
        obf_tag = '_obf' if self.obfuscate_code else ''
        base = f"procrustes_{output_tag}_seed{self.seed}_a{alpha_fmt}_b{beta_fmt}{obf_tag}"

        # CSV
        csv_path = os.path.join(actual_dir, f"{base}.csv")
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                "loop_idx", "loop_project", "loop_filename", "loop_type",
                "rank", "kernel_idx", "kernel_name", "kernel_filename",
                "overall_similarity"
            ])
            for lr in results["retrieval_results"]:
                li = lr["loop_info"]
                for entry in lr["rankings"]:
                    writer.writerow([
                        lr["loop_idx"], li["project"], li["filename"], li["loop_type"],
                        entry["rank"], entry["kernel_idx"],
                        entry["kernel_name"], entry["kernel_filename"],
                        f"{entry['similarity']['overall_similarity']:.6f}"
                    ])
        print(f"CSV 已保存: {csv_path}")

        # JSON（默认关闭，数据量大时很慢；需要 debug 时传 save_json=True）
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
    流程：
      1. 加载配对数据，训练 Procrustes（特征带缓存）
      2. 用全量 C loops 在全量 CUDA kernels 中检索
      3. 输出结果（含特征相关性分析）
    """

    def __init__(self, retrieval: ProcrustesRetrieval, data_dir: str):
        self.retrieval = retrieval
        self.data_dir = data_dir
        self._loader = None

    def _get_loader(self) -> DataLoader:
        if self._loader is None:
            loader = DataLoader(self.data_dir, obfuscate_code=self.retrieval.obfuscate_code)
            if not loader.load_cache():
                loader.load()
                loader.save_cache()
            self._loader = loader
        return self._loader

    def _extract_paired_vecs(self, c_vecs: dict, cuda_vecs: dict) -> tuple:
        """从缓存中取配对数据的特征向量，用于训练 Procrustes"""
        loader = self._get_loader()
        kid_to_code = {k['id']: k['code'] for k in loader.cuda_codes}
        paired_c, paired_k = [], []

        for c in loader.c_codes:
            for kid in c.get('pair_kernel_ids', []):
                if kid in kid_to_code and c['id'] in c_vecs and kid in cuda_vecs:
                    paired_c.append(c_vecs[c['id']])
                    paired_k.append(cuda_vecs[kid])

        print(f"配对数据对数: {len(paired_c)}")
        return np.array(paired_c), np.array(paired_k)

    def run_correlation_analysis(self, V_C: np.ndarray, V_K: np.ndarray):
        """特征相关性分析 + 分布诊断"""
        print(f"\n{'特征':<25} {'r':>8}")
        print("-" * 35)
        high_corr = 0
        for i in range(FEATURE_DIM):
            if V_C[:, i].std() > 0 and V_K[:, i].std() > 0:
                r, _ = pearsonr(V_C[:, i], V_K[:, i])
            else:
                r = float('nan')
            marker = "🟢" if abs(r) > 0.5 else ("🟡" if abs(r) > 0.3 else "🔴")
            print(f"  {FEATURE_NAMES[i]:<25} {r:>8.3f} {marker}")
            if not np.isnan(r) and abs(r) > 0.5:
                high_corr += 1
        print(f"|r|>0.5: {high_corr}/{FEATURE_DIM}")

        print("\n" + "=" * 75)
        print("特征分布诊断 (C vs CUDA)")
        print(f"{'特征名称':<22} | {'C 均值':<8} {'CUDA 均值':<10} | {'C 非零率':<10} {'CUDA 非零率':<10}")
        print("-" * 75)
        for i in range(FEATURE_DIM):
            c_col, k_col = V_C[:, i], V_K[:, i]
            c_nz = np.mean(c_col > 1e-5)
            k_nz = np.mean(k_col > 1e-5)
            alert = ""
            if c_nz == 0 or k_nz == 0:
                alert = " ⚠️ 零值异常"
            elif abs(c_nz - k_nz) > 0.4:
                alert = " ⚠️ 提取断层"
            print(f"{FEATURE_NAMES[i]:<22} | {np.mean(c_col):<8.3f} {np.mean(k_col):<10.3f} | {c_nz:<10.1%} {k_nz:<10.1%}{alert}")
        print("=" * 75 + "\n")

    def run(self, output_dir: str, run_analysis: bool = False, save_json: bool = False):
        """
        执行 paired 评估。

        Args:
            output_dir   : 输出目录
            run_analysis : 是否在训练前运行特征相关性分析
        """
        loader = self._get_loader()

        # 提取特征（带缓存）
        print("\n提取特征（带缓存）...")
        c_vecs   = extract_c_vecs(loader.c_codes,    self.data_dir, self.retrieval.obfuscate_code)
        cuda_vecs = extract_cuda_vecs(loader.cuda_codes, self.data_dir, self.retrieval.obfuscate_code)

        # 训练 Procrustes
        V_C, V_K = self._extract_paired_vecs(c_vecs, cuda_vecs)

        if run_analysis:
            self.run_correlation_analysis(V_C, V_K)

        self.retrieval.train_procrustes(V_C, V_K)

        # 构造查询 loops
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
                "dwarf_vector": c.get('dwarf_vector', []),
            }
            for c in loader.c_codes
        ]

        # 构造 corpus kernels
        corpus_kernels = [
            {
                "id": k['id'],
                "filename": k['filename'],
                "kernel_name": k.get('kernel_name', ''),
                "source_code": k['code'],
                "dwarf_vector": k.get('dwarf_vector', []),
            }
            for k in loader.cuda_codes
        ]

        return self.retrieval.retrieval(
            query_loops, corpus_kernels, output_dir,
            output_tag="paired",
            save_json=save_json,
            c_vecs=c_vecs,
            cuda_vecs=cuda_vecs,
        )

# ============================================================
# 命令行入口
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Procrustes 跨语言检索")
    parser.add_argument('--parashift_dir', default='./code_datas')
    parser.add_argument('--cuda_only_dir', default='./CUDA-only')
    parser.add_argument('--output_dir', default='./res_rag/')
    parser.add_argument('--alpha', '-a', type=float, default=0.3, help='结构特征权重')
    parser.add_argument('--beta', '-b', type=float, default=0.7, help='矮人特征权重')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--obfuscate', action='store_true')
    parser.add_argument('--num_loops', type=int, default=10, help='随机选取 loop 数（cuda_only 任务）')
    parser.add_argument('--loop_ids_file', default=None, help='指定 loop IDs 的 JSON 文件（cuda_only 任务）')
    parser.add_argument('--save_json', action='store_true', help='保存完整 JSON 结果（默认关闭）')
    parser.add_argument('--run_analysis', action='store_true', help='运行特征相关性分析（paired 任务）')
    args = parser.parse_args()

    core = ProcrustesRetrieval(
        alpha=args.alpha,
        beta=args.beta,
        seed=args.seed,
        obfuscate_code=args.obfuscate
    )

    task = PairedEvaluation(core, args.parashift_dir)
    task.run(args.output_dir, run_analysis=args.run_analysis, save_json=args.save_json)
