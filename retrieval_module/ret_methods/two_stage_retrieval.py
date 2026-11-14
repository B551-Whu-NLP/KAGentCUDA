# ============================================================
# retrieval_module/two_stage_retrieval.py
#
# 两阶段检索：C 文件 → 1 个完整配对程序 + 2 个 standalone CUDA 源文件
#
# Stage 1: C → C（同语言，直接特征比较）
#   输入 C 文件 → 提取 loops → 在配对 C loop 库检索 → Top-1 个完整配对程序
#
# Stage 2: C → CUDA-only（跨语言，Procrustes 对齐）
#   同样的 loops → Procrustes 映射 → CUDA-only kernel 检索 → Top-2 个源文件
#
# 相似度公式（两阶段统一）:
#   sim = alpha * cosine(struct) + beta * cosine(dwarf)
# ============================================================

import os
import csv
import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from data_module.data_loader import DataLoader, obfuscate_ranking_code, RANKING_REPRESENTATION
from data_module.cuda_only_data_loader import CUDAOnlyDataLoader
from generation_module.reference_sources import paired_programs
from retrieval_module.ret_methods.source_validation import validate_standalone_sources
from retrieval_module.ret_methods.loop_exclusions import LoopExclusions, reviewed_paired_loops
from retrieval_module.features.c_loop_feature_class import CLoopFeatureExtractor
from retrieval_module.features.cuda_kernel_feature_class import CudaKernelFeatureExtractor

# 复用缓存工具
from retrieval_module.ret_methods.paired4eval_procrustes_retrieval import (
    extract_c_vecs,
    extract_cuda_vecs,
    extract_ranking_vecs,
    format_param,
    set_seed,
)

TOP_K1 = 1
TOP_K2 = 2
TOP_K = 3

# 13 Dwarf 模式顺序（与 dwarf_vector 下标对应）
DWARF_ORDER = [
    "dense_linear_algebra",
    "sparse_linear_algebra",
    "spectral_methods",
    "n_body_methods",
    "structured_grids",
    "unstructured_grids",
    "map_reduce",
    "combinational_logic",
    "graph_traversal",
    "dynamic_programming",
    "backtrack_branch_bound",
    "graphical_models",
    "finite_state_machines",
]
DWARF_DIM = len(DWARF_ORDER)  # 13


# ============================================================
# LLM 标注：提取 loop + 生成 dwarf_vector
# ============================================================
def _annotate_dwarf(loop_code: str, llm, system_prompt: str) -> list:
    """
    用 LLM 标注单个 loop，返回 13 维 dwarf_vector。
    标注失败时返回全零向量。
    """
    from prompts.classify_13_dwarf_prompt import Dwarf13CPrompt
    user_prompt = Dwarf13CPrompt.get_user_prompt(loop_code)
    try:
        response, _, _, _ = llm.chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            stream=False
        )
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:].rstrip("` \n")
        elif response.startswith("```"):
            response = response[3:].rstrip("` \n")
        result = json.loads(response)
        vec = [float(result.get(k, 0.0)) for k in DWARF_ORDER]
        return vec
    except Exception as e:
        print(f"    ⚠️  Dwarf 标注失败: {e}")
        return [0.0] * DWARF_DIM


def extract_loops_from_file(
    c_file: str,
    model_name: str = "qwen3-coder-next",
    org: str = "ali",
    use_llm: bool = True,
    loop_review=None,
    source_root=None,
    obfuscate_code=True,
) -> tuple[list, int]:
    """Extract reviewed C loops before any optional LLM annotation."""
    stem = Path(c_file).stem

    from data_module.c_file_process import extract_loops
    raw_loops = [
        {'source_code': loop.source_code, 'loop_index': i,
         'start_line': loop.start_line, 'loop_type': loop.loop_type,
         'filename': str(c_file)}
        for i, loop in enumerate(extract_loops(c_file))
    ]
    if not raw_loops:
        raise ValueError('No C loops extracted; cannot perform loop-level retrieval')
    excluded_count = 0
    if loop_review:
        if source_root is None:
            raise ValueError('Review source root is required')
        raw_loops, excluded = loop_review.select(raw_loops, 'query', source_root, require_all=False)
        excluded_count = len(excluded)
        if not raw_loops:
            raise ValueError('Loop review left no query loops')
    print(f"  保留 {len(raw_loops)} 个 loop，人工排除 {excluded_count} 个")

    # LLM 标注
    llm = None
    system_prompt = None
    if use_llm:
        try:
            from llm_utils import OpenAILLM
            from prompts.classify_13_dwarf_prompt import Dwarf13CPrompt
            llm = OpenAILLM(
                model_name=model_name,
                org=org,
                log_path="./logs",
                verbose=False,
                temperature=0.1,
                top_p=0.95,
            )
            system_prompt = Dwarf13CPrompt.get_system_prompt()
            print(f"  LLM 标注模型: {model_name}")
        except Exception as e:
            print(f"  ⚠️  LLM 初始化失败，dwarf_vector 将全零: {e}")
            use_llm = False

    loops = []
    for raw in tqdm(raw_loops, desc="标注 loops", unit="loop"):
        if use_llm and llm is not None:
            code = obfuscate_ranking_code(raw['source_code']) if obfuscate_code else raw['source_code']
            dwarf_vec = _annotate_dwarf(code, llm, system_prompt)
        else:
            dwarf_vec = [0.0] * DWARF_DIM

        loops.append({
            "id":           f"query_{stem}_{raw['loop_index']}",
            "source_code":  raw['source_code'],
            "dwarf_vector": dwarf_vec,
            "filename":     str(c_file),
            "project_name": "query",
            "loop_type":    raw['loop_type'],
            "start_line":   raw['start_line'],
        })

    return loops, excluded_count


# ============================================================
# 核心：两阶段检索
# ============================================================
class TwoStageRetrieval:
    """
    两阶段检索器。

    Stage 1: C → C（同语言，结构特征 + 矮人特征余弦加权）
    Stage 2: C → CUDA-only（Procrustes 跨语言对齐后余弦加权）
    """

    def __init__(
        self,
        parashift_dir: str,
        cuda_only_dir: str,
        alpha: float = 0.4,
        beta: float = 0.6,
        seed: int = 42,
        obfuscate_code: bool = True,
        model_name: str = "qwen3-coder-next",
        org: str = "ali",
        use_llm: bool = True,
        heldout_projects=None,
        loop_review=None,
    ):
        set_seed(seed)
        if heldout_projects is None:
            raise ValueError('Provide the held-out project set before fitting or retrieval')
        self.heldout_projects = set(heldout_projects)
        if len(self.heldout_projects) != 109 or len(self.heldout_projects) != len(heldout_projects):
            raise ValueError('Expected 109 distinct held-out project identities for end-to-end retrieval')
        self.parashift_dir  = parashift_dir
        self.cuda_only_dir  = cuda_only_dir
        self.alpha          = alpha
        self.beta           = beta
        self.seed           = seed
        self.obfuscate_code = obfuscate_code
        self.model_name     = model_name
        self.org            = org
        self.use_llm        = use_llm
        self.loop_review = LoopExclusions(loop_review) if loop_review else None
        self.loop_review_status = 'reviewed' if loop_review else 'not_reviewed'
        self.excluded_paired_loops = 0

        # Procrustes 参数（Stage 2 训练后填充）
        self.mu_C   = None
        self.mu_K   = None
        self.W      = None
        self._parashift_loader = None
        self._cuda_only_loader = None
        self._paired_sources = None
        self._standalone_sources = None

        print(f"{'='*60}")
        print(f"两阶段检索")
        print(f"  Alpha (结构特征权重) : {alpha}")
        print(f"  Beta  (矮人特征权重) : {beta}")
        print(f"  Top-K1 (C 文件)     : {TOP_K1}")
        print(f"  Top-K2 (CUDA files) : {TOP_K2}")
        print(f"  LLM 标注             : {'开启 (' + model_name + ')' if use_llm else '关闭'}")
        print(f"{'='*60}\n")

    def prepare(self):
        """
        预加载数据并训练 Procrustes。
        批量使用时在 run() 前调用一次，避免每个样本重复初始化。
        """
        print("[INFO] 初始化检索器（加载数据 + 训练 Procrustes）...")
        self._parashift_loader = self._load_parashift()
        self._c_vecs = self._ranking_vecs(self._parashift_loader.c_codes, self.parashift_dir, 'c')
        self._cuda_vecs = self._ranking_vecs(self._parashift_loader.cuda_codes, self.parashift_dir, 'cuda')
        self._train_procrustes(self._c_vecs, self._cuda_vecs, self._parashift_loader)

        print("[prepare] 加载 CUDA-only 数据...")
        self._cuda_only_loader = self._load_cuda_only()
        self._cuda_only_vecs = self._ranking_vecs(
            self._cuda_only_loader.cuda_codes, self.cuda_only_dir, 'cuda')
        print("[INFO] 检索器就绪\n")

    def _ranking_vecs(self, codes, data_dir, side):
        if self.obfuscate_code:
            return extract_ranking_vecs(codes, data_dir, side)
        extractor = extract_c_vecs if side == 'c' else extract_cuda_vecs
        return extractor(codes, data_dir, False)
    # ----------------------------------------------------------
    # 数据加载
    # ----------------------------------------------------------
    def _load_parashift(self) -> DataLoader:
        loader = DataLoader(self.parashift_dir, obfuscate_code=False)
        cache_dir = Path(__file__).resolve().parents[2] / '.data_cache'
        if not loader.load_cache(cache_dir):
            raise FileNotFoundError('Paired corpus cache required; refusing to silently rebuild a different split')
        c_projects = {entry['project'] for entry in loader.c_codes}
        cuda_projects = {entry['project'] for entry in loader.cuda_codes}
        projects = c_projects | cuda_projects
        if c_projects != cuda_projects:
            raise ValueError('Paired corpus project identities disagree between C and CUDA caches')
        overlap = projects & self.heldout_projects
        if overlap:
            raise ValueError(f'Held-out projects found in paired corpus: {sorted(overlap)}')
        if not projects:
            raise ValueError('Paired corpus is empty')
        if self.loop_review:
            self.loop_review.validate_sources('paired', self.parashift_dir)
            loader.c_codes, excluded = reviewed_paired_loops(
                loader.c_codes, self.loop_review, self.parashift_dir, False)
            self.excluded_paired_loops = len(excluded)
            print(f'Loop review: excluded {len(excluded)} paired C loops before feature extraction')
        else:
            print('Loop review: no manifest; output-only loops NOT excluded')
        print(f'Paired corpus: {len(projects)} development projects')
        return loader

    def _load_cuda_only(self) -> CUDAOnlyDataLoader:
        loader = CUDAOnlyDataLoader(self.cuda_only_dir)
        repo_root = Path(__file__).resolve().parents[2]
        cache = repo_root / '.data_cache' / 'cuda_only_codes_cache.json'
        if not loader.load_cache(cache.parent):
            raise FileNotFoundError('Committed standalone cache required; refusing to rebuild an unverified collection')
        self._standalone_sources = validate_standalone_sources(
            loader.cuda_codes, self.cuda_only_dir,
            repo_root / 'standalone_source_manifest.json', cache,
        )
        return loader

    # ----------------------------------------------------------
    # Procrustes 训练（Stage 2 用）
    # ----------------------------------------------------------
    def _train_procrustes(self, c_vecs: dict, cuda_vecs: dict, loader: DataLoader):
        """Fit a row-vector orthogonal map on mean-centered paired features."""
        print("\n训练 Procrustes W 矩阵...")
        kid_to_code = {k['id']: k['code'] for k in loader.cuda_codes}
        paired_c, paired_k = [], []

        for c in loader.c_codes:
            for kid in c.get('pair_kernel_ids', []):
                if kid in kid_to_code and c['id'] in c_vecs and kid in cuda_vecs:
                    paired_c.append(c_vecs[c['id']])
                    paired_k.append(cuda_vecs[kid])

        print(f"配对数据对数: {len(paired_c)}")
        V_C = np.array(paired_c)
        V_K = np.array(paired_k)

        if V_C.ndim != 2 or V_K.shape != V_C.shape or len(V_C) < 2:
            raise ValueError('Procrustes requires at least two aligned feature pairs')
        mu_C, mu_K = V_C.mean(axis=0), V_K.mean(axis=0)
        centered_C, centered_K = V_C - mu_C, V_K - mu_K
        U, _, Vt = np.linalg.svd(centered_K.T @ centered_C)
        row_map = U @ Vt
        residual = np.linalg.norm(centered_K @ row_map - centered_C, 'fro') / \
                   max(1e-10, np.linalg.norm(centered_C, 'fro'))
        print(f"残差: {residual:.4f}")

        self.mu_C, self.mu_K, self.W = mu_C, mu_K, row_map

    # ----------------------------------------------------------
    # 相似度工具
    # ----------------------------------------------------------
    @staticmethod
    def _l2_norm_matrix(M: np.ndarray) -> np.ndarray:
        """行 L2 归一化"""
        norms = np.linalg.norm(M, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        return M / norms

    def _sim_matrix(
        self,
        query_struct: np.ndarray,   # (n_q, d)
        query_dwarf:  np.ndarray,   # (n_q, dw)
        corpus_struct: np.ndarray,  # (n_c, d)
        corpus_dwarf:  np.ndarray,  # (n_c, dw)
    ) -> np.ndarray:
        """
        返回加权相似度矩阵 (n_q, n_c)：
          sim = alpha * cosine(struct) + beta * cosine(dwarf)
        """
        qs = self._l2_norm_matrix(query_struct)
        cs = self._l2_norm_matrix(corpus_struct)
        sim_s = qs @ cs.T                               # (n_q, n_c)

        qd = self._l2_norm_matrix(query_dwarf)
        cd = self._l2_norm_matrix(corpus_dwarf)
        sim_d = qd @ cd.T                               # (n_q, n_c)

        return self.alpha * sim_s + self.beta * sim_d   # (n_q, n_c)

    # ----------------------------------------------------------
    # Stage 1：C → C
    # ----------------------------------------------------------
    def _stage1(
        self,
        query_loops: list,
        parashift_loader: DataLoader,
        c_vecs: dict,
    ) -> list:
        """
        C → C 检索。

        对每个 query loop 与配对 C loop 比较，按项目/文件的最大得分
        选择一个可定位到唯一 C/CUDA 完整源文件的配对示例。

        Returns:
            list of {
                "c_filename": str, "c_code": str, "score": float,
                "pair_kernels": [{"code": str, "source_path": str}]
            }
        """
        print("\n[ Stage 1 ] C → C 检索...")

        # 提取 query loop 特征（现场提取，不写缓存）
        q_struct_list, q_dwarf_list = [], []
        for loop in query_loops:
            code = obfuscate_ranking_code(loop['source_code']) if self.obfuscate_code else loop['source_code']
            vec = CLoopFeatureExtractor(code).extract().to_vector()
            q_struct_list.append(vec)
            q_dwarf_list.append(np.array(loop.get('dwarf_vector', [0.0] * 13), dtype=np.float32))

        Q_struct = np.array(q_struct_list, dtype=np.float32)   # (n_q, d)
        Q_dwarf  = np.array(q_dwarf_list,  dtype=np.float32)   # (n_q, dw)

        # 构建配对语料 C loop 矩阵
        db_ids      = [c['id'] for c in parashift_loader.c_codes]
        db_struct   = np.array([c_vecs[cid] for cid in db_ids], dtype=np.float32)
        db_dwarf    = np.array(
            [parashift_loader.c_codes[i].get('dwarf_vector', [0.0]*13)
             for i, cid in enumerate(db_ids)],
            dtype=np.float32
        )

        # (n_q, n_db) 相似度矩阵
        sim = self._sim_matrix(Q_struct, Q_dwarf, db_struct, db_dwarf)

        # score1(f) = max_i max_{j in L(f)} sim(query_i, c_j)
        # 先对每个 query loop 取最大，再按文件聚合取最大
        loop_max_sim = sim.max(axis=0)   # (n_db,)  每个 db loop 对所有 query 的最大相似度

        if self._paired_sources is None:
            self._paired_sources = paired_programs(parashift_loader, self.parashift_dir)
        file_score = {}
        for i, c in enumerate(parashift_loader.c_codes):
            key = (c['project'], c['filename'])
            if key in self._paired_sources:
                file_score[key] = max(file_score.get(key, float('-inf')), float(loop_max_sim[i]))

        results = []
        for key, score in sorted(file_score.items(), key=lambda item: item[1], reverse=True)[:TOP_K1]:
            source = self._paired_sources[key]
            results.append({
                'project': key[0], 'c_filename': key[1], 'score': score,
                'c_code': source['c_code'], 'c_path': source['c_path'],
                'pair_kernels': [{'filename': Path(source['cuda_path']).name,
                                  'code': source['cuda_code'], 'source_path': source['cuda_path']}],
            })
            print(f"  [{len(results)}] {key[1]} score={score:.4f} paired_source={source['cuda_path']}")

        return results

    # ----------------------------------------------------------
    # Stage 2：C → CUDA-only
    # ----------------------------------------------------------
    def _stage2(
        self,
        query_loops: list,
        cuda_only_loader: CUDAOnlyDataLoader,
        cuda_only_vecs: dict,
    ) -> list:
        """
        C → CUDA-only 检索（Procrustes 跨语言对齐）。

        Returns:
            list of {
                "rank": int,
                "kernel_id": str,
                "kernel_name": str,
                "filename": str,
                "score": float,
                "code": str,
            }
        """
        assert self.W is not None, "Procrustes 未训练，请先调用 run()"
        print("\n[ Stage 2 ] C → CUDA-only 检索（Procrustes）...")

        q_struct_list, q_dwarf_list = [], []
        for loop in query_loops:
            code = obfuscate_ranking_code(loop['source_code']) if self.obfuscate_code else loop['source_code']
            vec = CLoopFeatureExtractor(code).extract().to_vector()
            q_struct_list.append(np.array(vec, dtype=np.float32))
            q_dwarf_list.append(np.array(loop.get('dwarf_vector', [0.0]*13), dtype=np.float32))

        Q_struct = np.array(q_struct_list, dtype=np.float32)   # (n_q, d)
        Q_dwarf  = np.array(q_dwarf_list,  dtype=np.float32)   # (n_q, dw)

        kernel_ids = [k['id'] for k in cuda_only_loader.cuda_codes]
        K_struct = np.array(
            [(cuda_only_vecs[kid] - self.mu_K) @ self.W + self.mu_C
             for kid in kernel_ids],
            dtype=np.float32
        )
        K_dwarf = np.array(
            [cuda_only_loader.cuda_codes[i].get('dwarf_vector', [0.0]*13)
             for i, kid in enumerate(kernel_ids)],
            dtype=np.float32
        )

        # (n_q, n_k) 相似度矩阵，取每个 kernel 对所有 query 的最大值
        sim = self._sim_matrix(Q_struct, Q_dwarf, K_struct, K_dwarf)
        kernel_score = sim.max(axis=0)   # (n_k,)

        results = []
        seen_files = set()
        for idx in np.argsort(kernel_score)[::-1]:
            k = cuda_only_loader.cuda_codes[idx]
            source = Path(self.cuda_only_dir) / 'db_cuda_codes' / k['filename']
            if k['filename'] in seen_files:
                continue
            code = self._standalone_sources[k['filename']]
            seen_files.add(k['filename'])
            rank = len(results) + 1
            results.append({
                'rank': rank, 'kernel_id': k['id'], 'kernel_name': k.get('kernel_name', ''),
                'filename': k['filename'], 'repo_name': k.get('repo_name', ''),
                'score': float(kernel_score[idx]), 'code': code, 'source_path': str(source),
            })
            if rank == TOP_K2:
                break

        return results

    # ----------------------------------------------------------
    # 主入口
    # ----------------------------------------------------------
    def run(self, c_file: str, output_dir: str) -> dict:
        """
        执行两阶段检索。
        若已调用 prepare()，直接复用缓存数据；否则临时加载。
        """
        print(f"\n查询文件: {c_file}")
        query = Path(c_file).resolve()
        data_root = Path(self.parashift_dir).resolve()
        try:
            relative = query.relative_to(data_root)
        except ValueError as exc:
            raise ValueError('Query is outside the paired source tree') from exc
        if (len(relative.parts) != 4 or not relative.parts[0].startswith('type-')
                or relative.parts[1] not in self.heldout_projects
                or relative.parts[2] != 'c' or query.suffix not in ('.c', '.cpp')):
            raise ValueError('Query must be a C source in a declared held-out project')

        query_loops, excluded_query_count = extract_loops_from_file(
            c_file,
            model_name=self.model_name,
            org=self.org,
            use_llm=self.use_llm,
            loop_review=self.loop_review,
            source_root=data_root,
            obfuscate_code=self.obfuscate_code,
        )
        if not self.loop_review:
            print('Loop review: no manifest; query output-only loops NOT excluded')

        # 复用 prepare() 的缓存，或临时加载
        if self._parashift_loader is not None:
            parashift_loader = self._parashift_loader
            c_vecs           = self._c_vecs
            cuda_only_loader = self._cuda_only_loader
            cuda_only_vecs   = self._cuda_only_vecs
        else:
            print("\n加载配对语料数据...")
            parashift_loader = self._load_parashift()
            c_vecs = self._ranking_vecs(parashift_loader.c_codes, self.parashift_dir, 'c')
            cuda_vecs = self._ranking_vecs(parashift_loader.cuda_codes, self.parashift_dir, 'cuda')
            self._train_procrustes(c_vecs, cuda_vecs, parashift_loader)

            print("\n加载 CUDA-only 数据...")
            cuda_only_loader = self._load_cuda_only()
            cuda_only_vecs = self._ranking_vecs(cuda_only_loader.cuda_codes, self.cuda_only_dir, 'cuda')

        # Stage 1 & 2
        stage1_results = self._stage1(query_loops, parashift_loader, c_vecs)
        stage2_results = self._stage2(query_loops, cuda_only_loader, cuda_only_vecs)
        if len(stage1_results) != TOP_K1 or len(stage2_results) != TOP_K2:
            raise ValueError('Cannot construct 1 executable paired example and 2 standalone CUDA source files')
        ranked = sorted([{'stage': 'stage1', **e} for e in stage1_results] +
                        [{'stage': 'stage2', **e} for e in stage2_results],
                        key=lambda e: e['score'], reverse=True)[:TOP_K]

        result = {
            "query_file": str(c_file),
            "config": {
                "alpha": self.alpha, "beta": self.beta,
                "top_k1": TOP_K1, "top_k2": TOP_K2, "top_k": TOP_K,
                "loop_review": self.loop_review_status,
                "ranking_representation": RANKING_REPRESENTATION if self.obfuscate_code else 'raw',
                "excluded_query_loops": excluded_query_count,
                "excluded_paired_loops": self.excluded_paired_loops,
            },
            "stage1": stage1_results,
            "stage2": stage2_results,
            "ranked": ranked,
        }
        self._save(result, output_dir, c_file)
        return result

    def _save(self, result: dict, output_dir: str, c_file: str):
        """保存 JSON + CSV"""
        os.makedirs(output_dir, exist_ok=True)
        stem = Path(c_file).stem
        alpha_fmt = format_param(self.alpha)
        beta_fmt  = format_param(self.beta)
        base = f"two_stage_{stem}_a{alpha_fmt}_b{beta_fmt}"

        # JSON（完整结果）
        json_path = os.path.join(output_dir, f"{base}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nJSON 已保存: {json_path}")

        # CSV（Stage 2 Top-K2，格式与其他方法一致）
        csv_path = os.path.join(output_dir, f"{base}_stage2.csv")
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                "rank", "kernel_id", "kernel_name", "filename", "repo_name", "score"
            ])
            for entry in result["stage2"]:
                writer.writerow([
                    entry["rank"], entry["kernel_id"], entry["kernel_name"],
                    entry["filename"], entry["repo_name"], f"{entry['score']:.6f}"
                ])
        print(f"CSV  已保存: {csv_path}")

        # Stage 1 摘要（方便查看 in-context examples）
        stage1_path = os.path.join(output_dir, f"{base}_stage1.txt")
        with open(stage1_path, 'w', encoding='utf-8') as f:
            f.write(f"Query: {result['query_file']}\n\n")
            for i, entry in enumerate(result["stage1"], 1):
                f.write(f"[{i}] {entry['c_filename']}  score={entry['score']:.4f}\n")
                for k in entry["pair_kernels"]:
                    f.write(f"    CUDA source: {k['source_path']}\n")
                f.write("\n")
        print(f"Stage1 已保存: {stage1_path}")


# ============================================================
# 命令行入口
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="两阶段 C → CUDA 检索")
    parser.add_argument('--input',          required=True, help='输入 C 文件路径')
    parser.add_argument('--parashift_dir',  default='./code_datas')
    parser.add_argument('--heldout_projects', required=True, help='JSON file with projects array')
    parser.add_argument('--cuda_only_dir',  default='./CUDA-only')
    parser.add_argument('--output_dir',     default='./res_rag/two_stage_output')
    parser.add_argument('--alpha', '-a',    type=float, default=0.4, help='结构特征权重')
    parser.add_argument('--beta',  '-b',    type=float, default=0.6, help='矮人特征权重')
    parser.add_argument('--seed',           type=int,   default=42)
    parser.add_argument('--no_obfuscate',   action='store_true', help='仅用于 raw-code 对照实验')
    parser.add_argument('--model_name',     default='qwen3-coder-next', help='LLM 模型名称')
    parser.add_argument('--org',            default='ali',              help='LLM 平台')
    parser.add_argument('--no_llm',         action='store_true',        help='跳过 LLM 标注（调试用）')
    parser.add_argument('--loop_review', help='Source-bound JSON review of output-only loops on both sides')
    args = parser.parse_args()
    heldout = json.loads(Path(args.heldout_projects).read_text(encoding='utf-8'))['projects']
    samples_path = Path(__file__).resolve().parents[2] / 'selected_samples' / 'selected_samples.json'
    selected_projects = {item['project_name'] for item in json.loads(samples_path.read_text(encoding='utf-8'))}
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

    retriever.run(args.input, args.output_dir)