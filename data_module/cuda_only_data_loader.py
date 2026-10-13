"""
CUDA-only 数据加载模块
专门用于加载 CUDA-only 数据集中的 kernel 代码
CUDA 侧无需混淆，只做纯代码提取
支持缓存机制
支持 13 矮人向量注入
"""

import sys
import re
import json
from pathlib import Path
from typing import List, Dict, Optional

sys.path.insert(0, str(Path(__file__).parent))

DWARF_DIM = 13


def _extract_kernel_source(file_content: str, kernel_name: str) -> Optional[str]:
    """从 CUDA 文件内容中提取指定 kernel 的源码（简单正则方式）"""
    # 匹配 __global__ 或 __device__ 函数定义
    pattern = rf'(?:__global__|__device__)\s+(?:\w+\s+)*(\w+)\s*\([^)]*\s*\{{'
    matches = list(re.finditer(pattern, file_content))

    for match in matches:
        found_name = match.group(1)
        if found_name == kernel_name:
            # 找到开始位置，匹配大括号找到结束
            start = match.start()
            brace_count = 0
            in_string = False
            in_comment = False

            for i, char in enumerate(file_content[start:]):
                if char == '"' and (i == 0 or file_content[start + i - 1] != '\\'):
                    in_string = not in_string
                elif char == '/' and i + 1 < len(file_content) - start:
                    next_char = file_content[start + i + 1]
                    if next_char == '/' and not in_string:
                        in_comment = True
                    elif next_char == '*' and not in_string:
                        in_comment = True
                elif char == '\n' and in_comment:
                    in_comment = False
                elif char == '{' and not in_string and not in_comment:
                    brace_count += 1
                elif char == '}' and not in_string and not in_comment:
                    brace_count -= 1
                    if brace_count == 0:
                        return file_content[start: start + i + 1]
            break

    return None


class CUDAOnlyDataLoader:
    """CUDA-only 数据集加载器 - 仅加载 CUDA kernel 代码"""

    def __init__(self, base_path: str | Path, dwarf_file: str | Path = None, model_name: str = 'qwen3-coder-next'):
        self.base_path = Path(base_path)
        self.db_cuda_dir = self.base_path / 'db_cuda_codes'
        self.model_name = model_name
        # 默认使用最新的标注结果
        if dwarf_file is None:
            label_dir = self.base_path / 'labeled_13dwaft_res' / 'label_kernels'
            # 找到最新的标注文件
            dwarf_files = sorted(label_dir.glob(f'{model_name}_*.json'))
            if dwarf_files:
                self.dwarf_file = dwarf_files[-1]  # 使用最新的
            else:
                self.dwarf_file = label_dir / f'{model_name}.json'
        else:
            self.dwarf_file = Path(dwarf_file)
        self.cuda_codes: List[Dict] = []
        self._loaded = False
        self._dwarf_data: Dict = {}

    def _get_cache_filename(self) -> str:
        """返回缓存文件名（CUDA-only 不需要混淆，固定名称）"""
        return 'cuda_only_codes_cache.json'

    # ----------------------------------------------------------------
    # 公开接口
    # ----------------------------------------------------------------

    def load(self, force_refresh: bool = False):
        """从 dwarf 标注文件加载 kernel 数据并提取源码"""
        if self._loaded and not force_refresh:
            return

        self.cuda_codes = []
        code_id = 0

        if not self.db_cuda_dir.exists():
            raise FileNotFoundError(f"CUDA codes directory not found: {self.db_cuda_dir}")

        # 第一步：加载 dwarf 标注数据
        self._load_dwarf_data()

        if not self._dwarf_data:
            print("No dwarf data found!")
            return

        print(f"Loading {len(self._dwarf_data)} kernels from dwarf data...")

        # 先建立文件名索引（stem -> full path）
        file_index = {}
        for f in self.db_cuda_dir.glob('*.cu'):
            stem = f.stem
            if stem not in file_index:
                file_index[stem] = []
            file_index[stem].append(f)

        # 对每个有标注的 kernel，读取源码文件并提取代码
        loaded = 0
        failed = 0

        for unique_key, dwarf_info in self._dwarf_data.items():
            # 从 dwarf_info 中获取信息
            kernel_name = dwarf_info.get('kernel_name', '')
            source_file = dwarf_info.get('source_file', '')

            if not kernel_name:
                failed += 1
                continue

            # 多种策略查找文件
            cu_file = None

            # 策略1: 从 source_file 中查找
            if source_file:
                source_stem = Path(source_file).stem
                if source_stem in file_index:
                    cu_file = file_index[source_stem][0]

            # 策略2: 从 unique_key 中尝试所有可能的文件名组合
            if cu_file is None:
                parts = unique_key.split('_')
                for i in range(len(parts) - 1, 0, -1):
                    candidate_stem = '_'.join(parts[:i])
                    if candidate_stem in file_index:
                        cu_file = file_index[candidate_stem][0]
                        break

            # 策略3: 直接查找 kernel_name
            if cu_file is None and kernel_name in file_index:
                cu_file = file_index[kernel_name][0]

            if cu_file is None or not cu_file.exists():
                failed += 1
                continue

            # 读取文件内容
            try:
                with open(cu_file, 'r', encoding='utf-8', errors='replace') as f:
                    file_content = f.read()
            except Exception:
                failed += 1
                continue

            # 优先使用 dwarf_info 中的 full_code，如果没有则提取
            code = dwarf_info.get('full_code', '')
            if not code:
                # 如果 full_code 为空，尝试从文件中提取
                code = _extract_kernel_source(file_content, kernel_name)
            if not code:
                # 如果提取失败，跳过
                failed += 1
                continue

            self.cuda_codes.append({
                'id': code_id,
                'filename': cu_file.name,
                'kernel_name': kernel_name,
                'unique_key': unique_key,
                'code': code,
                'start_line': dwarf_info.get('start_line', 0),
                'end_line': dwarf_info.get('end_line', 0),
                'repo_name': dwarf_info.get('repo_name', ''),
                'github_url': dwarf_info.get('github_url', ''),
                'dwarf_vector': dwarf_info.get('dwarf_vector', [0.0] * DWARF_DIM),
                'dwarf_scores': dwarf_info.get('dwarf_scores', {}),
                'success': dwarf_info.get('success', False),
                'reasoning': dwarf_info.get('reasoning', ''),
                'matched_patterns': dwarf_info.get('matched_patterns', []),
            })
            code_id += 1
            loaded += 1

            if loaded % 100 == 0:
                print(f"  [{loaded}/{len(self._dwarf_data)}] Loaded")

        print(f"\nTotal: {loaded} kernels loaded, {failed} failed")

        # 过滤掉没有有效 dwarf 向量的数据（全为 0）
        before_filter = len(self.cuda_codes)
        self.cuda_codes = [
            k for k in self.cuda_codes
            if any(v != 0 for v in k['dwarf_vector'])
        ]
        filtered = before_filter - len(self.cuda_codes)
        if filtered > 0:
            print(f"Filtered out {filtered} kernels with zero dwarf vector")

        # 重新分配 ID
        for i, k in enumerate(self.cuda_codes):
            k['id'] = i

        self._loaded = True
        self._print_summary()

    def save_cache(self, cache_dir: str | Path = None):
        """保存到缓存（含 13 矮人向量）"""
        if not self._loaded:
            self.load()

        cache_dir = (Path(cache_dir) if cache_dir
                     else self.base_path / '.data_cache')
        cache_dir.mkdir(parents=True, exist_ok=True)

        data = {
            'cuda_codes': self.cuda_codes,
        }

        cache_file = cache_dir / self._get_cache_filename()
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"Cache saved to: {cache_file}")

    def load_cache(self, cache_dir: str | Path = None) -> bool:
        """从缓存加载数据"""
        cache_dir = (Path(cache_dir) if cache_dir
                      else self.base_path / '.data_cache')
        cache_file = cache_dir / self._get_cache_filename()

        if not cache_file.exists():
            return False

        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.cuda_codes = data.get('cuda_codes', [])
            self._loaded = True

            self._print_summary()
            return True

        except Exception as e:
            print(f"Failed to load cache: {e}")
            return False

    def refresh_dwarf_and_save(self, cache_dir: str | Path = None):
        """
        重新注入最新的矮人向量并保存缓存
        当标注文件更新后调用
        """
        if not self._loaded:
            raise RuntimeError("请先调用 load() 或 load_cache()")
        self._load_dwarf_data()
        # dwarf 数据已经在 load 时加载了，这里不需要重新注入
        self.save_cache(cache_dir)
        print("缓存已更新")

    # ----------------------------------------------------------------
    # 私有方法
    # ----------------------------------------------------------------

    def _load_dwarf_data(self):
        """加载合并后的矮人标注数据 - 适配新的标注格式"""
        if not self.dwarf_file.exists():
            print(f"  [SKIP] Dwarf file not found: {self.dwarf_file}")
            self._dwarf_data = {}
            return

        try:
            with open(self.dwarf_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 新格式：外层有 results 数组，每个元素是一个 kernel 的标注结果
            if 'results' in data:
                # 转换为字典：unique_key -> result
                self._dwarf_data = {
                    r['unique_key']: r
                    for r in data['results']
                    if r.get('success', False)
                }
                print(f"  Loaded dwarf data for {len(self._dwarf_data)} kernels (from {self.dwarf_file.name})")
            else:
                # 旧格式：直接是字典
                self._dwarf_data = data
                print(f"  Loaded dwarf data for {len(self._dwarf_data)} kernels (old format)")
        except Exception as e:
            print(f"  [ERROR] Failed to load dwarf data: {e}")
            self._dwarf_data = {}

    def _print_summary(self):
        cuda_files = len(set(k['filename'] for k in self.cuda_codes))

        print(f"\nCUDA-only Dataset Summary:")
        print(f"  CUDA files:   {cuda_files}")
        print(f"  CUDA kernels: {len(self.cuda_codes)}")

        # 统计有有效矮人标注的数量（非全零向量）
        valid_dwarf = sum(
            1 for k in self.cuda_codes
            if any(v != 0 for v in k['dwarf_vector'])
        )
        print(f"  With dwarf:   {valid_dwarf}")


# ----------------------------------------------------------------
# CLI
# ----------------------------------------------------------------
if __name__ == '__main__':
    base_path = Path('./CUDA-only')

    loader = CUDAOnlyDataLoader(base_path)

    if not loader.load_cache():
        loader.load()
        loader.save_cache()

    print(f"\nCUDA kernels: {len(loader.cuda_codes)}")

    if loader.cuda_codes:
        k = loader.cuda_codes[0]
        print(f"\nFirst CUDA kernel:")
        print(f"  ID:           {k['id']}")
        print(f"  File:         {k['filename']}")
        print(f"  Kernel name:  {k['kernel_name']}")
        print(f"  Unique key:   {k['unique_key']}")
        print(f"  dwarf_vector: {k['dwarf_vector']}")
        print(f"  Code snippet: {k['code'][:80]}...")

        # 打印最后一个
        k = loader.cuda_codes[-1]
        print(f"\nLast CUDA kernel:")
        print(f"  ID:           {k['id']}")
        print(f"  Kernel name:  {k['kernel_name']}")
        print(f"  dwarf_vector: {k['dwarf_vector']}")
