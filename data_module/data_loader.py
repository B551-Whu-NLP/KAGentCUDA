"""
数据加载模块
加载配对的 C/CUDA 代码
支持缓存机制
支持13矮人向量注入
支持代码混淆
"""

import sys
import json
import re
from pathlib import Path
from typing import List, Dict, Optional

sys.path.insert(0, str(Path(__file__).parent))

from c_file_process import extract_loops, LoopInfo
from cuda_file_process import extract_kernels, KernelInfo

DWARF_DIM = 13

C_KEYWORDS = {
    'int', 'float', 'double', 'char', 'void', 'long', 'short',
    'unsigned', 'signed', 'const', 'static', 'extern', 'return',
    'if', 'else', 'for', 'while', 'do', 'break', 'continue',
    'struct', 'typedef', 'sizeof', 'NULL', 'true', 'false',
    'include', 'define', 'ifdef', 'endif', 'pragma',
}

CUDA_KEYWORDS = C_KEYWORDS | {
    '__global__', '__device__', '__host__', '__shared__',
    '__syncthreads', 'threadIdx', 'blockIdx', 'blockDim', 'gridDim',
    'atomicAdd', 'atomicSub', 'atomicMin', 'atomicMax',
    'cudaMalloc', 'cudaFree', 'cudaMemcpy',
}


RANKING_REPRESENTATION = 'identifier_v1'

_RANKING_TOKENS = re.compile(
    r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\''
    r'|(?:0[xX][0-9a-fA-F]+|0[bB][01]+|(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)[uUlLfF]*'
    r'|[a-zA-Z_]\w*|.', re.DOTALL
)
_RANKING_RESERVED = C_KEYWORDS | CUDA_KEYWORDS | {
    'auto', 'bool', 'class', 'enum', 'namespace', 'template', 'typename',
    'union', 'volatile', 'register', 'inline', 'restrict', 'size_t', 'uint',
    'nullptr', 'true', 'false', 'this', 'new', 'delete', 'float2', 'float3',
    'float4', 'int2', 'int3', 'int4', 'dim3', 'half', 'max', 'min', 'fmax',
    'fmin', 'fmaxf', 'fminf', 'fmaf', 'fma', '__restrict__', '__forceinline__',
    '__launch_bounds__', '__constant__', 'warpSize', 'x', 'y', 'z', 'w',
}


def obfuscate_ranking_code(code: str) -> str:
    """Identifier-only retrieval view; never use this as a compilable generation example."""
    mapping = {}
    output = []
    previous = ''
    for match in _RANKING_TOKENS.finditer(code):
        token = match.group()
        if re.fullmatch(r'[a-zA-Z_]\w*', token) and token not in _RANKING_RESERVED \
                and previous not in ('.', '->', '#'):
            if token not in mapping:
                mapping[token] = f'var_{len(mapping)}'
            output.append(mapping[token])
        else:
            output.append(token)
        if not token.isspace():
            previous = token
    return ''.join(output)


def obfuscate(code: str, keywords: set) -> str:
    """Cache/review obfuscation transform (also replaces numeric constants)."""
    tokens = re.findall(r'[a-zA-Z_]\w*|\d+\.?\d*|\S|\s+', code)
    mapping = {}
    id_counter  = [0]
    num_counter = [0]

    def replace(token):
        # 数字常量
        if re.match(r'^\d+\.?\d*$', token):
            if token not in mapping:
                mapping[token] = f'num_{num_counter[0]}'
                num_counter[0] += 1
            return mapping[token]
        # 非标识符
        if not re.match(r'^[a-zA-Z_]\w*$', token):
            return token
        # 保留关键字
        if token in keywords:
            return token
        # 替换标识符
        if token not in mapping:
            mapping[token] = f'var_{id_counter[0]}'
            id_counter[0] += 1
        return mapping[token]

    return ''.join(replace(t) for t in tokens)


class DataLoader:
    """配对 C/CUDA 数据加载器"""

    def __init__(self, base_path: str | Path, dwarf_dir: str | Path = None, obfuscate_code: bool = False):
        self.base_path = Path(base_path)
        self.dwarf_dir = Path(dwarf_dir) if dwarf_dir else Path('dwarf_labels')
        self.obfuscate_code = obfuscate_code
        self.c_codes: List[Dict] = []
        self.cuda_codes: List[Dict] = []
        self._loaded = False

    def _get_cache_filename(self) -> str:
        """根据是否混淆返回不同的缓存文件名"""
        return 'codes_cache_obfuscated.json' if self.obfuscate_code else 'codes_cache.json'

    # ----------------------------------------------------------------
    # 公开接口
    # ----------------------------------------------------------------

    def load(self, force_refresh: bool = False):
        """从原始文件加载所有数据"""
        if self._loaded and not force_refresh:
            return

        self.c_codes = []
        self.cuda_codes = []
        code_id = 0

        type_dirs = [
            d for d in sorted(self.base_path.iterdir())
            if d.is_dir() and d.name.startswith('type-')
        ]

        for type_idx, type_dir in enumerate(type_dirs, 1):
            print(f"\n[{type_idx}/{len(type_dirs)}] Processing {type_dir.name}...")

            project_dirs = [d for d in sorted(type_dir.iterdir()) if d.is_dir()]

            type_c_files   = set()
            type_cuda_files = set()
            type_loops     = 0
            type_kernels   = 0

            for project_dir in project_dirs:
                c_dir    = project_dir / 'c'
                cuda_dir = project_dir / 'cuda'

                if not c_dir.exists() or not cuda_dir.exists():
                    print(f"  [SKIP] {project_dir.name}: 缺少 c/ 或 cuda/ 目录")
                    continue

                c_files    = list(c_dir.glob('*.c')) + list(c_dir.glob('*.cpp'))
                cuda_files = list(cuda_dir.glob('*.cu'))

                if not c_files:
                    print(f"  [SKIP] {project_dir.name}: 没有 C 文件")
                    continue
                if not cuda_files:
                    print(f"  [SKIP] {project_dir.name}: 没有 CUDA 文件")
                    continue

                print(f"  [OK] {project_dir.name}: "
                      f"{len(c_files)} C files, {len(cuda_files)} CUDA files")

                project_loop_ids   = []
                project_kernel_ids = []
                all_c_files        = []
                all_cuda_files     = []

                for c_file in c_files:
                    loops = extract_loops(c_file)
                    if not loops:
                        print(f"  [NO LOOPS] {project_dir.name}: "
                              f"C 文件 {c_file.name} 没有提取到 loops")

                    type_c_files.add(c_file.name)
                    type_loops += len(loops)
                    all_c_files.append(c_file.name)

                    for loop in loops:
                        code = loop.source_code
                        if self.obfuscate_code:
                            code = obfuscate(code, C_KEYWORDS)
                        project_loop_ids.append(code_id)
                        self.c_codes.append({
                            'id':                   code_id,
                            'filename':             c_file.name,
                            'project':              project_dir.name,
                            'code':                 code,
                            'loop_type':            loop.loop_type,
                            'parent_function':      loop.parent_function or '',
                            'start_line':           loop.start_line,
                            'end_line':             loop.end_line,
                            'pair_kernel_ids':      [],
                            'pair_kernel_filenames': [],
                            'dwarf_vector':         [0.0] * DWARF_DIM,
                        })
                        code_id += 1

                for cuda_file in cuda_files:
                    kernels = extract_kernels(cuda_file)
                    if not kernels:
                        print(f"  [NO KERNELS] {project_dir.name}: "
                              f"CUDA 文件 {cuda_file.name} 没有提取到 kernels")

                    type_cuda_files.add(cuda_file.name)
                    type_kernels += len(kernels)
                    all_cuda_files.append(cuda_file.name)

                    for kernel in kernels:
                        code = kernel.source_code
                        project_kernel_ids.append(code_id)
                        self.cuda_codes.append({
                            'id':                  code_id,
                            'filename':            cuda_file.name,
                            'project':             project_dir.name,
                            'code':                code,
                            'kernel_name':         kernel.kernel_name,
                            'start_line':          kernel.start_line,
                            'end_line':            kernel.end_line,
                            'pair_loop_ids':       [],
                            'pair_loop_filenames': [],
                            'dwarf_vector':        [0.0] * DWARF_DIM,
                        })
                        code_id += 1

                # 建立互相引用
                n_loops   = len(project_loop_ids)
                n_kernels = len(project_kernel_ids)

                for i in range(len(self.c_codes) - n_loops, len(self.c_codes)):
                    self.c_codes[i]['pair_kernel_ids']      = project_kernel_ids.copy()
                    self.c_codes[i]['pair_kernel_filenames'] = all_cuda_files.copy()

                for i in range(len(self.cuda_codes) - n_kernels, len(self.cuda_codes)):
                    self.cuda_codes[i]['pair_loop_ids']       = project_loop_ids.copy()
                    self.cuda_codes[i]['pair_loop_filenames'] = all_c_files.copy()

            print(f"  → {type_dir.name}: "
                  f"{len(type_c_files)} C files, {len(type_cuda_files)} CUDA files, "
                  f"{type_loops} loops, {type_kernels} kernels")

        self._loaded = True
        self._inject_dwarf_vectors()
        self._print_summary()

    def save_cache(self, cache_dir: str | Path = None):
        """保存到缓存（含13矮人向量），混淆和非混淆使用不同文件名"""
        if not self._loaded:
            self.load()

        cache_dir = (Path(cache_dir) if cache_dir
                     else self.base_path.parent / '.data_cache')
        cache_dir.mkdir(parents=True, exist_ok=True)

        data = {
            'c_codes':    self.c_codes,
            'cuda_codes': self.cuda_codes,
        }

        cache_file = cache_dir / self._get_cache_filename()
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"Cache saved to: {cache_file}")

    def load_cache(self, cache_dir: str | Path = None) -> bool:
        """从缓存加载数据，并补充/更新13矮人向量，混淆和非混淆使用不同文件名"""
        cache_dir  = (Path(cache_dir) if cache_dir
                      else self.base_path.parent / '.data_cache')
        cache_file = cache_dir / self._get_cache_filename()

        if not cache_file.exists():
            return False

        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.c_codes    = data.get('c_codes', [])
            self.cuda_codes = data.get('cuda_codes', [])
            self._loaded    = True

            # 补充/更新矮人向量（缓存里可能是旧的零向量）
            self._inject_dwarf_vectors()
            self._print_summary()
            return True

        except Exception as e:
            print(f"Failed to load cache: {e}")
            return False

    def refresh_dwarf_and_save(self, cache_dir: str | Path = None):
        """
        重新注入最新的矮人向量并保存缓存
        当 loop_dwarf_by_id.json / kernel_dwarf_by_id.json 更新后调用
        """
        if not self._loaded:
            raise RuntimeError("请先调用 load() 或 load_cache()")
        self._inject_dwarf_vectors()
        self.save_cache(cache_dir)
        print("矮人向量已更新并保存到缓存")

    # ----------------------------------------------------------------
    # 私有方法
    # ----------------------------------------------------------------

    def _inject_dwarf_vectors(self):
        """把13矮人向量注入到每个 code 的字典里"""
        loop_file   = self.dwarf_dir / 'loop_dwarf_by_id.json'
        kernel_file = self.dwarf_dir / 'kernel_dwarf_by_id.json'

        # C loop 侧
        if loop_file.exists():
            with open(loop_file, 'r') as f:
                loop_dwarf = json.load(f)
            injected = 0
            for c in self.c_codes:
                vec = loop_dwarf.get(str(c['id']))
                if vec is not None:
                    c['dwarf_vector'] = vec
                    injected += 1
                elif 'dwarf_vector' not in c:
                    c['dwarf_vector'] = [0.0] * DWARF_DIM
            print(f"  Loop dwarf injected: {injected}/{len(self.c_codes)}")
        else:
            # 确保字段存在
            for c in self.c_codes:
                if 'dwarf_vector' not in c:
                    c['dwarf_vector'] = [0.0] * DWARF_DIM
            print(f"  [SKIP] loop_dwarf_by_id.json not found, using zero vectors")

        # CUDA kernel 侧
        if kernel_file.exists():
            with open(kernel_file, 'r') as f:
                kernel_dwarf = json.load(f)
            injected = 0
            for k in self.cuda_codes:
                vec = kernel_dwarf.get(str(k['id']))
                if vec is not None:
                    k['dwarf_vector'] = vec
                    injected += 1
                elif 'dwarf_vector' not in k:
                    k['dwarf_vector'] = [0.0] * DWARF_DIM
            print(f"  Kernel dwarf injected: {injected}/{len(self.cuda_codes)}")
        else:
            for k in self.cuda_codes:
                if 'dwarf_vector' not in k:
                    k['dwarf_vector'] = [0.0] * DWARF_DIM
            print(f"  [SKIP] kernel_dwarf_by_id.json not found, using zero vectors")

    def _print_summary(self):
        c_files    = len(set(c['filename'] for c in self.c_codes))
        cuda_files = len(set(k['filename'] for k in self.cuda_codes))
        c_projects = set(c['project'] for c in self.c_codes)
        cuda_projects = set(k['project'] for k in self.cuda_codes)
        projects   = len(c_projects)

        print(f"Loaded from cache:")
        print(f"  Projects:     {projects}")
        print(f"  C files:      {c_files}")
        print(f"  CUDA files:   {cuda_files}")
        print(f"  C loops:      {len(self.c_codes)}")
        print(f"  CUDA kernels: {len(self.cuda_codes)}")

        only_c    = c_projects - cuda_projects
        only_cuda = cuda_projects - c_projects

        if only_c:
            print(f"\n[WARN] 只有 C 代码的项目 ({len(only_c)}):")
            for p in sorted(only_c):
                print(f"  - {p}")

        if only_cuda:
            print(f"\n[WARN] 只有 CUDA 代码的项目 ({len(only_cuda)}):")
            for p in sorted(only_cuda):
                print(f"  - {p}")

