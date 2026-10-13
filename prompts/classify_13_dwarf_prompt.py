from typing import Dict, List, Tuple


class Dwarf13Prompt:
    """
    伯克利13矮人并行计算模式分类 - Prompt设计
    
    设计原则：
    1. 简洁清晰：避免冗余，核心定义优先
    2. 分层结构：先给核心规则，再给详细定义
    3. 实用导向：强调实际判断标准，减少理论描述
    """
    
    DWARFS = {
        "dense_linear_algebra": {
            "name": "稠密线性代数",
            "key": ["三重循环ijk", "矩阵乘法", "向量运算", "稠密数组线性操作"],
            "desc": "稠密数组上的线性代数运算"
        },
        "sparse_linear_algebra": {
            "name": "稀疏线性代数",
            "key": ["CSR/CSC/COO", "row_ptr/col_idx/values", "nnz", "SpMV"],
            "desc": "稀疏矩阵操作"
        },
        "spectral_methods": {
            "name": "谱方法",
            "key": ["FFT", "bit_reverse", "蝴蝶操作", "卷积"],
            "desc": "频率域操作"
        },
        "n_body_methods": {
            "name": "N体方法",
            "key": ["particle", "distance", "sqrt", "力场"],
            "desc": "粒子相互作用模拟"
        },
        "structured_grids": {
            "name": "结构化网格",
            "key": ["a[i-1][j]", "a[i][j-1]", "邻域访问", "模板计算"],
            "desc": "规则网格上的邻域操作"
        },
        "unstructured_grids": {
            "name": "非结构化网格",
            "key": ["mesh/element/node", "a[indices[i]]", "间接寻址"],
            "desc": "不规则网格间接访问"
        },
        "map_reduce": {
            "name": "MapReduce",
            "key": ["独立迭代", "无数据依赖", "sum/+=", "embarrassingly parallel"],
            "desc": "独立处理+聚合"
        },
        "combinational_logic": {
            "name": "组合逻辑",
            "key": ["&/|/^/~", "bit mask", "popcount", ">>/<<"],
            "desc": "位运算为主体"
        },
        "graph_traversal": {
            "name": "图遍历",
            "key": ["BFS/DFS", "queue/stack", "visited", "neighbor"],
            "desc": "图结构遍历"
        },
        "dynamic_programming": {
            "name": "动态规划",
            "key": ["dp[i][j]", "子问题依赖", "递推公式"],
            "desc": "子问题分解与记忆"
        },
        "backtrack_branch_bound": {
            "name": "回溯与分支定界",
            "key": ["递归", "剪枝", "状态回溯"],
            "desc": "搜索解空间"
        },
        "graphical_models": {
            "name": "图模型",
            "key": ["probability", "Bayesian", "Markov", "MCMC"],
            "desc": "概率图模型"
        },
        "finite_state_machines": {
            "name": "有限状态机",
            "key": ["state", "switch/case", "transition"],
            "desc": "状态转换"
        },
    }
    
    CONFUSION_RULES = [
        ("map_reduce vs structured_grids", 
         "map_reduce是'规约/独立处理'，structured_grids是'邻域访问'"),
        ("unstructured_grids vs graph_traversal", 
         "unstructured_grids只有间接访问，graph_traversal有明确的BFS/DFS"),
        ("map_reduce vs combinational_logic", 
         "只有大量位运算才是combinational_logic，单纯if/else是map_reduce"),
    ]
    
    @classmethod
    def build_cuda_prompt(cls, code: str) -> Tuple[str, str]:
        """构建CUDA代码分类的Prompt"""
        
        system_prompt = """你是GPU并行计算模式识别专家。
        
任务：分析CUDA代码，输出13种并行模式的置信度(0-1)。

分析要点：
1. 线程映射：threadIdx/blockIdx如何映射到数据
2. 内存模式：shared memory、coalesced访问
3. 同步与原子：__syncthreads、atomicAdd
4. 计算特征：循环结构、数据依赖

分类规则：
- 每个线程独立处理一个元素 → map_reduce (≥0.6)
- 有邻域访问a[i±1][j±1] → structured_grids
- 三重循环做矩阵运算 → dense_linear_algebra
- 间接访问a[indices[i]] → unstructured_grids
- 大量位运算&/|/^/~ → combinational_logic
- 每个线程独立处理一个元素（无atomicAdd无syncthreads）→ map_reduce ≥0.7
- 有if/else但无位运算 → 不是combinational_logic，combinational_logic ≤0.1

输出要求：
1. 必须输出所有13种模式的置信度
2. 格式为JSON，不要其他文字

{
  "dense_linear_algebra": 0.xx,
  "sparse_linear_algebra": 0.xx,
  "spectral_methods": 0.xx,
  "n_body_methods": 0.xx,
  "structured_grids": 0.xx,
  "unstructured_grids": 0.xx,
  "map_reduce": 0.xx,
  "combinational_logic": 0.xx,
  "graph_traversal": 0.xx,
  "dynamic_programming": 0.xx,
  "backtrack_branch_bound": 0.xx,
  "graphical_models": 0.xx,
  "finite_state_machines": 0.xx,
  "matched_patterns": [],
  "pattern_count": N,
  "reasoning": "判断理由"
}"""
        
        user_prompt = f"""分析以下CUDA代码：

```cuda
{code.strip()[:5000]}
{'...' if len(code) > 5000 else ''}
```

输出JSON格式的分类结果。"""
        
        return system_prompt, user_prompt
    
    @classmethod
    def build_c_prompt(cls, code: str) -> Tuple[str, str]:
        """构建C代码分类的Prompt"""
        
        system_prompt = """你是CPU并行模式识别专家。
        
任务：分析C代码，输出13种并行模式的置信度(0-1)。

分析要点：
1. 循环结构：嵌套层数、数据依赖
2. 数组访问：规则索引a[i][j] vs 间接索引a[b[i]]
3. 规约模式：sum/+=等聚合操作
4. 算法特征：DP表格、图遍历等

分类规则：
- 独立循环无依赖 → map_reduce
- 邻域访问a[i±1][j±1] → structured_grids
- 三重循环矩阵运算 → dense_linear_algebra
- 间接访问a[indices[i]] → unstructured_grids
- dp[i][j]递推 → dynamic_programming

输出要求：
1. 必须输出所有13种模式的置信度
2. matched_patterns取Top3 + 所有≥max_score*0.5的模式
3. 格式为JSON，不要其他文字

{
  "dense_linear_algebra": 0.xx,
  "sparse_linear_algebra": 0.xx,
  "spectral_methods": 0.xx,
  "n_body_methods": 0.xx,
  "structured_grids": 0.xx,
  "unstructured_grids": 0.xx,
  "map_reduce": 0.xx,
  "combinational_logic": 0.xx,
  "graph_traversal": 0.xx,
  "dynamic_programming": 0.xx,
  "backtrack_branch_bound": 0.xx,
  "graphical_models": 0.xx,
  "finite_state_machines": 0.xx,
  "matched_patterns": [],
  "pattern_count": N,
  "reasoning": "判断理由"
}"""
        
        user_prompt = f"""分析以下C代码：

```c
{code.strip()[:5000]}
{'...' if len(code) > 5000 else ''}
```

输出JSON格式的分类结果。"""
        
        return system_prompt, user_prompt


def get_cuda_prompt(code: str) -> Tuple[str, str]:
    """获取CUDA版Prompt"""
    return Dwarf13Prompt.build_cuda_prompt(code)


def get_c_prompt(code: str) -> Tuple[str, str]:
    """获取C代码版Prompt"""
    return Dwarf13Prompt.build_c_prompt(code)


# ========== 向后兼容性类 ==========
# 保留旧的类名，供其他文件导入使用

class Dwarf13PromptBase:
    """向后兼容的基类"""
    DWARF_DEFINITIONS = Dwarf13Prompt.DWARFS


class Dwarf13CudaPrompt(Dwarf13PromptBase):
    """向后兼容的CUDA Prompt类"""
    
    CUDA_SPECIFIC_HINTS = {
        "map_reduce": ["atomicAdd", "shared memory reduction", "blockReduce"],
        "structured_grids": ["thread mapping to grid", "2D/3D grid", "stencil computation"],
        "dense_linear_algebra": ["tile based gemm", "warp tile", "shared memory tile"],
        "sparse_linear_algebra": ["csr spmv", "ellpack format", "sparse kernel"],
    }
    
    @classmethod
    def get_system_prompt(cls) -> str:
        return Dwarf13Prompt.build_cuda_prompt("")[0]
    
    @classmethod
    def get_user_prompt(cls, code: str) -> str:
        return Dwarf13Prompt.build_cuda_prompt(code)[1]


class Dwarf13CPrompt(Dwarf13PromptBase):
    """向后兼容的C代码Prompt类"""
    
    @classmethod
    def get_system_prompt(cls) -> str:
        return Dwarf13Prompt.build_c_prompt("")[0]
    
    @classmethod
    def get_user_prompt(cls, code: str) -> str:
        return Dwarf13Prompt.build_c_prompt(code)[1]


if __name__ == "__main__":
    cuda_sys, cuda_user = Dwarf13Prompt.build_cuda_prompt("")
    c_sys, c_user = Dwarf13Prompt.build_c_prompt("")
    print("CUDA版Prompt长度:", len(cuda_sys))
    print("C代码版Prompt长度:", len(c_sys))
    print("\nPrompt已就绪！")
