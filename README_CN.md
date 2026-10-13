# KAGentCUDA

[中文文档](README_CN.md) | [English](README.md)

面向可靠 C 到 CUDA 翻译的知识增强框架，支持计算模式感知的检索和结构化诊断反馈。

## 项目结构

```text
KAGentCUDA/
├── data_module/                  # 数据加载与预处理
├── retrieval_module/             # 跨语言检索模块
│   ├── features/                 # C 循环与 CUDA Kernel 特征提取
│   └── ret_methods/              # 两阶段、Procrustes、CodeBERT、GraphCodeBERT 检索
├── generation_module/            # 代码生成模块
│   ├── ret_gen_iter.py           # 检索 + 迭代生成
│   ├── eval_types.py             # EvaluationResult、IterativeOptimizer
│   └── prompts_gen_iter.py       # Prompt 模板
├── evaluation_module/
│   ├── code_gen_eval/            # L1-L4 评估框架
│   │   ├── core/                 # 流水线、层级评估器、数据模型
│   │   ├── runners/              # 编译与运行封装
│   │   └── verifiers/            # 数值输出对比
│   └── code_ret_eval/            # 检索结果评估
├── prompts/                      # LLM Prompt 模板
├── main_scripts/                 # 基线和检索入口脚本
└── scripts/                      # 数据标注与检索工具
```

---

## 快速开始

安装 Python 3.10+ 和 `requirements.txt` 中的依赖。通过环境变量配置 LLM 接口（例如 `--org ali` 使用 `ALI_API_KEY`）。通过 `KAGENTCUDA_C_GRAMMAR_LIB` 与 `KAGENTCUDA_CUDA_GRAMMAR_LIB` 指定兼容的 tree-sitter 语法库。

在仓库根目录运行以下命令，并在 `code_datas/` 中准备任务源码、Makefile 和配套输入。启用检索的生成流程还需准备：

- `selected_samples/selected_samples.json` 及 `selected_samples/codes/` 中的查询 C 源码（也可使用清单中的有效 `code_path` 或 `code_datas/` 中对应的任务源码）。
- 仓库根目录的 `.data_cache/codes_cache.json`：包含配对关系和 Dwarf 标注的 C–CUDA 条目，与 `code_datas/type-*/<project>/c/` 和 `cuda/` 中的原始源码对应，且不包含留出项目。运行检索前须核对其划分。
- 仓库根目录的 `.data_cache/cuda_only_codes_cache.json` 和 `standalone_source_manifest.json`，以及与之对应的 `CUDA-only/db_cuda_codes/` 中的原始 CUDA 文件。运行带源码验证的独立 CUDA 检索前须补齐原始独立 CUDA 源码及其验证 manifest。manifest 中的 `cache_sha256`、`sources_sha256`、`kernel_count` 和 `file_count` 必须与缓存及源码集合一致。
- 留出划分 JSON，以及下列命令使用的循环审查 JSON，具体见第二步。这些输入须与源码匹配；检索读取已准备的缓存，不会重新构建。

### 第一步：生成 Gold Output（首次初始化）

L3 正确性验证需要 C baseline 的输出作为参照。准备好任务源码、Makefile 及配套输入后执行：

```bash
python main_scripts/run_all_c.py --dataset LARGE

# 断点续跑
python main_scripts/run_all_c.py --dataset LARGE --skip-completed

# 只跑指定类型
python main_scripts/run_all_c.py --dataset LARGE --types type-1 type-3
```

生成结果保存到 `outputs_gold_LARGE/`，由用户本地生成，不入库。

### 第二步：运行生成与评估

检索需要包含 `projects` 数组的留出划分 JSON，与 `selected_samples/selected_samples.json` 中不同的 `project_name` 一致。`--loop_review` JSON 使用 `version: 1` 和 `decisions` 数组，每个循环的审查条目包含 `side`（`query` 或 `paired`）、相对路径 `source`、`source_sha256`、`loop_sha256`、`decision`（`keep` 或 `exclude`）和 `reason`。

```bash
python generation_module/ret_gen_iter.py \
    --model qwen3-coder-next \
    --seed 42 \
    --max_samples 10 \
    --max_iterations 3 \
    --input_dir ./selected_samples \
    --parashift_dir ./code_datas \
    --cuda_only_dir ./CUDA-only \
    --heldout_projects ./test_projects.json \
    --loop_review ./loop_review.json
```

---

## 生成流水线

```text
C 文件 -> 两阶段检索 -> 参考程序 -> 首次翻译
                                        |
                                    L1-L4 评估
                                        | 失败时
                             基于诊断 Prompt 迭代修复
                              （后续迭代不重新检索）
```

检索使用标识符混淆的表示，生成使用原始源码。生成时提供一个配对 C–CUDA 示例和两个独立 CUDA 示例，控制器保留各次迭代中的最佳候选。

### BM25 基线与 30B 模型选择

`--retrieval_method bm25` 使用标识符混淆 BM25，`--retrieval_method raw-bm25` 使用原始代码排名；默认仍为 `ours`。两者使用标准正 IDF BM25（`k1=1.5`、`b=0.75`），分别建立配对 C 循环与独立 CUDA kernel 索引。对查询循环与候选循环/kernel 的得分取最大值，选出一个完整配对程序和两个不同的独立源码文件（Top-3）。混淆复用 `identifier_v1`，仅用于排名；生成 Prompt 始终使用原始完整源码。BM25 不拟合 Procrustes、不请求 Dwarf 标注，但仍执行留出划分、源码验证与循环审查约束，不会以缓存片段替代缺失的独立源码。

```bash
python generation_module/ret_gen_iter.py \
    --retrieval_method bm25 \
    --model qwen3-coder-30b-a3b-instruct --org ali \
    --seed 42 --max_samples 1 --max_iterations 3 \
    --input_dir ./selected_samples --parashift_dir ./code_datas \
    --cuda_only_dir ./CUDA-only \
    --heldout_projects ./test_projects.json --loop_review ./loop_review.json
```

将 `bm25` 替换为 `raw-bm25` 可运行原始代码基线，替换为 `ours` 使用默认方法。30B 是服务模型标识，不是独立算法：仅当配置的 Ali 兼容接口提供精确名称 **`qwen3-coder-30b-a3b-instruct`** 时才可调用；不附带模型权重或密钥。两种基线复用同一 L1-L4 结构化诊断与最佳候选回滚流程。基线生成结果及断点位于 `cuda_gen_results/<method>/`，诊断工作目录位于 `optimization_workspace/<method>/`，避免与 `ours` 混用；对比不同方法时应指定不同的 `--retrieval_output_dir`。

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | `qwen3-coder-next` | LLM 模型名称 |
| `--org` | `ali` | LLM 平台 |
| `--seed` | `42` | 随机种子 |
| `--max_samples` | `1` | 最大处理样本数 |
| `--max_iterations` | `3` | 每个样本最大迭代次数 |
| `--retrieval_method` | `ours` | `ours`、标识符混淆 `bm25` 或 `raw-bm25` |
| `--retrieval_alpha` | `0.4` | 结构特征权重（仅 `ours`） |
| `--retrieval_beta` | `0.6` | Dwarf 特征权重 |
| `--heldout_projects` | — | 检索留出项目划分 |
| `--loop_review` | — | 绑定源码的循环审查标注 |
| `--no_retrieval` | 关闭 | 使用随机参考示例降级 |
| `--retrieval_output_dir` | `./retrieval_results` | 检索结果目录 |

### 输出目录结构

```text
cuda_gen_results/cuda_codes/<model>/seed<seed>/<type>/
  <type>_<task_id>.cu               <- 当前最佳候选
  iter/
    <type>_<task_id>_iter1.cu       <- 每次迭代存档

optimization_workspace/<model>/seed<seed>/<type>_<task_id>/
  iter1/
    pipeline_result.json
    optimization_prompt.txt
```

每 5 个样本保存断点到 `cuda_gen_results/checkpoints/`，重启时加载断点。

---

## 评估层级（L1-L4）

| 层级 | 检查内容 | 通过判据 |
|------|----------|----------|
| L1 | 编译 | `nvcc` 返回 0 |
| L2 | 运行时安全 | `compute-sanitizer` 无违规报告 |
| L3 | 功能正确性 | 输出有限且匹配；`high_rel_error_ratio < 1%` 且 `rmse < 1e-3` |
| L4 | 性能 | 有效、可比计时且 `speedup > 1.0` |

任意层失败即短路，并向下一次迭代提供诊断反馈。L4 从 Nsight Systems 获取可用的 kernel 时间、H2D/D2H 传输时间、传输与计算时间比及重叠信息；无法获取的指标保留为未知。L4 需要 CPU 基线及经验证的计时来源；代理计时单独记录，不作为已验证加速比。

### 手动评估单个文件

准备好任务数据、基线输出和生成的 CUDA 源码后：

```bash
python -m evaluation_module.code_gen_eval.main pipeline type-1 c_236 \
    --model my-model --seed 42

python -m evaluation_module.code_gen_eval.main layer -l l3 type-1 c_236
```

---

## 两阶段检索（独立使用）

```bash
python main_scripts/run_two_stage.py \
    --input_dir ./code_datas/type-1/QUERY_PROJECT/c \
    --parashift_dir ./code_datas \
    --cuda_only_dir ./CUDA-only \
    --heldout_projects ./test_projects.json \
    --loop_review ./loop_review.json \
    --alpha 0.4 --beta 0.6 \
    --output_dir ./retrieval_results
```

将 `QUERY_PROJECT` 替换为留出项目。Stage 1 将 C 循环匹配到配对 C–CUDA 示例；Stage 2 使用结构与 Dwarf 模式特征及 Procrustes 跨语言对齐检索独立 CUDA kernel。检索器只初始化一次，在不同查询之间复用。

---

## 基础检索工具

对本地准备的任务数据和标签进行配对批量检索与评分：

```bash
python main_scripts/paired4eval_run_full_pipeline.py \
    --method procrustes --task paired \
    --alpha 0.4 --beta 0.6 \
    --parashift_dir ./code_datas
```

该工具针对所提供的输入串联检索、结果有效性检查、Top-K 评分和 NDCG 评分。CodeBERT 与 GraphCodeBERT 选项需要本地模型权重。

```bash
# 标识符混淆 BM25；移除 --obfuscate 即为原始代码 BM25
python main_scripts/paired4eval_run_full_pipeline.py \
    --method bm25 --task paired --obfuscate \
    --parashift_dir ./code_datas --output_dir ./res_rag
```

BM25 兼容入口使用已准备的原始配对缓存，并核验片段与本地原始源码一致。它与生成流程共享排名工具，但针对 C 循环到配对 CUDA kernel 进行基础评分，属于基础工具。可选 `--loop_review ./paired_loop_review.json` 应用绑定源码的配对循环审查。由于现有评分器使用固定结果路径，流水线要求输出到 `./res_rag`；结果为 `res_rag/bm25_output/retrieval_results_bm25[_obf].csv`。

---

## 数据

本仓库包含 `selected_samples/` 中的待翻译 C 查询、`code_datas/` 中的 C/CUDA 双侧项目及配套文件，以及根目录 `.data_cache/` 中的预处理缓存（含独立 CUDA kernel 代码片段）。独立 CUDA 原始源码、源码验证 manifest、留出划分 JSON 与循环审查 JSON 按「快速开始」所述在本地准备。

---

## 依赖

- Python 3.10+ 及 `requirements.txt` 中的包
- `gcc`/`g++` 与 `make`，用于任务基线
- `nvcc`（CUDA Toolkit）
- `compute-sanitizer`（L2）
- `nsys`（可选 L4 profiling）
- 兼容的 tree-sitter C/CUDA 语法库（tree-sitter 0.20.4；tree-sitter-cuda 0.21.1）
