# KAGentCUDA

[中文文档](README_CN.md) | [English](README.md)

Knowledge-augmented framework for reliable C-to-CUDA translation with computation-pattern-aware retrieval and structured diagnostic feedback.

## Project Structure

```text
KAGentCUDA/
├── data_module/                  # Data loading & preprocessing
├── retrieval_module/             # Cross-language retrieval module
│   ├── features/                 # Feature extractors (C loops, CUDA kernels)
│   └── ret_methods/              # Two-stage, Procrustes, CodeBERT and GraphCodeBERT retrieval
├── generation_module/            # Code generation module
│   ├── ret_gen_iter.py           # Retrieval + iterative generation
│   ├── eval_types.py             # EvaluationResult, IterativeOptimizer
│   └── prompts_gen_iter.py       # Prompt templates
├── evaluation_module/
│   ├── code_gen_eval/            # L1-L4 evaluation framework
│   │   ├── core/                 # Pipeline, layers, data models
│   │   ├── runners/              # Compile & run wrappers
│   │   └── verifiers/            # Numerical output verification
│   └── code_ret_eval/            # Retrieval result evaluation
├── prompts/                      # LLM prompt templates
├── main_scripts/                 # Baseline and retrieval entry scripts
└── scripts/                      # Data annotation and retrieval utilities
```

---

## Quick Start

Install Python 3.10+ and the packages in `requirements.txt`. Configure an LLM endpoint through environment variables (for example, `ALI_API_KEY` for `--org ali`). Provide compatible tree-sitter grammar libraries using `KAGENTCUDA_C_GRAMMAR_LIB` and `KAGENTCUDA_CUDA_GRAMMAR_LIB`.

Run the commands from the repository root with task sources, Makefiles and supporting inputs in `code_datas/`. For retrieval-enabled generation, prepare:

- `selected_samples/selected_samples.json` and query C sources in `selected_samples/codes/` (or valid manifest `code_path` entries or matching task sources in `code_datas/`).
- `.data_cache/codes_cache.json` at the repository root: paired C–CUDA entries with pair mappings and Dwarf annotations, matching the original sources under `code_datas/type-*/<project>/c/` and `cuda/` and excluding held-out projects. Verify its split before running retrieval.
- `.data_cache/cuda_only_codes_cache.json` at the repository root and `standalone_source_manifest.json` at the repository root, with matching original CUDA files in `CUDA-only/db_cuda_codes/`. Supply the original standalone CUDA sources and their validation manifest before running source-validated standalone retrieval. The manifest must contain `cache_sha256`, `sources_sha256`, `kernel_count` and `file_count` matching the cache and source collection.
- A held-out split JSON and, for the commands below, a loop-review JSON as described in Step 2. These inputs must match the source files; retrieval loads the prepared caches rather than rebuilding them.

### Step 1: Generate Gold Outputs (one-time setup)

L3 correctness checking requires C baseline outputs as ground truth. With task sources, Makefiles and supporting inputs available:

```bash
python main_scripts/run_all_c.py --dataset LARGE

# Resume after interruption
python main_scripts/run_all_c.py --dataset LARGE --skip-completed

# Only specific types
python main_scripts/run_all_c.py --dataset LARGE --types type-1 type-3
```

This produces `outputs_gold_LARGE/`, generated locally rather than committed.

### Step 2: Run Generation & Evaluation

Retrieval uses a held-out split JSON with a `projects` array matching the distinct `project_name` values in `selected_samples/selected_samples.json`. The `--loop_review` JSON uses `version: 1` and a `decisions` array containing `side` (`query` or `paired`), relative `source`, `source_sha256`, `loop_sha256`, `decision` (`keep` or `exclude`) and `reason` for each reviewed loop.

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

## Generation Pipeline

```text
C file -> Two-stage retrieval -> Reference programs -> First translation
                                                           |
                                                L1-L4 evaluation
                                                           | on failure
                                     Iterative fix using diagnostic prompt
                                        (no re-retrieval in later iterations)
```

Retrieval uses identifier-obfuscated representations; generation uses original source programs. One paired C–CUDA example and two standalone CUDA examples are used for generation. The controller retains the best candidate across iterations.

### BM25 baselines and 30B model selection

Use `--retrieval_method bm25` for identifier-obfuscated BM25 or `--retrieval_method raw-bm25` for raw-code ranking; the default remains `ours`. Both use standard positive-IDF BM25 (`k1=1.5`, `b=0.75`), with separate paired C-loop and standalone CUDA-kernel indexes. Scores are maximized over query loops and candidate loops/kernels, selecting one complete paired program and two distinct standalone source files (Top-3). Identifier obfuscation reuses `identifier_v1` only for ranking; prompts always receive original full sources. BM25 does not fit Procrustes or request Dwarf annotations. Held-out, source validation and loop-review checks still apply; missing standalone sources are not replaced by cached snippets.

```bash
python generation_module/ret_gen_iter.py \
    --retrieval_method bm25 \
    --model qwen3-coder-30b-a3b-instruct --org ali \
    --seed 42 --max_samples 1 --max_iterations 3 \
    --input_dir ./selected_samples --parashift_dir ./code_datas \
    --cuda_only_dir ./CUDA-only \
    --heldout_projects ./test_projects.json --loop_review ./loop_review.json
```

Replace `bm25` with `raw-bm25` for the raw baseline or `ours` for the default method. The 30B model is a service model identifier, not a separate algorithm: it can be called only if your configured Ali-compatible endpoint provides **exactly** `qwen3-coder-30b-a3b-instruct`. No weights or credentials are bundled. Both baselines reuse the same L1-L4 structured diagnostics and best-candidate rollback. Their generation outputs/checkpoints live under `cuda_gen_results/<method>/`, and diagnostic workspaces under `optimization_workspace/<method>/`, to avoid mixing baseline runs with `ours`. Use distinct `--retrieval_output_dir` values when comparing methods.

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `qwen3-coder-next` | LLM model name |
| `--org` | `ali` | LLM platform |
| `--seed` | `42` | Random seed |
| `--max_samples` | `1` | Max samples to process |
| `--max_iterations` | `3` | Max iterations per sample |
| `--retrieval_method` | `ours` | `ours`, identifier-obfuscated `bm25`, or `raw-bm25` |
| `--retrieval_alpha` | `0.4` | Structural feature weight (`ours` only) |
| `--retrieval_beta` | `0.6` | Dwarf feature weight |
| `--heldout_projects` | — | Held-out project split for retrieval |
| `--loop_review` | — | Source-bound loop review |
| `--no_retrieval` | off | Use random reference fallback |
| `--retrieval_output_dir` | `./retrieval_results` | Retrieval output directory |

### Output Structure

```text
cuda_gen_results/cuda_codes/<model>/seed<seed>/<type>/
  <type>_<task_id>.cu               <- current best
  iter/
    <type>_<task_id>_iter1.cu       <- per-iteration archive

optimization_workspace/<model>/seed<seed>/<type>_<task_id>/
  iter1/
    pipeline_result.json
    optimization_prompt.txt
```

Checkpoints are saved every 5 samples to `cuda_gen_results/checkpoints/` and loaded on restart.

---

## Evaluation Layers (L1-L4)

| Layer | Check | Pass Criterion |
|-------|-------|----------------|
| L1 | Compilation | `nvcc` returns 0 |
| L2 | Runtime safety | `compute-sanitizer` reports no violations |
| L3 | Correctness | Finite, matching outputs; `high_rel_error_ratio < 1%` and `rmse < 1e-3` |
| L4 | Performance | Valid comparable timing and `speedup > 1.0` |

Failure at any layer short-circuits the pipeline and provides diagnostic feedback for the next iteration. L4 diagnostics include kernel time, H2D/D2H transfer time, the transfer-to-kernel ratio and overlap when available from Nsight Systems; unavailable measurements remain unknown. L4 requires a CPU baseline and validated timing provenance. Proxy timing is recorded separately rather than reported as validated speedup.

### Evaluate a single file manually

With the task data, baseline outputs and a generated CUDA source available:

```bash
python -m evaluation_module.code_gen_eval.main pipeline type-1 c_236 \
    --model my-model --seed 42

python -m evaluation_module.code_gen_eval.main layer -l l3 type-1 c_236
```

---

## Two-Stage Retrieval (standalone)

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

Replace `QUERY_PROJECT` with a held-out project. Stage 1 matches C loops to paired C–CUDA examples. Stage 2 uses structural and Dwarf-pattern features with Procrustes cross-language alignment to retrieve standalone CUDA kernels. The retriever is prepared once and reused across queries.

---

## Basic Retrieval Utilities

For batch paired retrieval and scoring on locally prepared task data and labels:

```bash
python main_scripts/paired4eval_run_full_pipeline.py \
    --method procrustes --task paired \
    --alpha 0.4 --beta 0.6 \
    --parashift_dir ./code_datas
```

The utility connects retrieval, result validity checking, Top-K scoring and NDCG scoring for the supplied inputs. CodeBERT and GraphCodeBERT options require local model weights.

```bash
# Identifier-obfuscated BM25; omit --obfuscate for raw BM25
python main_scripts/paired4eval_run_full_pipeline.py \
    --method bm25 --task paired --obfuscate \
    --parashift_dir ./code_datas --output_dir ./res_rag
```

This BM25 compatibility entry uses the prepared raw paired cache and validates snippets against local original sources. It shares the ranking helpers with generation, but ranks C loops against paired CUDA kernels as a basic scoring utility. An optional `--loop_review ./paired_loop_review.json` applies source-bound paired-loop decisions. The pipeline requires `./res_rag` because existing scorers use fixed result paths. Its output is `res_rag/bm25_output/retrieval_results_bm25[_obf].csv`.

---

## Data

The repository includes C translation queries in `selected_samples/`, paired C/CUDA projects and supporting files in `code_datas/`, and prepared caches in the root `.data_cache/`, including standalone CUDA kernel code snippets. Prepare the standalone original CUDA sources, source-validation manifest, held-out split JSON and loop-review JSON locally as described in Quick Start.

---

## Dependencies

- Python 3.10+ and packages in `requirements.txt`
- `gcc`/`g++` and `make` for task baselines
- `nvcc` (CUDA Toolkit)
- `compute-sanitizer` for L2
- `nsys` for optional L4 profiling
- Compatible tree-sitter C/CUDA grammars (tree-sitter 0.20.4; tree-sitter-cuda 0.21.1)
