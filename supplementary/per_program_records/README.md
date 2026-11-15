# Per-Program Experiment Records

This directory contains per-program records supporting the generation,
refinement, community-speedup, and failure-case results.

## Files

| File | Granularity | Contents |
|---|---|---|
| `experiment_config.json` | experiment | Backbones, decoding parameters, verification thresholds, response budgets, retrieval settings, and stopping rules. |
| `01_generation_per_program.csv` | program × seed × method × k | Four-layer 0/1 outcomes `com` / `run` / `cor` / `fast` for every program. Seeds: 42 (all methods, @1/@3/@5) and 2026/10086 (LLM methods, @3/@5). |
| `02_rq4_trajectory.csv` | program × config × attempt (1–3) | Retained-candidate `fast` outcomes and controller records. |
| `03_community_speedup.csv` | 34 programs (30 PolyBench + 4 NPB) | Timing, speedup, and `S` for PPCG, Qwen3-Coder, Claude-4.5, KAGentCUDA, and the community kernel. |
| `04_failure_cases.csv` | 40 residual-failure cases | Final failing layer (L1–L4), error category, and whether the final attempt resolved it. |
| `05_three_shot_manifests.csv` | seed × program × slot (1–3) | The three C–CUDA exemplars drawn for Baseline-3-shot for each of the 109 evaluated programs under seeds 42 / 2026 / 10086 (981 rows = 109 × 3 × 3). For each query the three exemplars are sampled from the paired corpus excluding that query program. |
| `three_shot_examples.zip` | seed × program × slot | Self-contained copy of the actual exemplar source files: `seed<seed>/<program_id>_<query>/shot<slot>_<project>/{c,cuda}/...` (327 program folders, 1,962 source files), with an in-package `manifest.csv` (package-relative paths) and `README.txt`. No external path is needed. |

## Columns

`01_generation_per_program.csv`
: `program_id` (`P001`–`P109`), `seed`, `family`, `backbone`, `method`,
  `k` (1/3/5), and `com`,`run`,`cor`,`fast` ∈ {0,1}. The four layers are
  nested (`fast ⊆ cor ⊆ run ⊆ com`) and ranks satisfy @1 ⊆ @3 ⊆ @5.

`02_rq4_trajectory.csv`
: `program_id`, `config`, `attempt`, `fast_retained` ∈ {0,1},
  `controller_action`.

`03_community_speedup.csv`
: `source`, `benchmark`, and per-tool `*_time` / `*_speedup` / `*_S` columns,
  plus `community_time`, `community_speedup`, `sequential_time`.

`04_failure_cases.csv`
: `case_id`, `final_layer` (1–4), `error_category`, `resolved` ∈ {0,1}.

`05_three_shot_manifests.csv`
: `seed`, `program_id` (`P001`–`P109`, traversal order of the evaluated
  programs), `query_type`, `query_project`, `slot` (1–3), `shot_type`,
  `shot_project`, `c_file`, `cuda_file`, `c_path`, `cuda_path`. Each
  (seed, program_id) group has exactly three distinct exemplars and never
  contains the query program itself.

## Method naming

Methods use the manuscript's table names: `Baseline`, `Baseline-CoT`,
`Baseline-3-shot`, `Tool-Feedback`, `LASSI-inspired`, and `KAGentCUDA` across
the four backbones; the three compilers (PPCG / BabelTower / CodeRosetta);
the ablations (`w/o retrieval / refinement / sanitizer / structured diag /
rollback / rollback+raw diag / Procrustes`); and the RQ3 controlled-retrieval
configurations (`Retrieval: BM25 / raw-BM25 / CodeBERT / GraphCodeBERT`,
`w/o refinement (retrieval study)`, `w/o Procrustes`).
