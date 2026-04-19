# Tree-Structured Sparse Linear Equations for Fiber Optic Networks

This repository is the reproducibility package for the manuscript *On Large-Scale System of Sparse Linear Equations with Tree Structures and its Applications to Fiber Optic Networks*. It implements the manuscript's TSLE pipeline built on SCAD (Smoothly Clipped Absolute Deviation) estimation and tree-structure-based augmentation.

## Overview

This project implements and compares multiple regularization methods for sparse regression in hierarchical networks:

- **4-Step TSLE Procedure** (primary contribution)
  - Step 1: TSLE on the original matrix
  - Step 2: TSLE on the merged matrix with path reduction
  - Step 3: TSLE-Random with random augmentation
  - Step 4: TSLE-Addition with LUR-guided iterative augmentation
- **Comparison Methods**
  - Lasso
  - L0 regularization
  - MCP

## Installation

```bash
conda create -n jasa python=3.10.19
conda activate jasa

sudo apt-get update
sudo apt-get install r-base r-base-dev

Rscript -e "install.packages(c('ncvreg', 'L0Learn'), repos='https://cloud.r-project.org')"
pip install -r requirements.txt
```

The tested full-recomputation environment is:

- R `4.3.3`
- `ncvreg` `3.16.0`
- `L0Learn` `2.1.0`
- conda OpenBLAS `0.3.30`

See [`R_sessionInfo.txt`](R_sessionInfo.txt) for the recorded R session. Full experimental reruns should use these R package versions; newer CRAN releases may cause small numerical differences. If your CRAN mirror provides newer package versions, install the tested versions above with your preferred R package versioning tool before exact reruns.

## Reproducibility Guide

### Reproduce All Results

```bash
bash scripts/run_paper_tables.sh full all
bash scripts/run_lur_figures.sh full
```

The repository also ships canonical `results/` workbooks and rendered `paper_outputs/` files. Rebuilding outputs from those workbooks should reproduce the submitted tables and figures exactly. Full experimental recomputation is deterministic under the tested software stack above, but R package, BLAS, and platform differences can cause small numerical differences.

For a full table raw-data rerun after deleting generated table result workbooks, use `bash scripts/run_paper_tables.sh full all`. The script allows incomplete intermediate table renders while experiments are still running, then performs a strict final table rebuild after all table experiments finish. Run `bash scripts/run_lur_figures.sh full` separately for the LUR figure raw-data rerun. For experiment-only subset runs, set `SKIP_REFRESH_TABLES=1`.

## Runtime and Hardware Guidance

The repository supports two very different workloads:

- **Export-only rebuilds** from the shipped canonical workbooks in `results/`
- **Full experimental recomputation** from the raw city and synthetic data

For most reviewers, the expensive part is **not** rebuilding the publication PDFs/XLSX files; it is rerunning the experiments.

### Export-only rebuilds

These commands read the shipped canonical workbooks and regenerate `paper_outputs/`:

```bash
conda run -n jasa python -u paper_build/build_tables_table1_s1_s7.py
conda run -n jasa python -u paper_build/build_figures_figure5_s5_s6.py \
  --metrics-dir results/lur_figures \
  --output-dir paper_outputs/figures/readable
```

Observed runtime is typically **minutes**, not hours. A single CPU core and **< 8 GB RAM** are usually enough.

Important distinction:

- The `paper_build/*.py` commands above are **render-only**.
- The shell orchestrators in `scripts/` are **experiment runners**. They recompute results and then refresh the publication outputs.
- For example, `bash scripts/run_paper_tables.sh full synthetic` reruns the three synthetic scenarios before rebuilding `Table S.7`; it is not a simple table-render command.
- `paper_build/build_tables_table1_s1_s7.py` fails fast when expected table entries are missing; pass `--allow-missing` only for diagnostic rendering.

### Full experimental recomputation

The estimates below refer to **clean experimental reruns**, not to export-only rendering. They are planning numbers for this codebase on a large-memory Linux server, not guarantees.

| Workflow | Parallel run | Serial run | Observed memory profile |
|---|---:|---:|---:|
| `bash scripts/run_paper_tables.sh full table1` | ~6-24 hours | often >1 day | `city3` TSLE/SCAD jobs can reach ~140-180 GB RSS |
| `bash scripts/run_paper_tables.sh full other_methods` | ~8-24 hours | often >1 day | heavy `city3` jobs can transiently reach ~260 GB RSS |
| `bash scripts/run_paper_tables.sh full synthetic` | ~1-4 hours | ~4-8 hours | usually <32 GB RAM |
| `bash scripts/run_lur_figures.sh full` | ~4-12 hours | often >1 day | `city3` LUR can reach ~150-165 GB RSS |

Recommended planning numbers:

- **Reviewer-facing rebuild only**: 1 CPU core, 8 GB RAM is enough.
- **Full table rerun**: start with **>= 256 GB RAM** and multiple cores.
- **Full figure rerun**: start with **>= 256 GB RAM**.
- **Running full tables and full figures at the same time**: only do this if you control a machine in the **>1 TB RAM** class.

### Parallelism and checkpointing

- `scripts/run_paper_tables.sh` uses top-level task parallelism via `MAX_JOBS`.
- `scripts/run_lur_figures.sh` runs the three city figure jobs in parallel in `full` mode.
- The long-running experimental drivers all support checkpoint/resume and save progress at seed granularity:
  - `experiments/scad_main_experiment.py`
  - `experiments/other_methods_experiment.py`
  - `experiments/compute_lur_metrics.py`
- The shipped shell entrypoints already enable `--resume --checkpoint-interval 1` for these long jobs.

Operational guidance:

- On shared machines, start with a lower `MAX_JOBS` and scale up only after checking memory and I/O behavior.
- `city3` is the dominant cost driver for both TSLE table runs and LUR figure runs.
- If a long run is interrupted, restarting the same command will resume from the last checkpoint rather than starting over.

### Paper Output Mapping

Subset commands below assume the shipped canonical workbooks for the other tables remain present. If those workbooks were removed and only one subset should be recomputed, use `SKIP_REFRESH_TABLES=1` and rebuild the final tables after all required result workbooks are restored.

| Paper Output | Command | Input Data | Result File | Run Plan |
|---|---|---|---|---|
| Table 1 (TSLE, r=1%,5%,10%) | `bash scripts/run_paper_tables.sh full table1` | `data/city{1,2,3}.xlsx` | `results/main_scad/scad_city{N}_ratio{R}.xlsx` | 100 deterministic random-seed runs; launcher base seed 42; exact schedule: `configs/seeds/tables_current.json` → `tables.scad_main` |
| Table S.1 (tree structure summary) | `bash scripts/run_paper_tables.sh full table1` | `data/city{1,2,3}.xlsx` | `paper_outputs/tables/readable/table_s1_readable.pdf` | No random seed; deterministic topology summary from the city workbooks |
| Table S.2-S.3 (other methods, original) | `bash scripts/run_paper_tables.sh full other_methods` | `data/city{1,2,3}.xlsx` | `results/other_methods/othermethods_original_city{N}_ratio{R}.xlsx` | 100 deterministic random-seed runs; launcher base seed 0; exact schedule: `configs/seeds/tables_current.json` → `tables.other_methods_original` |
| Table S.4-S.5 (other methods, merged) | `bash scripts/run_paper_tables.sh full other_methods` | `data/city{1,2,3}.xlsx` | `results/other_methods/othermethods_merged_city{N}_ratio{R}.xlsx` | 100 deterministic random-seed runs; launcher base seed 0; exact schedule: `configs/seeds/tables_current.json` → `tables.other_methods_merged` |
| Table S.6 (TSLE, r=15%,20%,30%) | `bash scripts/run_paper_tables.sh full table_s6` | `data/city{1,2,3}.xlsx` | `results/main_scad/scad_city{N}_ratio{R}.xlsx` | 100 deterministic random-seed runs; launcher base seed 42; exact schedule: `configs/seeds/tables_current.json` → `tables.scad_main` |
| Table S.7 (synthetic) | `bash scripts/run_paper_tables.sh full synthetic` | `data/synthetic_topologies/synthetic_topologies_combined.xlsx` | `results/synthetic/synthetic_zeta{1pct,5pct,10pct}_alpha10.xlsx` | 100 deterministic random-seed runs; launcher base seed 42; exact schedule: `configs/seeds/tables_current.json` → `tables.synthetic_s7` |
| Figure 5 (City 1 LUR) | `bash scripts/run_lur_figures.sh full` | `data/city1.xlsx` | `results/lur_figures/lur_city1_metrics.xlsx` → `paper_outputs/figures/readable/figure5_city1.pdf` | 100 deterministic random-seed runs; launcher base seed 42; exact schedule: `configs/seeds/figures/city1_final.json` |
| Figure S.5 (City 2 LUR) | `bash scripts/run_lur_figures.sh full` | `data/city2.xlsx` | `results/lur_figures/lur_city2_metrics.xlsx` → `paper_outputs/figures/readable/figure_s5_city2.pdf` | 100 deterministic random-seed runs; launcher base seed 42; exact schedule: `configs/seeds/figures/city2_final.json` |
| Figure S.6 (City 3 LUR) | `bash scripts/run_lur_figures.sh full` | `data/city3.xlsx` | `results/lur_figures/lur_city3_metrics.xlsx` → `paper_outputs/figures/readable/figure_s6_city3.pdf` | 100 deterministic random-seed runs; launcher base seed 42; exact schedule: `configs/seeds/figures/city3_final.json` |

### How to Find Table 1 Numbers in Result Files

Each TSLE result workbook (for example [`results/main_scad/scad_city1_ratio0.01.xlsx`](results/main_scad/scad_city1_ratio0.01.xlsx)) contains:

- `Summary`: mean and standard deviation for Accuracy, Precision, Recall, and F0.5 by TSLE step
- `Metadata`: run configuration, including `base_seed`, `experiment_count`, sparsity ratio, and paper tag
- `Individual_Results`: per-experiment metrics with seed numbers
- `Seed_Status`: checkpoint/resume status for completed seeds, when present

Table 1 values correspond to the `*_mean` columns in `Summary`.

### Figure 5 Code Location

Figure 5 and supplementary Figures S.5 and S.6 are produced by:

1. [`experiments/compute_lur_metrics.py`](experiments/compute_lur_metrics.py)
2. [`postprocess/merge_lur_results.py`](postprocess/merge_lur_results.py)
3. [`paper_build/build_figures_figure5_s5_s6.py`](paper_build/build_figures_figure5_s5_s6.py)
4. [`scripts/run_lur_figures.sh`](scripts/run_lur_figures.sh)

In `full` mode, [`scripts/run_lur_figures.sh`](scripts/run_lur_figures.sh) recomputes the shipped article figures from deterministic 100-run seed plans. In `smoke` mode it falls back to a lighter sharded validation run.

## Data Documentation

### Real City Networks

`data/city1.xlsx`, `data/city2.xlsx`, and `data/city3.xlsx` store hierarchical network topologies. The loader reads topology endpoints from Excel columns B:C (`usecols=[1, 2]`) and treats them as:

- parent/source node
- child/target node

### Synthetic Topologies

The synthetic study uses `data/synthetic_topologies/synthetic_topologies_combined.xlsx`.
This workbook contains the three Table S.7 scenarios only:

| Manuscript label | Internal hard-edge count | Global alpha |
|--------|--------|--------|
| `zeta=1%` | `100` | `10%` |
| `zeta=5%` | `500` | `10%` |
| `zeta=10%` | `1000` | `10%` |

Current workbook sheet names follow the manuscript variables directly:

| Workbook sheet name | Manuscript interpretation |
|--------|--------|
| `synthetic_zeta1pct_alpha10` | `zeta=1%, alpha=10%` |
| `synthetic_zeta5pct_alpha10` | `zeta=5%, alpha=10%` |
| `synthetic_zeta10pct_alpha10` | `zeta=10%, alpha=10%` |

Each edge sheet contains the following variables:

| Variable | Meaning |
|--------|--------|
| `edge_id` | BFS-order edge index used to align workbook rows with `Tree.edges` and the design matrix columns |
| `parent_label` | Parent node label of the directed tree edge |
| `child_label` | Child node label of the directed tree edge |
| `x_true` | Template edge coefficient used when defining the hard region; in the experiment runner, hard edges copy this value and non-hard edges are regenerated at runtime |
| `is_hard` | Binary indicator for whether the edge belongs to the injected hard region |

The workbook also contains a `manifest` sheet. Its variables are:

| Variable | Meaning |
|--------|--------|
| `sheet` | Canonical sheet name for the synthetic case |
| `scenario_label` | Human-readable manuscript label for the synthetic case |
| `legacy_sheet` | Legacy sheet name kept for compatibility with older outputs |
| `zeta` | Proportion of injected hard-region edges used to name the Table S.7 scenario |
| `seed` | Random seed used when generating the stored topology workbook |
| `target_ratio` | Target global asymmetry ratio used when generating the topology workbook; in the current manuscript workflow this is always `0.10` |
| `alpha_merged` | Non-hard edge activation probability used for the merged topology during topology generation |
| `alpha_raw` | Non-hard edge activation probability used for the raw topology during topology generation |
| `left_hard_target` | Requested number of hard-region edges on the left-side chain |
| `left_hard_real` | Actual number of hard-region edges realized after respecting structural limits |
| `left_hard_backbone` | Number of hard-region backbone edges with nonzero coefficients |
| `left_hard_backbone_max_depth` | Maximum backbone depth included in the hard region |
| `left_hard_leaf_max_depth` | Maximum leaf depth included in the hard region |
| `left_edges_total` | Total number of left-side edges in the synthetic tree |
| `right_edges_merged` | Total number of right-side edges in the merged topology |
| `right_edges_raw` | Total number of right-side edges in the raw topology |
| `total_edges_merged` | Total number of edges in the merged topology |
| `total_edges_raw` | Total number of edges in the raw topology |
| `merged_leaves` | Number of leaves in the merged topology |
| `raw_leaves` | Number of leaves in the raw topology |
| `merged_nonzero` | Number of nonzero coefficients stored in the generated merged workbook template |
| `raw_nonzero` | Number of nonzero coefficients stored in the generated raw workbook template |
| `ratio_merged_actual` | `merged_nonzero / total_edges_merged` |
| `ratio_raw_actual` | `raw_nonzero / total_edges_raw` |

Important distinction:

- The workbook `x_true` values define the hard-region template and topology metadata.
- During each synthetic experiment, the runner reconstructs a fresh `x_clean`:
  hard edges are fixed from the workbook and all non-hard edges are resampled so
  that the global asymmetry ratio remains `alpha=10%`.

## Random Seeds

The shipped paper outputs are based on deterministic repeated runs.

- Real-city table results are shipped as `100`-run canonical workbooks.
- Final LUR figure results for `Figure 5`, `Figure S.5`, and `Figure S.6` are also shipped as `100`-run canonical workbooks.
- Synthetic `Table S.7` uses `100` runs for each of the three synthetic scenarios.
- Synthetic topology generation itself is fixed and deterministic.
- Table and figure seed schedules are stored under `configs/seeds/` and are read automatically by the reproducibility scripts in full mode. TSLE, synthetic, and LUR figure launchers use base seed 42; other-method launchers use base seed 0. In full mode, explicit seed lists from the JSON files listed above define the exact shipped schedules, including final replacements or ordering.

## Project Structure

```text
jasa_minor/
├── README.md
├── requirements.txt
├── configs/
├── src/
├── experiments/
├── scripts/
├── data/
├── results/
│   ├── main_scad/
│   ├── other_methods/
│   ├── synthetic/
│   └── lur_figures/
├── paper_outputs/
│   ├── tables/readable/
│   └── figures/
│       └── readable/
```

## Core Scripts

### experiments

- [`scad_main_experiment.py`](experiments/scad_main_experiment.py)
  - real-city SCAD 4-step experiments for Table 1 and Table S.6
- [`other_methods_experiment.py`](experiments/other_methods_experiment.py)
  - real-city L0 / Lasso / MCP experiments for Table S.2-S.5
- [`synthetic_experiment.py`](experiments/synthetic_experiment.py)
  - synthetic Table S.7 experiments
- [`compute_lur_metrics.py`](experiments/compute_lur_metrics.py)
  - compute per-seed LUR metrics for Figure 5, Figure S.5, Figure S.6

### data/synthetic_topologies

- [`build_synthetic_topologies.py`](data/synthetic_topologies/build_synthetic_topologies.py)
  - generate the canonical `synthetic_topologies_combined.xlsx` workbook

### postprocess

- [`merge_table_results.py`](postprocess/merge_table_results.py)
  - merge table experiment shards and aggregate canonical table result workbooks
- [`merge_lur_results.py`](postprocess/merge_lur_results.py)
  - merge LUR figure shard workbooks into canonical city workbooks

### paper_build

- [`build_tables_table1_s1_s7.py`](paper_build/build_tables_table1_s1_s7.py)
  - generate the final readable outputs for Table 1 and Tables S.1-S.7
- [`render_tables_readable.py`](paper_build/render_tables_readable.py)
  - render publication-style PDF/XLSX/CSV tables
- [`build_figures_figure5_s5_s6.py`](paper_build/build_figures_figure5_s5_s6.py)
  - generate Figure 5, Figure S.5, Figure S.6, plus their plot-data files

### configs/seeds

- [`figures/city1_final.json`](configs/seeds/figures/city1_final.json)
  - exact per-alpha seed list for the shipped Figure 5
- [`figures/city2_final.json`](configs/seeds/figures/city2_final.json)
  - exact per-alpha seed list for the shipped Figure S.5
- [`figures/city3_final.json`](configs/seeds/figures/city3_final.json)
  - exact per-alpha seed list for the shipped Figure S.6
- [`tables_current.json`](configs/seeds/tables_current.json)
  - current shipped table seed schedules for canonical outputs

### scripts

- [`run_paper_tables.sh`](scripts/run_paper_tables.sh)
  - master table orchestrator
- [`run_lur_figures.sh`](scripts/run_lur_figures.sh)
  - master figure orchestrator
- [`run_synthetic_experiments.sh`](scripts/run_synthetic_experiments.sh)
  - optional tmux convenience launcher for quick synthetic runs; canonical Table S.7 reproduction uses `bash scripts/run_paper_tables.sh full synthetic`

## Algorithm Details

### 4-Step TSLE Procedure

1. **Step 1: Original Matrix**
   - Apply SCAD estimation on the original design matrix
   - Use Wasserstein distance for lambda selection
2. **Step 2: Merged Matrix**
   - Apply path merging for dimension reduction
   - Compress linear paths to single edges
3. **Step 3: TSLE-Random**
   - Add random tree-based constraints
   - Provide the manuscript's random augmentation baseline
4. **Step 4: TSLE-Addition**
   - Add LUR-guided constraints using confidence-based node classification
   - Iteratively refine the active set until convergence

### LUR-Oriented Node Classification

- **Subhigh nodes**: connected to binary edges
- **High nodes**: restrictive edges based on tree structure
- **Ultra-high nodes**: non-restrictive edges

## Evaluation Metrics

- **Accuracy**: `1 - (errors / total_coefficients)`
- **Precision**: `TP / (TP + FP)`
- **Recall**: `TP / (TP + FN)`
- **F0.5 Score**: `(1.25 * P * R) / (0.25 * P + R)`
- **Coverage Accuracy**: accuracy on a selected coefficient subset

## Configuration

### Command Line Options

**TSLE main experiment**

```bash
python experiments/scad_main_experiment.py \
    --data DATA_FILE \
    --sparsity RATIO \
    --experiments N \
    --save OUTPUT_FILE
```

**Synthetic experiment**

```bash
python experiments/synthetic_experiment.py \
    --zeta 0.01 \
    --ratio 0.10 \
    --experiments N
```

**Other methods**

```bash
python experiments/other_methods_experiment.py \
    --data DATA_FILE \
    --method METHOD \
    --experiments N \
    --ratio SPARSITY \
    --merge \
    --seed SEED \
    --save OUTPUT_FILE
```

### Algorithm Parameters

- Lambda selection: Wasserstein distance with 5-step convergence
- SCAD penalty gamma: 3.7
- Matrix reduction: BFS-based edge ordering with path merging

## Requirements

**System**: tested on Linux/WSL. Large experiments benefit from high-memory machines. No GPU is required. macOS may require GNU bash/coreutils and R setup adjustments because the shell scripts use Linux-oriented tools such as `flock`, `wait -n`, and optional `tmux`.

See [`requirements.txt`](requirements.txt) for the Python dependency list.
