#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Seed Scheme
#
# In full mode, seeds are read from configs/seeds/tables_current.json.
# In smoke mode, contiguous seeds from base_seed are used for quick validation.
# Step 3 random augmentation: fixed seed=42
# =============================================================================

WORKDIR="$(cd "$(dirname "$0")/../.." && pwd)"
CONDA_ENV="${CONDA_ENV:-jasa}"
MODE="${1:-smoke}"
TARGET="${2:-all}"
PAPER_TAG="${PAPER_TAG:-paper_${MODE}_$(date +%Y%m%d_%H%M%S)}"
CPU_COUNT="$(nproc 2>/dev/null || echo 4)"
CITIES="${CITIES:-1 2 3}"

SEED_MANIFEST="${WORKDIR}/configs/seeds/tables_current.json"

FULL_EXPERIMENTS="${FULL_EXPERIMENTS:-100}"
FULL_CITY12_EXPERIMENTS="${FULL_CITY12_EXPERIMENTS:-$FULL_EXPERIMENTS}"
FULL_CITY3_EXPERIMENTS="${FULL_CITY3_EXPERIMENTS:-$FULL_EXPERIMENTS}"
OTHER_FULL_CITY12_EXPERIMENTS="${OTHER_FULL_CITY12_EXPERIMENTS:-$FULL_EXPERIMENTS}"
OTHER_FULL_CITY3_EXPERIMENTS="${OTHER_FULL_CITY3_EXPERIMENTS:-$FULL_EXPERIMENTS}"
SMOKE_EXPERIMENTS="${SMOKE_EXPERIMENTS:-5}"
SMOKE_CITY3_EXPERIMENTS="${SMOKE_CITY3_EXPERIMENTS:-3}"
SYNTHETIC_FULL_EXPERIMENTS="${SYNTHETIC_FULL_EXPERIMENTS:-$FULL_EXPERIMENTS}"
SYNTHETIC_SMOKE_EXPERIMENTS="${SYNTHETIC_SMOKE_EXPERIMENTS:-$SMOKE_EXPERIMENTS}"
OTHER_METHODS_THREADS="${OTHER_METHODS_THREADS:-8}"
EXPORT_TEX="${EXPORT_TEX:-0}"

if [[ "$MODE" == "full" ]]; then
  DEFAULT_MAX_JOBS=8
else
  DEFAULT_MAX_JOBS=12
fi
if (( CPU_COUNT < DEFAULT_MAX_JOBS )); then
  DEFAULT_MAX_JOBS="$CPU_COUNT"
fi
MAX_JOBS="${MAX_JOBS:-$DEFAULT_MAX_JOBS}"

LOG_DIR="results/logs/paper_tables/${PAPER_TAG}"

cd "$WORKDIR"

mkdir -p results/raw/main_scad
mkdir -p results/raw/other_methods
mkdir -p results/raw/synthetic
mkdir -p results/tables
mkdir -p "$LOG_DIR"

# Extract seed list from tables_current.json for a given key path
extract_seeds() {
  python3 -c "
import json, sys
d = json.load(open('${SEED_MANIFEST}'))['tables']
for k in sys.argv[1:]:
    d = d[k]
print(','.join(str(s) for s in d))
" "$@"
}

declare -a RUNNING_PIDS=()
declare -A PID_TO_NAME=()
declare -A PID_TO_LOG=()
declare -a FAILED_TASKS=()

refresh_tables() {
  local missing_mode="${1:-strict}"
  if [[ "${SKIP_REFRESH_TABLES:-0}" == "1" ]]; then
    echo "Skipping refresh_tables (SKIP_REFRESH_TABLES=1)"
    return
  fi
  echo
  local allow_arg=""
  if [[ "$missing_mode" == "allow-missing" ]]; then
    echo "Refreshing readable table outputs with --allow-missing..."
    allow_arg=" --allow-missing"
  else
    echo "Refreshing readable table outputs..."
  fi
  mkdir -p /tmp
  flock -x /tmp/jasa_refresh_tables.lock -c "conda run -n '$CONDA_ENV' python -u src/make_paper_results.py --tables${allow_arg}"
}

export_tex_if_requested() {
  if [[ "$EXPORT_TEX" != "1" ]]; then
    return
  fi

  echo
  echo "Exporting combined TeX summary..."
  conda run -n "$CONDA_ENV" python -u src/make_paper_results.py --tables --export-tex
}

experiments_for_city() {
  local city="$1"
  if [[ "$MODE" == "full" ]]; then
    if [[ "$city" == "3" ]]; then
      echo "$FULL_CITY3_EXPERIMENTS"
    else
      echo "$FULL_CITY12_EXPERIMENTS"
    fi
    return
  fi

  if [[ "$city" == "3" ]]; then
    echo "$SMOKE_CITY3_EXPERIMENTS"
  else
    echo "$SMOKE_EXPERIMENTS"
  fi
}

other_experiments_for_city() {
  local city="$1"
  if [[ "$MODE" == "full" ]]; then
    if [[ "$city" == "3" ]]; then
      echo "$OTHER_FULL_CITY3_EXPERIMENTS"
    else
      echo "$OTHER_FULL_CITY12_EXPERIMENTS"
    fi
    return
  fi

  if [[ "$city" == "3" ]]; then
    echo "$SMOKE_CITY3_EXPERIMENTS"
  else
    echo "$SMOKE_EXPERIMENTS"
  fi
}

synthetic_experiments() {
  if [[ "$MODE" == "full" ]]; then
    echo "$SYNTHETIC_FULL_EXPERIMENTS"
  else
    echo "$SYNTHETIC_SMOKE_EXPERIMENTS"
  fi
}

other_output_matches_expected() {
  local output_file="$1"
  local expected_tag="$2"
  local expected_experiments="$3"
  local expected_merge="$4"

  python - "$output_file" "$expected_tag" "$expected_experiments" "$expected_merge" <<'PY'
import sys
from pathlib import Path
import pandas as pd

path = Path(sys.argv[1])
expected_tag = sys.argv[2]
expected_experiments = int(sys.argv[3])
expected_merge = sys.argv[4].lower() == "true"

if not path.exists():
    sys.exit(1)

try:
    metadata = pd.read_excel(path, sheet_name="Metadata")
except Exception:
    sys.exit(1)

if "Key" not in metadata.columns or "Value" not in metadata.columns:
    sys.exit(1)

meta = dict(zip(metadata["Key"], metadata["Value"]))
paper_tag = str(meta.get("paper_tag", ""))
num_experiments = int(meta.get("num_experiments", 0))
use_merge_raw = meta.get("use_merge", 0)
use_merge = str(use_merge_raw).strip().lower() in {"1", "true", "yes"}

if paper_tag == expected_tag and num_experiments == expected_experiments and use_merge == expected_merge:
    sys.exit(0)
sys.exit(1)
PY
}

reap_finished_tasks() {
  local still_running=()
  local pid
  for pid in "${RUNNING_PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      still_running+=("$pid")
      continue
    fi

    local name="${PID_TO_NAME[$pid]}"
    local logfile="${PID_TO_LOG[$pid]}"
    if wait "$pid"; then
      echo "[done] $name"
    else
      echo "[fail] $name"
      echo "       log: $logfile"
      FAILED_TASKS+=("$name :: $logfile")
    fi
    unset 'PID_TO_NAME[$pid]'
    unset 'PID_TO_LOG[$pid]'
  done

  RUNNING_PIDS=("${still_running[@]}")
}

wait_for_slot() {
  while (( ${#RUNNING_PIDS[@]} >= MAX_JOBS )); do
    reap_finished_tasks
    sleep 1
  done
}

wait_for_all_tasks() {
  while (( ${#RUNNING_PIDS[@]} > 0 )); do
    reap_finished_tasks
    sleep 1
  done
}

submit_task() {
  local task_name="$1"
  local command="$2"
  local logfile="$LOG_DIR/${task_name}.log"

  wait_for_slot

  echo "[launch] $task_name"
  bash -lc "cd '$WORKDIR' && $command" >"$logfile" 2>&1 &
  local pid=$!

  RUNNING_PIDS+=("$pid")
  PID_TO_NAME["$pid"]="$task_name"
  PID_TO_LOG["$pid"]="$logfile"
}

finish_stage() {
  local stage_name="$1"
  local failed_before="$2"

  wait_for_all_tasks

  if [[ -d "results/raw/main_scad/shards/${PAPER_TAG}" ]]; then
    echo
    echo "Merging SCAD shard outputs for ${stage_name}..."
    conda run -n "$CONDA_ENV" python -u src/merge_results.py --merge-shards scad \
      --shard-dir "results/raw/main_scad/shards/${PAPER_TAG}" \
      --output-dir "results/raw/main_scad"
  fi

  if [[ -d "results/raw/other_methods/shards/${PAPER_TAG}" ]]; then
    echo
    echo "Merging other-method shard outputs for ${stage_name}..."
    conda run -n "$CONDA_ENV" python -u src/merge_results.py --merge-shards other \
      --shard-dir "results/raw/other_methods/shards/${PAPER_TAG}" \
      --output-dir "results/raw/other_methods"
  fi

  if [[ -d "results/raw/synthetic/shards/${PAPER_TAG}" ]]; then
    echo
    echo "Merging synthetic shard outputs for ${stage_name}..."
    conda run -n "$CONDA_ENV" python -u src/merge_results.py --merge-shards synthetic \
      --shard-dir "results/raw/synthetic/shards/${PAPER_TAG}" \
      --output-dir "results/raw/synthetic"
  fi

  refresh_tables allow-missing

  local failed_after=${#FAILED_TASKS[@]}
  if (( failed_after > failed_before )); then
    echo
    echo "Stage failed: $stage_name"
    printf '%s\n' "${FAILED_TASKS[@]}"
    exit 1
  fi
}

submit_scad_case() {
  local city="$1"
  local ratio="$2"
  local n_exp
  n_exp="$(experiments_for_city "$city")"

  local seed_flag=""
  if [[ "$MODE" == "full" ]]; then
    local seeds
    seeds="$(extract_seeds scad_main "city${city}" "${ratio}")"
    seed_flag="--seed-values '${seeds}'"
  fi

  local task_name="scad_c${city}_r${ratio}"
  local outfile="results/raw/main_scad/scad_city${city}_ratio${ratio}.xlsx"
  local cmd="conda run -n ${CONDA_ENV} python -u src/tsle_experiment.py \
    --data data/city${city}.xlsx \
    --experiments ${n_exp} \
    --sparsity ${ratio} \
    --base-seed 42 \
    ${seed_flag} \
    --resume \
    --checkpoint-interval 1 \
    --paper-tag '${PAPER_TAG}' \
    --save ${outfile}"
  submit_task "$task_name" "$cmd"
}

submit_other_case() {
  local matrix_type="$1"
  local city="$2"
  local ratio="$3"
  local n_exp
  n_exp="$(other_experiments_for_city "$city")"
  local task_name="other_${matrix_type}_c${city}_r${ratio}"
  local outfile="results/raw/other_methods/othermethods_${matrix_type}_city${city}_ratio${ratio}.xlsx"
  local merge_flag=""
  if [[ "$matrix_type" == "merged" ]]; then
    merge_flag="--merge"
  fi
  local expected_merge="false"
  if [[ "$matrix_type" == "merged" ]]; then
    expected_merge="true"
  fi

  if other_output_matches_expected "$outfile" "$PAPER_TAG" "$n_exp" "$expected_merge"; then
    echo "[skip] $task_name (existing output matches requested configuration)"
    return
  fi

  local env_prefix="OMP_NUM_THREADS=${OTHER_METHODS_THREADS} OPENBLAS_NUM_THREADS=${OTHER_METHODS_THREADS} MKL_NUM_THREADS=${OTHER_METHODS_THREADS} NUMEXPR_NUM_THREADS=${OTHER_METHODS_THREADS} VECLIB_MAXIMUM_THREADS=${OTHER_METHODS_THREADS}"

  local seed_flag=""
  if [[ "$MODE" == "full" ]]; then
    local seeds
    seeds="$(extract_seeds "other_methods_${matrix_type}" "city${city}" "${ratio}")"
    seed_flag="--seed-values '${seeds}'"
  fi

  local cmd="${env_prefix} conda run -n ${CONDA_ENV} python -u src/lasso_mcp_l0_experiment.py \
    --data data/city${city}.xlsx \
    --experiments ${n_exp} \
    --ratio ${ratio} \
    --base-seed 0 \
    ${seed_flag} \
    --resume \
    --checkpoint-interval 1 \
    ${merge_flag} \
    --paper-tag '${PAPER_TAG}' \
    --save ${outfile}"
  submit_task "$task_name" "$cmd"
}

submit_synthetic_case() {
  local city="$1"
  local ratio="$2"
  local n_exp
  n_exp="$(synthetic_experiments)"
  local zeta
  local zeta_tag
  case "$city" in
    1) zeta="0.01"; zeta_tag="1pct" ;;
    2) zeta="0.05"; zeta_tag="5pct" ;;
    3) zeta="0.10"; zeta_tag="10pct" ;;
    *) echo "Unsupported synthetic case index: $city" >&2; return 1 ;;
  esac
  local alpha_digits="${ratio/./}"
  local alpha_pct=$((10#${alpha_digits}))
  local task_name="synthetic_zeta${zeta_tag}_alpha${alpha_pct}"
  local outfile="results/raw/synthetic/synthetic_zeta${zeta_tag}_alpha${alpha_pct}.xlsx"

  local seed_flag=""
  if [[ "$MODE" == "full" ]]; then
    local seeds
    seeds="$(extract_seeds synthetic_s7 "synthetic_zeta${zeta_tag}_alpha${alpha_pct}")"
    seed_flag="--seed-values '${seeds}'"
  fi

  local cmd="conda run -n ${CONDA_ENV} python -u src/synthetic_experiment.py \
    --zeta ${zeta} \
    --ratio ${ratio} \
    --experiments ${n_exp} \
    --base-seed 42 \
    ${seed_flag} \
    --paper-tag '${PAPER_TAG}' \
    --save ${outfile}"
  submit_task "$task_name" "$cmd"
}

run_table1() {
  local failed_before=${#FAILED_TASKS[@]}
  echo
  echo "Submitting Table 1 batch..."
  for city in $CITIES; do
    for ratio in 0.01 0.05 0.10; do
      submit_scad_case "$city" "$ratio"
    done
  done
  finish_stage "Table 1" "$failed_before"
}

run_table_s6() {
  local failed_before=${#FAILED_TASKS[@]}
  echo
  echo "Submitting Table S.6 batch..."
  for city in $CITIES; do
    for ratio in 0.15 0.20 0.30; do
      submit_scad_case "$city" "$ratio"
    done
  done
  finish_stage "Table S.6" "$failed_before"
}

run_other_subset() {
  local matrix_type="$1"; shift
  local stage_name="$1"; shift
  local failed_before=${#FAILED_TASKS[@]}
  echo
  echo "Submitting ${stage_name} batch (${matrix_type})..."
  local city ratio
  for city in $CITIES; do
    for ratio in "$@"; do
      submit_other_case "$matrix_type" "$city" "$ratio"
    done
  done
  finish_stage "$stage_name" "$failed_before"
}

run_other_tables() {
  local failed_before
  failed_before=${#FAILED_TASKS[@]}
  echo
  echo "Submitting other-methods original batch..."
  for city in $CITIES; do
    for ratio in 0.01 0.05 0.10 0.15 0.20 0.30; do
      submit_other_case "original" "$city" "$ratio"
    done
  done
  finish_stage "Tables S.2-S.3" "$failed_before"

  failed_before=${#FAILED_TASKS[@]}
  echo
  echo "Submitting other-methods merged batch..."
  for city in $CITIES; do
    for ratio in 0.01 0.05 0.10 0.15 0.20 0.30; do
      submit_other_case "merged" "$city" "$ratio"
    done
  done
  finish_stage "Tables S.4-S.5" "$failed_before"
}

run_supp_tables() {
  run_table_s6
  run_other_tables
}

run_synthetic_tables() {
  local failed_before=${#FAILED_TASKS[@]}
  echo
  echo "Submitting synthetic Table S.7 batch..."
  for city in $CITIES; do
    submit_synthetic_case "$city" "0.10"
  done
  finish_stage "Table S.7" "$failed_before"
}

cleanup_on_interrupt() {
  if (( ${#RUNNING_PIDS[@]} > 0 )); then
    echo
    echo "Interrupt received. Stopping running jobs..."
    kill "${RUNNING_PIDS[@]}" 2>/dev/null || true
  fi
}
trap cleanup_on_interrupt INT TERM

echo "=============================================="
echo "Paper Tables Runner"
echo "=============================================="
echo "Working directory: $WORKDIR"
echo "Conda env: $CONDA_ENV"
echo "Mode: $MODE"
echo "Target: $TARGET"
echo "Paper tag: $PAPER_TAG"
echo "CPU count: $CPU_COUNT"
echo "Max concurrent jobs: $MAX_JOBS"
echo "Logs: $LOG_DIR"

refresh_tables allow-missing

case "$TARGET" in
  table1)
    run_table1
    ;;
  table_s1)
    # Table S.1 is a deterministic topology summary; rendering only.
    ;;
  table_s2)
    run_other_subset original "Table S.2" 0.01 0.05 0.10
    ;;
  table_s3)
    run_other_subset original "Table S.3" 0.15 0.20 0.30
    ;;
  table_s4)
    run_other_subset merged "Table S.4" 0.01 0.05 0.10
    ;;
  table_s5)
    run_other_subset merged "Table S.5" 0.15 0.20 0.30
    ;;
  table_s6)
    run_table_s6
    ;;
  scad_tables)
    run_table1
    run_table_s6
    ;;
  other_methods)
    run_other_tables
    ;;
  supp_tables)
    run_supp_tables
    ;;
  synthetic)
    run_synthetic_tables
    ;;
  all)
    run_table1
    run_supp_tables
    run_synthetic_tables
    ;;
  *)
    echo "Unknown target: $TARGET"
    echo "Usage: bash scripts/lib/tables_engine.sh [smoke|full] [table1|table_s1|table_s2|table_s3|table_s4|table_s5|table_s6|scad_tables|other_methods|supp_tables|synthetic|all]"
    exit 1
    ;;
esac

# For a single/partial target (e.g. reproducing only Table 1), the other tables'
# raw workbooks may be absent, so render leniently to avoid failing on them.
# A full ("all") run stays strict to surface any genuinely incomplete results.
if [[ "$TARGET" == "all" ]]; then
  refresh_tables
else
  refresh_tables allow-missing
fi
export_tex_if_requested

echo
echo "Completed. Outputs:"
echo "  Tables: results/tables"
echo "  Logs: $LOG_DIR"
