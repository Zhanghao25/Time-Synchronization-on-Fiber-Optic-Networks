#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Seed Scheme
#
# smoke mode:
#   contiguous seeds with sharding for fast validation
#
# full mode:
#   final article seed manifests under configs/seeds/figures/
#   to reproduce the shipped Figure 5 / Figure S.5 / Figure S.6 exactly
# =============================================================================

WORKDIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$WORKDIR"

MODE="${1:-full}"
CONDA_ENV="${CONDA_ENV:-jasa}"
MAX_JOBS="${MAX_JOBS:-12}"
ALPHAS="${ALPHAS:-0.01,0.05,0.10,0.15,0.20,0.30}"
BASE_SEED="${BASE_SEED:-42}"
RUN_TAG="${RUN_TAG:-lur_figures_$(date +%Y%m%d_%H%M%S)}"
USE_FINAL_MANIFESTS="${USE_FINAL_MANIFESTS:-1}"
SEED_CONFIG_DIR="${SEED_CONFIG_DIR:-configs/seeds/figures}"

if [[ "${MODE}" == "smoke" ]]; then
  CITY12_EXPERIMENTS="${CITY12_EXPERIMENTS:-10}"
  CITY3_EXPERIMENTS="${CITY3_EXPERIMENTS:-5}"
  CITY12_SHARD_SIZE="${CITY12_SHARD_SIZE:-5}"
  CITY3_SHARD_SIZE="${CITY3_SHARD_SIZE:-5}"
else
  CITY12_EXPERIMENTS="${CITY12_EXPERIMENTS:-100}"
  CITY3_EXPERIMENTS="${CITY3_EXPERIMENTS:-100}"
  CITY12_SHARD_SIZE="${CITY12_SHARD_SIZE:-10}"
  CITY3_SHARD_SIZE="${CITY3_SHARD_SIZE:-10}"
fi

RESULTS_DIR="results/raw/lur_figures"
SHARD_DIR="${RESULTS_DIR}/shards/${RUN_TAG}"
LOG_DIR="results/logs/lur_figures/${RUN_TAG}"
OUTPUT_DIR="results/figures"

mkdir -p "${RESULTS_DIR}" "${SHARD_DIR}" "${LOG_DIR}" "${OUTPUT_DIR}"

run_job() {
  local dataset="$1"
  local city_label="$2"
  local city_tag="$3"
  local shard_size="$4"
  local base_seed_offset="$5"
  local shard_name="$6"
  local save_path="${SHARD_DIR}/${shard_name}.xlsx"
  local log_path="${LOG_DIR}/${shard_name}.log"

  echo "[launch] ${shard_name} -> ${save_path}"
  conda run -n "${CONDA_ENV}" python -u src/compute_lur_metrics.py \
    --data "${dataset}" \
    --city-label "${city_label}" \
    --alphas "${ALPHAS}" \
    --experiments "${shard_size}" \
    --base-seed "${base_seed_offset}" \
    --resume \
    --checkpoint-interval 1 \
    --paper-tag "${RUN_TAG}" \
    --save "${save_path}" > "${log_path}" 2>&1 &
}

run_manifest_job() {
  local dataset="$1"
  local city_label="$2"
  local city_tag="$3"
  local manifest="$4"
  local save_path="${RESULTS_DIR}/lur_${city_tag}_metrics.xlsx"
  local log_path="${LOG_DIR}/${city_tag}.log"

  echo "[launch] ${city_tag} manifest=${manifest} -> ${save_path}"
  conda run -n "${CONDA_ENV}" python -u src/compute_lur_metrics.py \
    --data "${dataset}" \
    --city-label "${city_label}" \
    --alphas "${ALPHAS}" \
    --experiments 1 \
    --base-seed "${BASE_SEED}" \
    --seed-manifest "${manifest}" \
    --resume \
    --checkpoint-interval 1 \
    --paper-tag "${RUN_TAG}" \
    --save "${save_path}" > "${log_path}" 2>&1 &
}

wait_for_slot() {
  while [[ "$(jobs -pr | wc -l)" -ge "${MAX_JOBS}" ]]; do
    wait -n
  done
}

submit_city_jobs() {
  local dataset="$1"
  local city_label="$2"
  local city_tag="$3"
  local total_experiments="$4"
  local shard_size="$5"

  local shard_index=1
  local offset=0
  while [[ "${offset}" -lt "${total_experiments}" ]]; do
    local remaining=$(( total_experiments - offset ))
    local current_size="${shard_size}"
    if [[ "${remaining}" -lt "${current_size}" ]]; then
      current_size="${remaining}"
    fi
    wait_for_slot
    run_job "${dataset}" "${city_label}" "${city_tag}" "${current_size}" "$(( BASE_SEED + offset ))" "${city_tag}_shard${shard_index}"
    shard_index=$(( shard_index + 1 ))
    offset=$(( offset + current_size ))
  done
}

echo "[run] mode=${MODE} max_jobs=${MAX_JOBS} tag=${RUN_TAG}"

CITIES="${CITIES:-1 2 3}"

if [[ "${MODE}" == "full" && "${USE_FINAL_MANIFESTS}" == "1" ]]; then
  for c in $CITIES; do
    wait_for_slot
    run_manifest_job "data/city${c}.xlsx" "City ${c}" "city${c}" "${SEED_CONFIG_DIR}/city${c}_final.json"
  done
  wait
else
  for c in $CITIES; do
    if [[ "$c" == "3" ]]; then
      submit_city_jobs "data/city${c}.xlsx" "City ${c}" "city${c}" "${CITY3_EXPERIMENTS}" "${CITY3_SHARD_SIZE}"
    else
      submit_city_jobs "data/city${c}.xlsx" "City ${c}" "city${c}" "${CITY12_EXPERIMENTS}" "${CITY12_SHARD_SIZE}"
    fi
  done

  wait

  echo "[merge] merging city shard workbooks"
  for c in $CITIES; do
    conda run -n "${CONDA_ENV}" python -u src/merge_results.py --lur \
      --inputs "${SHARD_DIR}"/city${c}_shard*.xlsx \
      --save "${RESULTS_DIR}/lur_city${c}_metrics.xlsx"
  done
fi

echo "[figure] building canonical readable figures"
conda run -n "${CONDA_ENV}" python -u src/make_paper_results.py --figures \
  --metrics-dir "${RESULTS_DIR}" --cities "${CITIES}"

echo "[done] outputs in ${OUTPUT_DIR}"
