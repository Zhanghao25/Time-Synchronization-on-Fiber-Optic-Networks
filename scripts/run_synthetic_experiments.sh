#!/bin/bash
# Launch 3 parallel tmux sessions for Table S.7 synthetic experiments
# Each session runs one zeta scenario at alpha=0.10

WORKDIR="$(cd "$(dirname "$0")/.." && pwd)"
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
CONDA_ENV="${CONDA_ENV:-jasa}"

cd "$WORKDIR"
mkdir -p results/logs results/synthetic

ZETA_VALUES=("0.01" "0.05" "0.10")
ZETA_TAGS=("1pct" "5pct" "10pct")
ALPHA="0.10"

for idx in "${!ZETA_VALUES[@]}"; do
    session="synthetic_zeta${ZETA_TAGS[$idx]}_alpha10"
    tmux kill-session -t "$session" 2>/dev/null
done

echo "Starting 3 parallel synthetic experiments for Table S.7..."
echo "Cases: zeta=1%, 5%, 10%; alpha=0.10"
echo ""

for idx in "${!ZETA_VALUES[@]}"; do
    zeta="${ZETA_VALUES[$idx]}"
    zeta_tag="${ZETA_TAGS[$idx]}"
    session="synthetic_zeta${zeta_tag}_alpha10"
    logfile="${WORKDIR}/results/logs/${session}.log"
    outfile="${WORKDIR}/results/synthetic/synthetic_zeta${zeta_tag}_alpha10.xlsx"

    cmd="source ${CONDA_BASE}/etc/profile.d/conda.sh && conda activate ${CONDA_ENV} && cd ${WORKDIR} && python -u experiments/synthetic_experiment.py --zeta ${zeta} --ratio ${ALPHA} --experiments 10 --save ${outfile} 2>&1 | tee ${logfile}; echo DONE; read"

    echo "Starting ${session}: zeta=${zeta}, alpha=${ALPHA}"
    tmux new-session -d -s "$session" "bash -c '$cmd'"
done

echo ""
echo "All 3 sessions started."
echo ""
echo "Commands:"
echo "  tmux ls"
echo "  tmux attach -t synthetic_zeta1pct_alpha10"
echo "  tail -f results/logs/synthetic_zeta1pct_alpha10.log"
