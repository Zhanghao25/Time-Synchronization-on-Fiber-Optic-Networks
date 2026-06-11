#!/usr/bin/env bash
# Reproduce Figure 5 (City 1 LUR).
# Usage: bash scripts/run_figure5.sh [full|smoke]   (default: full)
# Outputs: results/raw/lur_figures/lur_city1_metrics.xlsx -> results/figures/figure5_city1.*
CITIES=1 exec bash "$(cd "$(dirname "$0")" && pwd)/run_all_figures.sh" "${1:-full}"
