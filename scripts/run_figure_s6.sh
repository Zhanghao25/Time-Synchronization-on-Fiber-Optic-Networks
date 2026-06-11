#!/usr/bin/env bash
# Reproduce Figure S.6 (City 3 LUR).
# Usage: bash scripts/run_figure_s6.sh [full|smoke]   (default: full)
# Outputs: results/raw/lur_figures/lur_city3_metrics.xlsx -> results/figures/figure_s6_city3.*
CITIES=3 exec bash "$(cd "$(dirname "$0")" && pwd)/run_all_figures.sh" "${1:-full}"
