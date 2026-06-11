#!/usr/bin/env bash
# Reproduce Figure S.5 (City 2 LUR).
# Usage: bash scripts/run_figure_s5.sh [full|smoke]   (default: full)
# Outputs: results/raw/lur_figures/lur_city2_metrics.xlsx -> results/figures/figure_s5_city2.*
CITIES=2 exec bash "$(cd "$(dirname "$0")" && pwd)/run_all_figures.sh" "${1:-full}"
