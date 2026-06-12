#!/usr/bin/env bash
# Reproduce Figure S.5 (City 2 LUR analysis).
# Usage: bash scripts/run_figure_s5.sh [full|smoke]   (default: full)
# Output: results/figures/figure_s5_city2.png
CITIES=2 exec bash "$(cd "$(dirname "$0")" && pwd)/lib/figures_engine.sh" "${1:-full}"
