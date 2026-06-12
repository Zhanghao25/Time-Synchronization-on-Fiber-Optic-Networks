#!/usr/bin/env bash
# Reproduce Figure S.6 (City 3 LUR analysis).
# Usage: bash scripts/run_figure_s6.sh [full|smoke]   (default: full)
# Output: results/figures/figure_s6_city3.png
CITIES=3 exec bash "$(cd "$(dirname "$0")" && pwd)/lib/figures_engine.sh" "${1:-full}"
