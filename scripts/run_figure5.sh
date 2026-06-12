#!/usr/bin/env bash
# Reproduce Figure 5 (City 1 LUR analysis).
# Usage: bash scripts/run_figure5.sh [full|smoke]   (default: full)
# Output: results/figures/figure5_city1.png
CITIES=1 exec bash "$(cd "$(dirname "$0")" && pwd)/lib/figures_engine.sh" "${1:-full}"
