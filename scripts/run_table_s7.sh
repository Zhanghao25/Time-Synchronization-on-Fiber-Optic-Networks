#!/usr/bin/env bash
# Reproduce Table S.7 (synthetic networks, zeta=1%,5%,10%).
# Usage: bash scripts/run_table_s7.sh [full|smoke]   (default: full)
# Output: results/tables/table_s7.csv
exec bash "$(cd "$(dirname "$0")" && pwd)/lib/tables_engine.sh" "${1:-full}" synthetic
