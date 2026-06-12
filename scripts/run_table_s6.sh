#!/usr/bin/env bash
# Reproduce Table S.6 (TSLE on three cities, alpha=15%,20%,30%).
# Usage: bash scripts/run_table_s6.sh [full|smoke]   (default: full)
# Output: results/tables/table_s6.csv
exec bash "$(cd "$(dirname "$0")" && pwd)/lib/tables_engine.sh" "${1:-full}" table_s6
