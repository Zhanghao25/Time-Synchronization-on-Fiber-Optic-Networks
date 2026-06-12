#!/usr/bin/env bash
# Reproduce Table 1 (TSLE on three cities, alpha=1%,5%,10%).
# Usage: bash scripts/run_table_1.sh [full|smoke]   (default: full)
# Output: results/tables/table1.csv
exec bash "$(cd "$(dirname "$0")" && pwd)/lib/tables_engine.sh" "${1:-full}" table1
