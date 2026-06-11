#!/usr/bin/env bash
# Reproduce Table 1 (TSLE, r=1%,5%,10%) and Table S.1 (tree structure summary).
# Usage: bash scripts/run_table1_s1.sh [full|smoke]   (default: full)
# Outputs: results/raw/main_scad/*.xlsx -> results/tables/table1.* and table_s1.*
exec bash "$(cd "$(dirname "$0")" && pwd)/run_all_tables.sh" "${1:-full}" table1
