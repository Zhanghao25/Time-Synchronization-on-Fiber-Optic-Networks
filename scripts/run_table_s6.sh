#!/usr/bin/env bash
# Reproduce Table S.6 (TSLE, r=15%,20%,30%).
# Usage: bash scripts/run_table_s6.sh [full|smoke]   (default: full)
# Outputs: results/raw/main_scad/*.xlsx -> results/tables/table_s6.*
exec bash "$(cd "$(dirname "$0")" && pwd)/run_all_tables.sh" "${1:-full}" table_s6
