#!/usr/bin/env bash
# Reproduce Table S.7 (synthetic zeta=1%,5%,10% scenarios).
# Usage: bash scripts/run_table_s7.sh [full|smoke]   (default: full)
# Outputs: results/raw/synthetic/*.xlsx -> results/tables/table_s7.*
exec bash "$(cd "$(dirname "$0")" && pwd)/run_all_tables.sh" "${1:-full}" synthetic
