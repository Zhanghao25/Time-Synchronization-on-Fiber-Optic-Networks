#!/usr/bin/env bash
# Reproduce Table S.2 (Lasso/L0/MCP, original matrix, alpha=1%,5%,10%).
# Usage: bash scripts/run_table_s2.sh [full|smoke]   (default: full)
# Output: results/tables/table_s2.csv
exec bash "$(cd "$(dirname "$0")" && pwd)/lib/tables_engine.sh" "${1:-full}" table_s2
