#!/usr/bin/env bash
# Reproduce Table S.5 (Lasso/L0/MCP, merged matrix, alpha=15%,20%,30%).
# Usage: bash scripts/run_table_s5.sh [full|smoke]   (default: full)
# Output: results/tables/table_s5.csv
exec bash "$(cd "$(dirname "$0")" && pwd)/lib/tables_engine.sh" "${1:-full}" table_s5
