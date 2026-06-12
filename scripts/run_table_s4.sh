#!/usr/bin/env bash
# Reproduce Table S.4 (Lasso/L0/MCP, merged matrix, alpha=1%,5%,10%).
# Usage: bash scripts/run_table_s4.sh [full|smoke]   (default: full)
# Output: results/tables/table_s4.csv
exec bash "$(cd "$(dirname "$0")" && pwd)/lib/tables_engine.sh" "${1:-full}" table_s4
