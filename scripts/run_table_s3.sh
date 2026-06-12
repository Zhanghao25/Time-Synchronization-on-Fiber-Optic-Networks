#!/usr/bin/env bash
# Reproduce Table S.3 (Lasso/L0/MCP, original matrix, alpha=15%,20%,30%).
# Usage: bash scripts/run_table_s3.sh [full|smoke]   (default: full)
# Output: results/tables/table_s3.csv
exec bash "$(cd "$(dirname "$0")" && pwd)/lib/tables_engine.sh" "${1:-full}" table_s3
