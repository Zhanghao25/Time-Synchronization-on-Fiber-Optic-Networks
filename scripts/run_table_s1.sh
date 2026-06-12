#!/usr/bin/env bash
# Reproduce Table S.1 (network summary; render-only, no experiments).
# Usage: bash scripts/run_table_s1.sh [full|smoke]   (default: full)
# Output: results/tables/table_s1.csv
exec bash "$(cd "$(dirname "$0")" && pwd)/lib/tables_engine.sh" "${1:-full}" table_s1
