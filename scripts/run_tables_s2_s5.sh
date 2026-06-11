#!/usr/bin/env bash
# Reproduce Tables S.2-S.5 (Lasso / L0 / MCP on original and merged matrices).
# Usage: bash scripts/run_tables_s2_s5.sh [full|smoke]   (default: full)
# Outputs: results/raw/other_methods/*.xlsx -> results/tables/table_s2.* ... table_s5.*
exec bash "$(cd "$(dirname "$0")" && pwd)/run_all_tables.sh" "${1:-full}" other_methods
