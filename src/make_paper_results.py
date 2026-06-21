#!/usr/bin/env python3
"""
Make all final paper results from the canonical workbooks in results/raw/:

- Tables:  Table 1 and Tables S.1-S.7 -> results/tables/ (CSV)
- Figures: Figure 5, Figure S.5, Figure S.6 -> results/figures/ (PNG)

Usage:
    python src/make_paper_results.py             # tables + figures
    python src/make_paper_results.py --tables    # tables only
    python src/make_paper_results.py --figures   # figures only
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MultipleLocator, PercentFormatter




# =============================================================================
# Table rendering helpers
# =============================================================================

ALPHA_COL = "$\\alpha$"
ZETA_COL = "$\\zeta$"
CITY_GROUPS = ["City 1", "City 2", "City 3"]
CITY_METRICS = ["Accuracy", "Precision", "Recall", "$F_{0.5}$"]
SIMPLE_METRICS = ["Accuracy", "Precision", "Recall", "$F_{0.5}$"]

PLAIN_METHOD_MAP_MAIN = {
    "TSLESCAD": "TSLE_SCAD",
    "TSLEMerged": "TSLE_Merged",
    "TSLERandom": "TSLE_Random",
    "TSLEAddition": "TSLE_Addition",
}

PLAIN_METHOD_MAP_OTHER = {
    "L0": "TSLE_L0",
    "Lasso": "TSLE_Lasso",
    "MCP": "TSLE_MCP",
}


@dataclass(frozen=True)
class TableSpec:
    name: str
    title: str
    kind: str
    ratio_col: str | None = None
    method_plain_map: dict[str, str] | None = None


TABLE_SPECS = [
    TableSpec("table1", "Table 1", "grouped_city", ALPHA_COL, PLAIN_METHOD_MAP_MAIN),
    TableSpec("table_s1", "Table S.1", "simple"),
    TableSpec("table_s2", "Table S.2", "grouped_city", ALPHA_COL, PLAIN_METHOD_MAP_OTHER),
    TableSpec("table_s3", "Table S.3", "grouped_city", ALPHA_COL, PLAIN_METHOD_MAP_OTHER),
    TableSpec("table_s4", "Table S.4", "grouped_city", ALPHA_COL, PLAIN_METHOD_MAP_OTHER),
    TableSpec("table_s5", "Table S.5", "grouped_city", ALPHA_COL, PLAIN_METHOD_MAP_OTHER),
    TableSpec("table_s6", "Table S.6", "grouped_city", ALPHA_COL, PLAIN_METHOD_MAP_MAIN),
    TableSpec("table_s7", "Table S.7", "grouped_metric", ZETA_COL, PLAIN_METHOD_MAP_MAIN),
]


def clean_cell(text: object) -> str:
    if pd.isna(text):
        return "--"
    return str(text).replace(r"\%", "%").strip()


def split_mean_std(text: object) -> tuple[str, str]:
    cleaned = clean_cell(text)
    if cleaned == "--":
        return "--", ""
    if " (" in cleaned and cleaned.endswith(")"):
        mean, std = cleaned.split(" (", 1)
        return mean, f"({std}"
    return cleaned, ""


def ratio_label_for_csv(col_name: str | None) -> str:
    if col_name == ZETA_COL:
        return "zeta"
    return "alpha"


def build_grouped_city_readable(df: pd.DataFrame, ratio_col: str, method_plain_map: dict[str, str]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for method in df["Method"].drop_duplicates():
        sub = df[df["Method"] == method].reset_index(drop=True)
        for idx, (_, record) in enumerate(sub.iterrows()):
            mean_row: dict[str, str] = {"Method": method_plain_map.get(method, method), ratio_label_for_csv(ratio_col): f"{float(record[ratio_col]):.2f}"}
            std_row: dict[str, str] = {"Method": "", ratio_label_for_csv(ratio_col): ""}
            if idx > 0:
                mean_row["Method"] = ""
            for city in CITY_GROUPS:
                for metric in CITY_METRICS:
                    mean, std = split_mean_std(record[f"{city} | {metric}"])
                    mean_row[f"{city} {metric}"] = mean
                    std_row[f"{city} {metric}"] = std
            rows.extend([mean_row, std_row])
    return pd.DataFrame(rows)


def build_grouped_metric_readable(df: pd.DataFrame, ratio_col: str, method_plain_map: dict[str, str]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for method in df["Method"].drop_duplicates():
        sub = df[df["Method"] == method].reset_index(drop=True)
        for idx, (_, record) in enumerate(sub.iterrows()):
            mean_row = {"Method": method_plain_map.get(method, method), ratio_label_for_csv(ratio_col): f"{float(record[ratio_col]):.2f}"}
            std_row = {"Method": "", ratio_label_for_csv(ratio_col): ""}
            if idx > 0:
                mean_row["Method"] = ""
            for metric in SIMPLE_METRICS:
                mean, std = split_mean_std(record[metric])
                mean_row[metric] = mean
                std_row[metric] = std
            rows.extend([mean_row, std_row])
    return pd.DataFrame(rows)


def build_simple_readable(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "City" in out.columns:
        out["City"] = out["City"].astype(str)
    return out


def render_from_dataframe(spec: TableSpec, df: pd.DataFrame, output_dir: Path) -> None:
    if spec.kind == "grouped_city":
        readable = build_grouped_city_readable(df, spec.ratio_col or ALPHA_COL, spec.method_plain_map or {})
    elif spec.kind == "grouped_metric":
        readable = build_grouped_metric_readable(df, spec.ratio_col or ZETA_COL, spec.method_plain_map or {})
    else:
        readable = build_simple_readable(df)

    readable.to_csv(output_dir / f"{spec.name}.csv", index=False)


def export_tables_from_frames(frames: dict[str, pd.DataFrame], output_dir: Path | str = "results/tables") -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for spec in TABLE_SPECS:
        df = frames.get(spec.name)
        if df is None:
            raise KeyError(f"Missing source dataframe for readable export: {spec.name}")
        render_from_dataframe(spec, df, output_dir)

    return output_dir


# =============================================================================
# Table assembly from canonical workbooks
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "raw"
DATA_DIR = ROOT / "data"
TABLES_OUTPUT_DIR = ROOT / "results" / "tables"
FIGURES_OUTPUT_DIR = ROOT / "results" / "figures"
METRICS_DIR = RESULTS_DIR / "lur_figures"


def repo_relpath(path: Path) -> str:
    """Return portable repository-relative paths in machine-readable outputs."""
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


SCAD_METHOD_MAP = {
    "step1": "TSLESCAD",
    "step2": "TSLEMerged",
    "step3": "TSLERandom",
    "step4": "TSLEAddition",
}

SYNTHETIC_METHOD_MAP = {
    "step1_original": "TSLESCAD",
    "step2_merged": "TSLEMerged",
    "step4_iterative": "TSLEAddition",
}

SCAD_METHOD_ORDER = ["TSLESCAD", "TSLEMerged", "TSLERandom", "TSLEAddition"]
SYNTHETIC_METHOD_ORDER = ["TSLESCAD", "TSLEMerged", "TSLEAddition"]
OTHER_METHOD_ORDER = ["L0", "Lasso", "MCP"]
CITY_ORDER = ["city1", "city2", "city3"]
CITY_LABELS = {
    "city1": "City 1",
    "city2": "City 2",
    "city3": "City 3",
}

LOW_RATIOS = [0.01, 0.05, 0.10]
HIGH_RATIOS = [0.15, 0.20, 0.30]
ZETA_ORDER = [0.01, 0.05, 0.10]
ZETA_MAP = {
    100: 0.01,
    500: 0.05,
    1000: 0.10,
}

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f05": r"$F_{0.5}$",
}
METRIC_KEYS = list(METRIC_LABELS.keys())

CURRENT_RESULT_COLUMNS = [
    "table_name",
    "city",
    "ratio",
    "method",
    "accuracy",
    "accuracy_std",
    "precision",
    "precision_std",
    "recall",
    "recall_std",
    "f05",
    "f05_std",
    "n_exp_current",
    "source_path",
]

SYNTHETIC_RESULT_COLUMNS = [
    "table_name",
    "alpha",
    "zeta",
    "method",
    "accuracy",
    "accuracy_std",
    "precision",
    "precision_std",
    "recall",
    "recall_std",
    "f05",
    "f05_std",
    "n_exp_current",
    "source_path",
]

TABLE_CAPTIONS = {
    "Table 1": (
        "Averaged accuracy, precision, recall and $F_{0.5}$ score across various methods "
        "and sparsity levels ($\\alpha \\in \\{0.01, 0.05, 0.10\\}$) for three cities; "
        "standard deviations in parentheses."
    ),
    "Table S.1": (
        "Number of nodes and edges in the fiber optic networks of three cities in China."
    ),
    "Table S.2": (
        "Averaged accuracy, precision, recall and $F_{0.5}$ performance across L0-norm, "
        "Lasso and MCP regularization methods and sparsity levels "
        "($\\alpha \\in \\{0.01, 0.05, 0.10\\}$) for three cities before merging, "
        "with standard deviations in parentheses."
    ),
    "Table S.3": (
        "Averaged accuracy, precision, recall and $F_{0.5}$ performance across L0-norm, "
        "Lasso and MCP regularization methods and sparsity levels "
        "($\\alpha \\in \\{0.15, 0.20, 0.30\\}$) for three cities before merging, "
        "with standard deviations in parentheses."
    ),
    "Table S.4": (
        "Averaged accuracy, precision, recall and $F_{0.5}$ performance across L0-norm, "
        "Lasso and MCP regularization methods and sparsity levels "
        "($\\alpha \\in \\{0.01, 0.05, 0.10\\}$) for three cities after merging, "
        "with standard deviations in parentheses."
    ),
    "Table S.5": (
        "Averaged accuracy, precision, recall and $F_{0.5}$ performance across L0-norm, "
        "Lasso and MCP regularization methods and sparsity levels "
        "($\\alpha \\in \\{0.15, 0.20, 0.30\\}$) for three cities after merging, "
        "with standard deviations in parentheses."
    ),
    "Table S.6": (
        "Averaged accuracy, precision, recall and $F_{0.5}$ score across various methods "
        "and sparsity levels ($\\alpha \\in \\{0.15, 0.20, 0.30\\}$) for three cities; "
        "standard deviations in parentheses."
    ),
    "Table S.7": (
        "Averaged accuracy, precision, recall and $F_{0.5}$ score across various methods "
        "and proportions of injected extreme local configurations "
        "($\\zeta \\in \\{0.01, 0.05, 0.10\\}$, $\\alpha = 0.10$) for the synthetic "
        "network; standard deviations in parentheses."
    ),
}

TABLE_LABELS = {
    "Table 1": "tab:table1",
    "Table S.1": "tab:table_s1",
    "Table S.2": "tab:table_s2",
    "Table S.3": "tab:table_s3",
    "Table S.4": "tab:table_s4",
    "Table S.5": "tab:table_s5",
    "Table S.6": "tab:table_s6",
    "Table S.7": "tab:table_s7",
}

TABLE_FILE_STEMS = {
    "Table 1": "table1",
    "Table S.1": "table_s1",
    "Table S.2": "table_s2",
    "Table S.3": "table_s3",
    "Table S.4": "table_s4",
    "Table S.5": "table_s5",
    "Table S.6": "table_s6",
    "Table S.7": "table_s7",
}


def ensure_output_dirs() -> None:
    TABLES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def ratio_to_str(value: Optional[float]) -> Optional[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return f"{float(value):.2f}"


def zeta_to_str(value: Optional[float]) -> Optional[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return f"{float(value):.2f}"


def parse_ratio_fragment(value: str) -> float:
    return round(float(value), 2)


def format_percent(mean: Any, std: Any) -> str:
    if pd.isna(mean):
        return "--"
    if pd.isna(std):
        std = 0.0
    return f"{float(mean) * 100:.2f}\\% ({float(std):.4f})"


def format_decimal(mean: Any, std: Any, mean_digits: int = 3, std_digits: int = 3) -> str:
    if pd.isna(mean):
        return "--"
    if pd.isna(std):
        std = 0.0
    return f"{float(mean):.{mean_digits}f} ({float(std):.{std_digits}f})"


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    flattened = df.copy()
    if isinstance(flattened.columns, pd.MultiIndex):
        flattened.columns = [
            " | ".join(str(part) for part in col if str(part) != "")
            for col in flattened.columns.to_flat_index()
        ]
    return flattened


def build_wide_flat(wide_df: pd.DataFrame) -> pd.DataFrame:
    return flatten_columns(wide_df.reset_index())


def render_table_latex(table_key: str, wide_df: pd.DataFrame) -> str:
    return wide_df.to_latex(
        escape=False,
        na_rep="--",
        multicolumn=True,
        multirow=True,
        caption=TABLE_CAPTIONS[table_key],
        label=TABLE_LABELS[table_key],
    )


def write_table_artifacts(
    table_key: str,
    wide_df: pd.DataFrame,
    long_df: pd.DataFrame,
    export_tex: bool = False,
) -> Tuple[Optional[str], pd.DataFrame]:
    wide_flat = build_wide_flat(wide_df)

    if not export_tex:
        return None, wide_flat

    return render_table_latex(table_key, wide_df), wide_flat


def load_current_scad_results(results_root: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    scad_dir = results_root / "main_scad"
    if not scad_dir.exists():
        return pd.DataFrame(columns=CURRENT_RESULT_COLUMNS)

    for path in sorted(scad_dir.glob("scad_city*_ratio*.xlsx")):
        match = re.search(r"scad_city(\d+)_ratio([0-9.]+)\.xlsx", path.name)
        if not match:
            continue

        city = f"city{match.group(1)}"
        ratio = parse_ratio_fragment(match.group(2))
        summary = pd.read_excel(path, sheet_name="Summary")

        table_name = "Table 1" if ratio in LOW_RATIOS else "Table S.6" if ratio in HIGH_RATIOS else None
        for _, row in summary.iterrows():
            step = str(row["Step"]).strip()
            paper_method = SCAD_METHOD_MAP.get(step)
            if paper_method is None:
                continue

            completed = row.get("Completed")
            if pd.isna(completed) or int(completed) == 0:
                accuracy = pd.NA
                accuracy_std = pd.NA
                precision = pd.NA
                precision_std = pd.NA
                recall = pd.NA
                recall_std = pd.NA
                f05 = pd.NA
                f05_std = pd.NA
            else:
                accuracy = row.get("accuracy_mean")
                accuracy_std = row.get("accuracy_std")
                precision = row.get("precision_mean")
                precision_std = row.get("precision_std")
                recall = row.get("recall_mean")
                recall_std = row.get("recall_std")
                f05 = row.get("f05_score_mean")
                f05_std = row.get("f05_score_std")

            rows.append({
                "table_name": table_name,
                "city": city,
                "ratio": ratio_to_str(ratio),
                "method": paper_method,
                "accuracy": accuracy,
                "accuracy_std": accuracy_std,
                "precision": precision,
                "precision_std": precision_std,
                "recall": recall,
                "recall_std": recall_std,
                "f05": f05,
                "f05_std": f05_std,
                "n_exp_current": completed,
                "source_path": repo_relpath(path),
            })

    return pd.DataFrame(rows, columns=CURRENT_RESULT_COLUMNS)


def load_current_other_results(results_root: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    other_dir = results_root / "other_methods"
    if not other_dir.exists():
        return pd.DataFrame(columns=CURRENT_RESULT_COLUMNS + ["matrix_type"])

    for path in sorted(other_dir.glob("othermethods_*_city*_ratio*.xlsx")):
        match = re.search(r"othermethods_(original|merged)_city(\d+)_ratio([0-9.]+)\.xlsx", path.name)
        if not match:
            continue

        matrix_type, city_num, ratio_raw = match.groups()
        city = f"city{city_num}"
        ratio = parse_ratio_fragment(ratio_raw)

        if ratio in LOW_RATIOS:
            table_name = "Table S.2" if matrix_type == "original" else "Table S.4"
        elif ratio in HIGH_RATIOS:
            table_name = "Table S.3" if matrix_type == "original" else "Table S.5"
        else:
            table_name = None

        stats = pd.read_excel(path, sheet_name="Statistical_Analysis")
        for _, row in stats.iterrows():
            method = str(row["Method"]).strip()
            rows.append({
                "table_name": table_name,
                "city": city,
                "ratio": ratio_to_str(ratio),
                "method": method,
                "accuracy": row.get("Mean_Accuracy"),
                "accuracy_std": row.get("Std_Accuracy"),
                "precision": row.get("Mean_Precision"),
                "precision_std": row.get("Std_Precision"),
                "recall": row.get("Mean_Recall"),
                "recall_std": row.get("Std_Recall"),
                "f05": row.get("Mean_F05_Score"),
                "f05_std": row.get("Std_F05_Score"),
                "n_exp_current": row.get("Successful_Experiments"),
                "matrix_type": matrix_type,
                "source_path": repo_relpath(path),
            })

    return pd.DataFrame(rows, columns=CURRENT_RESULT_COLUMNS + ["matrix_type"])


def load_current_synthetic_results(results_root: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    synthetic_dir = results_root / "synthetic"
    if not synthetic_dir.exists():
        return pd.DataFrame(columns=SYNTHETIC_RESULT_COLUMNS)

    for path in sorted(synthetic_dir.glob("*.xlsx")):
        name_lower = path.name.lower()
        if "combined" in name_lower:
            continue
        if not (name_lower.startswith("9cases_v3_city") or name_lower.startswith("synthetic_zeta")):
            continue

        try:
            summary = pd.read_excel(path, sheet_name="Summary")
        except Exception:
            continue

        if "Method" not in summary.columns:
            continue

        for _, row in summary.iterrows():
            method = SYNTHETIC_METHOD_MAP.get(str(row["Method"]).strip())
            if method is None:
                continue

            alpha_ratio = row.get("Ratio")
            if pd.isna(alpha_ratio):
                match_old = re.search(r"_r(\d+)\.xlsx", path.name)
                match_new = re.search(r"alpha(\d+)\.xlsx", path.name)
                if match_old:
                    alpha_ratio = parse_ratio_fragment(f"0.{match_old.group(1)}")
                elif match_new:
                    alpha_ratio = parse_ratio_fragment(f"0.{match_new.group(1)}")
                else:
                    continue

            hard_k = row.get("Hard_K")
            if pd.isna(hard_k):
                continue
            hard_k = int(hard_k)
            zeta = ZETA_MAP.get(hard_k)
            rows.append({
                "table_name": "Table S.7" if alpha_ratio == 0.10 else None,
                "alpha": ratio_to_str(alpha_ratio),
                "zeta": zeta_to_str(zeta),
                "method": method,
                "accuracy": row.get("accuracy_mean"),
                "accuracy_std": row.get("accuracy_std"),
                "precision": row.get("precision_mean"),
                "precision_std": row.get("precision_std"),
                "recall": row.get("recall_mean"),
                "recall_std": row.get("recall_std"),
                "f05": row.get("f05_score_mean"),
                "f05_std": row.get("f05_score_std"),
                "n_exp_current": row.get("N_Experiments"),
                "source_path": repo_relpath(path),
            })

    return pd.DataFrame(rows, columns=SYNTHETIC_RESULT_COLUMNS)


def build_table_s1() -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for city in CITY_ORDER:
        path = DATA_DIR / f"{city}.xlsx"
        if not path.exists():
            rows.append({
                "City": CITY_LABELS[city],
                "Nodes": pd.NA,
                "Edges": pd.NA,
                "source_path": repo_relpath(path),
            })
            continue

        df = pd.read_excel(path, usecols=[1, 2])
        nodes = pd.unique(pd.concat([df.iloc[:, 0], df.iloc[:, 1]], ignore_index=True))
        rows.append({
            "City": CITY_LABELS[city],
            "Nodes": len(nodes),
            "Edges": len(df),
            "source_path": repo_relpath(path),
        })

    return pd.DataFrame(rows)


def build_city_metric_wide(
    current_df: pd.DataFrame,
    table_name: str,
    methods: Sequence[str],
    ratios: Sequence[float],
    allow_missing: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if current_df.empty or "table_name" not in current_df.columns:
        filtered = pd.DataFrame(columns=CURRENT_RESULT_COLUMNS)
    else:
        filtered = current_df[current_df["table_name"] == table_name].copy()
    if not filtered.empty:
        filtered["ratio"] = filtered["ratio"].astype(str)

    columns = pd.MultiIndex.from_product(
        [[CITY_LABELS[city] for city in CITY_ORDER], list(METRIC_LABELS.values())]
    )

    rows: List[List[str]] = []
    index: List[Tuple[str, str]] = []
    missing: List[str] = []

    for method in methods:
        for ratio in ratios:
            ratio_key = ratio_to_str(ratio)
            row_values: List[str] = []
            for city in CITY_ORDER:
                match = filtered[
                    (filtered["method"] == method)
                    & (filtered["city"] == city)
                    & (filtered["ratio"] == ratio_key)
                ]
                if match.empty:
                    missing.append(f"{table_name}: city={city}, alpha={ratio_key}, method={method}")
                    row_values.extend(["--"] * len(METRIC_KEYS))
                    continue

                record = match.iloc[0]
                row_values.extend([
                    format_percent(record["accuracy"], record["accuracy_std"]),
                    format_percent(record["precision"], record["precision_std"]),
                    format_percent(record["recall"], record["recall_std"]),
                    format_percent(record["f05"], record["f05_std"]),
                ])

            rows.append(row_values)
            index.append((method, ratio_key))

    if missing and not allow_missing:
        preview = "\n".join(missing[:20])
        suffix = "" if len(missing) <= 20 else f"\n... and {len(missing) - 20} more"
        raise ValueError(
            "Missing expected table entries. Re-run the corresponding experiments "
            "or pass --allow-missing for diagnostic rendering only:\n"
            f"{preview}{suffix}"
        )

    wide_df = pd.DataFrame(
        rows,
        index=pd.MultiIndex.from_tuples(index, names=["Method", r"$\alpha$"]),
        columns=columns,
    )
    return wide_df, filtered


def build_table_s7_wide(current_df: pd.DataFrame, allow_missing: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if current_df.empty or "table_name" not in current_df.columns:
        filtered = pd.DataFrame(columns=SYNTHETIC_RESULT_COLUMNS)
    else:
        filtered = current_df[current_df["table_name"] == "Table S.7"].copy()
    if not filtered.empty:
        filtered["zeta"] = filtered["zeta"].astype(str)

    rows: List[List[str]] = []
    index: List[Tuple[str, str]] = []
    columns = pd.Index(list(METRIC_LABELS.values()))
    missing: List[str] = []

    for method in SYNTHETIC_METHOD_ORDER:
        for zeta in ZETA_ORDER:
            zeta_key = zeta_to_str(zeta)
            match = filtered[(filtered["method"] == method) & (filtered["zeta"] == zeta_key)]
            if match.empty:
                missing.append(f"Table S.7: zeta={zeta_key}, method={method}")
                row_values = ["--"] * len(METRIC_KEYS)
            else:
                record = match.iloc[0]
                row_values = [
                    format_decimal(record["accuracy"], record["accuracy_std"]),
                    format_decimal(record["precision"], record["precision_std"]),
                    format_decimal(record["recall"], record["recall_std"]),
                    format_decimal(record["f05"], record["f05_std"]),
                ]

            rows.append(row_values)
            index.append((method, zeta_key))

    if missing and not allow_missing:
        preview = "\n".join(missing[:20])
        raise ValueError(
            "Missing expected Table S.7 entries. Re-run the corresponding synthetic "
            "experiments or pass --allow-missing for diagnostic rendering only:\n"
            f"{preview}"
        )

    wide_df = pd.DataFrame(
        rows,
        index=pd.MultiIndex.from_tuples(index, names=["Method", r"$\zeta$"]),
        columns=columns,
    )
    return wide_df, filtered


def build_s1_wide(df: pd.DataFrame) -> pd.DataFrame:
    wide = df[["City", "Nodes", "Edges"]].copy()
    return wide.set_index("City")


def build_all_tables(results_root: Path, allow_missing: bool = False, export_tex: bool = False) -> None:
    """Build Table 1 and Tables S.1-S.7 into results/tables/."""
    ensure_output_dirs()

    scad_current = load_current_scad_results(results_root)
    other_current = load_current_other_results(results_root)
    synthetic_current = load_current_synthetic_results(results_root)

    latex_blocks: List[Tuple[str, str]] = []
    readable_frames: Dict[str, pd.DataFrame] = {}

    s1_df = build_table_s1()
    s1_wide = build_s1_wide(s1_df)
    latex, wide_flat = write_table_artifacts("Table S.1", s1_wide, s1_df, export_tex=export_tex)
    readable_frames["table_s1"] = wide_flat
    if latex:
        latex_blocks.append(("Table S.1", latex))

    table1_wide, table1_long = build_city_metric_wide(
        scad_current, "Table 1", SCAD_METHOD_ORDER, LOW_RATIOS, allow_missing=allow_missing
    )
    latex, wide_flat = write_table_artifacts("Table 1", table1_wide, table1_long, export_tex=export_tex)
    readable_frames["table1"] = wide_flat
    if latex:
        latex_blocks.append(("Table 1", latex))

    s6_wide, s6_long = build_city_metric_wide(
        scad_current, "Table S.6", SCAD_METHOD_ORDER, HIGH_RATIOS, allow_missing=allow_missing
    )
    latex, wide_flat = write_table_artifacts("Table S.6", s6_wide, s6_long, export_tex=export_tex)
    readable_frames["table_s6"] = wide_flat
    if latex:
        latex_blocks.append(("Table S.6", latex))

    for table_name, ratios in [
        ("Table S.2", LOW_RATIOS),
        ("Table S.3", HIGH_RATIOS),
        ("Table S.4", LOW_RATIOS),
        ("Table S.5", HIGH_RATIOS),
    ]:
        wide, long_df = build_city_metric_wide(
            other_current, table_name, OTHER_METHOD_ORDER, ratios, allow_missing=allow_missing
        )
        latex, wide_flat = write_table_artifacts(table_name, wide, long_df, export_tex=export_tex)
        readable_frames[TABLE_FILE_STEMS[table_name]] = wide_flat
        if latex:
            latex_blocks.append((table_name, latex))

    s7_wide, s7_long = build_table_s7_wide(synthetic_current, allow_missing=allow_missing)
    latex, wide_flat = write_table_artifacts("Table S.7", s7_wide, s7_long, export_tex=export_tex)
    readable_frames["table_s7"] = wide_flat
    if latex:
        latex_blocks.append(("Table S.7", latex))

    export_tables_from_frames(readable_frames, TABLES_OUTPUT_DIR)

    if export_tex:
        tex_lines = ["% Auto-generated by src/make_paper_results.py", ""]
        for table_name, block in latex_blocks:
            tex_lines.append(f"% {table_name}")
            tex_lines.append(block)
            tex_lines.append("")
        (TABLES_OUTPUT_DIR / "all_tables.tex").write_text("\n".join(tex_lines), encoding="utf-8")

    print(f"Paper tables written to: {TABLES_OUTPUT_DIR}")



# =============================================================================
# Figures (Figure 5, S.5, S.6)
# =============================================================================

CITY_META = {
    "City 1": {
        "figure_id": "Figure 5",
        "stem": "figure5_city1",
        "metrics_xlsx": METRICS_DIR / "lur_city1_metrics.xlsx",
    },
    "City 2": {
        "figure_id": "Figure S.5",
        "stem": "figure_s5_city2",
        "metrics_xlsx": METRICS_DIR / "lur_city2_metrics.xlsx",
    },
    "City 3": {
        "figure_id": "Figure S.6",
        "stem": "figure_s6_city3",
        "metrics_xlsx": METRICS_DIR / "lur_city3_metrics.xlsx",
    },
}

SUMMARY_COLUMNS = [
    "Figure",
    "City",
    "alpha",
    "alpha_label",
    "experiments_completed",
    "P_err_LUR",
    "P_err_LUR_std",
    "R_asy_LUR",
    "R_asy_LUR_std",
    "R_asy_nonLUR",
    "R_asy_nonLUR_std",
    "UHCE_Edges_Count",
    "UHCE_Edges_Count_std",
    "HCE_Edges_Count",
    "HCE_Edges_Count_std",
    "SHCE_Edges_Count",
    "SHCE_Edges_Count_std",
    "UHCE_Confidence",
    "UHCE_Confidence_std",
    "HCE_Confidence",
    "HCE_Confidence_std",
    "SHCE_Confidence",
    "SHCE_Confidence_std",
]


def load_city_summary(city_name: str, metrics_dir: Path) -> pd.DataFrame:
    meta = dict(CITY_META[city_name])
    xlsx_path = metrics_dir / meta["metrics_xlsx"].name
    summary = pd.read_excel(xlsx_path, sheet_name="Summary").copy().sort_values("alpha").reset_index(drop=True)
    summary.insert(0, "City", city_name)
    summary.insert(0, "Figure", meta["figure_id"])
    return summary[SUMMARY_COLUMNS]


def build_plot_figure(city_table: pd.DataFrame) -> plt.Figure:
    proportions = [f"{int(round(value * 100))}%" for value in city_table["alpha"].tolist()]
    uhces_numbers = city_table["UHCE_Edges_Count"].tolist()
    hces_numbers = city_table["HCE_Edges_Count"].tolist()
    shces_numbers = city_table["SHCE_Edges_Count"].tolist()
    uhces_confidence = city_table["UHCE_Confidence"].tolist()
    hces_confidence = city_table["HCE_Confidence"].tolist()
    shces_confidence = city_table["SHCE_Confidence"].tolist()
    y1 = [value * 100 for value in city_table["P_err_LUR"].tolist()]
    y2 = [value * 100 for value in city_table["R_asy_LUR"].tolist()]
    y3 = [value * 100 for value in city_table["R_asy_nonLUR"].tolist()]

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.linewidth": 1.2,
            "grid.linewidth": 0.8,
            "lines.linewidth": 2.5,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    ax3, ax4, ax1, ax2 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.96, wspace=0.18, hspace=0.32)

    x = list(range(len(proportions)))
    colors = {
        "UHCE": "#1f77b4",
        "HCE": "#ff7f0e",
        "SHCE": "#2ca02c",
        "P_err": "#821fb4",
        "R_asy": "#ffdb0e",
        "R_non": "#a04d2c",
    }

    ax3.text(0.5, -0.16, "(a)", transform=ax3.transAxes, ha="center", fontsize=20)
    ax4.text(0.5, -0.16, "(b)", transform=ax4.transAxes, ha="center", fontsize=20)
    ax1.text(0.5, -0.18, "(c)", transform=ax1.transAxes, ha="center", fontsize=20)
    ax2.text(0.5, -0.18, "(d)", transform=ax2.transAxes, ha="center", fontsize=20)

    width = 0.25
    ax1.bar([item - width for item in x], uhces_numbers, width, label="UHCE", color=colors["UHCE"], edgecolor="white")
    ax1.bar(x, hces_numbers, width, label="HCE", color=colors["HCE"], edgecolor="white")
    ax1.bar([item + width for item in x], shces_numbers, width, label="SHCE", color=colors["SHCE"], edgecolor="white")
    ax1.set_xlabel("α", fontsize=16)
    ax1.set_ylabel("Number of edges", fontsize=16)
    ax1.set_ylim(0, max(max(uhces_numbers), max(hces_numbers), max(shces_numbers)) * 1.1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(proportions, fontsize=14)
    ax1.legend(loc="upper left", frameon=True, fontsize=12)
    ax1.set_axisbelow(True)

    ax2.plot(x, uhces_confidence, marker="o", label="UHCE", color=colors["UHCE"], markersize=7, markerfacecolor="white", markeredgewidth=2)
    ax2.plot(x, hces_confidence, marker="s", label="HCE", color=colors["HCE"], markersize=7, markerfacecolor="white", markeredgewidth=2)
    ax2.plot(x, shces_confidence, marker="^", label="SHCE", color=colors["SHCE"], markersize=7, markerfacecolor="white", markeredgewidth=2)
    min_conf = min(min(uhces_confidence), min(hces_confidence), min(shces_confidence)) * 0.98
    ax2.set_ylim(min_conf, 1.005)
    ax2.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax2.set_xlabel("α", fontsize=16)
    ax2.set_ylabel("Precision", fontsize=16)
    ax2.set_xticks(x)
    ax2.set_xticklabels(proportions, fontsize=14)
    ax2.legend(loc="lower left", ncol=1, frameon=True, fontsize=12, title_fontsize=11)
    ax2.set_axisbelow(True)

    ax3.plot(x, y1, "o-", label=r"$P_{\mathrm{err}}^{\mathrm{LUR}}$", color=colors["P_err"], markersize=7)
    ax3.set_ylim(95, 100.5)
    ax3.yaxis.set_major_locator(MultipleLocator(2))
    ax3.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax3.set_xlabel(r"$\alpha$", fontsize=16)
    ax3.set_ylabel(r"$\beta$", fontsize=16, rotation=360)
    ax3.set_xticks(x)
    ax3.set_xticklabels(proportions, fontsize=14)
    ax3.set_axisbelow(True)

    ax4.plot(x, y2, "s-", label="LUR", color=colors["R_asy"], markersize=7)
    ax4.plot(x, y3, "^-", label="Non-LUR", color=colors["R_non"], markersize=7)
    ax4.set_ylim(0, 40)
    ax4.yaxis.set_major_locator(MultipleLocator(10))
    ax4.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax4.set_xlabel(r"$\alpha$", fontsize=16)
    ax4.set_ylabel("Proportion of asymmetric edges", fontsize=16)
    ax4.set_xticks(x)
    ax4.set_xticklabels(proportions, fontsize=14)
    ax4.legend(loc="upper left", frameon=True, fontsize=12, title_fontsize=11)
    ax4.set_axisbelow(True)
    return fig


def export_readable_outputs(city_tables: Dict[str, pd.DataFrame], readable_dir: Path) -> None:
    readable_dir.mkdir(parents=True, exist_ok=True)
    for city_name, city_table in city_tables.items():
        stem = CITY_META[city_name]["stem"]
        fig = build_plot_figure(city_table)
        fig.savefig(readable_dir / f"{stem}.png", format="png", dpi=900, bbox_inches="tight")
        plt.close(fig)


def build_all_figures(metrics_dir: Path, figures_dir: Path, cities: Optional[List[str]] = None) -> None:
    selected = list(CITY_META) if not cities else [city for city in CITY_META if city in cities]
    if cities:
        unknown = [c for c in cities if c not in CITY_META]
        if unknown:
            raise ValueError(f"Unknown city/cities for figures: {unknown}. Valid: {list(CITY_META)}")
    city_tables = {city: load_city_summary(city, metrics_dir) for city in selected}
    export_readable_outputs(city_tables, figures_dir)
    print(f"Paper figures written to: {figures_dir}")


def parse_city_filter(raw: Optional[str]) -> Optional[List[str]]:
    """Map a user string like '1', 'city1', '1 2 3' to canonical ['City 1', ...]."""
    if not raw:
        return None
    name_by_num = {"1": "City 1", "2": "City 2", "3": "City 3"}
    cities: List[str] = []
    for tok in re.split(r"[,\s]+", raw.strip()):
        if not tok:
            continue
        key = tok.lower().replace("city", "").strip()
        cities.append(name_by_num.get(key, tok))
    return cities or None


def main() -> None:
    parser = argparse.ArgumentParser(description="Make all final paper tables and figures.")
    parser.add_argument("--tables", action="store_true", help="Build only the tables.")
    parser.add_argument("--figures", action="store_true", help="Build only the figures.")
    parser.add_argument("--results-root", type=Path, default=RESULTS_DIR,
                        help="Directory containing the canonical raw result workbooks.")
    parser.add_argument("--metrics-dir", type=Path, default=None,
                        help="Directory containing the LUR figure metric workbooks "
                             "(default: <results-root>/lur_figures).")
    parser.add_argument("--export-tex", action="store_true",
                        help="Also export a combined all_tables.tex file.")
    parser.add_argument("--allow-missing", action="store_true",
                        help="Render missing expected table entries as '--' instead of failing.")
    parser.add_argument("--cities", type=str, default=None,
                        help="Restrict figure building to these cities (e.g. '1', 'city1', '1 2 3'). "
                             "Default: all cities. Lets a single figure be reproduced without the others.")
    args = parser.parse_args()

    do_tables = args.tables or not args.figures
    do_figures = args.figures or not args.tables
    metrics_dir = args.metrics_dir or (args.results_root / "lur_figures")

    if do_tables:
        build_all_tables(args.results_root, allow_missing=args.allow_missing, export_tex=args.export_tex)
    if do_figures:
        FIGURES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        build_all_figures(metrics_dir, FIGURES_OUTPUT_DIR, cities=parse_city_filter(args.cities))


if __name__ == "__main__":
    main()
