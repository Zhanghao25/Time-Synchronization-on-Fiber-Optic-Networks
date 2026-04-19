#!/usr/bin/env python3
"""
Merge sharded Figure 5 / Figure S.5 / Figure S.6 metric workbooks.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def normalize_empty_group_confidence(individual_df: pd.DataFrame) -> pd.DataFrame:
    df = individual_df.copy()
    mapping = [
        ("SHCE_Confidence", "SHCE_Edges_Count"),
        ("HCE_Confidence", "HCE_Edges_Count"),
        ("UHCE_Confidence", "UHCE_Edges_Count"),
    ]
    for confidence_col, count_col in mapping:
        if confidence_col in df.columns and count_col in df.columns:
            df.loc[df[count_col] <= 0, confidence_col] = float("nan")
    return df


def aggregate_metrics(individual_df: pd.DataFrame) -> pd.DataFrame:
    if individual_df.empty:
        return pd.DataFrame()

    individual_df = normalize_empty_group_confidence(individual_df)
    exclude = {"seed", "predicted_nonzero_indices", "true_nonzero_indices", "error_locations"}
    numeric_cols = [col for col in individual_df.columns if col not in exclude and col != "alpha"]

    summary_rows: List[Dict[str, Any]] = []
    for alpha, group in individual_df.groupby("alpha", sort=True):
        row: Dict[str, Any] = {
            "alpha": alpha,
            "alpha_label": f"{alpha:.2f}",
            "experiments_completed": len(group),
        }
        for col in numeric_cols:
            if pd.api.types.is_numeric_dtype(group[col]):
                row[col] = float(group[col].mean())
                row[f"{col}_std"] = float(group[col].std(ddof=0))
        summary_rows.append(row)

    return pd.DataFrame(summary_rows).sort_values("alpha").reset_index(drop=True)


def infer_experiments_per_alpha(individual_df: pd.DataFrame) -> int | str:
    if individual_df.empty:
        return 0
    counts = individual_df.groupby("alpha").size().astype(int).tolist()
    return counts[0] if len(set(counts)) == 1 else ",".join(str(item) for item in counts)


def read_metadata(xlsx_path: Path) -> Dict[str, Any]:
    metadata_df = pd.read_excel(xlsx_path, sheet_name="Metadata")
    return dict(zip(metadata_df["Key"], metadata_df["Value"]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge LUR figure shard outputs.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Shard workbooks to merge.")
    parser.add_argument("--save", required=True, help="Merged workbook path.")
    args = parser.parse_args()

    input_paths = [Path(item) for item in args.inputs]
    save_path = Path(args.save)

    individual_frames = []
    metadata = None
    for path in input_paths:
        individual_frames.append(pd.read_excel(path, sheet_name="Individual_Results"))
        if metadata is None:
            metadata = read_metadata(path)

    individual_df = normalize_empty_group_confidence(pd.concat(individual_frames, ignore_index=True))
    summary_df = aggregate_metrics(individual_df)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(save_path) as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        individual_df.to_excel(writer, sheet_name="Individual_Results", index=False)
        if metadata is not None:
            metadata = dict(metadata)
            metadata["experiments"] = infer_experiments_per_alpha(individual_df)
            metadata["unique_seed_count"] = int(individual_df["seed"].dropna().nunique())
            pd.DataFrame([{"Key": key, "Value": value} for key, value in metadata.items()]).to_excel(
                writer,
                sheet_name="Metadata",
                index=False,
            )
        pd.DataFrame({"Shard_Source": [str(path) for path in input_paths]}).to_excel(
            writer,
            sheet_name="Shard_Sources",
            index=False,
        )

    print(f"Merged {len(input_paths)} shards into {save_path}")


if __name__ == "__main__":
    main()
