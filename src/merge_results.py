#!/usr/bin/env python3
"""
Merge experiment results into summary tables.

Handles:
1. Synthetic Table S.7 results (results/raw/synthetic/)
2. SCAD 4-Step results (results/raw/main_scad/)
3. Other Methods results (results/raw/other_methods/)

Usage:
    python src/merge_results.py
    python src/merge_results.py --synthetic  # Only merge synthetic results
    python src/merge_results.py --scad       # Only merge SCAD results
    python src/merge_results.py --other      # Only merge Other Methods results
"""

import argparse
from datetime import datetime
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Directory configuration
BASE_DIR = Path(__file__).resolve().parents[1]
SYNTHETIC_RESULTS_DIR = BASE_DIR / "results" / "raw" / "synthetic"
SCAD_RESULTS_DIR = BASE_DIR / "results" / "raw" / "main_scad"
OTHER_RESULTS_DIR = BASE_DIR / "results" / "raw" / "other_methods"

# Experiment configuration
CITIES = [1, 2, 3]
RATIOS = [0.01, 0.05, 0.10]
SCAD_STEP_ORDER = ["step1", "step2", "step3", "step4"]
SCAD_METRICS = ["accuracy", "precision", "recall", "f05_score"]
OTHER_METHOD_ORDER = ["L0", "Lasso", "MCP"]
SYNTHETIC_METHOD_ORDER = ["step1_original", "step2_merged", "step4_iterative"]


def read_metadata_sheet(path: Path, sheet_name: str = "Metadata") -> Dict[str, object]:
    try:
        metadata_df = pd.read_excel(path, sheet_name=sheet_name)
    except Exception:
        return {}
    if "Key" not in metadata_df.columns or "Value" not in metadata_df.columns:
        return {}
    return dict(zip(metadata_df["Key"], metadata_df["Value"]))


def merge_scad_shard_group(paths: List[Path], output_path: Path) -> None:
    individuals: List[pd.DataFrame] = []
    seed_statuses: List[pd.DataFrame] = []
    shard_records: List[Dict[str, object]] = []

    for shard_path in sorted(paths):
        try:
            individual_df = pd.read_excel(shard_path, sheet_name="Individual_Results")
        except Exception:
            continue
        if individual_df.empty:
            continue
        individual_df = individual_df.copy()
        individual_df["Shard_File"] = str(shard_path)
        individuals.append(individual_df)

        try:
            seed_status_df = pd.read_excel(shard_path, sheet_name="Seed_Status")
        except Exception:
            seed_status_df = pd.DataFrame()
        if not seed_status_df.empty:
            seed_status_df = seed_status_df.copy()
            seed_status_df["Shard_File"] = str(shard_path)
            seed_statuses.append(seed_status_df)

        metadata = read_metadata_sheet(shard_path)
        shard_records.append({
            "shard_file": str(shard_path),
            "paper_tag": metadata.get("paper_tag"),
            "base_seed": metadata.get("base_seed"),
            "experiment_count": metadata.get("experiment_count"),
            "dataset": metadata.get("dataset"),
            "sparsity_ratio": metadata.get("sparsity_ratio"),
        })

    if not individuals:
        raise ValueError(f"No shard Individual_Results found for {output_path.name}")

    merged_individual = pd.concat(individuals, ignore_index=True, sort=False)
    if "Seed" in merged_individual.columns:
        merged_individual = merged_individual.sort_values(["Seed", "Step"], kind="stable").reset_index(drop=True)
        experiment_lookup = (
            merged_individual[["Seed"]]
            .drop_duplicates()
            .sort_values("Seed", kind="stable")
            .reset_index(drop=True)
        )
        experiment_lookup["Experiment"] = np.arange(1, len(experiment_lookup) + 1)
        merged_individual = merged_individual.drop(columns=["Experiment"], errors="ignore").merge(
            experiment_lookup,
            on="Seed",
            how="left",
        )
    else:
        merged_individual["Experiment"] = np.arange(1, len(merged_individual) + 1)

    merged_seed_status = pd.DataFrame()
    if seed_statuses:
        merged_seed_status = pd.concat(seed_statuses, ignore_index=True, sort=False)
        if "Seed" in merged_seed_status.columns:
            merged_seed_status = (
                merged_seed_status
                .sort_values(["Seed"], kind="stable")
                .drop_duplicates(subset=["Seed"], keep="last")
                .reset_index(drop=True)
            )
            experiment_lookup = (
                merged_seed_status[["Seed"]]
                .drop_duplicates()
                .sort_values("Seed", kind="stable")
                .reset_index(drop=True)
            )
            experiment_lookup["Experiment"] = np.arange(1, len(experiment_lookup) + 1)
            merged_seed_status = merged_seed_status.drop(columns=["Experiment"], errors="ignore").merge(
                experiment_lookup,
                on="Seed",
                how="left",
            )
            merged_seed_status = merged_seed_status[["Experiment"] + [c for c in merged_seed_status.columns if c != "Experiment"]]

    summary_rows: List[Dict[str, object]] = []
    for step in SCAD_STEP_ORDER:
        step_df = merged_individual[merged_individual["Step"] == step].copy()
        completed_df = step_df[step_df["Status"] == "Completed"].copy()
        skipped_df = step_df[step_df["Status"] == "Skipped"].copy()
        row: Dict[str, object] = {
            "Step": step,
            "Completed": int(len(completed_df)),
            "Skipped": int(len(skipped_df)),
            "Total": int(len(step_df)),
        }
        for metric in SCAD_METRICS:
            if completed_df.empty or metric not in completed_df.columns:
                row[f"{metric}_mean"] = 0.0
                row[f"{metric}_std"] = 0.0
            else:
                values = completed_df[metric].dropna().astype(float).to_numpy()
                row[f"{metric}_mean"] = float(np.mean(values)) if len(values) else 0.0
                row[f"{metric}_std"] = float(np.std(values)) if len(values) else 0.0
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)

    shard_meta_df = pd.DataFrame(shard_records)
    metadata: Dict[str, object] = {
        "dataset": shard_meta_df["dataset"].dropna().iloc[0] if shard_meta_df["dataset"].notna().any() else "",
        "sparsity_ratio": shard_meta_df["sparsity_ratio"].dropna().iloc[0] if shard_meta_df["sparsity_ratio"].notna().any() else math.nan,
        "paper_tag": shard_meta_df["paper_tag"].dropna().iloc[0] if shard_meta_df["paper_tag"].notna().any() else "",
        "base_seed": int(pd.to_numeric(shard_meta_df["base_seed"], errors="coerce").dropna().min()) if shard_meta_df["base_seed"].notna().any() else (int(merged_individual["Seed"].min()) if "Seed" in merged_individual.columns else ""),
        "experiment_count": int(merged_individual["Experiment"].nunique()),
        "num_experiments": int(merged_individual["Experiment"].nunique()),
        "shard_count": len(paths),
        "seed_min": int(merged_individual["Seed"].min()) if "Seed" in merged_individual.columns else "",
        "seed_max": int(merged_individual["Seed"].max()) if "Seed" in merged_individual.columns else "",
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path) as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        pd.DataFrame([{"Key": key, "Value": value} for key, value in metadata.items()]).to_excel(
            writer, sheet_name="Metadata", index=False
        )
        merged_individual.to_excel(writer, sheet_name="Individual_Results", index=False)
        if not merged_seed_status.empty:
            merged_seed_status.to_excel(writer, sheet_name="Seed_Status", index=False)
        shard_meta_df.to_excel(writer, sheet_name="Shard_Sources", index=False)


def merge_other_shard_group(paths: List[Path], output_path: Path) -> None:
    details: List[pd.DataFrame] = []
    seed_statuses: List[pd.DataFrame] = []
    shard_records: List[Dict[str, object]] = []

    for shard_path in sorted(paths):
        metadata = read_metadata_sheet(shard_path)
        shard_records.append({
            "shard_file": str(shard_path),
            "paper_tag": metadata.get("paper_tag"),
            "num_experiments": metadata.get("num_experiments"),
            "base_seed": metadata.get("base_seed"),
            "use_merge": metadata.get("use_merge"),
            "data_file": metadata.get("data_file"),
            "sparsity_ratio": metadata.get("sparsity_ratio"),
            "matrix_shape": metadata.get("matrix_shape"),
            "methods": metadata.get("methods"),
        })
        try:
            detailed_df = pd.read_excel(shard_path, sheet_name="Detailed_Results")
        except Exception:
            detailed_df = pd.DataFrame()
        if detailed_df.empty:
            continue
        detailed_df = detailed_df.copy()
        detailed_df["Shard_File"] = str(shard_path)
        details.append(detailed_df)

        try:
            seed_status_df = pd.read_excel(shard_path, sheet_name="Seed_Status")
        except Exception:
            seed_status_df = pd.DataFrame()
        if not seed_status_df.empty:
            seed_status_df = seed_status_df.copy()
            seed_status_df["Shard_File"] = str(shard_path)
            seed_statuses.append(seed_status_df)

    if not details:
        raise ValueError(f"No shard Detailed_Results found for {output_path.name}")

    merged_detailed = pd.concat(details, ignore_index=True, sort=False)
    merged_detailed = merged_detailed.sort_values(["Seed", "Method"], kind="stable").reset_index(drop=True)
    experiment_lookup = (
        merged_detailed[["Seed"]]
        .drop_duplicates()
        .sort_values("Seed", kind="stable")
        .reset_index(drop=True)
    )
    experiment_lookup["Experiment"] = np.arange(1, len(experiment_lookup) + 1)
    merged_detailed = merged_detailed.drop(columns=["Experiment"], errors="ignore").merge(
        experiment_lookup,
        on="Seed",
        how="left",
    )
    merged_detailed = merged_detailed[["Experiment"] + [c for c in merged_detailed.columns if c != "Experiment"]]

    merged_seed_status = pd.DataFrame()
    if seed_statuses:
        merged_seed_status = pd.concat(seed_statuses, ignore_index=True, sort=False)
        if "Seed" in merged_seed_status.columns:
            merged_seed_status = (
                merged_seed_status
                .sort_values(["Seed"], kind="stable")
                .drop_duplicates(subset=["Seed"], keep="last")
                .reset_index(drop=True)
            )
            experiment_lookup = (
                merged_seed_status[["Seed"]]
                .drop_duplicates()
                .sort_values("Seed", kind="stable")
                .reset_index(drop=True)
            )
            experiment_lookup["Experiment"] = np.arange(1, len(experiment_lookup) + 1)
            merged_seed_status = merged_seed_status.drop(columns=["Experiment"], errors="ignore").merge(
                experiment_lookup,
                on="Seed",
                how="left",
            )
            merged_seed_status = merged_seed_status[["Experiment"] + [c for c in merged_seed_status.columns if c != "Experiment"]]

    shard_meta_df = pd.DataFrame(shard_records)
    total_experiments = int(pd.to_numeric(shard_meta_df["num_experiments"], errors="coerce").fillna(0).sum())
    stats_rows: List[Dict[str, object]] = []
    for method in OTHER_METHOD_ORDER:
        method_df = merged_detailed[merged_detailed["Method"] == method].copy()
        if method_df.empty:
            continue
        successful = int(len(method_df))
        failed = max(0, int(total_experiments - successful))
        stats_rows.append({
            "Method": method,
            "Mean_Accuracy": float(method_df["Accuracy"].mean()),
            "Std_Accuracy": float(method_df["Accuracy"].std(ddof=0)),
            "Mean_F05_Score": float(method_df["F05_Score"].mean()),
            "Std_F05_Score": float(method_df["F05_Score"].std(ddof=0)),
            "Mean_Precision": float(method_df["Precision"].mean()),
            "Std_Precision": float(method_df["Precision"].std(ddof=0)),
            "Mean_Recall": float(method_df["Recall"].mean()),
            "Std_Recall": float(method_df["Recall"].std(ddof=0)),
            "Mean_Error_Count": float(method_df["Error_Count"].mean()),
            "Mean_Lambda": float(method_df["Optimal_Lambda"].mean()),
            "Successful_Experiments": successful,
            "Failed_Experiments": failed,
            "Success_Rate": successful / (successful + failed) if (successful + failed) else 0.0,
        })
    stats_df = pd.DataFrame(stats_rows).sort_values("Mean_F05_Score", ascending=False).reset_index(drop=True)

    use_merge_raw = shard_meta_df["use_merge"].dropna().iloc[0] if shard_meta_df["use_merge"].notna().any() else 0
    use_merge = str(use_merge_raw).strip().lower() in {"1", "true", "yes"}
    experiment_info = pd.DataFrame([{
        "data_file": shard_meta_df["data_file"].dropna().iloc[0] if shard_meta_df["data_file"].notna().any() else "",
        "sparsity_ratio": shard_meta_df["sparsity_ratio"].dropna().iloc[0] if shard_meta_df["sparsity_ratio"].notna().any() else np.nan,
        "num_experiments": total_experiments,
        "base_seed": int(merged_detailed["Seed"].min()) if "Seed" in merged_detailed.columns else 0,
        "use_merge": use_merge,
        "matrix_shape": shard_meta_df["matrix_shape"].dropna().iloc[0] if shard_meta_df["matrix_shape"].notna().any() else "",
        "methods": shard_meta_df["methods"].dropna().iloc[0] if shard_meta_df["methods"].notna().any() else "",
        "paper_tag": shard_meta_df["paper_tag"].dropna().iloc[0] if shard_meta_df["paper_tag"].notna().any() else "",
    }])
    metadata_df = pd.DataFrame([
        {"Key": "data_file", "Value": experiment_info.iloc[0]["data_file"]},
        {"Key": "sparsity_ratio", "Value": experiment_info.iloc[0]["sparsity_ratio"]},
        {"Key": "num_experiments", "Value": total_experiments},
        {"Key": "base_seed", "Value": experiment_info.iloc[0]["base_seed"]},
        {"Key": "use_merge", "Value": int(use_merge)},
        {"Key": "matrix_shape", "Value": experiment_info.iloc[0]["matrix_shape"]},
        {"Key": "methods", "Value": experiment_info.iloc[0]["methods"]},
        {"Key": "paper_tag", "Value": experiment_info.iloc[0]["paper_tag"]},
        {"Key": "shard_count", "Value": len(paths)},
        {"Key": "seed_min", "Value": int(merged_detailed["Seed"].min()) if "Seed" in merged_detailed.columns else 0},
        {"Key": "seed_max", "Value": int(merged_detailed["Seed"].max()) if "Seed" in merged_detailed.columns else 0},
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        stats_df.to_excel(writer, sheet_name="Statistical_Analysis", index=False)
        experiment_info.to_excel(writer, sheet_name="Experiment_Info", index=False)
        metadata_df.to_excel(writer, sheet_name="Metadata", index=False)
        merged_detailed.to_excel(writer, sheet_name="Detailed_Results", index=False)
        if not merged_seed_status.empty:
            merged_seed_status.to_excel(writer, sheet_name="Seed_Status", index=False)
        shard_meta_df.to_excel(writer, sheet_name="Shard_Sources", index=False)


def combine_mean_std(means: np.ndarray, stds: np.ndarray, counts: np.ndarray) -> Tuple[float, float]:
    total = int(counts.sum())
    if total <= 0:
        return math.nan, math.nan
    weighted_mean = float(np.sum(counts * means) / total)
    second_moment = float(np.sum(counts * (stds ** 2 + means ** 2)) / total)
    variance = max(0.0, second_moment - weighted_mean ** 2)
    return weighted_mean, math.sqrt(variance)


def merge_synthetic_shard_group(paths: List[Path], output_path: Path) -> None:
    summaries: List[pd.DataFrame] = []
    shard_records: List[Dict[str, object]] = []

    for shard_path in sorted(paths):
        try:
            summary_df = pd.read_excel(shard_path, sheet_name="Summary")
        except Exception:
            continue
        if summary_df.empty:
            continue
        summary_df = summary_df.copy()
        summary_df["Shard_File"] = str(shard_path)
        summaries.append(summary_df)
        metadata = read_metadata_sheet(shard_path)
        shard_records.append({
            "shard_file": str(shard_path),
            "paper_tag": metadata.get("paper_tag"),
            "base_seed": metadata.get("base_seed"),
            "experiments": metadata.get("experiments"),
            "data_workbook": metadata.get("data_workbook"),
        })

    if not summaries:
        raise ValueError(f"No shard Summary sheets found for {output_path.name}")

    merged_summary_input = pd.concat(summaries, ignore_index=True, sort=False)
    rows: List[Dict[str, object]] = []
    for method in SYNTHETIC_METHOD_ORDER:
        method_df = merged_summary_input[merged_summary_input["Method"] == method].copy()
        if method_df.empty:
            continue
        row = method_df.iloc[0][["Method", "City", "Scenario", "Legacy_Case", "Hard_K", "Zeta", "Ratio"]].to_dict()
        counts = pd.to_numeric(method_df["N_Experiments"], errors="coerce").fillna(0).to_numpy(dtype=float)
        row["N_Experiments"] = int(counts.sum())
        row["Base_Seed"] = min(pd.to_numeric(method_df["Base_Seed"], errors="coerce").dropna().astype(int).tolist() or [math.nan])
        row["Paper_Tag"] = next((str(v) for v in method_df["Paper_Tag"] if pd.notna(v) and str(v) != ""), "")
        for metric in SCAD_METRICS:
            means = pd.to_numeric(method_df[f"{metric}_mean"], errors="coerce").fillna(0).to_numpy(dtype=float)
            stds = pd.to_numeric(method_df[f"{metric}_std"], errors="coerce").fillna(0).to_numpy(dtype=float)
            mean_value, std_value = combine_mean_std(means, stds, counts)
            row[f"{metric}_mean"] = mean_value
            row[f"{metric}_std"] = std_value
        rows.append(row)

    merged_summary = pd.DataFrame(rows)
    shard_meta_df = pd.DataFrame(shard_records)
    metadata_df = pd.DataFrame([
        {"Key": "data_workbook", "Value": shard_meta_df["data_workbook"].dropna().iloc[0] if shard_meta_df["data_workbook"].notna().any() else ""},
        {"Key": "experiments", "Value": int(merged_summary["N_Experiments"].max()) if not merged_summary.empty else 0},
        {"Key": "base_seed", "Value": int(pd.to_numeric(shard_meta_df["base_seed"], errors="coerce").dropna().min()) if shard_meta_df["base_seed"].notna().any() else 42},
        {"Key": "paper_tag", "Value": shard_meta_df["paper_tag"].dropna().iloc[0] if shard_meta_df["paper_tag"].notna().any() else ""},
        {"Key": "shard_count", "Value": len(paths)},
        {"Key": "seed_min", "Value": int(pd.to_numeric(shard_meta_df["base_seed"], errors="coerce").dropna().min()) if shard_meta_df["base_seed"].notna().any() else 42},
        {"Key": "seed_max", "Value": int(pd.to_numeric(shard_meta_df["base_seed"], errors="coerce").dropna().max() + pd.to_numeric(shard_meta_df["experiments"], errors="coerce").fillna(0).max() - 1) if shard_meta_df["base_seed"].notna().any() else 42},
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        merged_summary.to_excel(writer, sheet_name="Summary", index=False)
        metadata_df.to_excel(writer, sheet_name="Metadata", index=False)
        shard_meta_df.to_excel(writer, sheet_name="Shard_Sources", index=False)


def merge_shard_directory(kind: str, shard_dir: Path, output_dir: Path) -> None:
    if not shard_dir.exists():
        print(f"No shard directory found: {shard_dir}")
        return

    if kind == "scad":
        grouped: Dict[Tuple[str, str], List[Path]] = {}
        for path in sorted(shard_dir.glob("scad_city*_ratio*_shard*.xlsx")):
            match = re.match(r"scad_city(\d+)_ratio([0-9.]+)_shard\d+\.xlsx$", path.name)
            if match:
                grouped.setdefault((f"city{match.group(1)}", match.group(2)), []).append(path)
        for (city, ratio), paths in grouped.items():
            output_path = output_dir / f"scad_{city}_ratio{ratio}.xlsx"
            print(f"Merging {len(paths)} shards into {output_path}")
            merge_scad_shard_group(paths, output_path)
        return

    if kind == "other":
        grouped_other: Dict[Tuple[str, str, str], List[Path]] = {}
        for path in sorted(shard_dir.glob("othermethods_*_shard*.xlsx")):
            match = re.match(r"othermethods_(original|merged)_city(\d+)_ratio([0-9.]+)_shard\d+\.xlsx$", path.name)
            if match:
                grouped_other.setdefault((match.group(1), f"city{match.group(2)}", match.group(3)), []).append(path)
        for (matrix_type, city, ratio), paths in grouped_other.items():
            output_path = output_dir / f"othermethods_{matrix_type}_{city}_ratio{ratio}.xlsx"
            print(f"Merging {len(paths)} shards into {output_path}")
            merge_other_shard_group(paths, output_path)
        return

    if kind == "synthetic":
        grouped_syn: Dict[Tuple[str, str], List[Path]] = {}
        for path in sorted(shard_dir.glob("synthetic_zeta*_alpha*_shard*.xlsx")):
            match = re.match(r"synthetic_zeta(\d+pct)_alpha(\d+)_shard\d+\.xlsx$", path.name)
            if match:
                grouped_syn.setdefault((match.group(1), match.group(2)), []).append(path)
        for (zeta_tag, alpha_tag), paths in grouped_syn.items():
            output_path = output_dir / f"synthetic_zeta{zeta_tag}_alpha{alpha_tag}.xlsx"
            print(f"Merging {len(paths)} shards into {output_path}")
            merge_synthetic_shard_group(paths, output_path)
        return

    raise ValueError(f"Unknown shard merge kind: {kind}")


def merge_synthetic_results():
    """
    Merge synthetic Table S.7 result files into one summary.

    Expected input files include both legacy names such as
    `9cases_v3_City1_r10.xlsx` and the current semantic names such as
    `synthetic_zeta1pct_alpha10.xlsx`.
    """
    print("=" * 60)
    print("Merging Synthetic Table S.7 Results")
    print("=" * 60)

    all_dfs = []

    for filepath in sorted(SYNTHETIC_RESULTS_DIR.glob("*.xlsx")):
        name_lower = filepath.name.lower()
        if "combined" in name_lower:
            continue
        if not (name_lower.startswith("9cases_v3_city") or name_lower.startswith("synthetic_zeta")):
            continue
        try:
            df = pd.read_excel(filepath, sheet_name='Summary')
        except Exception:
            continue
        all_dfs.append(df)
        print(f"  [LOADED] {filepath.name}")

    if not all_dfs:
        print("\nNo synthetic result files found!")
        return None

    merged = pd.concat(all_dfs, ignore_index=True)

    # Save combined results
    output_path = SYNTHETIC_RESULTS_DIR / "table_s7_combined.xlsx"

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        merged.to_excel(writer, sheet_name='Summary', index=False)

        # Create pivot tables for each metric
        for metric in ['accuracy', 'precision', 'recall', 'f05_score']:
            try:
                pivot = merged.pivot_table(
                    index=['Method', 'Ratio'],
                    columns='City' if 'City' in merged.columns else 'Zeta',
                    values=[f'{metric}_mean', f'{metric}_std'],
                    aggfunc='first'
                )
                pivot.to_excel(writer, sheet_name=f'{metric}_pivot')
            except:
                pass

    print(f"\nSaved: {output_path}")

    # Print summary table
    print("\n" + "=" * 80)
    print("Synthetic Results Summary")
    print("=" * 80)

    for method in ['step1_original', 'step2_merged', 'step4_iterative']:
        method_df = merged[merged['Method'] == method]
        if method_df.empty:
            continue
        print(f"\n{method}:")
        print("-" * 60)
        for _, row in method_df.iterrows():
            print(f"  {row['City']} r={row['Ratio']:.2f}: "
                  f"acc={row['accuracy_mean']:.4f}±{row['accuracy_std']:.4f} "
                  f"F0.5={row['f05_score_mean']:.4f}±{row['f05_score_std']:.4f}")

    return merged


def merge_scad_results():
    """
    Merge SCAD 4-step experiment results.

    Expected input files: scad_city{1,2,3}_ratio{0.01,0.05,0.10}.xlsx
    Each file contains Summary and Individual_Results sheets.
    """
    print("=" * 60)
    print("Merging SCAD 4-Step Results")
    print("=" * 60)

    all_summary = []
    all_individual = []

    for city in CITIES:
        for ratio in RATIOS:
            # Format ratio for filename (e.g., 0.01 -> 0.01)
            filename = f"scad_city{city}_ratio{ratio}.xlsx"
            filepath = SCAD_RESULTS_DIR / filename

            if not filepath.exists():
                print(f"  [MISSING] {filename}")
                continue

            print(f"  [LOADED] {filename}")

            try:
                # Read Summary sheet
                summary_df = pd.read_excel(filepath, sheet_name='Summary')
                summary_df['City'] = f"City{city}"
                summary_df['Ratio'] = ratio
                all_summary.append(summary_df)

                # Read Individual_Results sheet if exists
                try:
                    individual_df = pd.read_excel(filepath, sheet_name='Individual_Results')
                    individual_df['City'] = f"City{city}"
                    individual_df['Ratio'] = ratio
                    all_individual.append(individual_df)
                except:
                    pass

            except Exception as e:
                print(f"  [ERROR] {filename}: {e}")

    if not all_summary:
        print("\nNo SCAD result files found!")
        return None

    # Merge all summaries
    merged_summary = pd.concat(all_summary, ignore_index=True)

    # Merge all individual results
    merged_individual = pd.concat(all_individual, ignore_index=True) if all_individual else None

    # Create output file
    output_path = SCAD_RESULTS_DIR / "scad_combined_summary.xlsx"

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Summary sheet
        merged_summary.to_excel(writer, sheet_name='Summary', index=False)

        # Individual results sheet
        if merged_individual is not None:
            merged_individual.to_excel(writer, sheet_name='Individual_Results', index=False)

        # Create pivot tables for each metric by Step
        metrics = ['accuracy', 'precision', 'recall', 'f05_score']

        for metric in metrics:
            mean_col = f'{metric}_mean'
            std_col = f'{metric}_std'

            if mean_col in merged_summary.columns:
                # Pivot: Step × Ratio for each City
                pivot_data = []
                for _, row in merged_summary.iterrows():
                    pivot_data.append({
                        'Step': row['Step'],
                        'City': row['City'],
                        'Ratio': row['Ratio'],
                        'Mean': row.get(mean_col, np.nan),
                        'Std': row.get(std_col, np.nan)
                    })

                if pivot_data:
                    pivot_df = pd.DataFrame(pivot_data)
                    pivot_df.to_excel(writer, sheet_name=f'{metric}_details', index=False)

        # Create a nice formatted summary table
        summary_table = create_scad_summary_table(merged_summary)
        if summary_table is not None:
            summary_table.to_excel(writer, sheet_name='Formatted_Summary', index=True)

    print(f"\nSaved: {output_path}")

    # Print summary
    print_scad_summary(merged_summary)

    return merged_summary


def create_scad_summary_table(df):
    """Create a formatted summary table for SCAD results."""
    if df.empty:
        return None

    # Create pivot table: rows = (Step, City), columns = Ratio, values = accuracy_mean
    try:
        pivot = df.pivot_table(
            index=['Step', 'City'],
            columns='Ratio',
            values='accuracy_mean',
            aggfunc='first'
        )
        return pivot
    except:
        return None


def print_scad_summary(df):
    """Print SCAD results summary to console."""
    print("\n" + "=" * 80)
    print("SCAD 4-Step Results Summary")
    print("=" * 80)

    steps = df['Step'].unique() if 'Step' in df.columns else []

    for step in sorted(steps):
        step_df = df[df['Step'] == step]
        print(f"\n{step}:")
        print("-" * 70)

        for _, row in step_df.iterrows():
            city = row.get('City', 'N/A')
            ratio = row.get('Ratio', 'N/A')
            acc_mean = row.get('accuracy_mean', 0)
            acc_std = row.get('accuracy_std', 0)
            f05_mean = row.get('f05_score_mean', 0)
            f05_std = row.get('f05_score_std', 0)

            print(f"  {city} ratio={ratio}: "
                  f"Acc={acc_mean:.4f}±{acc_std:.4f}  "
                  f"F0.5={f05_mean:.4f}±{f05_std:.4f}")


def merge_other_methods_results():
    """
    Merge Other Methods (Lasso, L0, MCP) experiment results.

    Expected input files:
      - othermethods_original_city{1,2,3}_ratio{0.01,0.05,0.10}.xlsx
      - othermethods_merged_city{1,2,3}_ratio{0.01,0.05,0.10}.xlsx
    """
    print("\n" + "=" * 60)
    print("Merging Other Methods Results (Lasso, L0, MCP)")
    print("=" * 60)

    all_summary = []
    all_detailed = []

    for matrix_type in ['original', 'merged']:
        for city in CITIES:
            for ratio in RATIOS:
                filename = f"othermethods_{matrix_type}_city{city}_ratio{ratio}.xlsx"
                filepath = OTHER_RESULTS_DIR / filename

                if not filepath.exists():
                    print(f"  [MISSING] {filename}")
                    continue

                print(f"  [LOADED] {filename}")

                try:
                    # Read Statistical_Analysis sheet (summary)
                    try:
                        summary_df = pd.read_excel(filepath, sheet_name='Statistical_Analysis')
                        summary_df['City'] = f"City{city}"
                        summary_df['Ratio'] = ratio
                        summary_df['Matrix'] = matrix_type
                        all_summary.append(summary_df)
                    except:
                        pass

                    # Read Detailed_Results sheet
                    try:
                        detailed_df = pd.read_excel(filepath, sheet_name='Detailed_Results')
                        detailed_df['City'] = f"City{city}"
                        detailed_df['Ratio'] = ratio
                        detailed_df['Matrix'] = matrix_type
                        all_detailed.append(detailed_df)
                    except:
                        pass

                except Exception as e:
                    print(f"  [ERROR] {filename}: {e}")

    if not all_summary and not all_detailed:
        print("\nNo Other Methods result files found!")
        return None

    # Merge all data
    merged_summary = pd.concat(all_summary, ignore_index=True) if all_summary else pd.DataFrame()
    merged_detailed = pd.concat(all_detailed, ignore_index=True) if all_detailed else pd.DataFrame()

    # Create output file
    output_path = OTHER_RESULTS_DIR / "othermethods_combined_summary.xlsx"

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Summary sheet
        if not merged_summary.empty:
            merged_summary.to_excel(writer, sheet_name='Summary', index=False)

        # Detailed results sheet
        if not merged_detailed.empty:
            merged_detailed.to_excel(writer, sheet_name='Detailed_Results', index=False)

        # Create method comparison table
        if not merged_summary.empty:
            comparison = create_other_methods_comparison(merged_summary)
            if comparison is not None:
                comparison.to_excel(writer, sheet_name='Method_Comparison', index=True)

        # Create pivot tables for each metric
        metrics = ['Mean_Accuracy', 'Mean_F05_Score', 'Mean_Precision', 'Mean_Recall']

        for metric in metrics:
            if metric in merged_summary.columns:
                try:
                    pivot = merged_summary.pivot_table(
                        index=['Method', 'City'],
                        columns='Ratio',
                        values=metric,
                        aggfunc='first'
                    )
                    sheet_name = metric.replace('Mean_', '') + '_pivot'
                    pivot.to_excel(writer, sheet_name=sheet_name)
                except:
                    pass

    print(f"\nSaved: {output_path}")

    # Print summary
    print_other_methods_summary(merged_summary)

    return merged_summary


def create_other_methods_comparison(df):
    """Create a comparison table for Other Methods results."""
    if df.empty:
        return None

    try:
        # Pivot: rows = (Method, Matrix), columns = (City, Ratio), values = Mean_Accuracy
        pivot = df.pivot_table(
            index=['Method', 'Matrix'],
            columns=['City', 'Ratio'],
            values='Mean_Accuracy',
            aggfunc='first'
        )
        return pivot
    except:
        return None


def print_other_methods_summary(df):
    """Print Other Methods results summary to console."""
    if df.empty:
        print("\nNo summary data available.")
        return

    print("\n" + "=" * 80)
    print("Other Methods Results Summary (Lasso, L0, MCP)")
    print("=" * 80)

    for matrix_type in ['original', 'merged']:
        matrix_df = df[df['Matrix'] == matrix_type] if 'Matrix' in df.columns else df
        if matrix_df.empty:
            continue

        print(f"\n[{matrix_type.upper()} Matrix]")

        methods = matrix_df['Method'].unique() if 'Method' in matrix_df.columns else []

        for method in sorted(methods):
            method_df = matrix_df[matrix_df['Method'] == method]
            print(f"\n  {method}:")
            print("  " + "-" * 60)

            for _, row in method_df.iterrows():
                city = row.get('City', 'N/A')
                ratio = row.get('Ratio', 'N/A')
                acc = row.get('Mean_Accuracy', 0)
                acc_std = row.get('Std_Accuracy', 0)
                f05 = row.get('Mean_F05_Score', 0)
                f05_std = row.get('Std_F05_Score', 0)

                print(f"    {city} ratio={ratio}: "
                      f"Acc={acc:.4f}±{acc_std:.4f}  "
                      f"F0.5={f05:.4f}±{f05_std:.4f}")


def read_lur_metadata(xlsx_path: Path) -> Dict[str, object]:
    metadata_df = pd.read_excel(xlsx_path, sheet_name="Metadata")
    return dict(zip(metadata_df["Key"], metadata_df["Value"]))


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


def aggregate_lur_metrics(individual_df: pd.DataFrame) -> pd.DataFrame:
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


def merge_lur_shards(inputs: List[Path], save_path: Path) -> None:
    """Merge sharded Figure 5 / S.5 / S.6 LUR metric workbooks."""
    individual_frames = []
    metadata = None
    for path in inputs:
        individual_frames.append(pd.read_excel(path, sheet_name="Individual_Results"))
        if metadata is None:
            metadata = read_lur_metadata(path)

    individual_df = normalize_empty_group_confidence(pd.concat(individual_frames, ignore_index=True))
    summary_df = aggregate_lur_metrics(individual_df)

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
        pd.DataFrame({"Shard_Source": [str(path) for path in inputs]}).to_excel(
            writer,
            sheet_name="Shard_Sources",
            index=False,
        )

    print(f"Merged {len(inputs)} shards into {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Merge experiment results")
    parser.add_argument('--synthetic', action='store_true', help='Only merge synthetic results')
    parser.add_argument('--scad', action='store_true', help='Only merge SCAD results')
    parser.add_argument('--other', action='store_true', help='Only merge Other Methods results')
    parser.add_argument('--merge-shards', choices=['scad', 'other', 'synthetic'], help='Merge shard workbooks for one result type')
    parser.add_argument('--lur', action='store_true', help='Merge LUR figure shard workbooks (use with --inputs/--save)')
    parser.add_argument('--inputs', nargs='+', default=None, help='LUR shard workbooks to merge')
    parser.add_argument('--save', default=None, help='Merged LUR workbook path')
    parser.add_argument('--shard-dir', type=Path, default=None, help='Shard directory for --merge-shards')
    parser.add_argument('--output-dir', type=Path, default=None, help='Output directory for --merge-shards')

    args = parser.parse_args()

    print(f"\nMerge Results Tool")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Base directory: {BASE_DIR}")

    if args.lur:
        if not args.inputs or not args.save:
            raise ValueError("--lur requires both --inputs and --save.")
        merge_lur_shards([Path(item) for item in args.inputs], Path(args.save))
        return

    if args.merge_shards:
        if args.shard_dir is None or args.output_dir is None:
            raise ValueError("--merge-shards requires both --shard-dir and --output-dir.")
        merge_shard_directory(args.merge_shards, args.shard_dir, args.output_dir)
        print("\n" + "=" * 60)
        print("Done!")
        print("=" * 60)
        return

    # If no specific flag, merge all
    merge_all = not args.synthetic and not args.scad and not args.other

    if args.synthetic or merge_all:
        merge_synthetic_results()

    if args.scad or merge_all:
        merge_scad_results()

    if args.other or merge_all:
        merge_other_methods_results()

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
