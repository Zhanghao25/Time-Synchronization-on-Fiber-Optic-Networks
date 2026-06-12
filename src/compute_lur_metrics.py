#!/usr/bin/env python3
"""
Compute Figure 5 / Figure S.5 / Figure S.6 metrics from reproducible SCAD step-2 runs.

Official LUR definition:
- LUR nodes = SHCN union HCN
- LUR edges = all edges incident to those nodes

Panel metrics:
- (a) P_err_LUR = proportion of incorrect predicted edges that fall in LUR
- (b) R_asy_LUR / R_asy_nonLUR = proportion of predicted asymmetric edges
      within the LUR / non-LUR partitions, normalized by partition size
- (c) SHCE / HCE / UHCE counts
- (d) SHCE / HCE / UHCE confidence
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tsle_methods import tsle_step2_merged
from tree_structure import Tree
from utils import generate_synthetic_data, load_network_data
from evaluation_metrics import Evaluator


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "results" / "raw" / "lur_figures"
DEFAULT_ALPHAS = [0.01, 0.05, 0.10, 0.15, 0.20, 0.30]


def parse_alphas(value: str) -> List[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_seed_values(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def load_seed_sequence(base_seed: int, experiments: int, seed_values: str = "", seed_file: str = "") -> List[int]:
    if seed_values and seed_file:
        raise ValueError("Use either --seed-values or --seed-file, not both.")
    if seed_values:
        seeds = parse_seed_values(seed_values)
    elif seed_file:
        text = Path(seed_file).read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"Seed file is empty: {seed_file}")
        seeds = parse_seed_values(text.replace("\n", ","))
    else:
        seeds = [base_seed + index for index in range(experiments)]

    if len(seeds) != experiments:
        raise ValueError(
            f"Expected {experiments} seeds but got {len(seeds)}. "
            "Ensure --experiments matches the supplied seed list."
        )
    return seeds


def normalize_alpha_key(alpha: float) -> str:
    return f"{alpha:.2f}"


def load_seed_plan(
    *,
    alphas: Sequence[float],
    base_seed: int,
    experiments: int,
    seed_values: str = "",
    seed_file: str = "",
    seed_manifest: str = "",
) -> Dict[float, List[int]]:
    if seed_manifest:
        if seed_values or seed_file:
            raise ValueError("Use either --seed-manifest or --seed-values/--seed-file, not both.")
        manifest_path = Path(seed_manifest)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        alpha_mapping = manifest.get("alphas", manifest)
        plan: Dict[float, List[int]] = {}
        missing: List[str] = []
        for alpha in alphas:
            key = normalize_alpha_key(alpha)
            values = alpha_mapping.get(key)
            if values is None:
                missing.append(key)
                continue
            plan[alpha] = [int(value) for value in values]
        if missing:
            raise ValueError(
                f"Seed manifest {manifest_path} is missing alpha keys: {', '.join(missing)}"
            )
        return plan

    seeds = load_seed_sequence(
        base_seed=base_seed,
        experiments=experiments,
        seed_values=seed_values,
        seed_file=seed_file,
    )
    return {alpha: list(seeds) for alpha in alphas}


def build_city_label(dataset_path: Path, provided: str = "") -> str:
    if provided:
        return provided
    stem = dataset_path.stem.lower()
    if stem.startswith("city"):
        suffix = stem.replace("city", "")
        if suffix.isdigit():
            return f"City {int(suffix)}"
    return dataset_path.stem


def union_incident_edges(tree: Tree, nodes: Iterable[int]) -> Set[int]:
    result: Set[int] = set()
    for node in nodes:
        result.update(tree.find_edges_for_node(node))
    return result


def ratio_or_default(numerator: int, denominator: int, default: float = 0.0) -> float:
    if denominator <= 0:
        return default
    return numerator / denominator


def confidence_or_nan(value: float, edge_count: int) -> float:
    if edge_count <= 0:
        return float("nan")
    return float(value)


def compute_lur_metrics(
    *,
    total_edges: int,
    predicted_nonzero: Set[int],
    error_locations: Set[int],
    lur_edges: Set[int],
) -> Dict[str, float]:
    pred_in_lur = len(predicted_nonzero & lur_edges)
    pred_out_lur = len(predicted_nonzero - lur_edges)
    err_in_lur = len(error_locations & lur_edges)
    lur_region_count = len(lur_edges)
    nonlur_region_count = total_edges - lur_region_count

    return {
        "P_err_LUR": 1.0 if len(error_locations) == 0 else err_in_lur / len(error_locations),
        "R_asy_LUR": ratio_or_default(pred_in_lur, lur_region_count),
        "R_asy_nonLUR": ratio_or_default(pred_out_lur, nonlur_region_count),
        "LUR_region_prop": ratio_or_default(lur_region_count, total_edges),
        "nonLUR_region_prop": ratio_or_default(nonlur_region_count, total_edges),
        "pred_edges_in_LUR": pred_in_lur,
        "pred_edges_out_LUR": pred_out_lur,
        "LUR_edge_count": lur_region_count,
        "nonLUR_edge_count": nonlur_region_count,
    }


def compute_single_experiment_metrics(
    *,
    tree: Tree,
    design_matrix: np.ndarray,
    alpha: float,
    seed: int,
) -> Dict[str, Any]:
    x_true, x_raw, observations, true_nonzero = generate_synthetic_data(design_matrix, alpha, seed)

    step2_result = tsle_step2_merged(tree, observations)
    predicted_nonzero = step2_result["nonzero_indices"]

    evaluator = Evaluator(x_true, true_nonzero)
    performance = evaluator.evaluate_all_metrics(step2_result["x_hat"], predicted_nonzero)
    confidence_results = tree.find_subhigh_nodes(
        list(range(design_matrix.shape[1])),
        predicted_nonzero,
        true_nonzero,
    )

    predicted_set = set(predicted_nonzero)
    error_set = set(performance["error_locations"])

    lur_nodes = set(confidence_results["subhigh_nodes"]) | set(confidence_results["high_nodes"])
    lur_edges = union_incident_edges(tree, lur_nodes)

    lur_metrics = compute_lur_metrics(
        total_edges=design_matrix.shape[1],
        predicted_nonzero=predicted_set,
        error_locations=error_set,
        lur_edges=lur_edges,
    )

    row: Dict[str, Any] = {
        "alpha": alpha,
        "seed": seed,
        "total_edges": design_matrix.shape[1],
        "true_nonzero_count": len(true_nonzero),
        "predicted_nonzero_count": len(predicted_nonzero),
        "error_count": performance["n_errors"],
        "accuracy": performance["accuracy"],
        "precision": performance["precision"],
        "recall": performance["recall"],
        "f05_score": performance["f05_score"],
        "SHCE_Edges_Count": confidence_results["group_counts"]["subhigh_edges_count"],
        "HCE_Edges_Count": confidence_results["group_counts"]["high_edges_count"],
        "UHCE_Edges_Count": confidence_results["group_counts"]["ultrahigh_edges_count"],
        "SHCE_Confidence": confidence_or_nan(
            confidence_results["confidence_stats"]["subhigh_confidence"],
            confidence_results["group_counts"]["subhigh_edges_count"],
        ),
        "HCE_Confidence": confidence_or_nan(
            confidence_results["confidence_stats"]["high_confidence"],
            confidence_results["group_counts"]["high_edges_count"],
        ),
        "UHCE_Confidence": confidence_or_nan(
            confidence_results["confidence_stats"]["ultrahigh_confidence"],
            confidence_results["group_counts"]["ultrahigh_edges_count"],
        ),
        "SHCN_Count": confidence_results["group_counts"]["subhigh_nodes_count"],
        "HCN_Count": confidence_results["group_counts"]["high_nodes_count"],
        "UHCN_Count": confidence_results["group_counts"]["ultrahigh_nodes_count"],
        "LUR_Node_Count": len(lur_nodes),
        "predicted_nonzero_indices": json.dumps(predicted_nonzero),
        "true_nonzero_indices": json.dumps(true_nonzero),
        "error_locations": json.dumps(performance["error_locations"]),
    }

    for key, value in lur_metrics.items():
        row[key] = value

    return row


def aggregate_metrics(individual_df: pd.DataFrame) -> pd.DataFrame:
    if individual_df.empty:
        return pd.DataFrame()

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


def save_results(
    *,
    save_path: Path,
    metadata: Dict[str, Any],
    seed_plan_df: pd.DataFrame,
    seed_status_df: pd.DataFrame,
    individual_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(suffix=".xlsx", dir=save_path.parent)
    Path(tmp_name).unlink(missing_ok=True)
    temp_path = Path(tmp_name)
    with pd.ExcelWriter(temp_path) as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        individual_df.to_excel(writer, sheet_name="Individual_Results", index=False)
        seed_plan_df.to_excel(writer, sheet_name="Seed_Plan", index=False)
        seed_status_df.to_excel(writer, sheet_name="Seed_Status", index=False)
        pd.DataFrame([{"Key": key, "Value": value} for key, value in metadata.items()]).to_excel(
            writer,
            sheet_name="Metadata",
            index=False,
        )
    temp_path.replace(save_path)


def load_checkpoint_results(
    *,
    save_path: Path,
    metadata: Dict[str, Any],
    resume: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not resume or not save_path.exists():
        return pd.DataFrame(), pd.DataFrame()
    try:
        metadata_df = pd.read_excel(save_path, sheet_name="Metadata")
        existing = dict(zip(metadata_df["Key"], metadata_df["Value"]))
    except Exception:
        return pd.DataFrame(), pd.DataFrame()

    keys_to_match = ["dataset", "city_label", "alphas", "base_seed"]
    for key in keys_to_match:
        if key in existing and str(existing.get(key)) != str(metadata.get(key)):
            return pd.DataFrame(), pd.DataFrame()
    if metadata.get("paper_tag") and str(existing.get("paper_tag", "")) != str(metadata.get("paper_tag")):
        return pd.DataFrame(), pd.DataFrame()

    try:
        individual_df = pd.read_excel(save_path, sheet_name="Individual_Results")
    except Exception:
        individual_df = pd.DataFrame()
    try:
        seed_status_df = pd.read_excel(save_path, sheet_name="Seed_Status")
    except Exception:
        seed_status_df = pd.DataFrame()
    return individual_df, seed_status_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute LUR figure metrics from reproducible SCAD merged runs.")
    parser.add_argument("--data", type=str, required=True, help="Path to city dataset Excel file.")
    parser.add_argument("--city-label", type=str, default="", help="Human-readable city label.")
    parser.add_argument("--alphas", type=str, default="0.01,0.05,0.10,0.15,0.20,0.30")
    parser.add_argument("--experiments", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--seed-values", type=str, default="", help="Optional comma-separated explicit seed list.")
    parser.add_argument("--seed-file", type=str, default="", help="Optional text file containing comma/newline separated seeds.")
    parser.add_argument("--seed-manifest", type=str, default="", help="Optional JSON file mapping alpha -> ordered seed list.")
    parser.add_argument("--paper-tag", type=str, default="")
    parser.add_argument("--save", type=str, required=True, help="Output .xlsx path.")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                        help="Resume from an existing matching workbook if present (default: True)")
    parser.add_argument("--checkpoint-interval", type=int, default=1,
                        help="Save checkpoint every N completed alpha-seed runs (default: 1)")
    args = parser.parse_args()

    dataset_path = Path(args.data)
    save_path = Path(args.save)
    alphas = parse_alphas(args.alphas)
    city_label = build_city_label(dataset_path, args.city_label)
    seed_plan = load_seed_plan(
        alphas=alphas,
        base_seed=args.base_seed,
        experiments=args.experiments,
        seed_values=args.seed_values,
        seed_file=args.seed_file,
        seed_manifest=args.seed_manifest,
    )

    graph, source = load_network_data(dataset_path)
    tree = Tree(graph, source, merge=True)
    design_matrix, _ = tree.get_Ax()

    experiments_meta = ",".join(str(len(seed_plan[alpha])) for alpha in alphas)
    seed_mode = "manifest" if args.seed_manifest else ("explicit" if (args.seed_values or args.seed_file) else "contiguous")
    metadata = {
        "dataset": str(dataset_path),
        "city_label": city_label,
        "alphas": ",".join(f"{alpha:.2f}" for alpha in alphas),
        "experiments": experiments_meta,
        "base_seed": args.base_seed,
        "seed_mode": seed_mode,
        "seed_values": "" if args.seed_manifest else ",".join(str(seed) for seed in seed_plan[alphas[0]]),
        "seed_manifest": args.seed_manifest,
        "paper_tag": args.paper_tag,
        "total_edges": design_matrix.shape[1],
    }

    individual_df, seed_status_df = load_checkpoint_results(
        save_path=save_path,
        metadata=metadata,
        resume=args.resume,
    )
    completed_pairs: Set[Tuple[float, int]] = set()
    if not seed_status_df.empty and {'alpha', 'seed'}.issubset(seed_status_df.columns):
        completed_pairs = {
            (float(alpha), int(seed))
            for alpha, seed in zip(seed_status_df['alpha'], seed_status_df['seed'])
            if pd.notna(alpha) and pd.notna(seed)
        }
    elif not individual_df.empty and {'alpha', 'seed'}.issubset(individual_df.columns):
        completed_pairs = {
            (float(alpha), int(seed))
            for alpha, seed in zip(individual_df['alpha'], individual_df['seed'])
            if pd.notna(alpha) and pd.notna(seed)
        }

    if completed_pairs:
        print(f"Resuming from checkpoint with {len(completed_pairs)} completed alpha-seed runs")

    seed_plan_rows: List[Dict[str, Any]] = []
    for alpha in alphas:
        alpha_seeds = seed_plan[alpha]
        for index, seed in enumerate(alpha_seeds):
            seed_plan_rows.append(
                {
                    "alpha": alpha,
                    "alpha_label": f"{alpha:.2f}",
                    "run_index": index + 1,
                    "seed": seed,
                }
            )
            if (alpha, seed) in completed_pairs:
                print(f"[{city_label}] alpha={alpha:.2f} experiment={index + 1}/{len(alpha_seeds)} seed={seed} - skipped (checkpoint)")
                continue
            print(f"[{city_label}] alpha={alpha:.2f} experiment={index + 1}/{len(alpha_seeds)} seed={seed}")
            row = compute_single_experiment_metrics(
                tree=tree,
                design_matrix=design_matrix,
                alpha=alpha,
                seed=seed,
            )
            row["City"] = city_label
            row["Dataset"] = str(dataset_path)
            row["Paper_Tag"] = args.paper_tag
            individual_df = pd.concat([individual_df, pd.DataFrame([row])], ignore_index=True)
            seed_status_df = pd.concat(
                [
                    seed_status_df,
                    pd.DataFrame(
                        [{
                            "alpha": alpha,
                            "alpha_label": f"{alpha:.2f}",
                            "run_index": index + 1,
                            "seed": seed,
                            "completed": True,
                        }]
                    ),
                ],
                ignore_index=True,
            )
            completed_pairs.add((alpha, seed))
            if args.checkpoint_interval > 0 and len(completed_pairs) % args.checkpoint_interval == 0:
                seed_plan_df = pd.DataFrame(seed_plan_rows)
                summary_df = aggregate_metrics(individual_df)
                save_results(
                    save_path=save_path,
                    metadata=metadata,
                    seed_plan_df=seed_plan_df,
                    seed_status_df=seed_status_df,
                    individual_df=individual_df,
                    summary_df=summary_df,
                )

    summary_df = aggregate_metrics(individual_df)
    seed_plan_df = pd.DataFrame(seed_plan_rows)
    save_results(
        save_path=save_path,
        metadata=metadata,
        seed_plan_df=seed_plan_df,
        seed_status_df=seed_status_df,
        individual_df=individual_df,
        summary_df=summary_df,
    )
    print(f"Saved LUR figure metrics to {save_path}")


if __name__ == "__main__":
    main()
