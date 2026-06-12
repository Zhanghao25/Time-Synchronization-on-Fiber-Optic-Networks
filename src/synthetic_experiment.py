#!/usr/bin/env python3
"""
Synthetic Topology Experiment for SCAD Methods

Uses synthetic topology data with:
- BFS-ordered edges (matching Tree class)
- is_selected column for direct hard-edge identification
- 7000 leaves (A matrix ~7000 x n_edges)

Usage:
    python src/synthetic_experiment.py --zeta 0.01 --ratio 0.10 --experiments 10
"""

import numpy as np
import pandas as pd
import networkx as nx
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from sklearn.preprocessing import LabelEncoder
import warnings

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tsle_methods import (
    tsle_step1_original,
    tsle_step2_merged,
    tsle_step4_iterative
)
from tree_structure import Tree
from evaluation_metrics import Evaluator

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════
# Configuration
# ═══════════════════════════════════════
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "synthetic_topologies"
RAW_COMBINED_WORKBOOK = DATA_DIR / "synthetic_topologies_combined.xlsx"

CASES = [
    {"legacy_city": "City1", "hard_k": 100, "zeta": 0.01, "label": "zeta=1%"},
    {"legacy_city": "City2", "hard_k": 500, "zeta": 0.05, "label": "zeta=5%"},
    {"legacy_city": "City3", "hard_k": 1000, "zeta": 0.10, "label": "zeta=10%"},
]

RATIOS = [0.10]
METHODS = ["step1_original", "step2_merged", "step4_iterative"]
ZETA_MAP = {
    100: 0.01,
    500: 0.05,
    1000: 0.10,
}

LEGACY_CITY_TO_CASE = {idx: case for idx, case in enumerate(CASES, start=1)}
ZETA_TO_CASE = {case["zeta"]: case for case in CASES}


# ═══════════════════════════════════════
# Data loading (simplified: uses is_selected column)
# ═══════════════════════════════════════
def get_case_sheet_name(case: Dict, ratio: float) -> str:
    """Return the canonical sheet/file stem for a synthetic case."""
    ratio_int = int(ratio * 100)
    return f"synthetic_zeta{int(case['zeta'] * 100)}pct_alpha{ratio_int}"


def get_legacy_case_sheet_name(case: Dict, ratio: float) -> str:
    """Return the legacy workbook sheet/file stem for backwards compatibility."""
    ratio_int = int(ratio * 100)
    return f"{case['legacy_city']}_leftK{case['hard_k']}_r{ratio_int}"


def get_case_name_candidates(case: Dict, ratio: float) -> List[str]:
    """Return new and legacy case names in lookup order."""
    names = [get_case_sheet_name(case, ratio)]
    legacy_name = get_legacy_case_sheet_name(case, ratio)
    if legacy_name not in names:
        names.append(legacy_name)
    return names


def case_slug(case: Dict) -> str:
    return f"zeta{int(case['zeta'] * 100)}pct"


def resolve_case(city: Optional[int] = None, zeta: Optional[float] = None) -> Dict:
    if zeta is not None:
        for value, case in ZETA_TO_CASE.items():
            if abs(value - zeta) < 1e-12:
                return case
        raise ValueError(f"Unsupported zeta={zeta}")
    if city is not None:
        return LEGACY_CITY_TO_CASE[city]
    raise ValueError("Either city or zeta must be provided")


def build_tree_dataframe(tree: Tree, hard_edges: set) -> pd.DataFrame:
    """Materialize a Tree object back to a BFS-ordered edge DataFrame."""
    rows = []
    edge_values = nx.get_edge_attributes(tree.tree, 'value')
    for edge_id, (parent, child) in enumerate(tree.edges):
        rows.append({
            'edge_id': edge_id,
            'parent_label': parent,
            'child_label': child,
            'x_true': edge_values.get((parent, child), 0.0),
            'is_selected': 1 if (parent, child) in hard_edges else 0,
        })
    return pd.DataFrame(rows)


def reconstruct_merged_df_from_raw(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstruct the merged synthetic topology from the raw workbook.

    The committed repository currently only ships the raw combined workbook.
    The raw topology was created by expanding degree-2 right-side leaf chains,
    so running `Tree(..., merge=True)` reproduces the merged topology and edge
    coefficients needed by the synthetic experiment.
    """
    required_cols = {'parent_label', 'child_label', 'x_true', 'is_selected'}
    missing = required_cols - set(df_raw.columns)
    if missing:
        raise ValueError(f"Raw synthetic sheet missing required columns: {sorted(missing)}")

    graph = nx.Graph()
    hard_edges = set()
    for _, row in df_raw.iterrows():
        parent = row['parent_label']
        child = row['child_label']
        graph.add_edge(parent, child, value=row['x_true'])
        if int(row['is_selected']) == 1:
            hard_edges.add((parent, child))

    source_candidates = set(df_raw['parent_label']) - set(df_raw['child_label'])
    if not source_candidates:
        raise ValueError("Unable to infer synthetic source node from raw workbook")

    source = min(source_candidates)
    merged_tree = Tree(graph, source, merge=True)
    return build_tree_dataframe(merged_tree, hard_edges)


def load_case_data(case: Dict, ratio: float, matrix_type: str,
                   combined_workbook: Optional[Path] = None) -> Tuple[pd.DataFrame, Dict]:
    """Load a synthetic case from the canonical raw combined workbook."""
    workbook_path = Path(combined_workbook) if combined_workbook else RAW_COMBINED_WORKBOOK
    if not workbook_path.exists():
        raise FileNotFoundError(f"Data file not found for case {case_slug(case)}")

    matched_sheet = None
    with pd.ExcelFile(workbook_path) as workbook:
        for name in get_case_name_candidates(case, ratio):
            if name in workbook.sheet_names:
                matched_sheet = name
                break
    if matched_sheet is None:
        raise FileNotFoundError(
            f"No matching sheet found for {case_slug(case)} alpha={ratio:.2f} in {workbook_path}"
        )

    raw_df = pd.read_excel(workbook_path, sheet_name=matched_sheet)
    if matrix_type == "merged":
        df = reconstruct_merged_df_from_raw(raw_df)
        source_kind = "reconstructed_from_raw"
    else:
        df = raw_df
        source_kind = "combined_workbook"

    assert 'is_selected' in df.columns, "Missing is_selected column in the topology workbook."

    meta = {
        "filepath": str(workbook_path),
        "source_kind": source_kind or "unknown",
        "n_edges": len(df),
        "n_hard": int(df['is_selected'].sum()),
        "matched_name": matched_sheet or "",
    }
    return df, meta


def build_tree_from_df(df: pd.DataFrame, merge: bool) -> Tuple[Tree, np.ndarray, 'LabelEncoder']:
    """Build Tree from DataFrame."""
    le = LabelEncoder()
    all_nodes = pd.concat([df['parent_label'], df['child_label']])
    le.fit(all_nodes)

    df_enc = df.copy()
    df_enc['source'] = le.transform(df['parent_label'])
    df_enc['target'] = le.transform(df['child_label'])

    G = nx.Graph()
    for _, row in df_enc.iterrows():
        G.add_edge(row['source'], row['target'], value=row['x_true'])

    source_candidates = set(df_enc['source']) - set(df_enc['target'])
    source = min(source_candidates)

    tree = Tree(G, source, merge=merge)
    A, x_excel = tree.get_Ax()

    return tree, A, le


# ═══════════════════════════════════════
# Hybrid x_true generation (simplified)
# ═══════════════════════════════════════
def generate_hybrid_x_true(A: np.ndarray, df: pd.DataFrame,
                           target_ratio: float, seed: int,
                           min_val: float = 5000.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]:
    """
    Generate x_true: hard edges from excel, non-hard edges randomized.

    Since BFS order matches, df['x_true'] and df['is_selected'] align with Tree edges.
    """
    np.random.seed(seed)
    n_edges = len(df)

    is_selected = df['is_selected'].values.astype(bool)
    x_excel = df['x_true'].values

    x_clean = np.zeros(n_edges)

    # Hard edges: use excel values
    x_clean[is_selected] = x_excel[is_selected]

    # Count hard nonzero (BACKBONE hard edges have nonzero values)
    n_hard_nonzero = int(np.sum((is_selected) & (np.abs(x_excel) > 1e-12)))

    # Non-hard edges: randomly select and generate
    nonhard_idx = np.where(~is_selected)[0]
    target_total_nz = int(n_edges * target_ratio)
    needed_nz = max(0, target_total_nz - n_hard_nonzero)

    if needed_nz > 0 and len(nonhard_idx) > 0:
        n_select = min(needed_nz, len(nonhard_idx))
        selected = np.random.choice(nonhard_idx, size=n_select, replace=False)

        signal_strength = 2000000.0 / 3.0
        vals = np.random.normal(0, signal_strength, size=n_select)

        # Ensure minimum magnitude
        insuf = np.abs(vals) < min_val
        while insuf.any():
            vals[insuf] = np.random.normal(0, signal_strength, size=insuf.sum())
            insuf = np.abs(vals) < min_val

        x_clean[selected] = vals

    # Nonzero indices
    nz_idx = np.where(np.abs(x_clean) > 1e-12)[0].tolist()

    # Add noise
    coef_noise = np.random.normal(0, 20.0/3.0, n_edges)
    x_noisy = x_clean + coef_noise

    obs_noise = np.random.normal(0, 100.0/3.0, A.shape[0])
    b = A @ x_noisy + obs_noise

    return x_noisy, x_clean, b, nz_idx


# ═══════════════════════════════════════
# Single experiment
# ═══════════════════════════════════════
def run_single_experiment(case: Dict, ratio: float, seed: int,
                          tree_raw: Tree, A_raw: np.ndarray, df_raw: pd.DataFrame,
                          tree_merged: Tree, A_merged: np.ndarray, df_merged: pd.DataFrame
                          ) -> Dict[str, Any]:
    """Run one experiment."""
    results = {}

    # Generate data
    x_noisy_raw, _, b_raw, nz_raw = generate_hybrid_x_true(A_raw, df_raw, ratio, seed)
    x_noisy_merged, _, b_merged, nz_merged = generate_hybrid_x_true(A_merged, df_merged, ratio, seed)

    eval_raw = Evaluator(x_noisy_raw, nz_raw)
    eval_merged = Evaluator(x_noisy_merged, nz_merged)

    # Step 1
    try:
        r1 = tsle_step1_original(A_raw, b_raw)
        if 'x_hat' in r1:
            results['step1_original'] = eval_raw.evaluate_all_metrics(r1['x_hat'], r1['nonzero_indices'])
        else:
            results['step1_original'] = {'error': 'No x_hat'}
    except Exception as e:
        results['step1_original'] = {'error': str(e)}

    # Step 2
    try:
        r2 = tsle_step2_merged(tree_merged, b_merged)
        if 'x_hat' in r2:
            results['step2_merged'] = eval_merged.evaluate_all_metrics(r2['x_hat'], r2['nonzero_indices'])
        else:
            results['step2_merged'] = {'error': 'No x_hat'}
    except Exception as e:
        results['step2_merged'] = {'error': str(e)}

    # Step 4
    try:
        r4 = tsle_step4_iterative(A_merged, b_merged, tree_merged, x_noisy_merged, add_high=True)
        if 'x_hat' in r4:
            results['step4_iterative'] = eval_merged.evaluate_all_metrics(r4['x_hat'], r4['nonzero_indices'])
        else:
            results['step4_iterative'] = {'error': 'No x_hat'}
    except Exception as e:
        results['step4_iterative'] = {'error': str(e)}

    return results


# ═══════════════════════════════════════
# Main loop
# ═══════════════════════════════════════
def run_all_experiments(n_experiments: int = 10,
                        combined_workbook: Optional[Path] = None,
                        base_seed: int = 42,
                        paper_tag: str = "",
                        seed_list: Optional[List[int]] = None) -> pd.DataFrame:
    """Run all experiments."""
    if seed_list is not None:
        target_seeds = seed_list
        n_experiments = len(target_seeds)
    else:
        target_seeds = [base_seed + i for i in range(n_experiments)]

    all_results = []
    total = len(CASES) * len(RATIOS)
    idx = 0

    for case in CASES:
        for ratio in RATIOS:
            idx += 1
            tag = get_case_sheet_name(case, ratio)

            print(f"\n{'='*60}")
            print(f"[{idx}/{total}] {tag}")
            print(f"{'='*60}")

            # Load data
            df_raw, meta_raw = load_case_data(case, ratio, "raw", combined_workbook=combined_workbook)
            df_merged, meta_merged = load_case_data(case, ratio, "merged", combined_workbook=combined_workbook)

            tree_raw, A_raw, _ = build_tree_from_df(df_raw, merge=False)
            tree_merged, A_merged, _ = build_tree_from_df(df_merged, merge=True)

            print(f"  Raw:    {len(df_raw)} edges, A={A_raw.shape}, hard={meta_raw['n_hard']}")
            print(f"  Merged: {len(df_merged)} edges, A={A_merged.shape}, hard={meta_merged['n_hard']}")

            case_results = {m: [] for m in METHODS}

            for exp_i, seed in enumerate(target_seeds):
                print(f"\n  Exp {exp_i+1}/{n_experiments} (seed={seed})")

                try:
                    result = run_single_experiment(
                        case, ratio, seed,
                        tree_raw, A_raw, df_raw,
                        tree_merged, A_merged, df_merged
                    )
                    for m in METHODS:
                        if m in result and 'error' not in result[m]:
                            perf = result[m]
                            case_results[m].append(perf)
                            print(f"    {m}: acc={perf['accuracy']:.4f} prec={perf['precision']:.4f} "
                                  f"rec={perf['recall']:.4f} F0.5={perf['f05_score']:.4f}")
                        else:
                            print(f"    {m}: FAILED")
                except Exception as e:
                    print(f"    Error: {e}")
                    import traceback; traceback.print_exc()

            # Aggregate
            for m in METHODS:
                runs = case_results[m]
                if not runs:
                    continue
                row = {
                    'Method': m,
                    'City': case['label'],
                    'Scenario': case['label'],
                    'Legacy_Case': case['legacy_city'],
                    'Hard_K': case['hard_k'],
                    'Zeta': case['zeta'],
                    'Ratio': ratio,
                    'N_Experiments': len(runs),
                    'Base_Seed': base_seed,
                    'Paper_Tag': paper_tag,
                }
                for metric in ['accuracy', 'precision', 'recall', 'f05_score']:
                    vals = [r[metric] for r in runs if metric in r]
                    row[f'{metric}_mean'] = np.mean(vals) if vals else np.nan
                    row[f'{metric}_std'] = np.std(vals) if vals else np.nan
                all_results.append(row)

    return pd.DataFrame(all_results)


def save_results(df: pd.DataFrame, path: Path, metadata: Optional[Dict[str, Any]] = None):
    """Save results to Excel."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(path, engine='openpyxl') as w:
        df.to_excel(w, sheet_name='Summary', index=False)
        if metadata:
            metadata_rows = [{'Key': key, 'Value': value} for key, value in metadata.items()]
            pd.DataFrame(metadata_rows).to_excel(w, sheet_name='Metadata', index=False)
        if df.empty:
            print(f"\nSaved empty result set: {path}")
            return
        for m in ['accuracy', 'precision', 'recall', 'f05_score']:
            pivot = df.pivot_table(
                index=['Method', 'Ratio'],
                columns='City',
                values=[f'{m}_mean', f'{m}_std'],
                aggfunc='first'
            )
            pivot.to_excel(w, sheet_name=f'{m}_pivot')

    print(f"\nSaved: {path}")


def run_single_case(case: Dict, ratio: float, n_experiments: int,
                    combined_workbook: Optional[Path] = None,
                    base_seed: int = 42,
                    paper_tag: str = "",
                    seed_list: Optional[List[int]] = None) -> pd.DataFrame:
    """Run experiments for a single (zeta, alpha) configuration."""
    if seed_list is not None:
        target_seeds = seed_list
        n_experiments = len(target_seeds)
    else:
        target_seeds = [base_seed + i for i in range(n_experiments)]

    tag = get_case_sheet_name(case, ratio)

    print(f"\n{'='*60}")
    print(f"{tag} - {n_experiments} experiments")
    print(f"{'='*60}")

    # Load data
    df_raw, meta_raw = load_case_data(case, ratio, "raw", combined_workbook=combined_workbook)
    df_merged, meta_merged = load_case_data(case, ratio, "merged", combined_workbook=combined_workbook)

    tree_raw, A_raw, _ = build_tree_from_df(df_raw, merge=False)
    tree_merged, A_merged, _ = build_tree_from_df(df_merged, merge=True)

    print(f"  Raw:    {len(df_raw)} edges, A={A_raw.shape}, hard={meta_raw['n_hard']}")
    print(f"  Merged: {len(df_merged)} edges, A={A_merged.shape}, hard={meta_merged['n_hard']}")

    case_results = {m: [] for m in METHODS}

    for exp_i, seed in enumerate(target_seeds):
        print(f"\n  Exp {exp_i+1}/{n_experiments} (seed={seed})")

        try:
            result = run_single_experiment(
                case, ratio, seed,
                tree_raw, A_raw, df_raw,
                tree_merged, A_merged, df_merged
            )
            for m in METHODS:
                if m in result and 'error' not in result[m]:
                    perf = result[m]
                    case_results[m].append(perf)
                    print(f"    {m}: acc={perf['accuracy']:.4f} prec={perf['precision']:.4f} "
                          f"rec={perf['recall']:.4f} F0.5={perf['f05_score']:.4f}")
                else:
                    print(f"    {m}: FAILED")
        except Exception as e:
            print(f"    Error: {e}")
            import traceback; traceback.print_exc()

    # Aggregate
    all_results = []
    for m in METHODS:
        runs = case_results[m]
        if not runs:
            continue
        row = {
            'Method': m,
            'City': case['label'],
            'Scenario': case['label'],
            'Legacy_Case': case['legacy_city'],
            'Hard_K': case['hard_k'],
            'Zeta': case['zeta'],
            'Ratio': ratio,
            'N_Experiments': len(runs),
            'Base_Seed': base_seed,
            'Paper_Tag': paper_tag,
        }
        for metric in ['accuracy', 'precision', 'recall', 'f05_score']:
            vals = [r[metric] for r in runs if metric in r]
            row[f'{metric}_mean'] = np.mean(vals) if vals else np.nan
            row[f'{metric}_std'] = np.std(vals) if vals else np.nan
        all_results.append(row)

    return pd.DataFrame(all_results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiments', type=int, default=10)
    parser.add_argument('--city', type=int, choices=[1, 2, 3],
                        help='Legacy case index kept for backward compatibility: 1->zeta 1%%, 2->zeta 5%%, 3->zeta 10%%')
    parser.add_argument('--zeta', type=float, choices=[0.01, 0.05, 0.10],
                        help='Synthetic hard-region proportion zeta (recommended)')
    parser.add_argument('--ratio', type=float, choices=[0.10], default=0.10,
                        help='Global asymmetry ratio alpha. Table S.7 uses only alpha=0.10.')
    parser.add_argument('--save', type=str, default=None)
    parser.add_argument('--data-workbook', type=str, default=str(RAW_COMBINED_WORKBOOK),
                        help='Synthetic topology workbook path (default: combined raw workbook)')
    parser.add_argument('--base-seed', type=int, default=42,
                        help='Base random seed for reproducible experiment batches (default: 42)')
    parser.add_argument('--paper-tag', type=str, default='',
                        help='Optional paper run tag stored in output metadata')
    parser.add_argument('--seed-values', type=str, default=None,
                        help='Comma-separated explicit seed list (overrides --base-seed and --experiments)')
    args = parser.parse_args()

    # Parse seed list
    seed_list = None
    if args.seed_values:
        seed_list = [int(s.strip()) for s in args.seed_values.split(',') if s.strip()]

    print("="*60)
    print("Table S.7 Synthetic SCAD Experiment")
    print("="*60)
    print(f"Data: {DATA_DIR}")
    if seed_list:
        print(f"Seeds: {seed_list[0]}..{seed_list[-1]} ({len(seed_list)} total)")
    else:
        print(f"Experiments: {args.experiments}")
        print(f"Base seed: {args.base_seed}")
    if args.paper_tag:
        print(f"Paper tag: {args.paper_tag}")
    if args.city is not None and args.zeta is not None:
        legacy_case = resolve_case(city=args.city)
        if abs(legacy_case["zeta"] - args.zeta) > 1e-12:
            raise ValueError(f"--city {args.city} and --zeta {args.zeta} refer to different synthetic cases")

    start = time.time()

    # Single case mode or all cases mode
    if args.city is not None or args.zeta is not None:
        case = resolve_case(city=args.city, zeta=args.zeta)
        print(f"Mode: Single case ({case['label']}, alpha={args.ratio:.2f})")
        print("="*60)
        results_df = run_single_case(
            case,
            args.ratio,
            args.experiments,
            combined_workbook=Path(args.data_workbook),
            base_seed=args.base_seed,
            paper_tag=args.paper_tag,
            seed_list=seed_list,
        )
        save_name = args.save or f"results/raw/synthetic/synthetic_{case_slug(case)}_alpha{int(args.ratio * 100)}.xlsx"
    else:
        print("Mode: All 3 zeta cases (alpha=0.10)")
        print("="*60)
        results_df = run_all_experiments(
            args.experiments,
            combined_workbook=Path(args.data_workbook),
            base_seed=args.base_seed,
            paper_tag=args.paper_tag,
            seed_list=seed_list,
        )
        save_name = args.save or 'results/raw/synthetic/table_s7_combined.xlsx'

    print("\n" + "="*60)
    print("Results Summary")
    print("="*60)
    print(results_df.to_string())

    save_path = Path(__file__).resolve().parents[1] / save_name
    save_results(results_df, save_path, metadata={
        'data_workbook': args.data_workbook,
        'experiments': args.experiments,
        'base_seed': args.base_seed,
        'paper_tag': args.paper_tag,
    })

    print(f"\nTotal: {time.time()-start:.1f}s")


if __name__ == "__main__":
    main()
