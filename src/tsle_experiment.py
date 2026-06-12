#!/usr/bin/env python3
"""
SCAD Main Experiment

Simple 4-step SCAD algorithm implementation for real data.
- Step 1: Uses original matrix data (merge=False)
- Steps 2-4: Use merged matrix data (merge=True)

Usage:
    python src/tsle_experiment.py --data data/city1.xlsx --experiments 10 --save results/scad_results.xlsx
"""

import sys
import numpy as np
import time
import argparse
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List, Set, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Import existing functions
from tsle_methods import (
    tsle_step1_original,
    tsle_step2_merged, 
    tsle_step3_random,
    tsle_step4_iterative
)

from utils import load_network_data, generate_synthetic_data
from evaluation_metrics import Evaluator
from tree_structure import Tree

import pandas as pd


def run_four_step_scad(A_original: np.ndarray, A_merged: np.ndarray, 
                      b_original: np.ndarray, b_merged: np.ndarray,
                      tree_original: Tree, tree_merged: Tree,
                      x_true_original: np.ndarray, x_true_merged: np.ndarray,
                      true_nonzero_original: list, true_nonzero_merged: list) -> Dict[str, Any]:
    """
    Run 4-step SCAD algorithm with correct data for each step.
    
    Args:
        A_original: Design matrix from tree without merging (merge=False) 
        A_merged: Design matrix from tree with merging (merge=True)
        b_original: Observation vector for original matrix
        b_merged: Observation vector for merged matrix  
        tree_original: Tree object with merge=False
        tree_merged: Tree object with merge=True
        x_true_original: True coefficient vector for original matrix
        x_true_merged: True coefficient vector for merged matrix
        true_nonzero_original: True nonzero indices for original matrix
        true_nonzero_merged: True nonzero indices for merged matrix
    
    Returns:
        Dict with step1, step2, step3, step4 results
    """
    results = {}
    
    # Step 1: Original matrix (merge=False) with original data
    print("Running Step 1: SCAD on original matrix (merge=False)...")
    results['step1'] = tsle_step1_original(A_original, b_original)
    results['step1']['step'] = 1
    
    # Evaluate step1 with original true values
    evaluator_original = Evaluator(x_true_original, true_nonzero_original)
    if 'x_hat' in results['step1']:
        performance_step1 = evaluator_original.evaluate_all_metrics(
            results['step1']['x_hat'], results['step1']['nonzero_indices']
        )
        results['step1']['performance'] = performance_step1
        
        # Check if perfect accuracy achieved
        if performance_step1['accuracy'] == 1.0:
            print(f"Step 1 achieved perfect accuracy ({performance_step1['accuracy']:.4f}), skipping steps 2-4.")
            results['step2'] = {'skipped': True, 'reason': 'Perfect accuracy achieved in step 1'}
            results['step3'] = {'skipped': True, 'reason': 'Perfect accuracy achieved in step 1'}
            results['step4'] = {'skipped': True, 'reason': 'Perfect accuracy achieved in step 1'}
            return results
    
    # Step 2: Merged matrix (merge=True) with merged data
    print("Running Step 2: SCAD on merged matrix (merge=True)...")
    results['step2'] = tsle_step2_merged(tree_merged, b_merged)
    results['step2']['step'] = 2
    
    # Evaluate step2 with merged true values
    evaluator_merged = Evaluator(x_true_merged, true_nonzero_merged)
    if 'x_hat' in results['step2']:
        performance_step2 = evaluator_merged.evaluate_all_metrics(
            results['step2']['x_hat'], results['step2']['nonzero_indices']
        )
        results['step2']['performance'] = performance_step2
        
        # Check if perfect accuracy achieved in step2
        if performance_step2['accuracy'] == 1.0:
            print(f"Step 2 achieved perfect accuracy ({performance_step2['accuracy']:.4f}), skipping steps 3 and 4.")
            results['step3'] = {'skipped': True, 'reason': 'Perfect accuracy achieved in step 2'}
            results['step4'] = {'skipped': True, 'reason': 'Perfect accuracy achieved in step 2'}
            return results
    
    # Step 3: Random augmentation with merged data (step2 accuracy < 1.0)
    print("Running Step 3: SCAD with random augmentation...")
    results['step3'] = tsle_step3_random(A_merged, b_merged, tree_merged, x_true_merged)
    results['step3']['step'] = 3
    
    # Evaluate step3 with merged true values
    if 'x_hat' in results['step3']:
        performance_step3 = evaluator_merged.evaluate_all_metrics(
            results['step3']['x_hat'], results['step3']['nonzero_indices']
        )
        results['step3']['performance'] = performance_step3
    
    # Step 4: Iterative augmentation with merged data 
    # (step2 accuracy < 1.0, already satisfied since not returned above)
    print("Running Step 4: SCAD with iterative augmentation...")
    results['step4'] = tsle_step4_iterative(A_merged, b_merged, tree_merged, x_true_merged, add_high=True)
    results['step4']['step'] = 4
    
    # Evaluate step4 with merged true values
    if 'x_hat' in results['step4']:
        performance_step4 = evaluator_merged.evaluate_all_metrics(
            results['step4']['x_hat'], results['step4']['nonzero_indices']
        )
        results['step4']['performance'] = performance_step4
    
    return results


def run_single_experiment(dataset_path: Path, sparsity_ratio: float = 0.1, 
                         seed: int = 42) -> Dict[str, Any]:
    """
    Run single SCAD experiment with proper data separation.
    
    Args:
        dataset_path: Path to Excel dataset
        sparsity_ratio: Sparsity for synthetic data
        seed: Random seed
        
    Returns:
        Experiment results
    """
    # Load data
    graph, source_node = load_network_data(dataset_path)
    
    # Create two tree objects: one without merging, one with merging
    tree_original = Tree(graph, source_node, merge=False)  # For step 1
    tree_merged = Tree(graph, source_node, merge=True)     # For steps 2-4
    
    # Get design matrices
    A_original, _ = tree_original.get_Ax()  # Original matrix (merge=False)
    A_merged, _ = tree_merged.get_Ax()      # Merged matrix (merge=True)
    
    print(f"Original matrix shape: {A_original.shape}")
    print(f"Merged matrix shape: {A_merged.shape}")
    print(f"Dimension reduction: {A_original.shape[1] - A_merged.shape[1]} edges removed")
    
    # Generate synthetic data for each matrix separately
    # Step 1 uses original matrix data
    x_true_original, x_raw_original, b_original, true_nonzero_original = generate_synthetic_data(
        A_original, sparsity_ratio, seed
    )
    
    # Steps 2-4 use merged matrix data
    x_true_merged, x_raw_merged, b_merged, true_nonzero_merged = generate_synthetic_data(
        A_merged, sparsity_ratio, seed
    )
    
    print(f"True non-zero coefficients (original): {len(true_nonzero_original)}")
    print(f"True non-zero coefficients (merged): {len(true_nonzero_merged)}")
    
    # Run 4-step algorithm with appropriate data for each step
    results = run_four_step_scad(
        A_original, A_merged, 
        b_original, b_merged,
        tree_original, tree_merged, 
        x_true_original, x_true_merged,
        true_nonzero_original, true_nonzero_merged
    )
    
    # Add metadata
    results['metadata'] = {
        'dataset': str(dataset_path),
        'sparsity_ratio': sparsity_ratio,
        'seed': seed,
        'original_matrix_shape': A_original.shape,
        'merged_matrix_shape': A_merged.shape,
        'true_nonzero_count_original': len(true_nonzero_original),
        'true_nonzero_count_merged': len(true_nonzero_merged),
        'dimension_reduction': A_original.shape[1] - A_merged.shape[1]
    }
    
    # Add ground truth data for reference
    results['ground_truth'] = {
        'original': {
            'x_true': x_true_original,
            'x_raw': x_raw_original,
            'b': b_original,
            'true_nonzero_indices': true_nonzero_original
        },
        'merged': {
            'x_true': x_true_merged,
            'x_raw': x_raw_merged,
            'b': b_merged,
            'true_nonzero_indices': true_nonzero_merged
        }
    }
    
    return results


def run_multiple_experiments(dataset_path: Path, n_experiments: int = 10,
                           sparsity_ratio: float = 0.1,
                           save_path: Path = None,
                           base_seed: int = 42,
                           paper_tag: str = "",
                           resume: bool = True,
                           checkpoint_interval: int = 1,
                           seed_list: Optional[List[int]] = None) -> Dict[str, Any]:
    """
    Run multiple SCAD experiments with different seeds.

    Args:
        dataset_path: Path to dataset
        n_experiments: Number of experiments (ignored when seed_list is provided)
        sparsity_ratio: Sparsity ratio
        save_path: Path to save results
        seed_list: Explicit seed list; overrides base_seed + n_experiments

    Returns:
        Aggregated results
    """
    if seed_list is not None:
        target_seeds = seed_list
        n_experiments = len(target_seeds)
    else:
        target_seeds = [base_seed + i for i in range(n_experiments)]

    print(f"Running {n_experiments} SCAD experiments...")
    print(f"Dataset: {dataset_path}")
    print(f"Sparsity ratio: {sparsity_ratio}")
    print(f"Seeds: {target_seeds[0]}..{target_seeds[-1]} ({len(target_seeds)} total)")
    if paper_tag:
        print(f"Paper tag: {paper_tag}")

    run_config = {
        'dataset': str(dataset_path),
        'sparsity_ratio': sparsity_ratio,
        'base_seed': base_seed,
        'paper_tag': paper_tag,
        'seed_source': 'explicit' if seed_list is not None else 'contiguous',
    }

    individual_df, seed_status_df = load_checkpoint_results(
        save_path=save_path,
        run_config=run_config,
        resume=resume,
    )
    completed_seeds: Set[int] = set()
    if not seed_status_df.empty and 'Seed' in seed_status_df.columns:
        completed_seeds = set(seed_status_df['Seed'].dropna().astype(int).tolist())
    elif not individual_df.empty and 'Seed' in individual_df.columns:
        completed_seeds = set(individual_df['Seed'].dropna().astype(int).tolist())

    if completed_seeds:
        print(f"Resuming from checkpoint with {len(completed_seeds)} completed seeds")

    next_experiment = int(seed_status_df['Experiment'].max()) + 1 if not seed_status_df.empty else 1

    for i, seed in enumerate(target_seeds):
        if seed in completed_seeds:
            print(f"Skipping experiment {i+1}/{n_experiments} (seed={seed}) - already checkpointed")
            continue
        print(f"\n{'='*60}")
        print(f"Experiment {i+1}/{n_experiments} (seed={seed})")
        print(f"{'='*60}")
        
        try:
            result = run_single_experiment(dataset_path, sparsity_ratio, seed)
            experiment_rows, status_row = result_to_rows(
                result=result,
                experiment_number=next_experiment,
                paper_tag=paper_tag,
            )
            individual_df = pd.concat([individual_df, pd.DataFrame(experiment_rows)], ignore_index=True)
            seed_status_df = pd.concat([seed_status_df, pd.DataFrame([status_row])], ignore_index=True)
            completed_seeds.add(seed)
            next_experiment += 1
            if save_path and checkpoint_interval > 0 and len(completed_seeds) % checkpoint_interval == 0:
                save_checkpoint_results(
                    save_path=save_path,
                    run_config=run_config,
                    individual_df=individual_df,
                    seed_status_df=seed_status_df,
                )
            
            # Print quick summary
            print(f"\nExperiment {i+1} Summary:")
            for step_name in ['step1', 'step2', 'step3', 'step4']:
                if step_name in result:
                    step_result = result[step_name]
                    if step_result.get('skipped', False):
                        print(f"  {step_name}: SKIPPED - {step_result.get('reason', 'Unknown')}")
                    elif 'performance' in step_result:
                        perf = step_result['performance']
                        print(f"  {step_name}: Accuracy={perf['accuracy']:.4f}, "
                              f"Precision={perf['precision']:.4f}, "
                              f"Recall={perf['recall']:.4f}, "
                              f"F0.5={perf['f05_score']:.4f}")
                        
        except Exception as e:
            print(f"    Error in experiment {i+1}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    experiment_count = seed_status_df['Seed'].nunique() if not seed_status_df.empty else 0
    print(f"\nCompleted {experiment_count}/{n_experiments} experiments successfully")

    aggregated = build_aggregated_from_dataframes(
        individual_df=individual_df,
        seed_status_df=seed_status_df,
        run_config=run_config,
    )

    if save_path:
        save_checkpoint_results(
            save_path=save_path,
            run_config=run_config,
            individual_df=individual_df,
            seed_status_df=seed_status_df,
        )

    return aggregated


def aggregate_results(all_results: list,
                     dataset_path: Optional[Path] = None,
                     sparsity_ratio: Optional[float] = None,
                     base_seed: Optional[int] = None,
                     paper_tag: str = "") -> Dict[str, Any]:
    """Aggregate multiple experiment results."""
    if not all_results:
        return {}
    
    steps = ['step1', 'step2', 'step3', 'step4']
    metrics = ['accuracy', 'precision', 'recall', 'f05_score']
    
    aggregated = {
        'summary': {},
        'individual_results': all_results,
        'experiment_count': len(all_results),
        'run_config': {
            'dataset': str(dataset_path) if dataset_path else "",
            'sparsity_ratio': sparsity_ratio,
            'base_seed': base_seed,
            'paper_tag': paper_tag,
        }
    }
    
    # Calculate means and stds for each step
    for step in steps:
        step_metrics = {}
        for metric in metrics:
            values = []
            for result in all_results:
                if (step in result and 'performance' in result[step] and 
                    not result[step].get('skipped', False)):
                    values.append(result[step]['performance'][metric])
            
            if values:
                step_metrics[f'{metric}_mean'] = np.mean(values)
                step_metrics[f'{metric}_std'] = np.std(values)
                step_metrics[f'{metric}_values'] = values
                step_metrics['count'] = len(values)
            else:
                step_metrics[f'{metric}_mean'] = 0.0
                step_metrics[f'{metric}_std'] = 0.0
                step_metrics[f'{metric}_values'] = []
                step_metrics['count'] = 0
        
        # Count skipped experiments
        skipped_count = sum(1 for result in all_results 
                           if step in result and result[step].get('skipped', False))
        step_metrics['skipped_count'] = skipped_count
        step_metrics['completed_count'] = step_metrics['count']
        
        aggregated['summary'][step] = step_metrics
    
    return aggregated


def save_aggregated_results(aggregated: Dict[str, Any], save_path: Path,
                           save_individual: bool = True):
    """Save aggregated results to Excel."""
    individual_df = aggregated.get('individual_df')
    if individual_df is None:
        rows: List[Dict[str, Any]] = []
        for i, result in enumerate(aggregated.get('individual_results', []), start=1):
            experiment_rows, _ = result_to_rows(
                result=result,
                experiment_number=i,
                paper_tag=aggregated.get('run_config', {}).get('paper_tag', ""),
            )
            rows.extend(experiment_rows)
        individual_df = pd.DataFrame(rows)
    seed_status_df = aggregated.get('seed_status_df', pd.DataFrame())

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create summary DataFrame
    summary_data = []
    for step, metrics in aggregated['summary'].items():
        row = {
            'Step': step,
            'Completed': metrics.get('completed_count', 0),
            'Skipped': metrics.get('skipped_count', 0),
            'Total': metrics.get('completed_count', 0) + metrics.get('skipped_count', 0)
        }
        
        # Add metric statistics
        for metric in ['accuracy', 'precision', 'recall', 'f05_score']:
            row[f'{metric}_mean'] = metrics.get(f'{metric}_mean', 0.0)
            row[f'{metric}_std'] = metrics.get(f'{metric}_std', 0.0)
        
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    
    fd, tmp_name = tempfile.mkstemp(suffix=".xlsx", dir=save_path.parent)
    Path(tmp_name).unlink(missing_ok=True)
    temp_path = Path(tmp_name)

    with pd.ExcelWriter(temp_path) as writer:
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

        metadata_rows = []
        for key, value in aggregated.get('run_config', {}).items():
            metadata_rows.append({'Key': key, 'Value': value})
        metadata_rows.append({'Key': 'experiment_count', 'Value': aggregated.get('experiment_count', 0)})
        pd.DataFrame(metadata_rows).to_excel(writer, sheet_name='Metadata', index=False)
        
        # Save individual results
        if save_individual and individual_df is not None and not individual_df.empty:
            individual_df.to_excel(writer, sheet_name='Individual_Results', index=False)
        if seed_status_df is not None and not seed_status_df.empty:
            seed_status_df.to_excel(writer, sheet_name='Seed_Status', index=False)

    temp_path.replace(save_path)


def result_to_rows(result: Dict[str, Any], experiment_number: int, paper_tag: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    metadata = result.get('metadata', {})
    rows: List[Dict[str, Any]] = []
    completed_steps = 0
    skipped_steps = 0
    for step_name, step_result in result.items():
        if not step_name.startswith('step'):
            continue
        base_row = {
            'Experiment': experiment_number,
            'Step': step_name,
            'Seed': metadata.get('seed'),
            'Dataset': metadata.get('dataset'),
            'Sparsity_Ratio': metadata.get('sparsity_ratio'),
            'Paper_Tag': paper_tag,
        }

        if step_result.get('skipped', False):
            skipped_steps += 1
            row = {
                **base_row,
                'Status': 'Skipped',
                'Reason': step_result.get('reason', 'Unknown'),
            }
        elif 'performance' in step_result:
            completed_steps += 1
            row = {
                **base_row,
                'Status': 'Completed',
                **step_result['performance'],
            }
            if 'optimal_lambda' in step_result:
                row['optimal_lambda'] = step_result['optimal_lambda']
            if 'nonzero_indices' in step_result:
                row['predicted_nonzero_count'] = len(step_result['nonzero_indices'])
            if 'n_constraints_added' in step_result:
                row['constraints_added'] = step_result['n_constraints_added']
            elif 'total_constraints_added' in step_result:
                row['constraints_added'] = step_result['total_constraints_added']
        else:
            continue
        rows.append(row)

    status_row = {
        'Experiment': experiment_number,
        'Seed': metadata.get('seed'),
        'Dataset': metadata.get('dataset'),
        'Sparsity_Ratio': metadata.get('sparsity_ratio'),
        'Paper_Tag': paper_tag,
        'Completed_Steps': completed_steps,
        'Skipped_Steps': skipped_steps,
        'Completed': True,
    }
    return rows, status_row


def build_summary_from_individual_df(individual_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    steps = ['step1', 'step2', 'step3', 'step4']
    metrics = ['accuracy', 'precision', 'recall', 'f05_score']
    summary: Dict[str, Dict[str, Any]] = {}
    if individual_df.empty:
        for step in steps:
            summary[step] = {f'{metric}_mean': 0.0 for metric in metrics}
            summary[step].update({f'{metric}_std': 0.0 for metric in metrics})
            summary[step]['count'] = 0
            summary[step]['skipped_count'] = 0
            summary[step]['completed_count'] = 0
        return summary

    for step in steps:
        group = individual_df[individual_df['Step'] == step].copy()
        completed = group[group['Status'] == 'Completed']
        skipped = group[group['Status'] == 'Skipped']
        row: Dict[str, Any] = {
            'count': len(completed),
            'skipped_count': len(skipped),
            'completed_count': len(completed),
        }
        for metric in metrics:
            if metric in completed.columns and not completed.empty:
                row[f'{metric}_mean'] = float(completed[metric].mean())
                row[f'{metric}_std'] = float(completed[metric].std(ddof=0))
                row[f'{metric}_values'] = completed[metric].tolist()
            else:
                row[f'{metric}_mean'] = 0.0
                row[f'{metric}_std'] = 0.0
                row[f'{metric}_values'] = []
        summary[step] = row
    return summary


def build_aggregated_from_dataframes(
    *,
    individual_df: pd.DataFrame,
    seed_status_df: pd.DataFrame,
    run_config: Dict[str, Any],
) -> Dict[str, Any]:
    experiment_count = int(seed_status_df['Seed'].nunique()) if not seed_status_df.empty else 0
    return {
        'summary': build_summary_from_individual_df(individual_df),
        'experiment_count': experiment_count,
        'run_config': run_config,
        'individual_df': individual_df,
        'seed_status_df': seed_status_df,
        'individual_results': [],
    }


def load_checkpoint_results(
    *,
    save_path: Optional[Path],
    run_config: Dict[str, Any],
    resume: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not resume or save_path is None or not Path(save_path).exists():
        return pd.DataFrame(), pd.DataFrame()

    try:
        metadata_df = pd.read_excel(save_path, sheet_name='Metadata')
        metadata = dict(zip(metadata_df['Key'], metadata_df['Value']))
    except Exception:
        return pd.DataFrame(), pd.DataFrame()

    expected = {
        'dataset': run_config.get('dataset'),
        'sparsity_ratio': run_config.get('sparsity_ratio'),
        'base_seed': run_config.get('base_seed'),
    }
    for key, value in expected.items():
        if key in metadata and str(metadata.get(key)) != str(value):
            return pd.DataFrame(), pd.DataFrame()
    if run_config.get('paper_tag') and str(metadata.get('paper_tag', '')) != str(run_config.get('paper_tag')):
        return pd.DataFrame(), pd.DataFrame()

    try:
        individual_df = pd.read_excel(save_path, sheet_name='Individual_Results')
    except Exception:
        individual_df = pd.DataFrame()
    try:
        seed_status_df = pd.read_excel(save_path, sheet_name='Seed_Status')
    except Exception:
        seed_status_df = pd.DataFrame()
    return individual_df, seed_status_df


def save_checkpoint_results(
    *,
    save_path: Path,
    run_config: Dict[str, Any],
    individual_df: pd.DataFrame,
    seed_status_df: pd.DataFrame,
) -> None:
    aggregated = build_aggregated_from_dataframes(
        individual_df=individual_df,
        seed_status_df=seed_status_df,
        run_config=run_config,
    )
    save_aggregated_results(aggregated, save_path, save_individual=True)


def print_results_summary(aggregated: Dict[str, Any]):
    """Print summary of results."""
    print(f"\n{'='*100}")
    print(f"SCAD 4-Step Algorithm Results Summary")
    print(f"{'='*100}")
    
    if 'summary' not in aggregated:
        print("No results to display")
        return
    
    print(f"Total experiments: {aggregated.get('experiment_count', 0)}")
    print()
    
    # Header
    header = f"{'Step':<8} {'Count':<8} {'Skip':<6} {'Accuracy':<15} {'F0.5':<15} {'Precision':<15} {'Recall':<15}"
    print(header)
    print("-" * len(header))
    
    for step, metrics in aggregated['summary'].items():
        completed = metrics.get('completed_count', 0)
        skipped = metrics.get('skipped_count', 0)
        
        if completed == 0:
            print(f"{step:<8} {completed:<8} {skipped:<6} {'N/A':<15} {'N/A':<15} {'N/A':<15} {'N/A':<15}")
            continue
            
        acc_mean = metrics.get('accuracy_mean', 0)
        acc_std = metrics.get('accuracy_std', 0)
        f05_mean = metrics.get('f05_score_mean', 0)
        f05_std = metrics.get('f05_score_std', 0)
        prec_mean = metrics.get('precision_mean', 0)
        prec_std = metrics.get('precision_std', 0)
        rec_mean = metrics.get('recall_mean', 0)
        rec_std = metrics.get('recall_std', 0)
        
        print(f"{step:<8} {completed:<8} {skipped:<6} "
              f"{acc_mean:.3f}±{acc_std:.3f}    "
              f"{f05_mean:.3f}±{f05_std:.3f}    "
              f"{prec_mean:.3f}±{prec_std:.3f}    "
              f"{rec_mean:.3f}±{rec_std:.3f}")
    
    print(f"\n{'='*100}")


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="SCAD 4-Step Algorithm")
    parser.add_argument('--data', type=str, required=True, help='Path to dataset file')
    parser.add_argument('--experiments', type=int, default=10, help='Number of experiments')
    parser.add_argument('--sparsity', type=float, default=0.1, help='Sparsity ratio')
    parser.add_argument('--save', type=str, help='Path to save results')
    parser.add_argument('--base-seed', type=int, default=42,
                       help='Base random seed for reproducible experiment batches (default: 42)')
    parser.add_argument('--paper-tag', type=str, default="",
                       help='Optional paper run tag stored in output metadata')
    parser.add_argument('--save-individual', action=argparse.BooleanOptionalAction, default=True,
                       help='Whether to save the Individual_Results sheet (default: True)')
    parser.add_argument('--resume', action=argparse.BooleanOptionalAction, default=True,
                       help='Resume from an existing matching workbook if present (default: True)')
    parser.add_argument('--checkpoint-interval', type=int, default=1,
                       help='Save checkpoint every N completed experiments (default: 1)')
    parser.add_argument('--seed-values', type=str, default=None,
                       help='Comma-separated explicit seed list (overrides --base-seed and --experiments)')

    args = parser.parse_args()

    # Parse seed list
    seed_list = None
    if args.seed_values:
        seed_list = [int(s.strip()) for s in args.seed_values.split(',') if s.strip()]
    
    # Validate inputs
    dataset_path = Path(args.data)
    if not dataset_path.exists():
        print(f"Error: Dataset {dataset_path} not found")
        return 1
    
    save_path = Path(args.save) if args.save else None
    
    try:
        start_time = time.time()
        
        # Run experiments
        results = run_multiple_experiments(
            dataset_path=dataset_path,
            n_experiments=args.experiments,
            sparsity_ratio=args.sparsity,
            save_path=save_path,
            base_seed=args.base_seed,
            paper_tag=args.paper_tag,
            resume=args.resume,
            checkpoint_interval=args.checkpoint_interval,
            seed_list=seed_list,
        )

        if save_path:
            save_aggregated_results(results, save_path, save_individual=args.save_individual)
            print(f"Results saved to {save_path}")
        
        end_time = time.time()
        
        # Print summary
        print_results_summary(results)
        print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
