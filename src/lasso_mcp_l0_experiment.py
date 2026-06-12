#!/usr/bin/env python3
"""
Other Regularization Methods - Standalone Module

This module implements Lasso (L1), L0, and MCP regularization methods
as a standalone comparison tool for hierarchical network regression.

Key Features:
- Lasso (L1), L0, and MCP regularization baselines
- Wasserstein-based hyperparameter selection workflow
- Independent execution and evaluation
- Support for both original and merged matrices

Usage:
    python src/lasso_mcp_l0_experiment.py                                 # Run all methods with default data (10 experiments)
    python src/lasso_mcp_l0_experiment.py --data data/city1.xlsx          # Use custom data file
    python src/lasso_mcp_l0_experiment.py --method Lasso                  # Run single method
    python src/lasso_mcp_l0_experiment.py --merge                         # Use merged matrix
    python src/lasso_mcp_l0_experiment.py --experiments 10                # Run 10 experiments
    python src/lasso_mcp_l0_experiment.py --save results/output.xlsx      # Save results to file
"""

import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import sys
import time
import tempfile
from datetime import datetime
from typing import Dict, List, Tuple, Union, Optional, Any
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tree_structure import Tree, graph_from_pandas
from evaluation_metrics import Evaluator
from lasso_mcp_l0_methods import (
    lasso_regression_simple,
    l0_regression_simple,
    mcp_regression_wasserstein,
)

warnings.filterwarnings("ignore")


def generate_synthetic_data(A: np.ndarray, ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]:
    """Generate synthetic sparse data for regression experiments."""
    np.random.seed(seed)
    n_features = A.shape[1]
    
    # Initialize sparse coefficient vector
    x = np.zeros(n_features)
    
    # Select non-zero indices
    n_nonzero = max(1, int(n_features * ratio))
    non_zero_indices = np.random.choice(n_features, size=n_nonzero, replace=False).tolist()
    
    # Generate non-zero values
    signal_strength = 2000000.0 / 3.0
    random_values = np.random.normal(loc=0, scale=signal_strength, size=n_nonzero)
    
    # Ensure minimum magnitude
    min_val = 5000.0
    while np.any(np.abs(random_values) < min_val):
        insufficient = np.abs(random_values) < min_val
        random_values[insufficient] = np.random.normal(
            loc=0, scale=signal_strength, size=insufficient.sum()
        )
    
    x[non_zero_indices] = random_values
    
    # Add noise
    x += np.random.normal(loc=0, scale=20.0/3.0, size=n_features)
    
    # Generate observations
    b = A @ x + np.random.normal(loc=0, scale=100.0/3.0, size=A.shape[0])
    
    return x, x, b, non_zero_indices


def run_single_method(method_name: str, A: np.ndarray, b: np.ndarray, 
                     sparsity_ratio: float = 0.1) -> Dict[str, Any]:
    """Run a single regularization method."""
    if method_name.lower() == "lasso":
        return lasso_regression_simple(A, b)
    elif method_name.lower() == "l0":
        return l0_regression_simple(A, b, sparsity_ratio)
    elif method_name.lower() == "mcp":
        return mcp_regression_wasserstein(A, b)
    else:
        raise ValueError(f"Unknown method: {method_name}")


def run_single_experiment(A: np.ndarray, sparsity_ratio: float, seed: int, 
                         methods: List[str]) -> Dict[str, Any]:
    """
    Run a single experiment with all methods.
    
    Args:
        A: Design matrix
        sparsity_ratio: Sparsity ratio for synthetic data
        seed: Random seed
        methods: List of methods to test
        
    Returns:
        Dict containing experiment results
    """
    # Generate synthetic data
    x_true, _, b, true_nonzero_indices = generate_synthetic_data(A, sparsity_ratio, seed)
    
    # Initialize results for this experiment
    experiment_results = {
        'seed': seed,
        'true_nonzero_count': len(true_nonzero_indices),
        'true_nonzero_indices': true_nonzero_indices,
        'x_true': x_true,
        'methods': {}
    }
    
    # Run each method
    for method in methods:
        try:
            # Run method
            result = run_single_method(method, A, b, sparsity_ratio)
            
            # Evaluate result
            evaluator = Evaluator(x_true, true_nonzero_indices, threshold=500.0)
            evaluation = evaluator.comprehensive_evaluation(
                result['x_hat'], result['nonzero_indices']
            )
            
            # Store results
            method_results = {
                'x_hat': result['x_hat'],
                'nonzero_indices': result['nonzero_indices'],
                'optimal_lambda': result['optimal_lambda'],
                'accuracy': evaluation['accuracy'],
                'f05_score': evaluation['f05_score'],
                'precision': evaluation['precision'],
                'recall': evaluation['recall'],
                'error_count': evaluation['error_count']
            }
            
            # Add selection_reason only if it exists (for MCP method)
            if 'selection_reason' in result:
                method_results['selection_reason'] = result['selection_reason']
            
            experiment_results['methods'][method] = method_results
            
        except Exception as e:
            experiment_results['methods'][method] = {'error': str(e)}
    
    return experiment_results


def analyze_multiple_experiments(all_results: List[Dict[str, Any]], 
                               methods: List[str]) -> pd.DataFrame:
    """
    Analyze results from multiple experiments.
    
    Args:
        all_results: List of experiment results
        methods: List of methods tested
        
    Returns:
        DataFrame with statistical analysis
    """
    analysis_data = []
    
    for method in methods:
        # Collect metrics from all experiments
        accuracies = []
        f05_scores = []
        precisions = []
        recalls = []
        error_counts = []
        lambdas = []
        
        successful_experiments = 0
        failed_experiments = 0
        
        for result in all_results:
            if method in result['methods']:
                method_result = result['methods'][method]
                if 'error' not in method_result:
                    accuracies.append(method_result['accuracy'])
                    f05_scores.append(method_result['f05_score'])
                    precisions.append(method_result['precision'])
                    recalls.append(method_result['recall'])
                    error_counts.append(method_result['error_count'])
                    lambdas.append(method_result['optimal_lambda'])
                    successful_experiments += 1
                else:
                    failed_experiments += 1
        
        if accuracies:  # If we have successful experiments
            analysis_data.append({
                'Method': method,
                'Mean_Accuracy': np.mean(accuracies),
                'Std_Accuracy': np.std(accuracies),
                'Mean_F05_Score': np.mean(f05_scores),
                'Std_F05_Score': np.std(f05_scores),
                'Mean_Precision': np.mean(precisions),
                'Std_Precision': np.std(precisions),
                'Mean_Recall': np.mean(recalls),
                'Std_Recall': np.std(recalls),
                'Mean_Error_Count': np.mean(error_counts),
                'Mean_Lambda': np.mean(lambdas),
                'Successful_Experiments': successful_experiments,
                'Failed_Experiments': failed_experiments,
                'Success_Rate': successful_experiments / (successful_experiments + failed_experiments)
            })
    
    return pd.DataFrame(analysis_data)


def analyze_detailed_results(detailed_df: pd.DataFrame,
                             methods: List[str],
                             total_experiments: int) -> pd.DataFrame:
    """Analyze checkpointed detailed results without reconstructing full objects."""
    analysis_data = []
    if detailed_df.empty:
        return pd.DataFrame()

    for method in methods:
        method_df = detailed_df[detailed_df['Method'] == method].copy()
        successful_experiments = int(method_df['Seed'].nunique()) if 'Seed' in method_df.columns else len(method_df)
        failed_experiments = max(0, total_experiments - successful_experiments)
        if method_df.empty:
            continue
        analysis_data.append({
            'Method': method,
            'Mean_Accuracy': float(method_df['Accuracy'].mean()),
            'Std_Accuracy': float(method_df['Accuracy'].std(ddof=0)),
            'Mean_F05_Score': float(method_df['F05_Score'].mean()),
            'Std_F05_Score': float(method_df['F05_Score'].std(ddof=0)),
            'Mean_Precision': float(method_df['Precision'].mean()),
            'Std_Precision': float(method_df['Precision'].std(ddof=0)),
            'Mean_Recall': float(method_df['Recall'].mean()),
            'Std_Recall': float(method_df['Recall'].std(ddof=0)),
            'Mean_Error_Count': float(method_df['Error_Count'].mean()),
            'Mean_Lambda': float(method_df['Optimal_Lambda'].mean()),
            'Successful_Experiments': successful_experiments,
            'Failed_Experiments': failed_experiments,
            'Success_Rate': successful_experiments / total_experiments if total_experiments else 0.0
        })

    return pd.DataFrame(analysis_data)


def experiment_to_detailed_rows(exp_result: Dict[str, Any],
                                experiment_number: int,
                                experiment_info: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for method, method_result in exp_result['methods'].items():
        if 'error' in method_result:
            continue
        rows.append({
            'Experiment': experiment_number,
            'Seed': exp_result['seed'],
            'Method': method,
            'Data_File': experiment_info['data_file'],
            'Sparsity_Ratio': experiment_info['sparsity_ratio'],
            'Use_Merge': experiment_info['use_merge'],
            'Paper_Tag': experiment_info.get('paper_tag', ''),
            'Accuracy': method_result['accuracy'],
            'F05_Score': method_result['f05_score'],
            'Precision': method_result['precision'],
            'Recall': method_result['recall'],
            'Error_Count': method_result['error_count'],
            'Optimal_Lambda': method_result['optimal_lambda'],
            'Predicted_Nonzero': len(method_result['nonzero_indices']),
            'True_Nonzero': exp_result['true_nonzero_count']
        })
    return rows


def experiment_to_seed_status(exp_result: Dict[str, Any],
                              experiment_number: int,
                              experiment_info: Dict[str, Any],
                              methods: List[str]) -> Dict[str, Any]:
    successful = sum(
        1 for method in methods
        if method in exp_result['methods'] and 'error' not in exp_result['methods'][method]
    )
    failed = len(methods) - successful
    return {
        'Experiment': experiment_number,
        'Seed': exp_result['seed'],
        'Data_File': experiment_info['data_file'],
        'Sparsity_Ratio': experiment_info['sparsity_ratio'],
        'Use_Merge': experiment_info['use_merge'],
        'Paper_Tag': experiment_info.get('paper_tag', ''),
        'Successful_Methods': successful,
        'Failed_Methods': failed,
        'Completed': True,
    }


def load_checkpoint_results(output_file: Optional[str],
                            expected_info: Dict[str, Any],
                            resume: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not resume or not output_file:
        return pd.DataFrame(), pd.DataFrame()
    output_path = Path(output_file)
    if not output_path.exists():
        return pd.DataFrame(), pd.DataFrame()

    try:
        metadata_df = pd.read_excel(output_path, sheet_name='Metadata')
        metadata = dict(zip(metadata_df['Key'], metadata_df['Value']))
    except Exception:
        return pd.DataFrame(), pd.DataFrame()

    expected = {
        'data_file': expected_info['data_file'],
        'sparsity_ratio': expected_info['sparsity_ratio'],
        'base_seed': expected_info['base_seed'],
        'use_merge': expected_info['use_merge'],
    }
    for key, value in expected.items():
        if key in metadata and str(metadata.get(key)) != str(value):
            return pd.DataFrame(), pd.DataFrame()
    if expected_info.get('paper_tag') and str(metadata.get('paper_tag', '')) != str(expected_info.get('paper_tag')):
        return pd.DataFrame(), pd.DataFrame()

    try:
        detailed_df = pd.read_excel(output_path, sheet_name='Detailed_Results')
    except Exception:
        detailed_df = pd.DataFrame()
    try:
        seed_status_df = pd.read_excel(output_path, sheet_name='Seed_Status')
    except Exception:
        seed_status_df = pd.DataFrame()
    return detailed_df, seed_status_df


def run_methods_comparison(data_file: str = "data/city1.xlsx",
                          sparsity_ratio: float = 0.1, num_experiments: int = 10,
                          base_seed: int = 0, use_merge: bool = False,
                          methods: List[str] = None, paper_tag: str = "",
                          save_path: Optional[str] = None,
                          resume: bool = True,
                          checkpoint_interval: int = 1,
                          seed_list: Optional[List[int]] = None) -> Dict[str, Any]:
    """
    Run comparison of regularization methods with multiple experiments.

    Args:
        data_file: Path to Excel data file
        sparsity_ratio: Sparsity ratio for synthetic data
        num_experiments: Number of experiments to run (ignored when seed_list is provided)
        base_seed: Base random seed (will be incremented for each experiment)
        use_merge: Whether to use merged matrix
        methods: List of methods to test (default: all)
        seed_list: Explicit seed list; overrides base_seed + num_experiments

    Returns:
        Dict containing comparison results
    """
    if seed_list is not None:
        target_seeds = seed_list
        num_experiments = len(target_seeds)
    else:
        target_seeds = [base_seed + i for i in range(num_experiments)]

    print(f"\n{'='*60}")
    print(f"REGULARIZATION METHODS COMPARISON")
    print(f"{'='*60}")
    print(f"Data file: {data_file}")
    print(f"Sparsity ratio: {sparsity_ratio}")
    print(f"Number of experiments: {num_experiments}")
    print(f"Seeds: {target_seeds[0]}..{target_seeds[-1]} ({len(target_seeds)} total)")
    print(f"Matrix type: {'Merged' if use_merge else 'Original'}")
    if paper_tag:
        print(f"Paper tag: {paper_tag}")
    
    # Load data
    data_path = Path(data_file)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file {data_file} not found. Please provide a valid Excel file.")
    
    graph, source = graph_from_pandas(data_path)
    tree = Tree(graph, source, merge=use_merge)
    A, _ = tree.get_Ax()
    
    print(f"Network: {len(graph.nodes())} nodes, {len(graph.edges())} edges")
    print(f"Matrix shape: {A.shape}")
    
    # Set default methods
    if methods is None:
        methods = ['Lasso', 'L0', 'MCP']
    
    print(f"Methods: {', '.join(methods)}")
    
    # Run multiple experiments
    print(f"\nRunning {num_experiments} experiments...")
    print("-" * 40)
    
    experiment_info = {
        'data_file': data_file,
        'sparsity_ratio': sparsity_ratio,
        'num_experiments': num_experiments,
        'base_seed': base_seed,
        'use_merge': use_merge,
        'matrix_shape': A.shape,
        'methods': methods,
        'paper_tag': paper_tag,
        'seed_source': 'explicit' if seed_list is not None else 'contiguous'
    }

    detailed_df, seed_status_df = load_checkpoint_results(save_path, experiment_info, resume)
    completed_seeds = set(seed_status_df['Seed'].dropna().astype(int).tolist()) if not seed_status_df.empty else set()
    if completed_seeds:
        print(f"Resuming from checkpoint with {len(completed_seeds)} completed seeds")
    next_experiment = int(seed_status_df['Experiment'].max()) + 1 if not seed_status_df.empty else 1

    all_results = []
    
    for i, experiment_seed in enumerate(target_seeds):
        if experiment_seed in completed_seeds:
            print(f"Experiment {i+1}/{num_experiments} (seed: {experiment_seed}) - skipped (checkpoint)")
            continue
        print(f"Experiment {i+1}/{num_experiments} (seed: {experiment_seed})", end=" ")
        
        # Run single experiment
        experiment_result = run_single_experiment(A, sparsity_ratio, experiment_seed, methods)
        all_results.append(experiment_result)
        detailed_rows = experiment_to_detailed_rows(experiment_result, next_experiment, experiment_info)
        seed_status_row = experiment_to_seed_status(experiment_result, next_experiment, experiment_info, methods)
        if detailed_rows:
            detailed_df = pd.concat([detailed_df, pd.DataFrame(detailed_rows)], ignore_index=True)
        seed_status_df = pd.concat([seed_status_df, pd.DataFrame([seed_status_row])], ignore_index=True)
        completed_seeds.add(experiment_seed)
        next_experiment += 1
        
        # Print brief progress
        successful_methods = sum(1 for method in methods 
                               if method in experiment_result['methods'] 
                               and 'error' not in experiment_result['methods'][method])
        print(f"- {successful_methods}/{len(methods)} methods successful")

        if save_path and checkpoint_interval > 0 and len(completed_seeds) % checkpoint_interval == 0:
            checkpoint_results = {
                'statistical_analysis': analyze_detailed_results(detailed_df, methods, len(completed_seeds)),
                'all_experiment_results': [],
                'detailed_results_df': detailed_df,
                'seed_status_df': seed_status_df,
                'experiment_info': {**experiment_info, 'num_experiments': len(completed_seeds)},
            }
            save_results(checkpoint_results, save_path)
    
    # Analyze results
    print(f"\nAnalyzing results...")
    statistical_analysis = analyze_detailed_results(detailed_df, methods, len(completed_seeds))
    
    if not statistical_analysis.empty:
        statistical_analysis = statistical_analysis.sort_values('Mean_F05_Score', ascending=False)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"MULTIPLE EXPERIMENTS RESULTS")
    print(f"{'='*60}")
    
    if not statistical_analysis.empty:
        print(f"\nStatistical Summary (Mean ± Std):")
        print("-" * 50)
        for _, row in statistical_analysis.iterrows():
            print(f"{row['Method']:<8}: "
                  f"Acc={row['Mean_Accuracy']:.3f}±{row['Std_Accuracy']:.3f}, "
                  f"F0.5={row['Mean_F05_Score']:.3f}±{row['Std_F05_Score']:.3f}, "
                  f"Success={row['Success_Rate']:.1%}")
        
        best_method = statistical_analysis.iloc[0]
        print(f"\nBest performing method: {best_method['Method']}")
        print(f"   Mean F0.5 Score: {best_method['Mean_F05_Score']:.4f} ± {best_method['Std_F05_Score']:.4f}")
        print(f"   Mean Accuracy: {best_method['Mean_Accuracy']:.4f} ± {best_method['Std_Accuracy']:.4f}")
        print(f"   Success Rate: {best_method['Success_Rate']:.1%}")
    else:
        print("No successful method results to analyze")
    
    return {
        'statistical_analysis': statistical_analysis,
        'all_experiment_results': all_results,
        'detailed_results_df': detailed_df,
        'seed_status_df': seed_status_df,
        'experiment_info': {
            **experiment_info,
            'num_experiments': len(completed_seeds),
        }
    }


def save_results(results: Dict[str, Any], output_file: str = None) -> str:
    """Save comparison results to Excel file."""
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        matrix_type = "merged" if results['experiment_info']['use_merge'] else "original"
        num_exp = results['experiment_info']['num_experiments']
        output_file = f"other_methods_comparison_{matrix_type}_{num_exp}exp_{timestamp}.xlsx"
    
    output_path = Path(output_file)
    print(f"\nSaving results to {output_path}...")
    
    fd, tmp_name = tempfile.mkstemp(suffix=".xlsx", dir=output_path.parent)
    Path(tmp_name).unlink(missing_ok=True)
    temp_path = Path(tmp_name)

    with pd.ExcelWriter(temp_path, engine='openpyxl') as writer:
        # Save statistical analysis
        if not results['statistical_analysis'].empty:
            results['statistical_analysis'].to_excel(
                writer, sheet_name='Statistical_Analysis', index=False
            )

        # Save experiment info
        exp_info = pd.DataFrame([results['experiment_info']])
        exp_info.to_excel(writer, sheet_name='Experiment_Info', index=False)

        metadata_rows = [
            {'Key': key, 'Value': value}
            for key, value in results['experiment_info'].items()
        ]
        pd.DataFrame(metadata_rows).to_excel(writer, sheet_name='Metadata', index=False)

        detailed_df = results.get('detailed_results_df')
        if detailed_df is None:
            detailed_results = []
            for i, exp_result in enumerate(results['all_experiment_results']):
                detailed_results.extend(
                    experiment_to_detailed_rows(exp_result, i + 1, results['experiment_info'])
                )
            detailed_df = pd.DataFrame(detailed_results)

        if detailed_df is not None and not detailed_df.empty:
            detailed_df.to_excel(writer, sheet_name='Detailed_Results', index=False)

        seed_status_df = results.get('seed_status_df')
        if seed_status_df is not None and not seed_status_df.empty:
            seed_status_df.to_excel(writer, sheet_name='Seed_Status', index=False)

    temp_path.replace(output_path)
    
    print(f"Results saved to {output_path}")
    return str(output_path)


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Other Regularization Methods Comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python src/lasso_mcp_l0_experiment.py                                # Run all methods with default data (10 experiments)
    python src/lasso_mcp_l0_experiment.py --data your_data.xlsx          # Use custom data file
    python src/lasso_mcp_l0_experiment.py --method Lasso                 # Run single method
    python src/lasso_mcp_l0_experiment.py --merge                        # Use merged matrix
    python src/lasso_mcp_l0_experiment.py --ratio 0.15 --seed 123       # Custom parameters
    python src/lasso_mcp_l0_experiment.py --experiments 20               # Run 20 experiments
    python src/lasso_mcp_l0_experiment.py --save results.xlsx           # Save results to file
        """
    )
    
    parser.add_argument('--data', type=str, default='data/city1.xlsx',
                       help='Path to Excel data file')
    parser.add_argument('--method', type=str, choices=['Lasso', 'L0', 'MCP'], default=None,
                       help='Single method to run (default: run all)')
    parser.add_argument('--ratio', type=float, default=0.1,
                       help='Sparsity ratio (default: 0.1)')
    parser.add_argument('--experiments', type=int, default=10,
                       help='Number of experiments to run (default: 10)')
    parser.add_argument('--seed', '--base-seed', dest='base_seed', type=int, default=0,
                       help='Base random seed (default: 0)')
    parser.add_argument('--merge', action='store_true',
                       help='Use merged matrix instead of original')
    parser.add_argument('--save', type=str, default=None,
                       help='Save results to Excel file')
    parser.add_argument('--paper-tag', type=str, default='',
                       help='Optional paper run tag stored in output metadata')
    parser.add_argument('--resume', action=argparse.BooleanOptionalAction, default=True,
                       help='Resume from an existing matching workbook if present (default: True)')
    parser.add_argument('--checkpoint-interval', type=int, default=1,
                       help='Save checkpoint every N completed experiments (default: 1)')
    parser.add_argument('--seed-values', type=str, default=None,
                       help='Comma-separated explicit seed list (overrides --base-seed and --experiments)')

    args = parser.parse_args()

    print("Other Regularization Methods Comparison Tool")
    print("=" * 50)

    # Parse seed list
    seed_list = None
    if args.seed_values:
        seed_list = [int(s.strip()) for s in args.seed_values.split(',') if s.strip()]
    
    try:
        # Determine methods to run
        methods = [args.method] if args.method else ['Lasso', 'L0', 'MCP']
        
        # Run comparison
        results = run_methods_comparison(
            data_file=args.data,
            sparsity_ratio=args.ratio,
            num_experiments=args.experiments,
            base_seed=args.base_seed,
            use_merge=args.merge,
            methods=methods,
            paper_tag=args.paper_tag,
            save_path=args.save,
            resume=args.resume,
            checkpoint_interval=args.checkpoint_interval,
            seed_list=seed_list,
        )
        
        # Save results if requested
        if args.save:
            save_results(results, args.save)
        
        print(f"\nComparison completed successfully!")
        
    except KeyboardInterrupt:
        print(f"\nComparison interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nComparison failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
