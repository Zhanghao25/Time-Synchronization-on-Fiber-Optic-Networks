#!/usr/bin/env python3
"""
TSLE Methods Module for Hierarchical Network Regression

This module implements the manuscript's 4-step TSLE procedure built on
SCAD estimation:
1. TSLE on the original matrix (without merge)
2. TSLE on the merged matrix (with path merging)
3. TSLE-Random: random augmentation
4. TSLE-Addition: confidence-based (LUR-guided) iterative augmentation

Key Features:
- Wasserstein distance-based lambda selection
- Tree-structure-based constraint augmentation
- Random augmentation for comparison
- Enhanced confidence grouping
- RREF-based system reduction
"""

import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Any, Sequence
from scipy.stats import wasserstein_distance
import warnings
import threading

# R integration for SCAD regression
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
from rpy2.robjects.packages import importr

# Initialize R environment
rpy2.robjects.numpy2ri.activate()
try:
    ncvreg = importr("ncvreg")
except:
    print("Warning: ncvreg R package not available. Install with: install.packages('ncvreg')")
    ncvreg = None

from tree_structure import Tree, get_all_edge_indices_above

# Thread-safe lock for operations
operation_lock = threading.Lock()


def compute_lambda_max(X: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate the maximum lambda value for regularization path initialization.
    
    Args:
        X (np.ndarray): Feature matrix (n_samples, n_features)
        y (np.ndarray): Target vector (n_samples,)
        
    Returns:
        float: Maximum lambda value λ_max = max_j |X_j^T y| / n
    """
    if X.shape[0] != len(y):
        raise ValueError("X and y must have compatible dimensions")
    
    n_samples = X.shape[0]
    correlation_scores = np.abs(np.dot(X.T, y)) / n_samples
    lambda_max = np.max(correlation_scores)
    
    return lambda_max


def apply_scad_regression(A: np.ndarray, b: np.ndarray, 
                         lambda_min_ratio: float = 1e-6,
                         high_region_points: int = 10,
                         low_region_points: int = 90,
                         transition_lambda: float = 0.1,
                         max_zeros: int = 5) -> Dict[str, Any]:
    """
    Apply SCAD regression with Wasserstein distance-based lambda selection.
    
    Args:
        A (np.ndarray): Design matrix
        b (np.ndarray): Observation vector
        lambda_min_ratio (float): Minimum lambda as fraction of lambda_max
        high_region_points (int): Number of points in high lambda region
        low_region_points (int): Number of points in low lambda region
        transition_lambda (float): Transition point between regions
        max_zeros (int): Early stopping criterion
        
    Returns:
        Dict containing SCAD regression results
    """
    if ncvreg is None:
        raise RuntimeError("ncvreg R package not available")
    
    # Initialize SCAD parameters
    initial_beta = np.zeros(A.shape[1])
    xtx = np.sum(A ** 2, axis=0) / A.shape[0]  # Column-wise sum of squares
    penalty_factor = np.ones(A.shape[1])
    
    # Generate optimized lambda sequence
    lambda_max = compute_lambda_max(A, b)
    lambda_min = lambda_min_ratio * lambda_max
    
    # Intelligently adjust transition point
    if transition_lambda >= lambda_max:
        transition_lambda = lambda_max / 10
    elif transition_lambda <= lambda_min:
        transition_lambda = lambda_min * 10
    
    # Create two-phase lambda sequence
    if lambda_max > transition_lambda > lambda_min:
        high_lambdas = np.logspace(
            np.log10(lambda_max), np.log10(transition_lambda), 
            num=high_region_points, endpoint=False
        )
        low_lambdas = np.logspace(
            np.log10(transition_lambda), np.log10(lambda_min), 
            num=low_region_points, endpoint=True
        )
        lambda_sequence = np.concatenate([high_lambdas, low_lambdas])
    else:
        total_points = high_region_points + low_region_points
        lambda_sequence = np.logspace(np.log10(lambda_max), np.log10(lambda_min), num=total_points)
    
    # Solve SCAD for each lambda with convergence monitoring
    beta_solutions = {}
    wasserstein_distances = []
    convergence_indicators = []
    gamma = 3.7  # SCAD parameter
    
    # Early stopping parameters
    consecutive_convergence_threshold = max_zeros
    early_stopped = False

    for i, lam in enumerate(lambda_sequence):
        # Fit SCAD model
        scad_model = ncvreg.ncvfit(
            X=A, y=list(b), init=initial_beta, xtx=xtx,
            penalty="SCAD", gamma=gamma, penalty_factor=penalty_factor,
            **{'lambda': robjects.FloatVector([lam])}
        )
        
        current_beta = np.array(scad_model.rx2("beta")).flatten()
        initial_beta = current_beta  # Warm start for next iteration
        beta_solutions[lam] = current_beta
        
        # Monitor convergence via Wasserstein distance
        if i > 0:
            previous_lambda = lambda_sequence[i-1]
            previous_coefficients = beta_solutions[previous_lambda]
            
            distance = wasserstein_distance(previous_coefficients, current_beta)
            wasserstein_distances.append(distance)
            
            has_converged = distance < 1e-1
            convergence_indicators.append(has_converged)
            
            # Check early stopping condition
            if len(convergence_indicators) >= consecutive_convergence_threshold:
                recent_convergence = convergence_indicators[-consecutive_convergence_threshold:]
                if all(recent_convergence):
                    early_stopped = True
                    break
    
    # Select optimal lambda using sophisticated strategy
    optimal_idx = 0
    selection_reason = "Default selection (first lambda)"
    
    if len(wasserstein_distances) >= consecutive_convergence_threshold:
        # Look for sustained convergence
        convergence_found = False
        for i in range(len(convergence_indicators) - consecutive_convergence_threshold + 1):
            window = convergence_indicators[i:i + consecutive_convergence_threshold]
            if all(window):
                optimal_idx = i
                selection_reason = f"Sustained convergence over {consecutive_convergence_threshold} steps"
                convergence_found = True
                break
        
        if not convergence_found and wasserstein_distances:
            # Fall back to minimum distance
            optimal_idx = np.argmin(wasserstein_distances) + 1
            selection_reason = "Minimum Wasserstein distance"
    
    # Extract optimal solution
    optimal_lambda = lambda_sequence[optimal_idx]
    x_hat = beta_solutions[optimal_lambda]
    
    # Extract sparsity information
    sparsity_threshold = 500.0
    significant_mask = np.abs(x_hat) > sparsity_threshold
    nonzero_indices = np.where(significant_mask)[0].tolist()
    
    results = {
        'x_hat': x_hat,
        'optimal_lambda': optimal_lambda,
        'nonzero_indices': nonzero_indices,
        'selection_reason': selection_reason,
        'early_stopped': early_stopped,
        'total_lambdas_computed': len(beta_solutions)
    }
    
    return results


def rref(A: np.ndarray, tolerance: float = 1e-10) -> np.ndarray:
    """
    Convert matrix to Reduced Row Echelon Form (RREF) with numerical stability.
    
    Args:
        A (np.ndarray): Input matrix
        tolerance (float): Numerical tolerance for pivot detection
        
    Returns:
        np.ndarray: Matrix in RREF
    """
    A_rref = A.astype(float).copy()
    rows, cols = A_rref.shape
    current_row = 0
    
    for col_index in range(cols):
        if current_row >= rows:
            break
        
        # Find best pivot (largest absolute value) for numerical stability
        pivot_candidates = A_rref[current_row:, col_index]
        max_pivot_idx = np.argmax(np.abs(pivot_candidates))
        best_pivot_row = current_row + max_pivot_idx
        
        # Check if pivot is sufficiently large
        if abs(A_rref[best_pivot_row, col_index]) < tolerance:
            continue  # Skip this column
        
        # Swap rows to bring best pivot to current position
        if best_pivot_row != current_row:
            A_rref[[current_row, best_pivot_row]] = A_rref[[best_pivot_row, current_row]]
        
        # Scale pivot row to make pivot = 1
        pivot_value = A_rref[current_row, col_index]
        A_rref[current_row] /= pivot_value
        
        # Eliminate column in all other rows
        for row in range(rows):
            if row != current_row and abs(A_rref[row, col_index]) > tolerance:
                elimination_factor = A_rref[row, col_index]
                A_rref[row] -= elimination_factor * A_rref[current_row]
        
        current_row += 1
    
    # Clean up near-zero entries
    A_rref[np.abs(A_rref) < tolerance] = 0.0
    
    return A_rref


def rref_augmented(A: np.ndarray, b: np.ndarray, 
                  tol: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute RREF of augmented system [A|b] for constraint processing.
    
    Args:
        A (np.ndarray): Coefficient matrix (m × n)
        b (np.ndarray): Right-hand side vector (m,)
        tol (float): Numerical tolerance
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (A_rref, b_rref) in reduced form
    """
    # Form augmented matrix [A|b]
    augmented_matrix = np.hstack((A.astype(float), b.reshape(-1, 1).astype(float)))
    m, n_plus_1 = augmented_matrix.shape
    n = n_plus_1 - 1
    
    current_row = 0
    
    # Process each column (excluding augmented column)
    for col in range(n):
        if current_row >= m:
            break
        
        # Find pivot
        pivot_row = None
        for r in range(current_row, m):
            if abs(augmented_matrix[r, col]) > tol:
                pivot_row = r
                break
        
        if pivot_row is None:
            continue  # No pivot found, skip column
        
        # Swap pivot row into position
        if pivot_row != current_row:
            augmented_matrix[[current_row, pivot_row]] = augmented_matrix[[pivot_row, current_row]]
        
        # Normalize pivot row
        pivot_value = augmented_matrix[current_row, col]
        augmented_matrix[current_row] /= pivot_value
        
        # Eliminate column in other rows
        for r in range(m):
            if r != current_row and abs(augmented_matrix[r, col]) > tol:
                factor = augmented_matrix[r, col]
                augmented_matrix[r] -= factor * augmented_matrix[current_row]
        
        current_row += 1
    
    # Apply tolerance cleanup
    augmented_matrix[np.abs(augmented_matrix) < tol] = 0.0
    
    # Split back into A and b
    A_rref = augmented_matrix[:, :n]
    b_rref = augmented_matrix[:, n]
    
    return A_rref, b_rref


def augment_system_confidence(A: np.ndarray, b: np.ndarray, tree: Tree, 
                             pred_neg: Sequence[int], x_ori: np.ndarray, 
                             add_high: bool = False,
                             return_node_counts: bool = False) -> Union[Tuple[np.ndarray, np.ndarray, int], 
                                                                        Tuple[np.ndarray, np.ndarray, int, Dict[str, int]]]:
    """
    Augment linear system with tree-structure-based constraints (confidence-based).
    
    Args:
        A (np.ndarray): Current coefficient matrix
        b (np.ndarray): Current observation vector
        tree (Tree): Tree structure for constraint generation
        pred_neg (Sequence[int]): Predicted non-zero coefficient indices
        x_ori (np.ndarray): True coefficient vector (for constraint RHS)
        add_high (bool): Whether to include high-confidence nodes
        return_node_counts (bool): Whether to return subhigh/high node counts
        
    Returns:
        Tuple[np.ndarray, np.ndarray, int] or Tuple[..., Dict]: (augmented_A, augmented_b, n_added_rows, [node_counts])
    """
    # Calculate true nonzero indices from x_ori
    true_nonzero_indices = np.where(np.abs(x_ori) > 500)[0].tolist()
    
    # Ensure pred_neg is a list
    pred_neg_list = list(pred_neg) if not isinstance(pred_neg, list) else pred_neg
    
    # Get enhanced confidence grouping results
    confidence_results = tree.find_subhigh_nodes(
        list(range(A.shape[1])), 
        pred_neg_list, 
        true_nonzero_indices
    )
    subhigh_nodes = confidence_results['subhigh_nodes']
    high_nodes = confidence_results['high_nodes']
    
    node_counts = {
        'subhigh_nodes_count': len(subhigh_nodes),
        'high_nodes_count': len(high_nodes),
        'total_nodes_to_add': len(subhigh_nodes) + len(high_nodes) if add_high else len(subhigh_nodes)
    }
    
    # Select nodes for constraint generation
    if add_high:
        nodes_to_augment = subhigh_nodes.union(high_nodes)
    else:
        nodes_to_augment = subhigh_nodes
        
    if not nodes_to_augment:
        if return_node_counts:
            return A, b, 0, node_counts
        return A, b, 0  # No constraints to add
    
    # Generate constraint rows
    additional_rows = []
    for node in nodes_to_augment:
        # Get all edges on path from node to root
        path_edge_indices = get_all_edge_indices_above(tree, node)
        
        # Create constraint row: sum of path edges = total flow
        constraint_row = np.zeros(A.shape[1])
        constraint_row[path_edge_indices] = 1.0
        additional_rows.append(constraint_row)
    
    if not additional_rows:
        if return_node_counts:
            return A, b, 0, node_counts
        return A, b, 0
    
    additional_rows = np.vstack(additional_rows)
    n_added = additional_rows.shape[0]
    
    # Calculate constraint RHS using true coefficients
    b_new = additional_rows @ x_ori
    
    # Augment system
    A_augmented = np.vstack((A, additional_rows))
    b_augmented = np.concatenate((b, b_new))
    
    # Reduce to RREF to remove linear dependence
    A_rref, b_rref = rref_augmented(A_augmented, b_augmented)
    
    # Remove zero rows
    non_zero_rows = np.any(A_rref != 0, axis=1)
    A_final = A_rref[non_zero_rows]
    b_final = b_rref[non_zero_rows]
    
    if return_node_counts:
        return A_final, b_final, n_added, node_counts
    return A_final, b_final, n_added


def augment_system_random(A: np.ndarray, b: np.ndarray, tree: Tree, 
                         n_random_nodes: int, x_ori: np.ndarray, 
                         random_seed: int = None) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Augment linear system with randomly selected tree nodes for comparison analysis.
    
    Args:
        A (np.ndarray): Current coefficient matrix
        b (np.ndarray): Current observation vector
        tree (Tree): Tree structure for constraint generation
        n_random_nodes (int): Number of random nodes to select for augmentation
        x_ori (np.ndarray): True coefficient vector (for constraint RHS)
        random_seed (int): Random seed for reproducible node selection
        
    Returns:
        Tuple[np.ndarray, np.ndarray, int]: (augmented_A, augmented_b, n_added_rows)
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Get all available nodes (excluding leaves and source)
    all_nodes = list(tree.tree.nodes())
    available_nodes = [node for node in all_nodes 
                      if node != tree.source and node not in tree.leaves]
    
    if len(available_nodes) == 0 or n_random_nodes <= 0:
        return A, b, 0  # No nodes to augment
    
    # Randomly select nodes for augmentation
    n_select = min(n_random_nodes, len(available_nodes))
    if n_select == 1:
        # np.random.choice returns scalar when size=1, so ensure it's an array
        random_nodes = [np.random.choice(available_nodes, size=1, replace=False).item()]
    else:
        random_nodes = np.random.choice(available_nodes, size=n_select, replace=False).tolist()
    
    # Generate constraint rows for random nodes
    additional_rows = []
    for node in random_nodes:
        # Get all edges on path from node to root
        path_edge_indices = get_all_edge_indices_above(tree, node)
        
        # Create constraint row: sum of path edges = total flow
        constraint_row = np.zeros(A.shape[1])
        constraint_row[path_edge_indices] = 1.0
        additional_rows.append(constraint_row)
    
    if not additional_rows:
        return A, b, 0
    
    additional_rows = np.vstack(additional_rows)
    n_added = additional_rows.shape[0]
    
    # Calculate constraint RHS using true coefficients
    b_new = additional_rows @ x_ori
    
    # Augment system
    A_augmented = np.vstack((A, additional_rows))
    b_augmented = np.concatenate((b, b_new))
    
    # Reduce to RREF to remove linear dependence
    A_rref, b_rref = rref_augmented(A_augmented, b_augmented)
    
    # Remove zero rows
    non_zero_rows = np.any(A_rref != 0, axis=1)
    A_final = A_rref[non_zero_rows]
    b_final = b_rref[non_zero_rows]
    
    return A_final, b_final, n_added


def tsle_step1_original(A_original: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
    """
    Step 1: SCAD regression on original matrix (without path merging).
    
    Args:
        A_original (np.ndarray): Original design matrix
        b (np.ndarray): Observation vector
        
    Returns:
        Dict containing SCAD results for original matrix
    """
    print("Step 1: SCAD regression on original matrix...")
    
    result = apply_scad_regression(A_original, b)
    
    print(f"   Optimal lambda: {result['optimal_lambda']:.6e}")
    print(f"   Selection reason: {result['selection_reason']}")
    print(f"   Non-zero coefficients: {len(result['nonzero_indices'])}")
    print(f"   Early stopped: {result['early_stopped']}")
    
    return result


def tsle_step2_merged(tree: Tree, b: np.ndarray) -> Dict[str, Any]:
    """
    Step 2: SCAD regression on merged matrix (with path merging).
    
    Args:
        tree (Tree): Tree object (should have merge=True)
        b (np.ndarray): Observation vector
        
    Returns:
        Dict containing SCAD results for merged matrix
    """
    print("Step 2: SCAD regression on merged matrix...")
    
    # Get merged design matrix
    A_merged, _ = tree.get_Ax()
    
    print(f"   Matrix size reduction: merged matrix shape = {A_merged.shape}")
    
    result = apply_scad_regression(A_merged, b)
    
    print(f"   Optimal lambda: {result['optimal_lambda']:.6e}")
    print(f"   Selection reason: {result['selection_reason']}")
    print(f"   Non-zero coefficients: {len(result['nonzero_indices'])}")
    print(f"   Early stopped: {result['early_stopped']}")
    
    return result


def tsle_step3_random(A: np.ndarray, b: np.ndarray, tree: Tree, 
                            x_true: np.ndarray, seed: int = 42) -> Dict[str, Any]:
    """
    Step 3: SCAD regression with random augmentation.
    
    Args:
        A (np.ndarray): Design matrix
        b (np.ndarray): Observation vector
        tree (Tree): Tree structure
        x_true (np.ndarray): True coefficient vector
        seed (int): Random seed
        
    Returns:
        Dict containing SCAD results with random augmentation
    """
    print("Step 3: SCAD regression with random augmentation...")
    
    # First get initial SCAD result to determine number of nodes
    initial_result = apply_scad_regression(A, b)
    
    # Get confidence analysis to determine number of nodes to add
    true_nonzero_indices = np.where(np.abs(x_true) > 500)[0].tolist()
    confidence_results = tree.find_subhigh_nodes(
        list(range(A.shape[1])), 
        initial_result['nonzero_indices'], 
        true_nonzero_indices
    )
    
    n_subhigh = confidence_results['group_counts'].get('subhigh_nodes_count', 0)
    n_high = confidence_results['group_counts'].get('high_nodes_count', 0)
    n_nodes_to_add = n_subhigh + n_high
    
    print(f"   Adding constraints from {n_nodes_to_add} randomly selected nodes...")
    
    if n_nodes_to_add == 0:
        print("   No nodes to add, returning original result")
        return initial_result
    
    # Apply random augmentation
    A_random, b_random, n_added = augment_system_random(
        A, b, tree, n_nodes_to_add, x_true, random_seed=seed
    )
    
    print(f"   Added {n_added} constraints, new matrix shape: {A_random.shape}")
    
    # Apply SCAD to augmented system
    result = apply_scad_regression(A_random, b_random)
    
    print(f"   Optimal lambda: {result['optimal_lambda']:.6e}")
    print(f"   Selection reason: {result['selection_reason']}")
    print(f"   Non-zero coefficients: {len(result['nonzero_indices'])}")
    print(f"   Early stopped: {result['early_stopped']}")
    
    # Add augmentation info
    result['n_constraints_added'] = n_added
    result['augmentation_type'] = 'random'
    
    return result


def tsle_step4_iterative(A: np.ndarray, b: np.ndarray, tree: Tree, 
                               x_true: np.ndarray, max_iterations: int = 5,
                               add_high: bool = False,
                               detailed_logging: bool = False) -> Dict[str, Any]:
    """
    Step 4: SCAD regression with confidence-based iterative augmentation.
    
    Args:
        A (np.ndarray): Design matrix
        b (np.ndarray): Observation vector
        tree (Tree): Tree structure
        x_true (np.ndarray): True coefficient vector
        max_iterations (int): Maximum number of iterations
        add_high (bool): Whether to include high-confidence nodes
        detailed_logging (bool): Whether to record detailed metrics per iteration
        
    Returns:
        Dict containing SCAD results with iterative augmentation
    """
    print("Step 4: SCAD regression with confidence-based iterative augmentation...")
    
    A_current = A.copy()
    b_current = b.copy()
    initial_rows = A.shape[0]
    
    iterations = 0
    total_constraints_added = 0
    cumulative_effective_rows = 0
    iteration_history = []
    
    # Get true nonzero indices once
    true_nonzero_indices = np.where(np.abs(x_true) > 500)[0].tolist()
    true_nonzero_set = set(true_nonzero_indices)
    
    while iterations < max_iterations:
        print(f"   Iteration {iterations + 1}:")
        
        rows_before = A_current.shape[0]
        
        # Apply SCAD to current system
        scad_result = apply_scad_regression(A_current, b_current)
        x_current = scad_result['x_hat']
        predicted_nonzero = scad_result['nonzero_indices']
        predicted_nonzero_set = set(predicted_nonzero)
        
        print(f"     Current non-zero coefficients: {len(predicted_nonzero)}")
        
        # Check convergence (perfect accuracy)
        errors = np.abs(x_current - x_true)
        error_locations = np.where(errors > 500)[0]
        accuracy = 1.0 - len(error_locations) / len(x_true)
        
        print(f"     Current accuracy: {accuracy:.4f}")
        
        # Calculate precision, recall, f0.5 for detailed logging
        if detailed_logging:
            tp = len(true_nonzero_set & predicted_nonzero_set)
            fp = len(predicted_nonzero_set - true_nonzero_set)
            fn = len(true_nonzero_set - predicted_nonzero_set)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            beta = 0.5
            denom = beta**2 * precision + recall
            f05_score = (1 + beta**2) * precision * recall / denom if denom > 0 else 0.0
            
            # Record error edge details (x_ori, x_hat, parent, child)
            error_details = []
            for edge_idx in error_locations[:20]:  # Limit to first 20 for log readability
                edge_idx_int = int(edge_idx)
                parent, child = tree.edges[edge_idx_int]
                error_details.append({
                    'edge_idx': edge_idx_int,
                    'parent': parent,
                    'child': child,
                    'x_true': float(x_true[edge_idx]),
                    'x_hat': float(x_current[edge_idx]),
                    'abs_error': float(abs(x_current[edge_idx] - x_true[edge_idx]))
                })
            
            iteration_info = {
                'iteration': iterations + 1,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f05_score': f05_score,
                'error_edges': error_locations.tolist(),
                'error_details': error_details,  # x_ori, x_hat for each error edge
                'error_count': len(error_locations),
                'predicted_nonzero_count': len(predicted_nonzero),
                'optimal_lambda': scad_result['optimal_lambda'],
                'selection_reason': scad_result['selection_reason'],
                'matrix_shape': A_current.shape,
                'cumulative_effective_rows': A_current.shape[0] - initial_rows
            }
        else:
            iteration_info = {
                'iteration': iterations + 1,
                'accuracy': accuracy,
                'error_count': len(error_locations),
                'predicted_nonzero_count': len(predicted_nonzero),
                'optimal_lambda': scad_result['optimal_lambda'],
                'selection_reason': scad_result['selection_reason']
            }
        
        # Check if perfect accuracy achieved
        if accuracy == 1.0:
            print(f"     Perfect accuracy achieved, stopping iteration")
            if detailed_logging:
                iteration_info['constraints_added_this_iter'] = 0
                iteration_info['effective_rows_added'] = 0
                iteration_info['subhigh_nodes_count'] = 0
                iteration_info['high_nodes_count'] = 0
            iteration_history.append(iteration_info)
            break
        
        # Augment system with confidence-based constraints (with node counts)
        aug_result = augment_system_confidence(
            A_current, b_current, tree, predicted_nonzero, x_true, add_high,
            return_node_counts=True
        )
        A_augmented, b_augmented, n_new_constraints, node_counts = aug_result
        
        effective_rows_added = A_augmented.shape[0] - rows_before
        
        if detailed_logging:
            iteration_info['constraints_added_this_iter'] = n_new_constraints
            iteration_info['effective_rows_added'] = effective_rows_added
            iteration_info['subhigh_nodes_count'] = node_counts['subhigh_nodes_count']
            iteration_info['high_nodes_count'] = node_counts['high_nodes_count']
        
        iteration_history.append(iteration_info)
        
        if n_new_constraints == 0:
            print(f"     No new constraints to add, stopping iteration")
            break
        
        print(f"     Added {n_new_constraints} constraints (RREF: {effective_rows_added} effective), new matrix shape: {A_augmented.shape}")
        
        A_current = A_augmented
        b_current = b_augmented
        total_constraints_added += n_new_constraints
        cumulative_effective_rows = A_current.shape[0] - initial_rows
        iterations += 1
    
    # Final SCAD result
    final_result = apply_scad_regression(A_current, b_current)
    
    print(f"   Final results after {iterations} iterations:")
    print(f"   Total constraints added: {total_constraints_added}")
    print(f"   Effective rows added (RREF): {A_current.shape[0] - initial_rows}")
    print(f"   Final matrix shape: {A_current.shape}")
    print(f"   Final optimal lambda: {final_result['optimal_lambda']:.6e}")
    print(f"   Final selection reason: {final_result['selection_reason']}")
    print(f"   Final non-zero coefficients: {len(final_result['nonzero_indices'])}")
    print(f"   Early stopped: {final_result['early_stopped']}")
    
    # Add iteration info
    final_result['iterations'] = iterations
    final_result['total_constraints_added'] = total_constraints_added
    final_result['final_effective_rows'] = A_current.shape[0] - initial_rows
    final_result['initial_rows'] = initial_rows
    final_result['final_rows'] = A_current.shape[0]
    final_result['iteration_history'] = iteration_history
    final_result['augmentation_type'] = 'confidence_based'
    
    return final_result