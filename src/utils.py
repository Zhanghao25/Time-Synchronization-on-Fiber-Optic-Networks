#!/usr/bin/env python3
"""
Utility Functions for Hierarchical Network Regression

Contains data loading, synthetic data generation, evaluation metrics,
and result saving utilities.
"""

import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from sklearn.preprocessing import LabelEncoder
import warnings


def load_network_data(file_path: Path, columns: List[int] = [1, 2]) -> Tuple[Optional[nx.Graph], Optional[int]]:
    """
    Load hierarchical network data from Excel file (topology only: source, target).
    
    Args:
        file_path: Path to Excel data file
        columns: Column indices [source, target]; edge value is set to 0 (used with generate_synthetic_data).
        
    Returns:
        Tuple containing (NetworkX graph object, source node identifier)
    """
    try:
        df = pd.read_excel(file_path, usecols=columns)
        if df.empty:
            raise ValueError("Excel file contains no data")
        df.columns = ['source', 'target']
        df['value'] = 0
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Cannot find file: {file_path}")
    except Exception as e:
        raise ValueError(f"Cannot read file {file_path}: {e}")
    
    # Encode node labels to integers
    label_encoder = LabelEncoder()
    all_nodes = pd.concat([df['source'], df['target']])
    label_encoder.fit(all_nodes)
    
    df['source'] = label_encoder.transform(df['source'])
    df['target'] = label_encoder.transform(df['target'])
    
    # Create weighted graph
    graph = nx.from_pandas_edgelist(df, edge_attr="value")
    
    # Identify source node (appears in source column but not target column)
    source_candidates = set(df["source"]) - set(df["target"])
    
    if len(source_candidates) == 1:
        source = source_candidates.pop()
    elif len(source_candidates) == 0:
        warnings.warn("No unique source node found (graph may be cyclic)")
        source = None
    else:
        warnings.warn(f"Multiple source candidates found: {source_candidates}")
        source = min(source_candidates)  # Choose minimum as default
    
    return graph, source


def generate_synthetic_data(A: np.ndarray, ratio: float, seed: int, 
                          min_val: float = 5000.0) -> Tuple[np.ndarray, np.ndarray, 
                                                           np.ndarray, List[int]]:
    """
    Generate synthetic sparse data for regression experiments.
    
    Args:
        A: Design matrix
        ratio: Sparsity ratio (fraction of non-zero coefficients)
        seed: Random seed for reproducibility
        min_val: Minimum magnitude for non-zero coefficients
        
    Returns:
        Tuple containing (true coefficients, clean coefficients, observations, nonzero indices)
    """
    np.random.seed(seed)
    n_features = A.shape[1]
    
    # Initialize sparse coefficient vector
    x = np.zeros(n_features)
    
    # Select non-zero indices
    n_nonzero = int(n_features * ratio)
    if n_nonzero == 0:
        n_nonzero = 1  # Ensure at least one non-zero coefficient
    
    non_zero_indices = np.random.choice(n_features, size=n_nonzero, replace=False)
    
    # Ensure non_zero_indices is always a list for consistency
    if n_nonzero == 1:
        non_zero_indices = [non_zero_indices.item()]
    else:
        non_zero_indices = non_zero_indices.tolist()
    
    # Generate non-zero values with controlled magnitude
    signal_strength = 2000000.0 / 3.0
    random_values = np.random.normal(loc=0, scale=signal_strength, size=n_nonzero)
    
    # Ensure minimum magnitude requirement
    insufficient_magnitude = np.abs(random_values) < min_val
    while insufficient_magnitude.any():
        n_resample = insufficient_magnitude.sum()
        random_values[insufficient_magnitude] = np.random.normal(
            loc=0, scale=signal_strength, size=n_resample
        )
        insufficient_magnitude = np.abs(random_values) < min_val
    
    # Assign non-zero values
    x[non_zero_indices] = random_values
    x_raw = x.copy()  # Clean version without noise
    
    # Add small noise to coefficients
    coefficient_noise_scale = 20.0 / 3.0
    x += np.random.normal(loc=0, scale=coefficient_noise_scale, size=n_features)
    
    # Generate observations with noise
    observation_noise_scale = 100.0 / 3.0
    b = A @ x + np.random.normal(loc=0, scale=observation_noise_scale, size=A.shape[0])
    
    return x, x_raw, b, non_zero_indices


class Evaluator:
    """Evaluation framework for regression performance assessment."""
    
    def __init__(self, ground_truth: np.ndarray, true_nonzero_indices: List[int], 
                 threshold: float = 500.0) -> None:
        """Initialize evaluator with ground truth data."""
        self.ground_truth = ground_truth.copy()
        self.threshold = threshold
        self.true_nonzero_indices = set(true_nonzero_indices)
        
        # Validate inputs
        if len(ground_truth) == 0:
            raise ValueError("Ground truth vector cannot be empty")
        
        if not all(0 <= idx < len(ground_truth) for idx in true_nonzero_indices):
            raise ValueError("Invalid indices in true_nonzero_indices")
    
    def evaluate_accuracy(self, predictions: np.ndarray) -> Dict[str, Any]:
        """Evaluate prediction accuracy based on absolute error threshold."""
        if len(predictions) != len(self.ground_truth):
            raise ValueError("Predictions and ground truth must have same length")
        
        absolute_errors = np.abs(predictions - self.ground_truth)
        error_locations = np.where(absolute_errors > self.threshold)[0].tolist()
        accuracy = 1.0 - len(error_locations) / len(self.ground_truth)
        
        return {
            "accuracy": accuracy,
            "error_locations": error_locations,
            "n_errors": len(error_locations)
        }
    
    def evaluate_f05_score(self, predictions: np.ndarray, 
                          predicted_nonzero_indices: List[int]) -> Dict[str, float]:
        """Calculate F0.5 score emphasizing precision over recall."""
        predicted_nonzero_set = set(predicted_nonzero_indices)
        
        # Compute confusion matrix elements
        true_positives = len(self.true_nonzero_indices & predicted_nonzero_set)
        false_positives = len(predicted_nonzero_set - self.true_nonzero_indices)
        false_negatives = len(self.true_nonzero_indices - predicted_nonzero_set)
        
        # Calculate precision and recall with safe division
        precision = (true_positives / (true_positives + false_positives) 
                    if (true_positives + false_positives) > 0 else 0.0)
        recall = (true_positives / (true_positives + false_negatives) 
                 if (true_positives + false_negatives) > 0 else 0.0)
        
        # Calculate F0.5 score (β = 0.5, emphasizes precision)
        beta = 0.5
        denominator = beta**2 * precision + recall
        f05_score = ((1 + beta**2) * precision * recall / denominator 
                    if denominator > 0 else 0.0)
        
        return {
            "precision": precision,
            "recall": recall,
            "f05_score": f05_score,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives
        }
    
    def evaluate_all_metrics(self, predictions: np.ndarray, 
                           predicted_nonzero_indices: List[int]) -> Dict[str, Any]:
        """Evaluate all metrics at once."""
        accuracy_metrics = self.evaluate_accuracy(predictions)
        f05_metrics = self.evaluate_f05_score(predictions, predicted_nonzero_indices)
        
        return {**accuracy_metrics, **f05_metrics}
