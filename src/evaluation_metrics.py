#!/usr/bin/env python3
"""
Evaluation Metrics Module for Hierarchical Network Regression

This module provides comprehensive evaluation metrics for assessing
the performance of sparse regression methods in hierarchical networks.

Key Features:
- Accuracy based on absolute error threshold
- Precision, Recall, and F0.5 score for sparsity pattern recognition
- Coverage accuracy for selected coefficient subsets
- Comprehensive result evaluation and comparison
"""

import numpy as np
from typing import Dict, List, Union, Any


class Evaluator:
    """
    Comprehensive evaluation framework for hierarchical network regression.
    
    This class provides standardized metrics for evaluating the quality of
    sparse regression results, focusing on accuracy, precision, recall, and F0.5 scores
    tailored for hierarchical network analysis.
    
    Attributes:
        ground_truth (np.ndarray): True coefficient values
        true_nonzero_indices (set): Indices of true non-zero coefficients
        threshold (float): Threshold for coefficient significance determination
    """
    
    def __init__(self, ground_truth: np.ndarray, true_nonzero_indices: List[int], 
                 threshold: float = 500.0) -> None:
        """
        Initialize evaluator with ground truth data.
        
        Args:
            ground_truth (np.ndarray): True coefficient vector
            true_nonzero_indices (List[int]): Indices where coefficients are truly non-zero
            threshold (float): Absolute error threshold for accuracy determination
            
        Raises:
            ValueError: If ground truth is empty or indices are invalid
        """
        self.ground_truth = ground_truth.copy()
        self.threshold = threshold
        self.true_nonzero_indices = set(true_nonzero_indices)
        
        # Validate inputs
        if len(ground_truth) == 0:
            raise ValueError("Ground truth vector cannot be empty")
        
        if not all(0 <= idx < len(ground_truth) for idx in true_nonzero_indices):
            raise ValueError("Invalid indices in true_nonzero_indices")
    
    def evaluate_accuracy(self, predictions: np.ndarray) -> Dict[str, Union[float, List[int]]]:
        """
        Evaluate prediction accuracy based on absolute error threshold.
        
        Args:
            predictions (np.ndarray): Predicted coefficient vector
            
        Returns:
            Dict containing accuracy score and error locations
            
        Raises:
            ValueError: If prediction length doesn't match ground truth
        """
        if len(predictions) != len(self.ground_truth):
            raise ValueError("Predictions and ground truth must have same length")
        
        absolute_errors = np.abs(predictions - self.ground_truth)
        error_locations = np.where(absolute_errors > self.threshold)[0].tolist()
        accuracy = 1.0 - len(error_locations) / len(self.ground_truth)
        
        return {
            "accuracy": accuracy,
            "error_locations": error_locations,
            "error_count": len(error_locations),
            "total_coefficients": len(self.ground_truth)
        }
    
    def evaluate_sparsity_pattern(self, predictions: np.ndarray, 
                                 predicted_nonzero_indices: List[int]) -> Dict[str, float]:
        """
        Calculate precision, recall, and F0.5 score for sparsity pattern recognition.
        
        Args:
            predictions (np.ndarray): Predicted coefficient vector
            predicted_nonzero_indices (List[int]): Predicted non-zero indices
            
        Returns:
            Dict containing precision, recall, F0.5 score, and confusion matrix elements
        """
        predicted_nonzero_set = set(predicted_nonzero_indices)
        
        # Compute confusion matrix elements
        true_positives = len(self.true_nonzero_indices & predicted_nonzero_set)
        false_positives = len(predicted_nonzero_set - self.true_nonzero_indices)
        false_negatives = len(self.true_nonzero_indices - predicted_nonzero_set)
        true_negatives = len(self.ground_truth) - true_positives - false_positives - false_negatives
        
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
        
        # Calculate F1 score for comparison
        f1_denominator = precision + recall
        f1_score = (2 * precision * recall / f1_denominator 
                   if f1_denominator > 0 else 0.0)
        
        return {
            "precision": precision,
            "recall": recall,
            "f05_score": f05_score,
            "f1_score": f1_score,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "true_negatives": true_negatives
        }
    
    def evaluate_f05_score(self, predictions: np.ndarray,
                           predicted_nonzero_indices: List[int]) -> Dict[str, float]:
        """
        Calculate F0.5 score emphasizing precision over recall.

        Returns the core sparsity-pattern metrics (precision, recall, F0.5,
        and confusion matrix counts) used by the TSLE result workbooks.
        """
        sparsity = self.evaluate_sparsity_pattern(predictions, predicted_nonzero_indices)
        return {
            "precision": sparsity["precision"],
            "recall": sparsity["recall"],
            "f05_score": sparsity["f05_score"],
            "true_positives": sparsity["true_positives"],
            "false_positives": sparsity["false_positives"],
            "false_negatives": sparsity["false_negatives"],
        }

    def evaluate_all_metrics(self, predictions: np.ndarray,
                             predicted_nonzero_indices: List[int]) -> Dict[str, Any]:
        """
        Evaluate accuracy and sparsity-pattern metrics at once.

        Result keys ("accuracy", "error_locations", "n_errors", precision/recall/F0.5
        and confusion counts) match the columns of the shipped result workbooks.
        """
        accuracy_metrics = self.evaluate_accuracy(predictions)
        f05_metrics = self.evaluate_f05_score(predictions, predicted_nonzero_indices)

        return {
            "accuracy": accuracy_metrics["accuracy"],
            "error_locations": accuracy_metrics["error_locations"],
            "n_errors": accuracy_metrics["error_count"],
            **f05_metrics,
        }

    def evaluate_coverage(self, predictions: np.ndarray, 
                         selected_indices: List[int]) -> Dict[str, Union[float, List[int]]]:
        """
        Calculate coverage accuracy for a selected subset of coefficients.
        
        Args:
            predictions (np.ndarray): Predicted coefficient vector
            selected_indices (List[int]): Indices of selected coefficients to evaluate
            
        Returns:
            Dict containing coverage accuracy and subset error information
            
        Raises:
            ValueError: If indices are invalid
        """
        if len(selected_indices) == 0:
            return {
                "coverage_accuracy": 1.0, 
                "subset_error_locations": [],
                "subset_error_count": 0,
                "subset_size": 0
            }
        
        if not all(0 <= idx < len(predictions) for idx in selected_indices):
            raise ValueError("Invalid indices for coverage evaluation")
            
        # Extract subset predictions and ground truth
        subset_predictions = predictions[selected_indices]
        subset_ground_truth = self.ground_truth[selected_indices]
        
        # Calculate errors in subset
        absolute_errors = np.abs(subset_predictions - subset_ground_truth)
        subset_error_mask = absolute_errors > self.threshold
        subset_error_locations = np.where(subset_error_mask)[0].tolist()
        
        # Map back to original indices
        original_error_locations = [selected_indices[i] for i in subset_error_locations]
        
        coverage_accuracy = 1.0 - len(subset_error_locations) / len(selected_indices)
        
        return {
            "coverage_accuracy": coverage_accuracy,
            "subset_error_locations": original_error_locations,
            "subset_error_count": len(subset_error_locations),
            "subset_size": len(selected_indices)
        }
    
    def comprehensive_evaluation(self, predictions: np.ndarray, 
                               predicted_nonzero_indices: List[int]) -> Dict[str, Any]:
        """
        Perform comprehensive evaluation including all metrics.
        
        Args:
            predictions (np.ndarray): Predicted coefficient vector
            predicted_nonzero_indices (List[int]): Predicted non-zero indices
            
        Returns:
            Dict containing all evaluation metrics
        """
        # Basic accuracy evaluation
        accuracy_results = self.evaluate_accuracy(predictions)
        
        # Sparsity pattern evaluation
        sparsity_results = self.evaluate_sparsity_pattern(predictions, predicted_nonzero_indices)
        
        # Coverage evaluation for predicted non-zero subset
        coverage_results = self.evaluate_coverage(predictions, predicted_nonzero_indices)
        
        # Additional statistics
        prediction_stats = {
            "prediction_norm": float(np.linalg.norm(predictions)),
            "ground_truth_norm": float(np.linalg.norm(self.ground_truth)),
            "relative_error": float(np.linalg.norm(predictions - self.ground_truth) / np.linalg.norm(self.ground_truth)),
            "max_absolute_error": float(np.max(np.abs(predictions - self.ground_truth))),
            "mean_absolute_error": float(np.mean(np.abs(predictions - self.ground_truth))),
            "predicted_sparsity": len(predicted_nonzero_indices) / len(predictions),
            "true_sparsity": len(self.true_nonzero_indices) / len(self.ground_truth)
        }
        
        # Combine all results
        comprehensive_results = {
            **accuracy_results,
            **sparsity_results,
            **coverage_results,
            **prediction_stats
        }
        
        return comprehensive_results
