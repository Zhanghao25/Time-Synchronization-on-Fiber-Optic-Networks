#!/usr/bin/env python3
"""
Baseline regularization methods: Lasso (L1), L0, and MCP.

These are the comparison methods reported in Tables S.2-S.5. All three are
solved through R backends via rpy2:

- Lasso (L1), L0, and MCP regularization baselines
- Wasserstein-based hyperparameter selection workflow
"""

import numpy as np
from typing import Dict, Any

from scipy.stats import wasserstein_distance

# R integration for regularization methods
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
from rpy2.robjects.packages import importr

# Initialize R environment
rpy2.robjects.numpy2ri.activate()

try:
    ncvreg = importr("ncvreg")
    l0learn = importr("L0Learn")
    print("R packages (ncvreg, L0Learn) loaded successfully")
except Exception as e:
    print(f"Warning: R packages not available - {e}")
    ncvreg = None
    l0learn = None


def lasso_regression_simple(A: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
    """
    Lasso regression using traditional path selection (99th point).
    """
    if ncvreg is None:
        raise RuntimeError("ncvreg R package not available")

    # Fit Lasso using ncvreg
    lasso_model = ncvreg.ncvreg(
        A, list(b),
        family="gaussian",
        penalty="lasso",
        lambda_min=0.00001
    )

    # Extract coefficient matrix and lambda sequence
    model_dict = dict(lasso_model.items())
    beta_matrix = model_dict["beta"]
    lambda_sequence = model_dict["lambda"]

    # Use 99th point of regularization path
    coefficient_estimates = beta_matrix[1:, 99]  # Skip intercept, use 99th point
    optimal_lambda = lambda_sequence[99]

    # Identify non-zero coefficients
    nonzero_indices = np.where(np.abs(coefficient_estimates) > 500)[0].tolist()

    return {
        'x_hat': np.squeeze(coefficient_estimates),
        'optimal_lambda': optimal_lambda,
        'nonzero_indices': nonzero_indices,
        'method': 'Lasso'
    }


def l0_regression_simple(A: np.ndarray, b: np.ndarray, sparsity_ratio: float = 0.1) -> Dict[str, Any]:
    """
    L0 regression with coordinate descent.
    """
    if l0learn is None:
        raise RuntimeError("L0Learn R package not available")

    # Set L0 parameters
    l0_fit = l0learn.L0Learn_fit(
        A, b,
        penalty="L0",
        maxSuppSize=int(sparsity_ratio * min(A.shape)),
        intercept=False,
        maxIters=200,
        nLambda=int(0.2 * min(A.shape)),
        algorithm="CD"
    )

    # Extract solution (use last lambda)
    fit_result = dict(l0_fit.items())
    optimal_lambda = np.asarray(fit_result["lambda"])[0][-1]

    # Get coefficient estimates
    coef_function = robjects.r["coef"]
    coefficient_matrix = robjects.r["as.matrix"](
        coef_function(l0_fit, **{"lambda": optimal_lambda})
    )
    coefficient_estimates = np.squeeze(coefficient_matrix)

    # Identify non-zero coefficients
    nonzero_indices = np.where(np.abs(coefficient_estimates) > 500)[0].tolist()

    return {
        'x_hat': coefficient_estimates,
        'optimal_lambda': optimal_lambda,
        'nonzero_indices': nonzero_indices,
        'method': 'L0'
    }


def mcp_regression_wasserstein(A: np.ndarray, b: np.ndarray, gamma: float = 3.0) -> Dict[str, Any]:
    """
    MCP regression with Wasserstein distance lambda selection (same as SCAD).
    """
    if ncvreg is None:
        raise RuntimeError("ncvreg R package not available")

    # Generate lambda sequence (same as SCAD)
    lambda_max = np.max(np.abs(np.dot(A.T, b))) / A.shape[0]
    lambda_min = 1e-6 * lambda_max

    # Two-phase lambda sequence
    high_lambdas = np.logspace(np.log10(lambda_max), np.log10(0.1), num=10, endpoint=False)
    low_lambdas = np.logspace(np.log10(0.1), np.log10(lambda_min), num=90, endpoint=True)
    lambda_sequence = np.concatenate([high_lambdas, low_lambdas])

    # Initialize MCP parameters
    initial_beta = np.zeros(A.shape[1])
    xtx = np.sum(A ** 2, axis=0) / A.shape[0]
    penalty_factor = np.ones(A.shape[1])

    # Solve MCP for each lambda with Wasserstein monitoring
    beta_solutions = {}
    wasserstein_distances = []
    convergence_indicators = []

    for i, lam in enumerate(lambda_sequence):
        # Fit MCP model
        mcp_model = ncvreg.ncvfit(
            X=A, y=list(b), init=initial_beta, xtx=xtx,
            penalty="MCP", gamma=gamma, penalty_factor=penalty_factor,
            **{'lambda': robjects.FloatVector([lam])}
        )

        current_beta = np.array(mcp_model.rx2("beta")).flatten()
        initial_beta = current_beta  # Warm start
        beta_solutions[lam] = current_beta

        # Monitor convergence via Wasserstein distance
        if i > 0:
            previous_beta = beta_solutions[lambda_sequence[i-1]]
            distance = wasserstein_distance(previous_beta, current_beta)
            wasserstein_distances.append(distance)

            has_converged = distance < 1e-1
            convergence_indicators.append(has_converged)

            # Early stopping check
            if len(convergence_indicators) >= 5:
                if all(convergence_indicators[-5:]):
                    break

    # Select optimal lambda using Wasserstein criterion
    optimal_idx = 0
    selection_reason = "Default selection"

    if len(convergence_indicators) >= 5:
        # Look for sustained convergence
        for i in range(len(convergence_indicators) - 4):
            if all(convergence_indicators[i:i+5]):
                optimal_idx = i + 1
                selection_reason = "Sustained convergence (Wasserstein)"
                break

        if selection_reason == "Default selection" and wasserstein_distances:
            optimal_idx = np.argmin(wasserstein_distances) + 1
            selection_reason = "Minimum Wasserstein distance"

    # Extract optimal solution
    optimal_lambda = lambda_sequence[optimal_idx]
    x_hat = beta_solutions[optimal_lambda]

    # Identify non-zero coefficients
    nonzero_indices = np.where(np.abs(x_hat) > 500)[0].tolist()

    return {
        'x_hat': x_hat,
        'optimal_lambda': optimal_lambda,
        'nonzero_indices': nonzero_indices,
        'selection_reason': selection_reason,
        'method': 'MCP'
    }
