"""
PyReCo Model Wrapper

This module provides a wrapper for PyReCo's ReservoirComputer that inherits
from BaseTimeSeriesModel, enabling unified interface and fair comparison.

Supports:
- Standard PyReCo API (pyreco.models.ReservoirComputer)
- Hyperparameter tuning with cross-validation
- Multiple optimizer options

Author: Beginner-friendly version with English comments
"""

import time
import numpy as np
from typing import Dict, Any, List, Tuple
from pyreco.models import ReservoirComputer as RC

from .base_model import BaseTimeSeriesModel


class PyReCoStandardModel(BaseTimeSeriesModel):
    """
    Standard PyReCo Model Wrapper

    This wrapper uses PyReCo's standard API (pyreco.models.ReservoirComputer)
    and provides the same interface as other time series models.

    Key hyperparameters:
    - num_nodes: Number of reservoir nodes (default: 100)
    - spec_rad: Spectral radius of reservoir weights (default: 0.9)
    - leakage_rate: Leakage rate for leaky integrator neurons (default: 0.3)
    - input_scaling: Scaling factor for input weights (default: 1.0)
    - bias_scaling: Scaling factor for bias terms (default: 0.0)
    - optimizer: Readout training method ('ridge', 'pinv', etc.)
    - ridge_alpha: Regularization parameter for ridge regression (default: 1e-5)

    Example:
        model = PyReCoStandardModel(num_nodes=200, spec_rad=0.95)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        results = model.evaluate(X_test, y_test)
    """

    def __init__(self,
                 num_nodes: int = 100,
                 density: float = 0.8,
                 activation: str = 'tanh',
                 spec_rad: float = 0.9,
                 leakage_rate: float = 0.5,
                 fraction_input: float = 1.0,
                 fraction_output: float = 1.0,
                 optimizer: str = 'ridge',
                 verbose: bool = True):
        """
        Initialize PyReCo standard model

        Args:
            num_nodes: Number of reservoir nodes
            density: Density of reservoir connections (0-1)
            activation: Activation function ('tanh', 'relu', etc.)
            spec_rad: Spectral radius (controls reservoir dynamics)
            leakage_rate: Leakage rate (0=no memory, 1=full memory)
            fraction_input: Fraction of nodes EXCLUDED from input (0-1, PyReCo convention)
            fraction_output: Fraction of nodes connected to output (0-1)
            optimizer: Optimizer for readout training ('ridge', 'pinv', etc.)
            verbose: Whether to print training information
        """
        # Call parent class initialization
        super().__init__(
            name="PyReCo-Standard",
            config={
                'num_nodes': num_nodes,
                'density': density,
                'activation': activation,
                'spec_rad': spec_rad,
                'leakage_rate': leakage_rate,
                'fraction_input': fraction_input,
                'fraction_output': fraction_output,
                'optimizer': optimizer,
            }
        )

        self.verbose = verbose

        # Create PyReCo model
        self.model = RC(
            num_nodes=num_nodes,
            density=density,
            activation=activation,
            spec_rad=spec_rad,
            leakage_rate=leakage_rate,
            fraction_input=fraction_input,
            fraction_output=fraction_output,
            optimizer=optimizer,
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train the PyReCo model

        Args:
            X_train: Training input, shape (n_samples, n_timesteps, n_features)
            y_train: Training target, shape (n_samples, n_output_steps, n_features)
        """
        if self.verbose:
            print(f"    Training {self.name}...")

        start_time = time.time()

        # Train PyReCo model
        self.model.fit(X_train, y_train)

        # Update state
        self.training_time = time.time() - start_time
        self.is_trained = True

        if self.verbose:
            print(f"    Training complete! Took {self.training_time:.2f} seconds")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using PyReCo

        Args:
            X: Input data, shape (n_samples, n_timesteps, n_features)

        Returns:
            Predictions, shape (n_samples, n_output_steps, n_features)
        """
        if not self.is_trained:
            raise RuntimeError(f"{self.name} model is not trained yet!")

        start_time = time.time()

        # Make predictions
        predictions = self.model.predict(X)

        self.prediction_time = time.time() - start_time

        return predictions

    def get_reservoir_states(self, X: np.ndarray) -> np.ndarray:
        """
        Get reservoir states for input data

        This is useful for analyzing what the reservoir learned.

        Args:
            X: Input data, shape (n_samples, n_timesteps, n_features)

        Returns:
            Reservoir states, shape (n_samples, n_timesteps, num_nodes)
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained yet!")

        # PyReCo's internal method to get states
        if hasattr(self.model, 'get_states'):
            return self.model.get_states(X)
        else:
            raise NotImplementedError("This version of PyReCo doesn't support get_states()")


# ============================================================================
# Hyperparameter tuning helper functions
# ============================================================================

def tune_pyreco_hyperparameters(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        param_grid: Dict[str, List],
        default_spec_rad: float = 1.0,
        default_leakage: float = 0.3,
        default_density: float = 0.1,
        default_activation: str = "tanh",
        default_fraction_input: float = 0.5,
        verbose: bool = True) -> Dict[str, Any]:
    """
    Grid search for PyReCo hyperparameters (following train_pyreco_model.py)

    Args:
        X_train: Training input
        y_train: Training target
        X_val: Validation input
        y_val: Validation target
        param_grid: Dictionary of parameters to search, e.g.:
            {
                'num_nodes': [50, 100, 200],
                'spec_rad': [0.5, 0.9, 1.2],
                'leakage_rate': [0.1, 0.3, 0.5],
            }
        default_spec_rad: Default spectral radius (default: 1.0, same as train_pyreco_model.py)
        default_leakage: Default leakage rate (default: 0.3, same as train_pyreco_model.py)
        default_density: Default density (default: 0.1, same as train_pyreco_model.py)
        default_activation: Default activation (default: "tanh")
        default_fraction_input: Default fraction input (default: 0.5, same as train_pyreco_model.py)
        verbose: Whether to print progress

    Returns:
        Dictionary containing:
            - 'best_params': Best parameter combination
            - 'best_score': Best validation MSE
            - 'best_r2': Best validation R²
            - 'all_results': List of all results
    """
    from itertools import product

    # Generate all parameter combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(product(*values))

    best_score = float('inf')
    best_r2 = float('-inf')
    best_params = None
    all_results = []

    if verbose:
        print(f"\n{'='*60}")
        print(f"Starting PyReCo hyperparameter search...")
        print(f"Total combinations: {len(combinations)}")
        print(f"{'='*60}\n")

    for i, combo in enumerate(combinations):
        # using default values first, then override with grid (consistent with train_pyreco_model.py)
        spec_rad = default_spec_rad
        leakage_rate = default_leakage
        density = default_density
        activation = default_activation
        fraction_input = default_fraction_input

        # from grid retrieve other required parameters
        num_nodes = None
        fraction_output = None

        # using grid values to override defaults
        params_dict = dict(zip(keys, combo))
        for k, v in params_dict.items():
            if   k == "spec_rad":         spec_rad = float(v)
            elif k == "leakage_rate":     leakage_rate = float(v)
            elif k == "density":          density = float(v)
            elif k == "activation":       activation = str(v)
            elif k == "fraction_input":   fraction_input = float(v)
            elif k == "num_nodes":        num_nodes = int(v)
            elif k == "fraction_output":  fraction_output = float(v)

        # build the full parameter dictionary (explicit all parameters)
        full_params = {
            'num_nodes': num_nodes,
            'density': density,
            'activation': activation,
            'spec_rad': spec_rad,
            'leakage_rate': leakage_rate,
            'fraction_input': fraction_input,
            'fraction_output': fraction_output,
            'optimizer': 'ridge',
        }

        if verbose:
            print(f"[{i+1}/{len(combinations)}] Testing: {params_dict}")

        try:
            # Create and train model (implicitly passing all parameters)
            model = PyReCoStandardModel(**full_params, verbose=False)
            model.fit(X_train, y_train)

            # Evaluate on validation set
            val_results = model.evaluate(X_val, y_val, metrics=['mse', 'mae', 'r2'])
            val_mse = val_results['mse']
            val_r2 = val_results['r2']

            # Store results
            all_results.append({
                'params': full_params,
                'val_mse': val_mse,
                'val_r2': val_r2,
                'train_time': model.training_time,
                'success': True,
            })

            if verbose:
                print(f"  → Validation MSE: {val_mse:.6f}, R²: {val_r2:.4f}, Train time: {model.training_time:.2f}s")

            # Update best params
            if val_mse < best_score:
                best_score = val_mse
                best_r2 = val_r2
                best_params = full_params

        except Exception as e:
            if verbose:
                print(f"  ✗ Failed: {str(e)}")

            all_results.append({
                'params': full_params,
                'val_mse': float('inf'),
                'val_r2': float('-inf'),
                'train_time': 0.0,
                'success': False,
                'error': str(e),
            })

        if verbose:
            print()

    # Explicit garbage collection to free memory from temporary models
    import gc
    gc.collect()

    if verbose:
        print(f"{'='*60}")
        print(f"Hyperparameter search complete!")
        print(f"Best params: {best_params}")
        print(f"Best validation MSE: {best_score:.6f}")
        print(f"{'='*60}\n")

    return {
        'best_params': best_params,
        'best_score': best_score,
        'best_r2': best_r2,
        'all_results': all_results,
    }


def tune_pyreco_with_cv(
        raw_data: np.ndarray,
        param_grid: Dict[str, List],
        n_splits: int = 5,
        n_in: int = 100,
        n_out: int = 1,
        budget: int = None,
        max_nodes: int = 1000,
        # default values system (consistent with train_pyreco_model.py)
        default_spec_rad: float = 0.9,
        default_leakage: float = 0.5,
        default_density: float = 0.05,
        default_activation: str = "tanh",
        default_fraction_input: float = 0.3,
        verbose: bool = True) -> Dict[str, Any]:
    """
    Hyperparameter tuning with time series cross-validation.

    CV splits the RAW time series BEFORE creating sliding windows to avoid
    data leakage from overlapping windows (adjacent windows share n_in-1
    timesteps). Each fold independently standardizes and creates windows.

    If `budget` is set, num_nodes is dynamically computed for each
    (density, fraction_input) combination to satisfy the total parameter
    budget constraint: N^2*density + N*d_in*(1-fraction_input) + N*d_out = budget.
    Note: PyReCo's fraction_input = fraction of nodes EXCLUDED from input.

    Args:
        raw_data: Raw time series, shape (n_timesteps, n_features).
                  Should be the train+val portion (test excluded).
        param_grid: Parameter grid to search. May include density,
                    fraction_input, spec_rad, leakage_rate, num_nodes.
                    If budget is set, num_nodes in grid is ignored.
        n_splits: Number of forward-chaining CV splits (default: 5)
        n_in: Input window size (default: 100)
        n_out: Output window size (default: 1)
        budget: Total parameter budget. If set, num_nodes is computed
                dynamically per (density, fraction_input). If None,
                num_nodes must be in param_grid.
        max_nodes: Maximum allowed num_nodes (default: 1000)
        default_*: Default values for parameters not in param_grid
        verbose: Whether to print progress

    Returns:
        Dictionary containing:
            - 'best_params': Best parameter combination
            - 'best_score': Best CV mean MSE
            - 'best_r2': Best CV mean R²
            - 'cv_std': CV std of best combination
            - 'all_results': List of all results
    """
    from itertools import product as iterproduct
    from sklearn.preprocessing import StandardScaler

    d_in = raw_data.shape[1] if raw_data.ndim > 1 else 1
    d_out = d_in

    def _sliding_window(data, n_in, n_out):
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        T = len(data)
        N = T - n_in - n_out + 1
        if N <= 0:
            return np.array([]), np.array([])
        X = np.stack([data[i:i+n_in] for i in range(N)])
        Y = np.stack([data[i+n_in:i+n_in+n_out] for i in range(N)])
        return X, Y

    # Build CV splits on RAW time series (no window overlap leakage)
    n_total = len(raw_data)
    chunk_size = n_total // (n_splits + 1)
    min_len = n_in + n_out

    cv_splits = []
    for k in range(n_splits):
        train_end = (k + 1) * chunk_size
        val_end = min(train_end + chunk_size, n_total)
        if train_end < min_len or val_end - train_end < min_len:
            continue
        raw_train = raw_data[:train_end]
        raw_val = raw_data[train_end:val_end]
        scaler = StandardScaler()
        raw_train_s = scaler.fit_transform(raw_train)
        raw_val_s = scaler.transform(raw_val)
        X_tr, y_tr = _sliding_window(raw_train_s, n_in, n_out)
        X_va, y_va = _sliding_window(raw_val_s, n_in, n_out)
        if len(X_tr) > 0 and len(X_va) > 0:
            cv_splits.append((X_tr, y_tr, X_va, y_va))

    if verbose:
        print(f"\n{'='*60}")
        print(f"PyReCo hyperparameter tuning with {len(cv_splits)}-fold CV")
        if budget:
            print(f"Budget-constrained: {budget:,} total params, max_nodes={max_nodes}")
        print(f"{'='*60}\n")

    # Generate all parameter combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(iterproduct(*values))

    best_score = float('inf')
    best_r2 = float('-inf')
    best_params = None
    best_cv_std = float('inf')
    all_results = []

    for i, combo in enumerate(combinations):
        # Start from defaults, override with grid values
        spec_rad = default_spec_rad
        leakage_rate = default_leakage
        density = default_density
        activation = default_activation
        fraction_input = default_fraction_input
        num_nodes = None
        fraction_output = 1.0

        params_dict = dict(zip(keys, combo))
        for k, v in params_dict.items():
            if   k == "spec_rad":         spec_rad = float(v)
            elif k == "leakage_rate":     leakage_rate = float(v)
            elif k == "density":          density = float(v)
            elif k == "activation":       activation = str(v)
            elif k == "fraction_input":   fraction_input = float(v)
            elif k == "num_nodes":        num_nodes = int(v)
            elif k == "fraction_output":  fraction_output = float(v)

        # Dynamic N from budget constraint
        if budget is not None:
            from src.utils.budget_matching import esn_solve_num_nodes, esn_total_params
            num_nodes = esn_solve_num_nodes(budget, density, fraction_input,
                                            d_in, d_out, max_nodes=max_nodes)
            if num_nodes is None:
                continue  # Skip infeasible (N > max_nodes)

        if num_nodes is None:
            if verbose:
                print(f"  Skipping combo {i+1}: no num_nodes and no budget")
            continue

        full_params = {
            'num_nodes': num_nodes,
            'density': density,
            'activation': activation,
            'spec_rad': spec_rad,
            'leakage_rate': leakage_rate,
            'fraction_input': fraction_input,
            'fraction_output': fraction_output,
            'optimizer': 'ridge',
        }

        if verbose and (i + 1) % 10 == 0:
            print(f"[{i+1}/{len(combinations)}] N={num_nodes} d={density} "
                  f"fi={fraction_input} sr={spec_rad} leak={leakage_rate}",
                  flush=True)

        # Evaluate on all CV folds
        fold_mse_scores = []
        fold_r2_scores = []

        for fold_idx, (X_tr, y_tr, X_val, y_val) in enumerate(cv_splits):
            try:
                model = PyReCoStandardModel(**full_params, verbose=False)
                model.fit(X_tr, y_tr)
                val_results = model.evaluate(X_val, y_val, metrics=['mse', 'r2'])
                fold_mse_scores.append(val_results['mse'])
                fold_r2_scores.append(val_results['r2'])
            except Exception as e:
                if verbose:
                    print(f"  Fold {fold_idx+1} failed: {str(e)}")
                fold_mse_scores.append(float('inf'))
                fold_r2_scores.append(float('-inf'))

        cv_mean = np.mean(fold_mse_scores)
        cv_std = np.std(fold_mse_scores)
        cv_r2_mean = np.mean(fold_r2_scores)
        cv_r2_std = np.std(fold_r2_scores)

        # Compute param_info if budget is set
        param_info = None
        if budget is not None:
            from src.utils.budget_matching import esn_total_params
            param_info = esn_total_params(num_nodes, density, fraction_input, d_in, d_out)

        all_results.append({
            'params': full_params,
            'param_info': param_info,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'cv_r2_mean': cv_r2_mean,
            'cv_r2_std': cv_r2_std,
            'fold_mse_scores': fold_mse_scores,
            'fold_r2_scores': fold_r2_scores,
        })

        if cv_mean < best_score:
            best_score = cv_mean
            best_r2 = cv_r2_mean
            best_cv_std = cv_std
            best_params = full_params

    if verbose:
        print(f"\n{'='*60}")
        print(f"CV tuning complete!")
        if best_params:
            print(f"Best: N={best_params['num_nodes']} d={best_params['density']} "
                  f"fi={best_params['fraction_input']} sr={best_params['spec_rad']} "
                  f"leak={best_params['leakage_rate']}")
            print(f"Best CV MSE: {best_score:.6f} ± {best_cv_std:.6f}")
        print(f"{'='*60}\n")

    return {
        'best_params': best_params,
        'best_score': best_score,
        'best_r2': best_r2,
        'cv_std': best_cv_std,
        'all_results': all_results,
    }
