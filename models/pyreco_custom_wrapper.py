"""
PyReCo Custom Model Wrapper

This module provides a wrapper for PyReCo's custom_models API
that inherits from BaseTimeSeriesModel.

The custom API allows layer-by-layer construction for advanced users.

Author: Beginner-friendly version with English comments
"""

import time
import numpy as np
from typing import Dict, Any

from pyreco.custom_models import RC
from pyreco.layers import InputLayer, RandomReservoirLayer, ReadoutLayer

from .base_model import BaseTimeSeriesModel


class PyReCoCustomModel(BaseTimeSeriesModel):
    """
    PyReCo Custom Model Wrapper

    This wrapper uses PyReCo's custom API (pyreco.custom_models.RC)
    which allows layer-by-layer construction of the reservoir computer.

    Useful when you need:
    - Multiple reservoir layers
    - Custom layer configurations
    - More control over architecture

    Example:
        model = PyReCoCustomModel(
            num_nodes=100,
            spec_rad=0.9,
            leakage_rate=0.3
        )
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
    """

    def __init__(self,
                 num_nodes: int = 100,
                 density: float = 0.05,
                 activation: str = 'tanh',
                 spec_rad: float = 0.95,
                 leakage_rate: float = 0.3,
                 fraction_output: float = 1.0,
                 optimizer: str = 'ridge',
                 discard_transients: int = 20,
                 seed: int = None,
                 verbose: bool = True):
        """
        Initialize PyReCo custom model

        Args:
            num_nodes: Number of reservoir nodes
            density: Density of reservoir connections (0-1)
            activation: Activation function ('tanh', 'relu', etc.)
            spec_rad: Spectral radius
            leakage_rate: Leakage rate
            fraction_output: Fraction of reservoir nodes connected to output (0-1)
            optimizer: Optimizer for readout training
            discard_transients: Number of initial timesteps to discard
            seed: Random seed
            verbose: Whether to print information
        """
        super().__init__(
            name="PyReCo-Custom",
            config={
                'num_nodes': num_nodes,
                'density': density,
                'activation': activation,
                'spec_rad': spec_rad,
                'leakage_rate': leakage_rate,
                'fraction_output': fraction_output,
                'optimizer': optimizer,
                'discard_transients': discard_transients,
                'seed': seed,
            }
        )

        self.verbose = verbose

        # Store hyperparameters for model construction during fit()
        self.num_nodes = num_nodes
        self.density = density
        self.activation = activation
        self.spec_rad = spec_rad
        self.leakage_rate = leakage_rate
        self.fraction_output = fraction_output
        self.optimizer = optimizer
        self.discard_transients = discard_transients
        self.seed = seed

        # Model will be built during fit()
        self.model = None

    def _build_model(self, n_timesteps: int, n_features: int) -> RC:
        """
        Build the custom RC model with layers (following train_custom_model.py)

        Args:
            n_timesteps: Number of timesteps in input sequences
            n_features: Number of features per timestep

        Returns:
            Constructed RC model
        """
        # Create model instance
        m = RC()

        # Add input layer
        # input_shape format: (n_timesteps, n_features)
        # Actual input data shape: (batch, n_timesteps, n_features)
        m.add(InputLayer(input_shape=(n_timesteps, n_features)))

        # Add reservoir layer (use RandomReservoirLayer)
        m.add(RandomReservoirLayer(
            nodes=self.num_nodes,
            density=self.density,
            activation=self.activation,
            leakage_rate=self.leakage_rate,
            spec_rad=self.spec_rad,
        ))

        # Add readout layer
        # output_shape format: (None, n_features)
        # None means arbitrary output timesteps
        m.add(ReadoutLayer(
            output_shape=(None, n_features),
            fraction_out=self.fraction_output,
        ))

        # Compile the model (required for custom models)
        m.compile(
            optimizer=self.optimizer,
            metrics=["mse"],
            discard_transients=self.discard_transients
        )

        return m

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train the PyReCo custom model

        Args:
            X_train: Training input, shape (n_samples, n_timesteps, n_features)
            y_train: Training target, shape (n_samples, n_output_steps, n_features)
        """
        if self.verbose:
            print(f"    Training {self.name}...")

        start_time = time.time()

        # Get data dimensions
        n_samples, n_timesteps, n_features = X_train.shape

        # Build model if not already built
        if self.model is None:
            self.model = self._build_model(n_timesteps, n_features)

        # Train model
        self.model.fit(X_train, y_train)

        # Update state
        self.training_time = time.time() - start_time
        self.is_trained = True

        if self.verbose:
            print(f"    Training complete! Took {self.training_time:.2f} seconds")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using PyReCo custom model

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

    def get_model_summary(self) -> str:
        """
        Get summary of model architecture

        Returns:
            String describing the model layers
        """
        if self.model is None:
            return "Model not built yet"

        summary = f"{self.name} Architecture:\n"
        summary += "-" * 40 + "\n"

        if hasattr(self.model, 'layers'):
            for i, layer in enumerate(self.model.layers):
                layer_type = type(layer).__name__
                summary += f"Layer {i+1}: {layer_type}\n"

                # Add layer-specific info
                if hasattr(layer, 'num_nodes'):
                    summary += f"  - num_nodes: {layer.num_nodes}\n"
                if hasattr(layer, 'spec_rad'):
                    summary += f"  - spec_rad: {layer.spec_rad}\n"
                if hasattr(layer, 'leakage_rate'):
                    summary += f"  - leakage_rate: {layer.leakage_rate}\n"

        summary += "-" * 40

        return summary


# ============================================================================
# Helper function for creating multi-layer reservoirs
# ============================================================================

class PyReCoMultiLayerModel(BaseTimeSeriesModel):
    """
    Multi-layer PyReCo model

    Stacks multiple reservoir layers for increased capacity.

    Example:
        model = PyReCoMultiLayerModel(
            layer_configs=[
                {'num_nodes': 100, 'spec_rad': 0.9},
                {'num_nodes': 50, 'spec_rad': 0.8},
            ]
        )
    """

    def __init__(self,
                 layer_configs: list,
                 optimizer: str = 'ridge',
                 ridge_alpha: float = 1e-5,
                 verbose: bool = True):
        """
        Initialize multi-layer PyReCo model

        Args:
            layer_configs: List of layer configurations, each a dict with:
                - num_nodes: Number of nodes in this layer
                - spec_rad: Spectral radius
                - leakage_rate: Leakage rate (optional)
                - input_scaling: Input scaling (optional)
                - bias_scaling: Bias scaling (optional)
            optimizer: Optimizer for readout
            ridge_alpha: Ridge regularization
            verbose: Print information
        """
        super().__init__(
            name=f"PyReCo-MultiLayer-{len(layer_configs)}",
            config={
                'layer_configs': layer_configs,
                'optimizer': optimizer,
                'ridge_alpha': ridge_alpha,
            }
        )

        self.layer_configs = layer_configs
        self.optimizer = optimizer
        self.ridge_alpha = ridge_alpha
        self.verbose = verbose
        self.model = None

    def _build_model(self, n_timesteps: int, n_features: int) -> RC:
        """Build multi-layer model (following train_custom_model.py)"""
        m = RC()

        # Input layer
        m.add(InputLayer(input_shape=(n_timesteps, n_features)))

        # Add multiple reservoir layers (use RandomReservoirLayer)
        for i, config in enumerate(self.layer_configs):
            m.add(RandomReservoirLayer(
                nodes=config.get('num_nodes', 100),
                density=config.get('density', 0.05),
                activation=config.get('activation', 'tanh'),
                leakage_rate=config.get('leakage_rate', 0.3),
                spec_rad=config.get('spec_rad', 0.95),
            ))

        # Readout layer
        m.add(ReadoutLayer(
            output_shape=(None, n_features),  # None for arbitrary timesteps
            fraction_out=1.0,
        ))

        # Compile the model (required)
        m.compile(
            optimizer=self.optimizer,
            metrics=["mse"],
            discard_transients=20  # Default value
        )

        return m

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train multi-layer model"""
        if self.verbose:
            print(f"    Training {self.name}...")

        start_time = time.time()

        n_samples, n_timesteps, n_features = X_train.shape

        if self.model is None:
            self.model = self._build_model(n_timesteps, n_features)

        self.model.fit(X_train, y_train)

        self.training_time = time.time() - start_time
        self.is_trained = True

        if self.verbose:
            print(f"    Training complete! Took {self.training_time:.2f} seconds")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise RuntimeError(f"{self.name} model is not trained yet!")

        start_time = time.time()
        predictions = self.model.predict(X)
        self.prediction_time = time.time() - start_time

        return predictions


# ============================================================================
# Hyperparameter Tuning Functions
# ============================================================================

def tune_pyreco_custom_hyperparameters(
        X_train, y_train, X_val, y_val,
        param_grid,
        default_spec_rad: float = 0.95,
        default_leakage: float = 0.3,
        default_density: float = 0.05,
        default_activation: str = 'tanh',
        default_fraction_output: float = 1.0,
        default_discard_transients: int = 20,
        verbose: bool = True):
    """
    Tune PyReCo Custom model hyperparameters using simple train/val split.

    This function mirrors tune_pyreco_hyperparameters() from pyreco_wrapper.py
    but uses PyReCoCustomModel instead of PyReCoStandardModel.

    Args:
        X_train: Training input, shape (n_samples, n_timesteps, n_features)
        y_train: Training target, shape (n_samples, n_output_steps, n_features)
        X_val: Validation input
        y_val: Validation target
        param_grid: Dictionary with parameter names as keys and lists of values
            Example: {
                'num_nodes': [100, 200],
                'spec_rad': [0.8, 0.9, 1.0],
                'leakage_rate': [0.2, 0.3, 0.5],
                'density': [0.05, 0.1],
                'fraction_output': [0.8, 1.0],
            }
        default_spec_rad: Default spectral radius (if not in grid)
        default_leakage: Default leakage rate (if not in grid)
        default_density: Default density (if not in grid)
        default_activation: Default activation function
        default_fraction_output: Default fraction of output nodes
        default_discard_transients: Default number of transient steps to discard
        verbose: Whether to print progress

    Returns:
        Dictionary with:
            - 'best_params': Best hyperparameters found
            - 'best_score': Best validation MSE
            - 'final_model': Trained model with best parameters
            - 'all_results': List of all results for analysis
    """
    import itertools

    if verbose:
        print("\n" + "="*80)
        print("Hyperparameter Tuning for PyReCo Custom Model")
        print("="*80)

    # Generate all combinations
    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    combinations = list(itertools.product(*values))

    if verbose:
        print(f"Total combinations to test: {len(combinations)}")
        print(f"Parameter grid: {param_grid}\n")

    best_score = float('inf')
    best_params = None
    all_results = []

    for i, combo in enumerate(combinations):
        # Start with default values (following train_pyreco_model.py pattern)
        spec_rad = default_spec_rad
        leakage_rate = default_leakage
        density = default_density
        activation = default_activation
        fraction_output = default_fraction_output
        discard_transients = default_discard_transients

        # Required parameters from grid
        num_nodes = None

        # Override with grid values
        params_dict = dict(zip(keys, combo))
        for k, v in params_dict.items():
            if k == "spec_rad":
                spec_rad = float(v)
            elif k == "leakage_rate":
                leakage_rate = float(v)
            elif k == "density":
                density = float(v)
            elif k == "activation":
                activation = v
            elif k == "num_nodes":
                num_nodes = int(v)
            elif k == "fraction_output":
                fraction_output = float(v)
            elif k == "discard_transients":
                discard_transients = int(v)
            else:
                raise ValueError(f"Unknown parameter: {k}")

        # Validate required parameters
        if num_nodes is None:
            raise ValueError("param_grid must contain 'num_nodes'")

        # Build full parameter dictionary
        full_params = {
            'num_nodes': num_nodes,
            'density': density,
            'activation': activation,
            'spec_rad': spec_rad,
            'leakage_rate': leakage_rate,
            'fraction_output': fraction_output,
            'discard_transients': discard_transients,
            'optimizer': 'ridge',
            'verbose': False,
        }

        if verbose:
            print(f"[{i+1}/{len(combinations)}] Testing: {params_dict}")

        try:
            # Create and train model
            model = PyReCoCustomModel(**full_params)
            model.fit(X_train, y_train)

            # Evaluate on validation set
            val_results = model.evaluate(X_val, y_val, metrics=['mse'])
            val_mse = val_results['mse']

            # Track results
            all_results.append({
                'params': full_params.copy(),
                'val_mse': val_mse,
            })

            if verbose:
                print(f"  → Validation MSE: {val_mse:.6f}, Train time: {model.training_time:.2f}s")

            # Update best
            if val_mse < best_score:
                best_score = val_mse
                best_params = full_params.copy()

        except Exception as e:
            if verbose:
                print(f"  ✗ Failed: {str(e)}")
            continue

        if verbose:
            print()

    # Explicit garbage collection to free memory from temporary models
    import gc
    gc.collect()

    # Train final model with best parameters
    # COMMENTED OUT: To prevent memory leak in parallel experiments
    # if verbose:
    #     print("\n" + "="*80)
    #     print("Training final model with best parameters...")
    #     print("="*80)
    #     print(f"Best parameters: {best_params}")
    #     print(f"Best validation MSE: {best_score:.6f}\n")

    # final_model = PyReCoCustomModel(**{k: v for k, v in best_params.items()
    #                                    if k != 'verbose'}, verbose=verbose)
    # final_model.fit(X_train, y_train)

    if verbose:
        print(f"\n✓ Hyperparameter search complete")
        print(f"Best validation MSE: {best_score:.6f}\n")

    return {
        'best_params': best_params,
        'best_score': best_score,
        # 'final_model': final_model,  # COMMENTED OUT: Memory leak prevention
        'all_results': all_results,
    }


def tune_pyreco_custom_with_cv(
        X_train, y_train,
        param_grid,
        n_splits: int = 5,
        default_spec_rad: float = 0.95,
        default_leakage: float = 0.3,
        default_density: float = 0.05,
        default_activation: str = 'tanh',
        default_fraction_output: float = 1.0,
        default_discard_transients: int = 20,
        verbose: bool = True):
    """
    Tune PyReCo Custom model hyperparameters using time series cross-validation.

    Uses forward chaining (time series safe) to prevent data leakage.

    Args:
        X_train: Training input, shape (n_samples, n_timesteps, n_features)
        y_train: Training target, shape (n_samples, n_output_steps, n_features)
        param_grid: Dictionary with parameter names as keys and lists of values
        n_splits: Number of CV folds (default: 5)
        default_spec_rad: Default spectral radius
        default_leakage: Default leakage rate
        default_density: Default density
        default_activation: Default activation function
        default_fraction_output: Default fraction of output nodes
        default_discard_transients: Default transient steps to discard
        verbose: Whether to print progress

    Returns:
        Dictionary with:
            - 'best_params': Best hyperparameters found
            - 'best_score': Best mean CV MSE
            - 'final_model': Model trained on full training set with best params
            - 'all_results': List of all results with mean ± std
    """
    import itertools
    from src.utils.process_datasets import timeseries_cv_split

    if verbose:
        print("\n" + "="*80)
        print("Cross-Validation Tuning for PyReCo Custom Model")
        print("="*80)

    # Generate all combinations
    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    combinations = list(itertools.product(*values))

    if verbose:
        print(f"Total combinations: {len(combinations)}")
        print(f"CV splits: {n_splits}")
        print(f"Parameter grid: {param_grid}\n")

    best_score = float('inf')
    best_params = None
    all_results = []

    for i, combo in enumerate(combinations):
        # Start with defaults
        spec_rad = default_spec_rad
        leakage_rate = default_leakage
        density = default_density
        activation = default_activation
        fraction_output = default_fraction_output
        discard_transients = default_discard_transients

        num_nodes = None

        # Override with grid values
        params_dict = dict(zip(keys, combo))
        for k, v in params_dict.items():
            if k == "spec_rad":
                spec_rad = float(v)
            elif k == "leakage_rate":
                leakage_rate = float(v)
            elif k == "density":
                density = float(v)
            elif k == "activation":
                activation = v
            elif k == "num_nodes":
                num_nodes = int(v)
            elif k == "fraction_output":
                fraction_output = float(v)
            elif k == "discard_transients":
                discard_transients = int(v)
            else:
                raise ValueError(f"Unknown parameter: {k}")

        if num_nodes is None:
            raise ValueError("param_grid must contain 'num_nodes'")

        # Build full parameter dictionary
        full_params = {
            'num_nodes': num_nodes,
            'density': density,
            'activation': activation,
            'spec_rad': spec_rad,
            'leakage_rate': leakage_rate,
            'fraction_output': fraction_output,
            'discard_transients': discard_transients,
            'optimizer': 'ridge',
            'verbose': False,
        }

        try:
            # Perform cross-validation
            cv_splits = timeseries_cv_split(X_train, y_train, n_splits)
            cv_scores = []

            for fold_idx, (X_tr, y_tr, X_va, y_va) in enumerate(cv_splits):
                # Train and evaluate
                model = PyReCoCustomModel(**full_params)
                model.fit(X_tr, y_tr)
                val_results = model.evaluate(X_va, y_va, metrics=['mse'])
                cv_scores.append(val_results['mse'])

            # Calculate mean and std
            mean_score = np.mean(cv_scores)
            std_score = np.std(cv_scores)

            all_results.append({
                'params': full_params.copy(),
                'mean_mse': mean_score,
                'std_mse': std_score,
                'cv_scores': cv_scores,
            })

            # Update best
            if mean_score < best_score:
                best_score = mean_score
                best_params = full_params.copy()

            if verbose and (i + 1) % max(1, len(combinations) // 10) == 0:
                print(f"Progress: {i+1}/{len(combinations)} | "
                      f"Current: {mean_score:.6f}±{std_score:.6f} | "
                      f"Best: {best_score:.6f}")

        except Exception as e:
            if verbose:
                print(f"Error with params {params_dict}: {str(e)}")
            continue

    # Train final model with best parameters on full training set
    if verbose:
        print("\n" + "="*80)
        print("Training final model with best parameters on full training set...")
        print("="*80)
        print(f"Best parameters: {best_params}")
        print(f"Best mean CV MSE: {best_score:.6f}\n")

    final_model = PyReCoCustomModel(**{k: v for k, v in best_params.items()
                                       if k != 'verbose'}, verbose=verbose)
    final_model.fit(X_train, y_train)

    return {
        'best_params': best_params,
        'best_score': best_score,
        'final_model': final_model,
        'all_results': all_results,
    }
