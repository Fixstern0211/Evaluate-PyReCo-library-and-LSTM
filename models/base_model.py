"""
Simplified Abstract Base Class - BaseTimeSeriesModel

Purpose: Ensure all models (PyReCo, LSTM, etc.) use the same interface and evaluation methods
This allows fair comparison and avoids confusion from different interfaces

Author: Beginner-friendly version
"""

import time
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any

# Use PyReCo's metrics to ensure evaluation consistency
from pyreco.metrics import mse, mae, r2


class BaseTimeSeriesModel(ABC):
    """
    Abstract base class for time series models (simplified version)

    What is an abstract base class?
    - Like a "standard contract" that specifies functions all models must implement
    - Both PyReCo and LSTM must follow this "contract"
    - Benefit: Can train and evaluate all models in a unified way

    Methods you must implement (subclass must provide):
    1. fit() - Train the model
    2. predict() - Make predictions

    Already implemented methods (subclass can use directly):
    1. evaluate() - Evaluate model (uses PyReCo's metrics)
    2. get_info() - Get model information
    """

    def __init__(self, name: str, config: Dict[str, Any] = None):
        """
        Initialize base class

        Args:
            name: Model name, e.g., "PyReCo-Standard", "LSTM"
            config: Model configuration dictionary (optional)
        """
        self.name = name
        self.config = config if config is not None else {}

        # Track state
        self.is_trained = False       # Whether model is trained
        self.training_time = 0.0      # Training time (seconds)
        self.prediction_time = 0.0    # Prediction time (seconds)

    # ========================================================================
    # Abstract methods - Subclass must implement these
    # ========================================================================

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train the model - Must implement!

        Args:
            X_train: Training input, shape=(n_samples, n_timesteps, n_features)
            y_train: Training target, shape=(n_samples, n_output_steps, n_features)

        Example:
            model.fit(X_train, y_train)

        Note:
        - This method should internally record training time
        - After training, set self.is_trained = True
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions - Must implement!

        Args:
            X: Input data, shape=(n_samples, n_timesteps, n_features)

        Returns:
            Predictions, shape=(n_samples, n_output_steps, n_features)

        Example:
            y_pred = model.predict(X_test)

        Note:
        - This method should internally record prediction time
        """
        pass

    # ========================================================================
    # Implemented methods - Subclass can use directly
    # ========================================================================

    def evaluate(self, X: np.ndarray, y: np.ndarray,
                 metrics: list = None) -> Dict[str, float]:
        """
        Evaluate model - Already implemented, all models use this method uniformly

        ✨ Key: This forces all models to use PyReCo's metrics, ensuring fair comparison!

        Args:
            X: Input data
            y: True labels
            metrics: List of metrics to compute, default ['mse', 'mae', 'r2']

        Returns:
            Dictionary of metric results, e.g., {'mse': 0.123, 'mae': 0.234, 'r2': 0.89}

        Example:
            results = model.evaluate(X_test, y_test)
            print(f"MSE: {results['mse']}")
        """
        if not self.is_trained:
            raise RuntimeError(f"{self.name} model is not trained yet! Please call fit() first.")

        # Default: compute MSE, MAE, R²
        if metrics is None:
            metrics = ['mse', 'mae', 'r2']

        # Get predictions
        y_pred = self.predict(X)

        # Compute metrics using PyReCo's functions (ensures consistency)
        results = {}
        for metric_name in metrics:
            metric_name_lower = metric_name.lower()

            if metric_name_lower == 'mse':
                results['mse'] = float(mse(y, y_pred))
            elif metric_name_lower == 'mae':
                results['mae'] = float(mae(y, y_pred))
            elif metric_name_lower == 'r2':
                results['r2'] = float(r2(y, y_pred))
            elif metric_name_lower == 'rmse':
                # RMSE = sqrt(MSE)
                results['rmse'] = float(np.sqrt(mse(y, y_pred)))
            else:
                print(f"Warning: Unknown metric '{metric_name}', skipped")

        return results

    def get_info(self) -> Dict[str, Any]:
        """
        Get model information

        Returns:
            Dictionary containing model information

        Example:
            info = model.get_info()
            print(f"Model name: {info['name']}")
            print(f"Training time: {info['training_time']:.2f} seconds")
        """
        return {
            'name': self.name,
            'config': self.config,
            'is_trained': self.is_trained,
            'training_time': self.training_time,
            'prediction_time': self.prediction_time,
        }

    def __repr__(self):
        """Print model information"""
        status = "Trained" if self.is_trained else "Not Trained"
        return f"{self.name} ({status})"


# ============================================================================
# Helper function - Convenient for batch model comparison
# ============================================================================

def compare_models(models: list, X_train: np.ndarray, y_train: np.ndarray,
                  X_test: np.ndarray, y_test: np.ndarray,
                  metrics: list = None) -> Dict[str, Dict[str, float]]:
    """
    Train and evaluate multiple models in batch, convenient for comparison

    This is the biggest benefit of abstract base class: one function handles all models!

    Args:
        models: List of models, all should inherit from BaseTimeSeriesModel
        X_train, y_train: Training data
        X_test, y_test: Test data
        metrics: Metrics to compute

    Returns:
        Evaluation results dictionary for all models
        Format: {'PyReCo': {'mse': 0.1, ...}, 'LSTM': {'mse': 0.2, ...}}

    Example:
        models = [PyReCoModel(...), LSTMModel(...)]
        results = compare_models(models, X_train, y_train, X_test, y_test)

        for model_name, metrics in results.items():
            print(f"{model_name}: MSE={metrics['mse']:.4f}")
    """
    if metrics is None:
        metrics = ['mse', 'mae', 'r2']

    all_results = {}

    print("="*60)
    print("Starting batch training and evaluation...")
    print("="*60 + "\n")

    for model in models:
        print(f"📊 Processing: {model.name}")
        print("-" * 40)

        # Train
        print("  ⏳ Training...")
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        print(f"  ✓ Training complete, took {train_time:.2f} seconds")

        # Evaluate
        print("  ⏳ Evaluating...")
        results = model.evaluate(X_test, y_test, metrics=metrics)
        print(f"  ✓ Evaluation complete")

        # Print results
        print("  📈 Results:")
        for metric_name, value in results.items():
            print(f"     {metric_name.upper()}: {value:.6f}")
        print()

        # Save results
        all_results[model.name] = results

    print("="*60)
    print("All models evaluation complete!")
    print("="*60 + "\n")

    return all_results
