"""
Advanced Multi-Step Time Series Evaluation Module

This module provides comprehensive evaluation capabilities for time series models,
supporting multi-step prediction, teacher forcing vs free-run modes, and advanced metrics.

Key Features:
- Multi-horizon evaluation (1-50 steps)
- Teacher forcing vs free-run prediction modes
- Advanced metrics: NRMSE, spectral similarity, trajectory divergence time
- Statistical consistency checks
- Standardized evaluation protocols

Author: Updated for PyReCo vs LSTM comprehensive comparison
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Union
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import signal, stats
from scipy.fft import fft, fftfreq
import warnings


def normalized_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Normalized Root Mean Square Error (NRMSE)

    NRMSE = RMSE / std(y_true)

    Benefits:
    - Scale-invariant metric (0-1 typically, but can exceed 1 for very poor predictions)
    - Easier to compare across different datasets and variables
    - NRMSE < 0.1: Excellent, 0.1-0.3: Good, 0.3-0.5: Fair, >0.5: Poor

    Args:
        y_true: Ground truth values, shape (n_samples, n_features)
        y_pred: Predicted values, shape (n_samples, n_features)

    Returns:
        NRMSE value (float)
    """
    mse = mean_squared_error(y_true.flatten(), y_pred.flatten())
    rmse = np.sqrt(mse)
    std_true = np.std(y_true)

    if std_true == 0:
        warnings.warn("Standard deviation of true values is zero, returning inf")
        return float('inf')

    return rmse / std_true


def spectral_similarity(y_true: np.ndarray, y_pred: np.ndarray,
                       fs: float = 1.0, freq_bands: Optional[List[Tuple[float, float]]] = None) -> Dict[str, float]:
    """
    Spectral similarity analysis between true and predicted time series

    Compares power spectral density (PSD) to assess if the model captures
    the frequency characteristics of the dynamical system.

    Args:
        y_true: Ground truth time series, shape (n_samples, n_features)
        y_pred: Predicted time series, shape (n_samples, n_features)
        fs: Sampling frequency (default: 1.0)
        freq_bands: List of frequency bands to analyze, e.g., [(0, 0.1), (0.1, 0.5)]
                   If None, analyzes the full spectrum

    Returns:
        Dictionary with spectral similarity metrics:
        - 'psd_correlation': Pearson correlation between PSDs
        - 'spectral_rmse': RMSE between PSDs
        - 'dominant_freq_error': Error in dominant frequency
        - 'band_power_errors': If freq_bands provided, power error in each band
    """
    # Ensure 2D arrays and take first feature if multivariate
    if y_true.ndim > 1:
        y_true = y_true[:, 0]
    if y_pred.ndim > 1:
        y_pred = y_pred[:, 0]

    # Compute power spectral densities
    freqs_true, psd_true = signal.welch(y_true, fs=fs, nperseg=min(256, len(y_true)//4))
    freqs_pred, psd_pred = signal.welch(y_pred, fs=fs, nperseg=min(256, len(y_pred)//4))

    # Ensure same frequency bins
    min_len = min(len(freqs_true), len(freqs_pred))
    freqs_true = freqs_true[:min_len]
    freqs_pred = freqs_pred[:min_len]
    psd_true = psd_true[:min_len]
    psd_pred = psd_pred[:min_len]

    results = {}

    # 1. PSD correlation
    if len(psd_true) > 1:
        psd_corr, _ = stats.pearsonr(psd_true, psd_pred)
        results['psd_correlation'] = float(psd_corr) if not np.isnan(psd_corr) else 0.0
    else:
        results['psd_correlation'] = 0.0

    # 2. Spectral RMSE (log scale to handle wide dynamic range)
    log_psd_true = np.log10(psd_true + 1e-12)
    log_psd_pred = np.log10(psd_pred + 1e-12)
    results['spectral_rmse'] = float(np.sqrt(mean_squared_error(log_psd_true, log_psd_pred)))

    # 3. Dominant frequency error
    dom_freq_true = freqs_true[np.argmax(psd_true)]
    dom_freq_pred = freqs_pred[np.argmax(psd_pred)]
    results['dominant_freq_error'] = float(abs(dom_freq_true - dom_freq_pred))

    # 4. Band power analysis if requested
    if freq_bands is not None:
        band_errors = []
        for low_freq, high_freq in freq_bands:
            mask_true = (freqs_true >= low_freq) & (freqs_true <= high_freq)
            mask_pred = (freqs_pred >= low_freq) & (freqs_pred <= high_freq)

            power_true = np.trapz(psd_true[mask_true], freqs_true[mask_true])
            power_pred = np.trapz(psd_pred[mask_pred], freqs_pred[mask_pred])

            if power_true > 0:
                relative_error = abs(power_true - power_pred) / power_true
            else:
                relative_error = float('inf') if power_pred > 0 else 0.0

            band_errors.append(float(relative_error))

        results['band_power_errors'] = band_errors

    return results


def trajectory_divergence_time(y_true: np.ndarray, y_pred: np.ndarray,
                              threshold: float = 0.1, metric: str = 'euclidean') -> float:
    """
    Calculate trajectory divergence time - when predictions diverge from truth

    This measures how long the prediction stays close to the true trajectory,
    which is crucial for chaotic systems where small errors grow exponentially.

    Args:
        y_true: Ground truth trajectory, shape (n_steps, n_features)
        y_pred: Predicted trajectory, shape (n_steps, n_features)
        threshold: Distance threshold for considering trajectories diverged
        metric: Distance metric ('euclidean', 'manhattan', 'max')

    Returns:
        Divergence time (number of steps until divergence)
        Returns n_steps if never diverges
    """
    n_steps = min(len(y_true), len(y_pred))

    for step in range(n_steps):
        if metric == 'euclidean':
            distance = np.linalg.norm(y_true[step] - y_pred[step])
        elif metric == 'manhattan':
            distance = np.sum(np.abs(y_true[step] - y_pred[step]))
        elif metric == 'max':
            distance = np.max(np.abs(y_true[step] - y_pred[step]))
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if distance > threshold:
            return float(step)

    return float(n_steps)  # Never diverged within the sequence


def long_term_statistics_consistency(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Check long-term statistical consistency between true and predicted series

    For chaotic systems, even if short-term prediction fails, the model should
    preserve the statistical properties of the attractor (mean, variance, distribution).

    Args:
        y_true: Ground truth series, shape (n_samples, n_features)
        y_pred: Predicted series, shape (n_samples, n_features)

    Returns:
        Dictionary with statistical consistency metrics:
        - 'mean_error': Relative error in mean
        - 'variance_error': Relative error in variance
        - 'skewness_error': Absolute error in skewness
        - 'kurtosis_error': Absolute error in kurtosis
        - 'ks_statistic': Kolmogorov-Smirnov test statistic
        - 'ks_pvalue': KS test p-value (>0.05 indicates similar distributions)
    """
    # Flatten to 1D for statistical analysis
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    results = {}

    # Mean error
    mean_true = np.mean(y_true_flat)
    mean_pred = np.mean(y_pred_flat)
    if mean_true != 0:
        results['mean_error'] = float(abs(mean_true - mean_pred) / abs(mean_true))
    else:
        results['mean_error'] = float(abs(mean_pred))

    # Variance error
    var_true = np.var(y_true_flat)
    var_pred = np.var(y_pred_flat)
    if var_true != 0:
        results['variance_error'] = float(abs(var_true - var_pred) / var_true)
    else:
        results['variance_error'] = float(abs(var_pred))

    # Higher-order moments
    skew_true = stats.skew(y_true_flat)
    skew_pred = stats.skew(y_pred_flat)
    results['skewness_error'] = float(abs(skew_true - skew_pred))

    kurt_true = stats.kurtosis(y_true_flat)
    kurt_pred = stats.kurtosis(y_pred_flat)
    results['kurtosis_error'] = float(abs(kurt_true - kurt_pred))

    # Distribution similarity (Kolmogorov-Smirnov test)
    ks_stat, ks_pvalue = stats.ks_2samp(y_true_flat, y_pred_flat)
    results['ks_statistic'] = float(ks_stat)
    results['ks_pvalue'] = float(ks_pvalue)

    return results


def multi_step_predict(model, X_test: np.ndarray, n_steps: int,
                      mode: str = 'free_run') -> np.ndarray:
    """
    Generate multi-step predictions using either teacher forcing or free-run mode

    Args:
        model: Trained model with predict() method
        X_test: Initial input sequences, shape (n_samples, n_timesteps, n_features)
        n_steps: Number of steps to predict
        mode: 'teacher_forcing' or 'free_run'
            - teacher_forcing: Use true values as input at each step
            - free_run: Use model's own predictions as input (autonomous prediction)

    Returns:
        Predictions, shape (n_samples, n_steps, n_features)
    """
    if mode == 'teacher_forcing':
        # For teacher forcing, we need access to the true sequence
        # This should be called with the appropriate ground truth
        raise NotImplementedError("Teacher forcing requires ground truth sequence")

    elif mode == 'free_run':
        # Free-run prediction: use model's own outputs as inputs
        n_samples, n_timesteps, n_features = X_test.shape
        predictions = []

        # Start with initial input
        current_input = X_test.copy()

        for step in range(n_steps): 
            # Get next prediction
            pred = model.predict(current_input)  # Shape: (n_samples, 1, n_features)
            predictions.append(pred)

            # Update input: shift left and append prediction
            current_input = np.concatenate([
                current_input[:, 1:, :],  # Remove first timestep
                pred  # Add prediction as last timestep
            ], axis=1)

        return np.concatenate(predictions, axis=1)  # (n_samples, n_steps, n_features)

    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'teacher_forcing' or 'free_run'")


def evaluate_multi_step(model, X_test: np.ndarray, y_test: np.ndarray,
                       horizons: List[int] = None, mode: str = 'free_run',
                       include_advanced_metrics: bool = True) -> Dict[int, Dict[str, float]]:
    """
    Comprehensive multi-step evaluation of time series model

    Args:
        model: Trained model with predict() method
        X_test: Test input sequences, shape (n_samples, n_timesteps, n_features)
        y_test: Test target sequences, shape (n_samples, n_steps_total, n_features)
        horizons: List of prediction horizons to evaluate (default: [1, 5, 10, 20, 50])
        mode: Prediction mode ('free_run' or 'teacher_forcing')
        include_advanced_metrics: Whether to compute spectral and trajectory metrics

    Returns:
        Results dictionary: {horizon: {metric: value}}

    Example:
        results = evaluate_multi_step(model, X_test, y_test, horizons=[1, 10, 50])
        print(f"10-step MSE: {results[10]['mse']:.6f}")
        print(f"10-step NRMSE: {results[10]['nrmse']:.6f}")
    """
    if horizons is None:
        horizons = [1, 5, 10, 20, 50]

    max_horizon = max(horizons)
    n_samples, n_timesteps, n_features = X_test.shape

    # Generate predictions for maximum horizon
    print(f"Generating {max_horizon}-step {mode} predictions...")
    start_time = time.time()

    if mode == 'free_run':
        y_pred_all = multi_step_predict(model, X_test, max_horizon, mode='free_run')
    else:
        raise NotImplementedError("Teacher forcing mode not yet implemented")

    pred_time = time.time() - start_time
    print(f"Prediction completed in {pred_time:.2f} seconds")

    # Evaluate at each horizon
    results = {}

    for horizon in horizons:
        print(f"Evaluating horizon {horizon}...")

        # Extract predictions and targets for this horizon
        y_pred_h = y_pred_all[:, :horizon, :]  # (n_samples, horizon, n_features)
        y_true_h = y_test[:, :horizon, :]      # (n_samples, horizon, n_features)

        # Basic metrics
        horizon_results = {}
        horizon_results['mse'] = float(mean_squared_error(y_true_h.flatten(), y_pred_h.flatten()))
        horizon_results['mae'] = float(mean_absolute_error(y_true_h.flatten(), y_pred_h.flatten()))
        horizon_results['rmse'] = float(np.sqrt(horizon_results['mse']))
        horizon_results['nrmse'] = float(normalized_rmse(y_true_h, y_pred_h))

        # R² can be negative for very poor predictions
        # Reshape to 2D (n_samples*horizon, n_features) for correct per-feature R²
        n_s, n_h, n_f = y_true_h.shape
        r2 = r2_score(y_true_h.reshape(-1, n_f), y_pred_h.reshape(-1, n_f),
                       multioutput='uniform_average')
        horizon_results['r2'] = float(r2)

        # Advanced metrics (computationally expensive for large horizons)
        if include_advanced_metrics and horizon <= 20:  # Limit to shorter horizons
            try:
                # Average trajectory divergence time across samples
                div_times = []
                for i in range(min(10, n_samples)):  # Sample subset for efficiency
                    div_time = trajectory_divergence_time(
                        y_true_h[i], y_pred_h[i],
                        threshold=0.1 * np.std(y_true_h[i])  # Adaptive threshold
                    )
                    div_times.append(div_time)

                horizon_results['avg_divergence_time'] = float(np.mean(div_times))
                horizon_results['std_divergence_time'] = float(np.std(div_times))

                # Spectral similarity (use first sample for efficiency)
                spectral_metrics = spectral_similarity(y_true_h[0], y_pred_h[0])
                for key, value in spectral_metrics.items():
                    if key != 'band_power_errors':  # Skip complex nested metrics for now
                        horizon_results[f'spectral_{key}'] = value

                # Long-term statistics (use all samples)
                stats_metrics = long_term_statistics_consistency(y_true_h, y_pred_h)
                for key, value in stats_metrics.items():
                    horizon_results[f'stats_{key}'] = value

            except Exception as e:
                print(f"Warning: Advanced metrics failed for horizon {horizon}: {e}")

        results[horizon] = horizon_results

    return results


def create_evaluation_protocol_document():
    """
    Create standardized evaluation protocol documentation

    This documents the exact procedures, metrics, and parameters used
    for reproducible multi-step time series evaluation.
    """
    protocol = {
        "evaluation_protocol_version": "1.0",
        "description": "Standardized multi-step time series prediction evaluation",
        "standard_horizons": [1, 5, 10, 20, 50],
        "prediction_modes": {
            "free_run": "Autonomous prediction using model's own outputs as inputs",
            "teacher_forcing": "Use true values as inputs at each step (oracle setting)"
        },
        "core_metrics": {
            "mse": "Mean Squared Error - primary loss metric",
            "mae": "Mean Absolute Error - robust to outliers",
            "rmse": "Root Mean Squared Error - same scale as data",
            "nrmse": "Normalized RMSE - scale-invariant (target: <0.3)",
            "r2": "R-squared coefficient - explained variance (-inf to 1)"
        },
        "advanced_metrics": {
            "trajectory_divergence_time": "Steps until prediction diverges (chaotic systems)",
            "spectral_correlation": "Frequency domain similarity (0 to 1)",
            "statistical_consistency": "Long-term attractor properties preservation"
        },
        "evaluation_parameters": {
            "divergence_threshold": "10% of target standard deviation",
            "spectral_nperseg": "min(256, sequence_length//4)",
            "max_samples_for_advanced_metrics": 10
        },
        "interpretation_guidelines": {
            "nrmse_excellent": "<0.1",
            "nrmse_good": "0.1-0.3",
            "nrmse_fair": "0.3-0.5",
            "nrmse_poor": ">0.5",
            "r2_excellent": ">0.9",
            "r2_good": "0.7-0.9",
            "r2_fair": "0.3-0.7",
            "r2_poor": "<0.3"
        }
    }

    return protocol