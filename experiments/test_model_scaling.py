"""
Model Scaling Test: Performance across Different Parameter Budgets

Tests PyReCo-Standard and LSTM at three parameter scales:
- Small: ~1,000 total parameters
- Medium: ~10,000 total parameters
- Large: ~50,000 total parameters

Fair comparison features:
- Total parameter budget matching (not just trainable params)
- Both models use hyperparameter tuning when --tune-pyreco is set:
  - PyReCo: 36 combinations (spec_rad, leakage_rate, density, fraction_input)
  - LSTM: 40 combinations (num_layers, learning_rate, dropout)

Green computing metrics:
- inference_time_total: Total prediction time on test set (seconds)
- inference_time_per_sample_ms: Per-sample inference latency (milliseconds)
- final_train_time: Single model training time (seconds)
- tune_time: Total hyperparameter search time (seconds)
- memory_peak_mb: Peak memory usage during training (MB) [via tracemalloc]
- energy_kwh: Energy consumption in kWh (requires CodeCarbon)
- emissions_kg_co2: Carbon emissions in kg CO2 (requires CodeCarbon)

Literature support for LSTM tuning:
- Greff et al. (2017) "LSTM: A Search Space Odyssey": learning_rate is most critical
- Zaremba et al. (2014): dropout 0.1-0.3 for regularization
- Gal & Ghahramani (2016): dropout rates 0.1-0.3 optimal for sequence tasks
"""

import argparse
import json
import math
import time
import tracemalloc
import gc
from datetime import datetime
import numpy as np
import torch  # For GPU memory management
from sklearn.preprocessing import StandardScaler
import sys
import os

# Optional: CodeCarbon for energy/carbon tracking
try:
    from codecarbon import OfflineEmissionsTracker
    CODECARBON_AVAILABLE = True
except ImportError:
    CODECARBON_AVAILABLE = False


class GreenMetricsTracker:
    """Track green computing metrics: memory usage and energy consumption."""

    def __init__(self, track_carbon=False, country_iso_code="USA"):
        self.track_carbon = track_carbon and CODECARBON_AVAILABLE
        self.country_iso_code = country_iso_code
        self.carbon_tracker = None
        self.memory_start = 0
        self.memory_peak = 0

    def start(self):
        """Start tracking metrics."""
        gc.collect()
        tracemalloc.start()
        self.memory_start = tracemalloc.get_traced_memory()[0]

        if self.track_carbon:
            self.carbon_tracker = OfflineEmissionsTracker(
                country_iso_code=self.country_iso_code,
                log_level='error'
            )
            self.carbon_tracker.start()

    def stop(self):
        """Stop tracking and return metrics."""
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        metrics = {
            'memory_current_mb': current / 1024 / 1024,
            'memory_peak_mb': peak / 1024 / 1024,
            'memory_delta_mb': (current - self.memory_start) / 1024 / 1024,
        }

        if self.track_carbon and self.carbon_tracker:
            emissions = self.carbon_tracker.stop()
            if emissions is not None:
                metrics['emissions_kg_co2'] = float(emissions)
                metrics['energy_kwh'] = float(self.carbon_tracker._total_energy.kWh) if hasattr(self.carbon_tracker, '_total_energy') else None
            else:
                metrics['emissions_kg_co2'] = 0.0
                metrics['energy_kwh'] = 0.0

        return metrics

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.load_dataset import load_data, load as local_load, set_seed
from src.utils import process_datasets
from src.utils.node_number import best_num_nodes_and_fraction_out, compute_readout_F_from_budget
from src.utils.budget_matching import (
    esn_solve_num_nodes, esn_total_params,
    lstm_solve_hidden_size, lstm_total_params, lstm_layer_hidden_map,
)

# Import new PyReCo datasets interface
try:
    from pyreco.datasets import load as pyreco_load
    USE_PYRECO_DATASETS = True
except ImportError:
    USE_PYRECO_DATASETS = False

# Dataset routing: PyReCo supports lorenz/mackeyglass, local_load supports all including santafe
PYRECO_SUPPORTED_DATASETS = {'lorenz', 'mackeyglass', 'mackey-glass', 'mg'}

# Import all model types
from models.pyreco_wrapper import PyReCoStandardModel, tune_pyreco_hyperparameters
from models.pyreco_custom_wrapper import PyReCoCustomModel, tune_pyreco_custom_hyperparameters
from models.lstm_model import LSTMModel, tune_lstm_hyperparameters


def calculate_model_params(model_type, config):
    """
    Calculate parameter counts from the ACTUAL model configuration.

    Uses budget_matching module for correct formulas:
    - ESN: N²×δ + N×d_in×f_in + N×d_out
    - LSTM: 4h(h+d_in+2) + (L-1)×4h(2h+2) + h×d_out + d_out
            (matches PyTorch nn.LSTM with both bias_ih and bias_hh)
    """
    if model_type in ['pyreco_standard', 'pyreco_custom']:
        num_nodes = config.get('num_nodes', 100)
        density = config.get('density', 0.1)
        fraction_input = config.get('fraction_input', 0.5)
        d_in = config.get('n_input_features', 3)
        d_out = config.get('n_output_features', 3)

        return esn_total_params(num_nodes, density, fraction_input, d_in, d_out)

    elif model_type == 'lstm':
        hidden_size = config.get('hidden_size', 64)
        num_layers = config.get('num_layers', 2)
        d_in = config.get('n_input_features', 3)
        d_out = config.get('n_output_features', 3)

        return lstm_total_params(hidden_size, d_in, d_out, num_layers)

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def compute_lstm_hidden_size(budget, num_layers, n_input, n_output):
    """Compute LSTM hidden_size that fits the parameter budget for given num_layers.

    Correct formula matching PyTorch nn.LSTM (includes both bias_ih and bias_hh):
        Layer 1:    4h(h + d_in + 2)
        Layer k>1:  4h(2h + 2)
        FC output:  h × d_out + d_out

    Delegates to budget_matching.lstm_solve_hidden_size.
    """
    h = lstm_solve_hidden_size(budget, n_input, n_output, num_layers)
    return max(8, min(h, 512))


def get_model_configs_for_budget(budget, n_input_features=3, n_output_features=3):
    """
    Generate model configurations for a given parameter budget

    IMPORTANT: budget refers to TOTAL PARAMETERS
    This enables fair comparison of computational and memory cost:
    - RC: Total = Input + Reservoir (fixed) + Readout (trainable)
    - LSTM: Total = All parameters (all trainable)

    Literature Support:
    - Lukoševičius (2012): ESN comparisons use total network size
    - Jaeger et al. (2007): Reservoir size (total params) as primary metric
    - Common practice: Compare models with similar total parameter count

    Args:
        budget: Target number of TOTAL parameters
        n_input_features: Number of input features
        n_output_features: Number of output features

    Returns:
        Dictionary of model configs
    """
    configs = {}

    # PyReCo: Calculate num_nodes from total parameter budget
    # Total params ≈ N*D_in*f_in + N²*density + N*f_out*D_out
    # Dominated by reservoir: N² * density ≈ budget
    # Therefore: N ≈ sqrt(budget / density)

    density = 0.1
    fraction_input = 0.5
    fraction_output = 1.0

    # Calculate num_nodes from total budget
    num_nodes_approx = int(math.sqrt(budget / density))

    # Select closest candidate
    candidates = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    num_nodes = min(candidates, key=lambda x: abs(x - num_nodes_approx))

    # PyReCo Standard config
    configs['pyreco_standard'] = {
        'num_nodes': num_nodes,
        'density': 0.1,  # Will be tuned
        'activation': 'tanh',
        'spec_rad': 1.0,  # Will be tuned
        'leakage_rate': 0.3,  # Will be tuned
        'fraction_input': 0.5,  # Will be tuned
        'fraction_output': fraction_output,
        'optimizer': 'ridge',
        'n_input_features': n_input_features,
        'n_output_features': n_output_features,
    }

    # PyReCo Custom config - REMOVED
    # Analysis shows Custom API uses the same underlying implementation as Standard
    # (pyreco.models.ReservoirComputer internally uses pyreco.custom_models.RC)
    # Both have the same severe performance issues with large reservoirs (1149s for 3000 nodes)
    # Removed to focus on meaningful comparison: Single-layer RC (Standard) vs Multi-layer LSTM

    # configs['pyreco_custom'] = {
    #     'num_nodes': num_nodes,
    #     'density': 0.05,
    #     'activation': 'tanh',
    #     'spec_rad': 0.95,
    #     'leakage_rate': 0.3,
    #     'fraction_output': fraction_output,
    #     'discard_transients': 20,
    #     'optimizer': 'ridge',
    #     'n_input_features': n_input_features,
    #     'n_output_features': n_output_features,
    # }

    # LSTM: Calculate hidden_size from total parameter budget
    # General formula for L-layer LSTM:
    #   Layer 1: 4h(input_size + h), Layers 2..L: 4h(2h) each, Output: h × output_size
    #   Total = (8L-4)h² + (4×input_size + output_size)h
    # Solve: ah² + bh - budget = 0, where a = 8L-4, b = 4×n_in + n_out
    # h = (-b + sqrt(b² + 4a×budget)) / (2a)
    #
    # Reference: PyTorch LSTM documentation
    # https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html

    num_layers = 2  # Default for non-tuning mode
    h = compute_lstm_hidden_size(budget, num_layers, n_input_features, n_output_features)

    configs['lstm'] = {
        'hidden_size': h,
        'num_layers': num_layers,
        'dropout': 0.2,
        'learning_rate': 0.001,
        'epochs': 100,
        'batch_size': 32,
        'patience': 10,
        'n_input_features': n_input_features,
        'n_output_features': n_output_features,
    }

    return configs


def test_model_at_scale(model_type, config, X_train, y_train, X_val, y_val, X_test, y_test,
                        use_tuning=False, param_grid=None, lstm_device='auto', verbose=True,
                        track_green_metrics=False, country_iso_code="USA",
                        layer_hidden_map=None):
    """
    Test a model with given configuration

    Args:
        model_type: 'pyreco_standard', 'pyreco_custom', 'lstm'
        config: Model configuration
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Test data
        use_tuning: Whether to use hyperparameter tuning
        param_grid: Grid for hyperparameter search
        verbose: Print progress
        track_green_metrics: Whether to track memory and energy consumption
        country_iso_code: Country code for carbon intensity calculation

    Returns:
        Dictionary of results
    """
    # Convert 'auto' to None for LSTM device auto-detection
    if lstm_device == 'auto':
        lstm_device = None

    if verbose:
        print(f"\n{'='*80}")
        print(f"Testing {model_type.upper()}")
        print(f"{'='*80}")
        print(f"Configuration: {config}")

    # Initialize green metrics tracker
    green_tracker = None
    if track_green_metrics:
        green_tracker = GreenMetricsTracker(
            track_carbon=True,
            country_iso_code=country_iso_code
        )
        green_tracker.start()
        if verbose:
            print(f"  [Green Metrics] Tracking memory and {'energy' if CODECARBON_AVAILABLE else 'memory only (CodeCarbon not installed)'}")

    start_time = time.time()

    # Create and train model
    if model_type == 'pyreco_standard':
        if use_tuning and param_grid:
            # Use hyperparameter tuning
            results = tune_pyreco_hyperparameters(
                X_train, y_train, X_val, y_val,
                param_grid=param_grid,
                default_spec_rad=config['spec_rad'],
                default_leakage=config['leakage_rate'],
                default_density=config['density'],
                default_activation=config['activation'],
                default_fraction_input=config['fraction_input'],
                verbose=verbose
            )
            # Memory fix: train final model ourselves instead of using returned model
            best_params = results['best_params']
            val_score = results['best_score']
            # Extract best combination's training time from all_results
            best_combo_train_time = None
            for result in results['all_results']:
                if result['params'] == best_params and result['success']:
                    best_combo_train_time = result['train_time']
                    break
            del results  # Explicitly delete results dict to free memory
            import gc
            gc.collect()

            model = PyReCoStandardModel(**{k: v for k, v in best_params.items()
                                          if k not in ['n_input_features', 'n_output_features']})
            # Train on combined train+val data
            X_full = np.concatenate([X_train, X_val], axis=0)
            y_full = np.concatenate([y_train, y_val], axis=0)

            if verbose:
                print(f"\n    Training PyReCo-Standard...")
            model.fit(X_full, y_full)
            final_train_time = model.training_time
            if verbose:
                print(f"    Training complete! Took {final_train_time:.2f} seconds")
        else:
            # Simple training
            model = PyReCoStandardModel(**{k: v for k, v in config.items()
                                          if k not in ['n_input_features', 'n_output_features']})
            model.fit(X_train, y_train)
            val_results = model.evaluate(X_val, y_val, metrics=['mse'])
            best_params = config
            val_score = val_results['mse']
            best_combo_train_time = None
            final_train_time = model.training_time

    elif model_type == 'pyreco_custom':
        if use_tuning and param_grid:
            # Use hyperparameter tuning
            results = tune_pyreco_custom_hyperparameters(
                X_train, y_train, X_val, y_val,
                param_grid=param_grid,
                default_spec_rad=config['spec_rad'],
                default_leakage=config['leakage_rate'],
                default_density=config['density'],
                default_activation=config.get('activation', 'tanh'),
                default_fraction_output=config['fraction_output'],
                verbose=verbose
            )
            # Memory fix: train final model ourselves instead of using returned model
            best_params = results['best_params']
            val_score = results['best_score']
            # Extract best combination's training time from all_results
            best_combo_train_time = None
            for result in results['all_results']:
                if result['params'] == best_params and result['success']:
                    best_combo_train_time = result['train_time']
                    break
            del results  # Explicitly delete results dict to free memory
            import gc
            gc.collect()

            model = PyReCoCustomModel(**{k: v for k, v in best_params.items()
                                         if k not in ['n_input_features', 'n_output_features']})
            # Train on combined train+val data
            X_full = np.concatenate([X_train, X_val], axis=0)
            y_full = np.concatenate([y_train, y_val], axis=0)

            if verbose:
                print(f"\n    Training PyReCo-Custom...")
            model.fit(X_full, y_full)
            final_train_time = model.training_time
            if verbose:
                print(f"    Training complete! Took {final_train_time:.2f} seconds")
        else:
            # Simple training
            model = PyReCoCustomModel(**{k: v for k, v in config.items()
                                         if k not in ['n_input_features', 'n_output_features']})
            model.fit(X_train, y_train)
            val_results = model.evaluate(X_val, y_val, metrics=['mse'])
            best_params = config
            val_score = val_results['mse']
            best_combo_train_time = None
            final_train_time = model.training_time

    elif model_type == 'lstm':
        if use_tuning and param_grid:
            # Use hyperparameter tuning
            results = tune_lstm_hyperparameters(
                X_train, y_train, X_val, y_val,
                param_grid=param_grid,
                lstm_device=lstm_device,
                verbose=verbose,
                layer_hidden_map=layer_hidden_map
            )
            best_params = results['best_params']
            val_score = results['best_score']
            # Extract best combination's training time and best epoch from all_results
            best_combo_train_time = None
            best_epoch = None
            for result in results['all_results']:
                if result['params'] == best_params:
                    best_combo_train_time = result.get('train_time')
                    best_epoch = result.get('best_epoch')
                    break
            del results  # Explicitly delete results dict to free memory
            import gc
            gc.collect()

            # Train final model with best params on combined train+val data
            X_full = np.concatenate([X_train, X_val], axis=0)
            y_full = np.concatenate([y_train, y_val], axis=0)

            if verbose:
                print(f"\n    Starting training LSTM...")
            model = LSTMModel(**best_params, device=lstm_device, verbose=verbose)
            # Fix B1: use best_epoch from tuning to avoid overfitting without val data
            if best_epoch and best_epoch > 0:
                model.epochs = best_epoch
                if verbose:
                    print(f"    Using best_epoch={best_epoch} from tuning (fix B1)")
            model.fit(X_full, y_full)
            final_train_time = model.training_time
        else:
            # Simple training
            model = LSTMModel(**{k: v for k, v in config.items()
                                if k not in ['n_input_features', 'n_output_features']},
                            device=lstm_device)
            model.fit(X_train, y_train)
            val_results = model.evaluate(X_val, y_val, metrics=['mse'])
            best_params = config
            val_score = val_results['mse']
            best_combo_train_time = None
            final_train_time = model.training_time

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    tune_time = time.time() - start_time

    # Evaluate on test set
    test_results = model.evaluate(X_test, y_test, metrics=['mse', 'mae', 'r2'])

    # =========================================================================
    # GREEN COMPUTING METRIC: Inference Time Measurement
    # Measure prediction time on test set (important for deployment scenarios)
    # =========================================================================
    n_test_samples = X_test.shape[0]

    # Warm-up run (first run may include JIT compilation overhead)
    _ = model.predict(X_test[:min(10, n_test_samples)])

    # Measure total inference time
    inference_start = time.time()
    _ = model.predict(X_test)
    total_inference_time = time.time() - inference_start

    # Calculate per-sample inference time (in milliseconds)
    per_sample_inference_ms = (total_inference_time / n_test_samples) * 1000

    # Calculate parameter count from ACTUAL best params (not default config)
    param_config = dict(best_params)
    param_config['n_input_features'] = config.get('n_input_features', 3)
    param_config['n_output_features'] = config.get('n_output_features', 3)
    param_info = calculate_model_params(model_type, param_config)

    # Stop green metrics tracking
    green_metrics = {}
    if green_tracker:
        green_metrics = green_tracker.stop()

    if verbose:
        print(f"\n✅ Training Complete!")
        print(f"Trainable parameters: {param_info['trainable']:,}")
        print(f"Total parameters: {param_info['total']:,}")
        if best_combo_train_time is not None:
            print(f"Best combo train time: {best_combo_train_time:.2f}s (single hyperparameter combination)")
        print(f"Final model train time: {final_train_time:.2f}s (final model with best params)")
        print(f"Total tune time: {tune_time:.2f}s (entire tuning process)")
        print(f"Inference time: {total_inference_time:.4f}s ({per_sample_inference_ms:.4f}ms/sample)")
        if green_metrics:
            print(f"Memory peak: {green_metrics.get('memory_peak_mb', 0):.2f} MB")
            if 'emissions_kg_co2' in green_metrics:
                print(f"Energy: {green_metrics.get('energy_kwh', 0):.6f} kWh")
                print(f"CO2 emissions: {green_metrics.get('emissions_kg_co2', 0):.6f} kg")
        print(f"Validation MSE: {val_score:.6f}")
        print(f"Test MSE: {test_results['mse']:.6f}")
        print(f"Test MAE: {test_results['mae']:.6f}")
        print(f"Test R²: {test_results['r2']:.6f}")

    result = {
        'model_type': model_type,
        'config': best_params,
        'param_info': param_info,
        'tune_time': tune_time,  # Total time for hyperparameter search + final training
        'best_combo_train_time': best_combo_train_time,  # Training time of best hyperparameter combination
        'final_train_time': final_train_time,  # Training time of final model
        # Green computing metrics - time
        'inference_time_total': total_inference_time,  # Total inference time on test set (seconds)
        'inference_time_per_sample_ms': per_sample_inference_ms,  # Per-sample inference time (milliseconds)
        'n_test_samples': n_test_samples,  # Number of test samples
        # Evaluation metrics
        'val_mse': val_score,
        'test_mse': test_results['mse'],
        'test_mae': test_results['mae'],
        'test_r2': test_results['r2'],
    }

    # Add green metrics if tracked
    if green_metrics:
        result['memory_peak_mb'] = green_metrics.get('memory_peak_mb')
        result['memory_delta_mb'] = green_metrics.get('memory_delta_mb')
        if 'emissions_kg_co2' in green_metrics:
            result['energy_kwh'] = green_metrics.get('energy_kwh')
            result['emissions_kg_co2'] = green_metrics.get('emissions_kg_co2')

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='lorenz',
                       help='Dataset: lorenz, mackeyglass, santafe')
    parser.add_argument('--length', type=int, default=5000,
                       help='Dataset length')
    parser.add_argument('--train-frac', '--train-ratio', type=float, default=0.6,
                       dest='train_frac',
                       help='Training fraction (0-1)')
    parser.add_argument('--n-in', type=int, default=100,
                       help='Input window size')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--tune-pyreco', action='store_true',
                       help='Use hyperparameter tuning for PyReCo (36 combos) AND LSTM (9 combos)')
    parser.add_argument('--tune-all', action='store_true',
                       help='Alias for --tune-pyreco (both enable fair comparison with tuning)')
    parser.add_argument('--lstm-device', type=str, default='auto',
                       choices=['auto', 'cpu', 'mps', 'cuda'],
                       help='Device for LSTM: auto (default, uses MPS if available), cpu (fair comparison), mps, cuda')
    parser.add_argument('--output', type=str, default='results_scaling.json',
                       help='Output file for results')
    # Green computing metrics options
    parser.add_argument('--track-green', action='store_true',
                       help='Track green computing metrics (memory usage, energy consumption)')
    parser.add_argument('--country-code', type=str, default='USA',
                       help='Country ISO code for carbon intensity calculation (default: USA)')
    args = parser.parse_args()

    # Print CodeCarbon availability
    if args.track_green:
        if CODECARBON_AVAILABLE:
            print("✅ CodeCarbon available - will track energy consumption and CO2 emissions")
        else:
            print("⚠️  CodeCarbon not installed - will only track memory usage")
            print("   Install with: pip install codecarbon")

    set_seed(args.seed)

    # ============================================================================
    # Pre-check: Validate train_fraction + val_fraction < 1.0
    # ============================================================================
    VAL_FRACTION = 0.15  # Fixed validation fraction used throughout
    if args.train_frac + VAL_FRACTION >= 1.0:
        raise ValueError(
            f"Invalid configuration: train_fraction ({args.train_frac}) + val_fraction ({VAL_FRACTION}) = "
            f"{args.train_frac + VAL_FRACTION:.2f} >= 1.0. "
            f"This leaves no data for testing. Use train_fraction <= {1.0 - VAL_FRACTION - 0.05:.2f} "
            f"to ensure at least 5% test data."
        )

    print("\n" + "="*100)
    print("MODEL SCALING TEST: Performance Across Parameter Budgets")
    print("="*100)
    print(f"\nDataset: {args.dataset}")
    print(f"Length: {args.length}")
    print(f"Seed: {args.seed}")

    # 1) Load and prepare data
    print("\n" + "="*100)
    print("STEP 1: Load and Prepare Data")
    print("="*100)

    # ============================================================================
    # Original manual data processing (commented out)
    # ============================================================================
    # data, meta = load_data(args.dataset, length=args.length, seed=args.seed)
    # Dout = meta["Dout"]
    # print(f"Data shape: {data.shape}, Output dimension: {Dout}")
    #
    # # Split
    # train, test, split = process_datasets.split_datasets(data, args.train_frac)
    # n_tr = int(0.85 * len(train))
    # series_train = train[:n_tr]
    # series_val = train[n_tr:]
    #
    # print(f"Train: {series_train.shape}, Val: {series_val.shape}, Test: {test.shape}")
    #
    # # Standardize
    # scaler = StandardScaler().fit(series_train)
    # series_train = scaler.transform(series_train)
    # series_val = scaler.transform(series_val)
    # series_test = scaler.transform(test)
    #
    # # Create windows
    # Xtr, Ytr = process_datasets.sliding_window(series_train, args.n_in, 1)
    # Xva, Yva = process_datasets.sliding_window(series_val, args.n_in, 1)
    # Xte, Yte = process_datasets.sliding_window(series_test, args.n_in, 1)
    #
    # print(f"Windows - Train: {Xtr.shape}, Val: {Xva.shape}, Test: {Xte.shape}")

    # ============================================================================
    # Data loading: PyReCo for lorenz/mackeyglass, local_load for santafe
    # Both use consistent split ratios: train=train_frac, val=0.15, test=remaining
    # ============================================================================

    use_pyreco = USE_PYRECO_DATASETS and args.dataset.lower() in PYRECO_SUPPORTED_DATASETS

    if use_pyreco:
        # Use PyReCo datasets.load() for supported datasets
        print(f"Using PyReCo datasets.load() for {args.dataset}...")

        Xtr, Ytr, Xva, Yva, Xte, Yte, scaler = pyreco_load(
            args.dataset,
            n_samples=args.length,
            seed=args.seed,
            val_fraction=0.15,
            train_fraction=args.train_frac,
            n_in=args.n_in,
            n_out=1,
            standardize=True
        )
    else:
        # Use local_load (same interface as pyreco_load)
        print(f"Using local_load() for {args.dataset}...")

        Xtr, Ytr, Xva, Yva, Xte, Yte, scaler = local_load(
            args.dataset,
            n_samples=args.length,
            seed=args.seed,
            val_fraction=0.15,
            train_fraction=args.train_frac,
            n_in=args.n_in,
            n_out=1,
            standardize=True
        )

    Dout = Xtr.shape[-1]  # Get output dimension from data shape

    print(f"✅ Data loaded and preprocessed automatically")
    print(f"Dataset: {args.dataset}, Length: {args.length}, Output dimension: {Dout}")
    print(f"Windows - Train: {Xtr.shape}, Val: {Xva.shape}, Test: {Xte.shape}")
    print(f"Data is standardized with mean≈0, std≈1")
    print(f"Train data mean: {Xtr.mean():.3f}, std: {Xtr.std():.3f}")

    # 2) Define parameter budgets (Total Parameters)
    budgets = {
        'small': 1000,      # ~100 nodes for RC, ~16 hidden for LSTM
        'medium': 10000,    # ~300 nodes for RC, ~28 hidden for LSTM
        'large': 50000,     # ~700 nodes for RC, ~48 hidden for LSTM
    }

    print("\n" + "="*100)
    print("STEP 2: Parameter Budgets (Total Parameters)")
    print("="*100)
    print("Comparing models with equal total parameter count")
    print("  - RC: Total = Input + Reservoir (fixed) + Readout (trainable)")
    print("  - LSTM: Total = All parameters (all trainable)")
    print("  - Literature: Lukoševičius (2012), Jaeger et al. (2007)")
    print("")
    for scale, budget in budgets.items():
        print(f"{scale.capitalize():>10}: {budget:>10,} total parameters")

    # 3) Test all models at all scales
    all_results = {}

    for scale, budget in budgets.items():
        print("\n" + "="*100)
        print(f"STEP 3: Testing Models at {scale.upper()} Scale ({budget:,} parameters)")
        print("="*100)

        # Get configs for this budget
        configs = get_model_configs_for_budget(budget, n_input_features=Dout, n_output_features=Dout)

        scale_results = []

        # Test each model type (pyreco_custom removed - same implementation as standard)
        for model_type in ['pyreco_standard', 'lstm']:
            config = configs[model_type]

            # Define param_grid and decide whether to use tuning
            param_grid = None
            use_tuning = False
            layer_hidden_map = None

            if model_type == 'pyreco_standard':
                if args.tune_all or args.tune_pyreco:
                    use_tuning = True
                    # Minimal grids covering all 3 budgets' optimal params
                    # Based on 5-fold CV pretuning (pretuning_{dataset}_merged.json)
                    if args.dataset == 'lorenz':
                        # small: sr=0.95, lr=0.7; medium: sr=0.99, lr=0.8; large: sr=0.99, lr=0.7
                        param_grid = {
                            'num_nodes': [config['num_nodes']],
                            'spec_rad': [0.95, 0.99],
                            'leakage_rate': [0.7, 0.8, 1.0],
                            'density': [0.01],
                            'fraction_input': [0.1],
                            'fraction_output': [config['fraction_output']],
                        }  # 2×3×1×1 = 6 combinations
                    elif args.dataset == 'mackeyglass':
                        # small/large: fi=0.1; medium: fi=0.3
                        param_grid = {
                            'num_nodes': [config['num_nodes']],
                            'spec_rad': [0.99],
                            'leakage_rate': [0.7, 1.0],
                            'density': [0.03],
                            'fraction_input': [0.1, 0.3],
                            'fraction_output': [config['fraction_output']],
                        }  # 1×2×1×2 = 4 combinations
                    elif args.dataset == 'santafe':
                        # small: sr=0.95; medium/large: sr=0.99
                        param_grid = {
                            'num_nodes': [config['num_nodes']],
                            'spec_rad': [0.95, 0.99],
                            'leakage_rate': [0.8, 1.0],
                            'density': [0.01],
                            'fraction_input': [0.1],
                            'fraction_output': [config['fraction_output']],
                        }  # 2×2×1×1 = 4 combinations
                    else:
                        param_grid = {
                            'num_nodes': [config['num_nodes']],
                            'spec_rad': [0.95, 0.99],
                            'leakage_rate': [0.7, 0.8, 1.0],
                            'density': [0.01, 0.03],
                            'fraction_input': [0.1],
                            'fraction_output': [config['fraction_output']],
                        }  # 2×2×2×1 = 8 combinations

            elif model_type == 'pyreco_custom':
                if args.tune_all:
                    use_tuning = True
                    param_grid = {
                        'num_nodes': [config['num_nodes']],  # Fixed by budget
                        'spec_rad': [0.85, 0.95, 1.0],
                        'leakage_rate': [0.2, 0.3, 0.4],
                        'density': [0.03, 0.05, 0.1],  # Sparser for custom
                        'fraction_output': [config['fraction_output']],  # Fixed by budget
                    }

            elif model_type == 'lstm':
                # LSTM hyperparameter tuning
                # Literature support:
                # - Greff et al. (2017) "LSTM: A Search Space Odyssey": learning_rate is most critical
                # - Zaremba et al. (2014): dropout 0.1-0.3 works well for regularization
                # - Gal & Ghahramani (2016): dropout rates 0.1-0.3 optimal for sequence tasks
                if args.tune_all or args.tune_pyreco:
                    use_tuning = True
                    # Compute hidden_size for each num_layers to match budget
                    layer_hidden_map = {
                        nl: compute_lstm_hidden_size(budget, nl, Dout, Dout)
                        for nl in [1, 2]
                    }
                    # LSTM grid: num_layers × learning_rate × dropout
                    # hidden_size is computed from layer_hidden_map (not in grid)
                    # 2 × 5 × 4 = 40 combinations
                    param_grid = {
                        'num_layers': [1, 2],
                        'learning_rate': [0.0005, 0.001, 0.002, 0.005, 0.01],
                        'dropout': [0.0, 0.1, 0.2, 0.3],
                    }

            try:
                result = test_model_at_scale(
                    model_type, config,
                    Xtr, Ytr, Xva, Yva, Xte, Yte,
                    use_tuning=use_tuning,
                    param_grid=param_grid,
                    lstm_device=args.lstm_device,
                    verbose=True,
                    track_green_metrics=args.track_green,
                    country_iso_code=args.country_code,
                    layer_hidden_map=layer_hidden_map
                )
                scale_results.append(result)
            except Exception as e:
                print(f"\n❌ Error testing {model_type}: {str(e)}")
                import traceback
                traceback.print_exc()

        all_results[scale] = scale_results

        # Memory optimization: garbage collection after each scale
        import gc
        gc.collect()
        # Also clear GPU cache if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # 4) Print summary
    print("\n" + "="*100)
    print("FINAL SUMMARY")
    print("="*100)

    for scale, results in all_results.items():
        print(f"\n{scale.upper()} Scale:")
        if args.track_green:
            print(f"{'Model':<20} {'Params':<10} {'Train(s)':<9} {'Mem(MB)':<9} {'Test MSE':<12} {'Test R²':<10}")
        else:
            print(f"{'Model':<25} {'Params':<12} {'Train(s)':<10} {'Test MSE':<12} {'Test R²':<10}")
        print("-" * 100)
        for r in results:
            # Display final_train_time (single model training time) instead of tune_time
            train_time_display = r.get('final_train_time', r.get('tune_time', 0))
            if args.track_green:
                mem_mb = r.get('memory_peak_mb', 0) or 0
                print(f"{r['model_type']:<20} {r['param_info']['trainable']:<10,} "
                      f"{train_time_display:<9.2f} {mem_mb:<9.1f} {r['test_mse']:<12.6f} {r['test_r2']:<10.6f}")
            else:
                print(f"{r['model_type']:<25} {r['param_info']['trainable']:<12,} "
                      f"{train_time_display:<10.2f} {r['test_mse']:<12.6f} {r['test_r2']:<10.6f}")

    # Print green metrics summary if tracked
    if args.track_green:
        print("\n" + "="*100)
        print("GREEN COMPUTING METRICS SUMMARY")
        print("="*100)
        for scale, results in all_results.items():
            print(f"\n{scale.upper()} Scale:")
            for r in results:
                mem_peak = r.get('memory_peak_mb', 'N/A')
                energy = r.get('energy_kwh', 'N/A')
                co2 = r.get('emissions_kg_co2', 'N/A')
                print(f"  {r['model_type']:<20}: Memory={mem_peak:.1f}MB" if isinstance(mem_peak, float) else f"  {r['model_type']:<20}: Memory=N/A", end="")
                if isinstance(energy, float):
                    print(f", Energy={energy:.6f}kWh, CO2={co2:.6f}kg")
                else:
                    print("")

    # 5) Save results
    output_data = {
        'metadata': {
            'dataset': args.dataset,
            'length': args.length,
            'seed': args.seed,
            'n_in': args.n_in,
            'train_frac': args.train_frac,
            'timestamp': datetime.now().isoformat(),
            'green_metrics_enabled': args.track_green,
            'codecarbon_available': CODECARBON_AVAILABLE if args.track_green else None,
            'country_code': args.country_code if args.track_green else None,
        },
        'budgets': budgets,
        'results': all_results,
    }

    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)

    print(f"\n✅ Results saved to: {args.output}")
    print("\n" + "="*100)
    print("TEST COMPLETE!")
    print("="*100)


if __name__ == '__main__':
    main()
