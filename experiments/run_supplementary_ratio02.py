"""
Supplementary Experiment: Broad PyReCo Grid Search at train_ratio=0.2

Purpose: Determine whether PyReCo's poor performance at low training data is due to:
  1. Parameter mismatch (pretuned grids optimized for higher ratios)
  2. Architectural limitation (ESN genuinely struggles with little data)

Approach: Use a BROAD PyReCo grid at train_ratio=0.2 (not the narrow pretuned grid)
and compare R² with the main experiment results.

If broad grid significantly improves R² → parameter mismatch (pretuned grid suboptimal)
If broad grid shows similar R² → architectural limitation (ESN needs more data)

Configuration:
  - Datasets: lorenz, mackeyglass, santafe
  - Seeds: 42, 43, 44, 45, 46
  - train_ratio: 0.2 (fixed)
  - Budgets: small (1K), medium (10K), large (50K)
  - PyReCo BROAD grid (180 combinations):
      spec_rad:      [0.5, 0.8, 0.95, 0.99]
      leakage_rate:  [0.1, 0.3, 0.5, 0.7, 1.0]
      density:       [0.01, 0.03, 0.1]
      fraction_input:[0.1, 0.3, 0.5]
  - LSTM: same 40 combinations as main experiment (baseline)
  - Total: 3 datasets × 5 seeds = 15 experiments

Usage:
    python run_supplementary_ratio02.py
    python run_supplementary_ratio02.py --quick          # 1 dataset × 1 seed
    python run_supplementary_ratio02.py --seeds 42 43    # specific seeds

    # Run in background:
    nohup python -u run_supplementary_ratio02.py > ../logs/exp_ratio02.log 2>&1 &
"""

import json
import math
import time
import gc
import argparse
import sys
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.load_dataset import load as local_load, set_seed
from src.utils.node_number import best_num_nodes_and_fraction_out

try:
    from pyreco.datasets import load as pyreco_load
    USE_PYRECO_DATASETS = True
except ImportError:
    USE_PYRECO_DATASETS = False

PYRECO_SUPPORTED_DATASETS = {'lorenz', 'mackeyglass', 'mackey-glass', 'mg'}

from models.pyreco_wrapper import PyReCoStandardModel, tune_pyreco_hyperparameters
from models.lstm_model import LSTMModel, tune_lstm_hyperparameters


def compute_lstm_hidden_size(budget, num_layers, n_input, n_output):
    """Compute LSTM hidden_size that fits the parameter budget for given num_layers."""
    a = 8 * num_layers - 4
    b = 4 * n_input + n_output
    h = int((-b + math.sqrt(b * b + 4 * a * budget)) / (2 * a))
    return max(8, min(h, 512))


def load_data(dataset, seed, train_ratio=0.2, n_in=100, length=5000):
    """Load and prepare data, returns (Xtr, Ytr, Xva, Yva, Xte, Yte, Dout)"""
    set_seed(seed)

    use_pyreco = USE_PYRECO_DATASETS and dataset.lower() in PYRECO_SUPPORTED_DATASETS
    load_fn = pyreco_load if use_pyreco else local_load

    Xtr, Ytr, Xva, Yva, Xte, Yte, scaler = load_fn(
        dataset,
        n_samples=length,
        seed=seed,
        val_fraction=0.15,
        train_fraction=train_ratio,
        n_in=n_in,
        n_out=1,
        standardize=True
    )

    Dout = Xtr.shape[-1]
    return Xtr, Ytr, Xva, Yva, Xte, Yte, Dout


def get_pyreco_num_nodes(budget, density=0.1):
    """Calculate num_nodes from total parameter budget."""
    num_nodes_approx = int(math.sqrt(budget / density))
    candidates = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    return min(candidates, key=lambda x: abs(x - num_nodes_approx))


def run_single_experiment(dataset, seed, output_dir):
    """Run one experiment: broad PyReCo grid + LSTM at train_ratio=0.2"""
    print(f"\n{'='*80}")
    print(f"Running: {dataset} | seed={seed} | train_ratio=0.2 (BROAD GRID)")
    print(f"{'='*80}")

    start_time = time.time()

    # Load data
    Xtr, Ytr, Xva, Yva, Xte, Yte, Dout = load_data(dataset, seed)
    print(f"Data: Train={Xtr.shape}, Val={Xva.shape}, Test={Xte.shape}, Dout={Dout}")

    budgets = {'small': 1000, 'medium': 10000, 'large': 50000}
    all_results = {}

    for scale, budget in budgets.items():
        print(f"\n--- {scale.upper()} ({budget:,} params) ---")
        scale_results = []

        # ====================================================================
        # PyReCo: BROAD grid search (180 combinations)
        # Key difference from main experiment: includes lower spec_rad and
        # leakage_rate values that might be better for small training sets
        # ====================================================================
        num_nodes = get_pyreco_num_nodes(budget)
        fraction_output = 1.0

        pyreco_grid = {
            'num_nodes': [num_nodes],
            'spec_rad': [0.5, 0.8, 0.95, 0.99],
            'leakage_rate': [0.1, 0.3, 0.5, 0.7, 1.0],
            'density': [0.01, 0.03, 0.1],
            'fraction_input': [0.1, 0.3, 0.5],
            'fraction_output': [fraction_output],
        }
        n_combos = 4 * 5 * 3 * 3  # 180
        print(f"  PyReCo: {n_combos} combinations (broad grid)")

        tune_start = time.time()
        pyreco_results = tune_pyreco_hyperparameters(
            Xtr, Ytr, Xva, Yva,
            param_grid=pyreco_grid,
            default_spec_rad=0.95,
            default_leakage=0.7,
            default_density=0.01,
            default_activation='tanh',
            default_fraction_input=0.1,
            verbose=True
        )

        best_params = pyreco_results['best_params']
        val_score = pyreco_results['best_score']

        # Extract best combo training time
        best_combo_train_time = None
        for r in pyreco_results['all_results']:
            if r['params'] == best_params and r['success']:
                best_combo_train_time = r['train_time']
                break

        del pyreco_results
        gc.collect()

        # Train final model on combined train+val
        model = PyReCoStandardModel(**{k: v for k, v in best_params.items()
                                       if k not in ['n_input_features', 'n_output_features']})
        X_full = np.concatenate([Xtr, Xva], axis=0)
        y_full = np.concatenate([Ytr, Yva], axis=0)
        model.fit(X_full, y_full)
        final_train_time = model.training_time

        pyreco_tune_time = time.time() - tune_start

        # Evaluate
        test_results = model.evaluate(Xte, Yte, metrics=['mse', 'mae', 'r2'])

        # Param info
        from experiments.test_model_scaling import calculate_model_params
        config_for_params = {**best_params, 'n_input_features': Dout, 'n_output_features': Dout}
        param_info = calculate_model_params('pyreco_standard', config_for_params)

        scale_results.append({
            'model_type': 'pyreco_standard',
            'grid_type': 'broad',
            'config': best_params,
            'param_info': param_info,
            'tune_time': pyreco_tune_time,
            'best_combo_train_time': best_combo_train_time,
            'final_train_time': final_train_time,
            'n_test_samples': Xte.shape[0],
            'val_mse': val_score,
            'test_mse': test_results['mse'],
            'test_mae': test_results['mae'],
            'test_r2': test_results['r2'],
        })

        print(f"  PyReCo best: {best_params}")
        print(f"  PyReCo R²={test_results['r2']:.6f}, MSE={test_results['mse']:.6f}")

        del model
        gc.collect()

        # ====================================================================
        # LSTM: same 40 combinations as main experiment (for comparison)
        # ====================================================================
        layer_hidden_map = {
            nl: compute_lstm_hidden_size(budget, nl, Dout, Dout)
            for nl in [1, 2]
        }
        lstm_grid = {
            'num_layers': [1, 2],
            'learning_rate': [0.0005, 0.001, 0.002, 0.005, 0.01],
            'dropout': [0.0, 0.1, 0.2, 0.3],
        }
        print(f"  LSTM: 40 combinations (same as main experiment)")

        lstm_tune_start = time.time()
        lstm_results = tune_lstm_hyperparameters(
            Xtr, Ytr, Xva, Yva,
            param_grid=lstm_grid,
            verbose=True,
            layer_hidden_map=layer_hidden_map
        )

        lstm_best = lstm_results['best_params']
        lstm_val_score = lstm_results['best_score']

        lstm_combo_time = None
        for r in lstm_results['all_results']:
            if r['params'] == lstm_best:
                lstm_combo_time = r['train_time']
                break

        del lstm_results
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        # Train final LSTM on combined train+val
        lstm_model = LSTMModel(**lstm_best, verbose=True)
        lstm_model.fit(X_full, y_full)
        lstm_final_time = lstm_model.training_time
        lstm_tune_time = time.time() - lstm_tune_start

        # Evaluate
        lstm_test = lstm_model.evaluate(Xte, Yte, metrics=['mse', 'mae', 'r2'])

        lstm_config_for_params = {**lstm_best, 'n_input_features': Dout, 'n_output_features': Dout}
        lstm_param_info = calculate_model_params('lstm', lstm_config_for_params)

        scale_results.append({
            'model_type': 'lstm',
            'config': lstm_best,
            'param_info': lstm_param_info,
            'tune_time': lstm_tune_time,
            'best_combo_train_time': lstm_combo_time,
            'final_train_time': lstm_final_time,
            'n_test_samples': Xte.shape[0],
            'val_mse': lstm_val_score,
            'test_mse': lstm_test['mse'],
            'test_mae': lstm_test['mae'],
            'test_r2': lstm_test['r2'],
        })

        print(f"  LSTM best: {lstm_best}")
        print(f"  LSTM R²={lstm_test['r2']:.6f}, MSE={lstm_test['mse']:.6f}")

        del lstm_model
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        all_results[scale] = scale_results

    total_time = time.time() - start_time

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"supp_ratio02_{dataset}_seed{seed}_{timestamp}.json"

    output_data = {
        'metadata': {
            'experiment_type': 'supplementary_ratio02_broad_grid',
            'dataset': dataset,
            'seed': seed,
            'train_frac': 0.2,
            'n_in': 100,
            'length': 5000,
            'timestamp': datetime.now().isoformat(),
            'pyreco_grid_size': 180,
            'lstm_grid_size': 40,
        },
        'budgets': {'small': 1000, 'medium': 10000, 'large': 50000},
        'results': all_results,
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n  Saved: {output_file} ({total_time:.0f}s)")
    return True, total_time, output_file


def main():
    parser = argparse.ArgumentParser(
        description='Supplementary: broad PyReCo grid at train_ratio=0.2')
    parser.add_argument('--quick', action='store_true',
                       help='Quick test: 1 dataset (lorenz) × 1 seed (42)')
    parser.add_argument('--seeds', nargs='+', type=int, default=None,
                       help='Specific seeds (default: 42-46)')
    parser.add_argument('--datasets', nargs='+', type=str, default=None,
                       help='Specific datasets (default: all 3)')
    parser.add_argument('--output-dir', type=str,
                       default='../results/supplementary_ratio02',
                       help='Output directory')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    if args.quick:
        datasets = ['lorenz']
        seeds = [42]
        print("\nQUICK TEST MODE")
    else:
        datasets = args.datasets or ['lorenz', 'mackeyglass', 'santafe']
        seeds = args.seeds or [42, 43, 44, 45, 46]
        print("\nFULL SUPPLEMENTARY MODE")

    total = len(datasets) * len(seeds)

    print(f"\n{'='*80}")
    print("SUPPLEMENTARY EXPERIMENT: Broad PyReCo Grid at train_ratio=0.2")
    print(f"{'='*80}")
    print(f"Datasets: {datasets}")
    print(f"Seeds: {seeds}")
    print(f"Train ratio: 0.2 (fixed)")
    print(f"Total experiments: {total}")
    print(f"Output: {output_dir}")
    print(f"\nPyReCo BROAD grid (180 combos):")
    print(f"  spec_rad:      [0.5, 0.8, 0.95, 0.99]")
    print(f"  leakage_rate:  [0.1, 0.3, 0.5, 0.7, 1.0]")
    print(f"  density:       [0.01, 0.03, 0.1]")
    print(f"  fraction_input:[0.1, 0.3, 0.5]")
    print(f"\nLSTM: 40 combos (same as main experiment)")
    print(f"{'='*80}")

    all_exp_results = []
    successful = 0
    failed = 0
    start_time = time.time()

    for i, dataset in enumerate(datasets):
        for j, seed in enumerate(seeds):
            exp_num = i * len(seeds) + j + 1

            print(f"\n{'='*80}")
            print(f"EXPERIMENT {exp_num}/{total} | "
                  f"Progress: {exp_num/total*100:.0f}% | "
                  f"Elapsed: {(time.time()-start_time)/3600:.2f}h")
            print(f"{'='*80}")

            success, runtime, output_file = run_single_experiment(
                dataset, seed, output_dir
            )

            if success:
                successful += 1
            else:
                failed += 1

            all_exp_results.append({
                'experiment': exp_num,
                'dataset': dataset,
                'seed': seed,
                'success': success,
                'runtime': runtime,
                'output_file': str(output_file) if output_file else None,
            })

            # Save progress
            progress_file = output_dir / 'experiment_progress.json'
            with open(progress_file, 'w') as f:
                json.dump({
                    'total': total,
                    'completed': exp_num,
                    'successful': successful,
                    'failed': failed,
                    'results': all_exp_results
                }, f, indent=2)

    total_time = time.time() - start_time

    print(f"\n{'='*80}")
    print("SUPPLEMENTARY EXPERIMENTS COMPLETE!")
    print(f"{'='*80}")
    print(f"Total: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total time: {total_time/3600:.2f} hours")
    if total > 0:
        print(f"Average: {total_time/total:.0f}s per experiment")
    print(f"Results: {output_dir}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
