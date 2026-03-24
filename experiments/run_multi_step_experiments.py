"""
Multi-Step Prediction Experiments with Best Configurations

This script runs multi-step (free-run) prediction experiments using the best
configurations found from the single-step tuning experiments.

Scientific Rationale:
- Each (dataset, train_ratio, budget) combination uses its own best configuration
- This ensures fair comparison: models are evaluated at their optimal settings
- Follows the principle that hyperparameters interact with data characteristics

Experimental Design:
- 3 datasets × 6 train_ratios × 3 budgets × 5 seeds = 270 experiments
- For each experiment:
  1. Load best config from single-step results
  2. Train model with best config
  3. Evaluate on multiple prediction horizons (1, 5, 10, 20, 50 steps)
  4. Record metrics: MSE, MAE, R², NRMSE at each horizon

Output: JSON files with multi-step prediction results
"""

import argparse
import json
import glob
import os
import sys
import time
import numpy as np
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.load_dataset import load as local_load, set_seed
from models.pyreco_wrapper import PyReCoStandardModel
from models.lstm_model import LSTMModel
from src.utils.evaluation import evaluate_multi_step, multi_step_predict
from pyreco.datasets import load as pyreco_load

# Lorenz/MG use pyreco_load, Santa Fe uses local_load
PYRECO_DATASETS = {'lorenz', 'mackeyglass'}

# Default horizons for multi-step evaluation
DEFAULT_HORIZONS = [1, 5, 10, 20, 50]


def load_best_configs(results_dir="results/final_v2"):
    """
    Load best configurations from single-step experiment results.

    For each (dataset, train_frac, budget) combination, finds the config
    with lowest average MSE across seeds.

    Returns:
        dict: Nested dict of best configs
              best_configs[dataset][train_frac][budget][model_type] = config
    """
    # Handle relative paths - look relative to project root
    if not os.path.isabs(results_dir):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        results_dir = os.path.join(project_root, results_dir)

    pattern = f"{results_dir}/results_*.json"
    files = glob.glob(pattern)

    if not files:
        raise FileNotFoundError(f"No result files found in {results_dir}")

    # Collect all results (supports both v1 and v2 JSON formats)
    records = []
    for file_path in files:
        with open(file_path, 'r') as f:
            data = json.load(f)

        metadata = data['metadata']
        # v2 format: metadata has budget_name, results keyed by budget_name
        # v1 format: budgets dict, results keyed by budget names
        if 'budget_name' in metadata:
            budget_names = [metadata['budget_name']]
        elif 'budgets' in data:
            budget_names = list(data['budgets'].keys())
        else:
            budget_names = list(data['results'].keys())

        for budget_name in budget_names:
            for model_result in data['results'].get(budget_name, []):
                records.append({
                    'dataset': metadata['dataset'],
                    'seed': metadata['seed'],
                    'train_frac': metadata['train_frac'],
                    'budget': budget_name,
                    'model_type': model_result['model_type'],
                    'val_mse': model_result['val_mse'],
                    'config': model_result['config']
                })

    # Find best config for each (dataset, train_frac, budget, model_type)
    # Uses val_mse (not test_mse) to avoid test-set leakage
    best_configs = {}

    for record in records:
        dataset = record['dataset']
        train_frac = record['train_frac']
        budget = record['budget']
        model_type = record['model_type']

        # Initialize nested dicts
        if dataset not in best_configs:
            best_configs[dataset] = {}
        if train_frac not in best_configs[dataset]:
            best_configs[dataset][train_frac] = {}
        if budget not in best_configs[dataset][train_frac]:
            best_configs[dataset][train_frac][budget] = {}
        if model_type not in best_configs[dataset][train_frac][budget]:
            best_configs[dataset][train_frac][budget][model_type] = {
                'configs': [],
                'mses': []
            }

        best_configs[dataset][train_frac][budget][model_type]['configs'].append(record['config'])
        best_configs[dataset][train_frac][budget][model_type]['mses'].append(record['val_mse'])

    # Average MSE across seeds and select best config
    final_configs = {}
    for dataset in best_configs:
        final_configs[dataset] = {}
        for train_frac in best_configs[dataset]:
            final_configs[dataset][train_frac] = {}
            for budget in best_configs[dataset][train_frac]:
                final_configs[dataset][train_frac][budget] = {}
                for model_type in best_configs[dataset][train_frac][budget]:
                    data = best_configs[dataset][train_frac][budget][model_type]

                    # Group by config and average MSE
                    config_mse = {}
                    for config, mse in zip(data['configs'], data['mses']):
                        config_key = json.dumps(config, sort_keys=True)
                        if config_key not in config_mse:
                            config_mse[config_key] = []
                        config_mse[config_key].append(mse)

                    # Find config with lowest average MSE
                    best_config_key = min(config_mse.keys(),
                                         key=lambda k: np.mean(config_mse[k]))
                    best_config = json.loads(best_config_key)
                    best_avg_mse = np.mean(config_mse[best_config_key])

                    final_configs[dataset][train_frac][budget][model_type] = {
                        'config': best_config,
                        'avg_mse': best_avg_mse
                    }

    return final_configs


def load_data_for_multistep(dataset, length, n_in, max_horizon, train_frac, seed):
    """
    Load data with extended targets for multi-step evaluation.

    Args:
        dataset: Dataset name
        length: Total sequence length
        n_in: Input window size
        max_horizon: Maximum prediction horizon
        train_frac: Training fraction
        seed: Random seed

    Returns:
        Training, validation, and test data with extended targets
    """
    set_seed(seed)
    load_func = pyreco_load if dataset.lower() in PYRECO_DATASETS else local_load

    # Load single-step data for training
    X_train, y_train, X_val, y_val, X_test_1, y_test_1, scaler = load_func(
        dataset,
        n_samples=length,
        seed=seed,
        val_fraction=0.15,
        train_fraction=train_frac,
        n_in=n_in,
        n_out=1,
        standardize=True
    )

    # Load multi-step targets for testing
    _, _, _, _, X_test_ms, y_test_ms, _ = load_func(
        dataset,
        n_samples=length,
        seed=seed,
        val_fraction=0.15,
        train_fraction=train_frac,
        n_in=n_in,
        n_out=max_horizon,
        standardize=True
    )

    return X_train, y_train, X_val, y_val, X_test_ms, y_test_ms, scaler


def run_single_experiment(dataset, train_frac, budget, seed, best_configs,
                          horizons, length=5000, n_in=100, output_dir="results/multi_step"):
    """
    Run multi-step prediction experiment for a single configuration.

    Args:
        dataset: Dataset name
        train_frac: Training fraction
        budget: Budget name (small/medium/large)
        seed: Random seed
        best_configs: Best configurations dict
        horizons: List of prediction horizons
        length: Dataset length
        n_in: Input window size
        output_dir: Output directory

    Returns:
        dict: Experiment results
    """
    max_horizon = max(horizons)

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset}, Train: {train_frac}, Budget: {budget}, Seed: {seed}")
    print(f"{'='*60}")

    # Get best configs for this combination
    try:
        pyreco_config = best_configs[dataset][train_frac][budget]['pyreco_standard']['config']
        lstm_config = best_configs[dataset][train_frac][budget]['lstm']['config']
    except KeyError as e:
        print(f"⚠️ Config not found: {e}")
        return None

    # Load data
    print("\n📊 Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = load_data_for_multistep(
        dataset, length, n_in, max_horizon, train_frac, seed
    )

    print(f"  Train: {X_train.shape}")
    print(f"  Test: {X_test.shape} -> {y_test.shape}")

    results = {
        'metadata': {
            'dataset': dataset,
            'train_frac': train_frac,
            'budget': budget,
            'seed': seed,
            'n_in': n_in,
            'length': length,
            'horizons': horizons,
            'timestamp': datetime.now().isoformat()
        },
        'models': {}
    }

    # Train and evaluate PyReCo
    print("\n🤖 Training PyReCo...")
    print(f"  Config: {pyreco_config}")

    try:
        # Train on train+val (consistent with run_final_v2.py)
        X_full = np.concatenate([X_train, X_val], axis=0)
        y_full_1step = np.concatenate([y_train, y_val], axis=0)

        set_seed(seed)
        start_time = time.time()
        pyreco_model = PyReCoStandardModel(**pyreco_config, verbose=False)
        pyreco_model.fit(X_full, y_full_1step)
        pyreco_train_time = time.time() - start_time

        print(f"  ✅ Trained in {pyreco_train_time:.2f}s")

        # Multi-step evaluation
        print("  📈 Evaluating multi-step predictions...")
        start_time = time.time()
        pyreco_horizon_results = evaluate_multi_step(
            model=pyreco_model,
            X_test=X_test,
            y_test=y_test,
            horizons=horizons,
            mode='free_run',
            include_advanced_metrics=False  # Speed up evaluation
        )
        pyreco_eval_time = time.time() - start_time

        results['models']['pyreco_standard'] = {
            'config': pyreco_config,
            'train_time': pyreco_train_time,
            'eval_time': pyreco_eval_time,
            'horizon_results': {h: {k: float(v) if isinstance(v, (np.floating, float)) else v
                                   for k, v in r.items()}
                               for h, r in pyreco_horizon_results.items()}
        }

        # Print summary
        for h in [1, 10, max_horizon]:
            if h in pyreco_horizon_results:
                r = pyreco_horizon_results[h]
                print(f"    Horizon {h}: MSE={r['mse']:.6f}, R²={r['r2']:.4f}")

    except Exception as e:
        print(f"  ❌ PyReCo failed: {e}")
        results['models']['pyreco_standard'] = {'error': str(e)}

    # Train and evaluate LSTM
    print("\n🧠 Training LSTM...")
    print(f"  Config: {lstm_config}")

    try:
        # Add fixed params for LSTM
        # High patience since training on X_full without validation —
        # early stopping monitors training loss which decreases more steadily
        full_lstm_config = {
            **lstm_config,
            'epochs': 100,
            'batch_size': 32,
            'patience': 20
        }

        # Train on train+val (consistent with run_final_v2.py)
        X_full = np.concatenate([X_train, X_val], axis=0)
        y_full_1step = np.concatenate([y_train, y_val], axis=0)

        set_seed(seed)
        start_time = time.time()
        lstm_model = LSTMModel(**full_lstm_config, verbose=False)
        lstm_model.fit(X_full, y_full_1step)
        lstm_train_time = time.time() - start_time

        print(f"  ✅ Trained in {lstm_train_time:.2f}s")

        # Multi-step evaluation
        print("  📈 Evaluating multi-step predictions...")
        start_time = time.time()
        lstm_horizon_results = evaluate_multi_step(
            model=lstm_model,
            X_test=X_test,
            y_test=y_test,
            horizons=horizons,
            mode='free_run',
            include_advanced_metrics=False
        )
        lstm_eval_time = time.time() - start_time

        results['models']['lstm'] = {
            'config': lstm_config,
            'train_time': lstm_train_time,
            'eval_time': lstm_eval_time,
            'horizon_results': {h: {k: float(v) if isinstance(v, (np.floating, float)) else v
                                   for k, v in r.items()}
                               for h, r in lstm_horizon_results.items()}
        }

        # Print summary
        for h in [1, 10, max_horizon]:
            if h in lstm_horizon_results:
                r = lstm_horizon_results[h]
                print(f"    Horizon {h}: MSE={r['mse']:.6f}, R²={r['r2']:.4f}")

    except Exception as e:
        print(f"  ❌ LSTM failed: {e}")
        results['models']['lstm'] = {'error': str(e)}

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(
        output_dir,
        f"multistep_{dataset}_seed{seed}_train{train_frac}_{budget}.json"
    )

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results saved to: {output_file}")

    return results


def run_all_experiments(best_configs, datasets=None, train_fracs=None, budgets=None,
                       seeds=None, horizons=None, output_dir="results/multi_step",
                       length=5000, n_in=100):
    """
    Run all multi-step prediction experiments.

    Args:
        best_configs: Best configurations dict
        datasets: List of datasets (default: all)
        train_fracs: List of train fractions (default: all)
        budgets: List of budgets (default: all)
        seeds: List of seeds (default: all)
        horizons: List of horizons
        output_dir: Output directory
        length: Dataset length
        n_in: Input window size

    Returns:
        Summary of all experiments
    """
    # Defaults
    if datasets is None:
        datasets = list(best_configs.keys())
    if train_fracs is None:
        train_fracs = [0.2, 0.3, 0.5, 0.6, 0.7, 0.8]
    if budgets is None:
        budgets = ['small', 'medium', 'large']
    if seeds is None:
        seeds = [42, 43, 44, 45, 46]
    if horizons is None:
        horizons = DEFAULT_HORIZONS

    total_experiments = len(datasets) * len(train_fracs) * len(budgets) * len(seeds)
    print(f"\n🚀 Starting {total_experiments} multi-step experiments")
    print(f"  Datasets: {datasets}")
    print(f"  Train fracs: {train_fracs}")
    print(f"  Budgets: {budgets}")
    print(f"  Seeds: {seeds}")
    print(f"  Horizons: {horizons}")

    summary = {
        'total': total_experiments,
        'completed': 0,
        'failed': 0,
        'results': []
    }

    experiment_num = 0
    for dataset in datasets:
        for train_frac in train_fracs:
            for budget in budgets:
                for seed in seeds:
                    experiment_num += 1
                    print(f"\n[{experiment_num}/{total_experiments}]", end="")

                    try:
                        result = run_single_experiment(
                            dataset=dataset,
                            train_frac=train_frac,
                            budget=budget,
                            seed=seed,
                            best_configs=best_configs,
                            horizons=horizons,
                            length=length,
                            n_in=n_in,
                            output_dir=output_dir
                        )

                        if result:
                            summary['completed'] += 1
                            summary['results'].append({
                                'dataset': dataset,
                                'train_frac': train_frac,
                                'budget': budget,
                                'seed': seed,
                                'success': True
                            })
                        else:
                            summary['failed'] += 1

                    except Exception as e:
                        print(f"\n❌ Experiment failed: {e}")
                        summary['failed'] += 1
                        summary['results'].append({
                            'dataset': dataset,
                            'train_frac': train_frac,
                            'budget': budget,
                            'seed': seed,
                            'success': False,
                            'error': str(e)
                        })

    # Save summary
    summary_file = os.path.join(output_dir, "experiment_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print("MULTI-STEP EXPERIMENTS COMPLETE")
    print(f"{'='*60}")
    print(f"Total: {total_experiments}")
    print(f"Completed: {summary['completed']}")
    print(f"Failed: {summary['failed']}")
    print(f"Summary saved to: {summary_file}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Run multi-step prediction experiments with best configurations"
    )
    parser.add_argument('--results-dir', type=str, default='results/final_v2',
                       help='Directory with single-step experiment results')
    parser.add_argument('--output-dir', type=str, default='results/multi_step',
                       help='Output directory for multi-step results')
    parser.add_argument('--dataset', type=str, default=None,
                       help='Single dataset to run (default: all)')
    parser.add_argument('--train-ratio', type=float, default=None,
                       help='Single train ratio to run (default: all)')
    parser.add_argument('--budget', type=str, default=None,
                       choices=['small', 'medium', 'large'],
                       help='Single budget to run (default: all)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Single seed to run (default: all)')
    parser.add_argument('--horizons', type=str, default='1,5,10,20,50',
                       help='Comma-separated prediction horizons')
    parser.add_argument('--length', type=int, default=5000,
                       help='Dataset length')
    parser.add_argument('--n-in', type=int, default=100,
                       help='Input window size')

    args = parser.parse_args()

    # Parse horizons
    horizons = [int(h) for h in args.horizons.split(',')]

    print("🔬 MULTI-STEP PREDICTION EXPERIMENTS")
    print("=" * 60)
    print(f"Results dir: {args.results_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Horizons: {horizons}")

    # Load best configs
    print("\n📂 Loading best configurations...")
    best_configs = load_best_configs(args.results_dir)

    # Count configs
    total_configs = sum(
        1 for d in best_configs.values()
        for t in d.values()
        for b in t.values()
        for m in b.values()
    )
    print(f"  ✅ Loaded {total_configs} best configurations")

    # Prepare experiment parameters
    datasets = [args.dataset] if args.dataset else None
    train_fracs = [args.train_ratio] if args.train_ratio else None
    budgets = [args.budget] if args.budget else None
    seeds = [args.seed] if args.seed else None

    # Run experiments
    summary = run_all_experiments(
        best_configs=best_configs,
        datasets=datasets,
        train_fracs=train_fracs,
        budgets=budgets,
        seeds=seeds,
        horizons=horizons,
        output_dir=args.output_dir,
        length=args.length,
        n_in=args.n_in
    )

    return summary


if __name__ == '__main__':
    main()
