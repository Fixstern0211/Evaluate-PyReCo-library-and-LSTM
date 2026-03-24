"""
Data Efficiency Experiments: Impact of Data Length on Model Performance

Scientific Rationale:
Data efficiency measures how well a model performs with limited training data.
This is crucial for real-world applications where data collection is expensive.

Key Research Questions:
1. How does model performance scale with data length?
2. Which model (PyReCo vs LSTM) is more data-efficient?
3. At what data length does performance saturate?
4. How does this interact with model complexity (budget)?

Experimental Design:
- Data lengths: [1000, 2000, 3000, 5000, 7000, 10000]
- Fixed train_frac: 0.7 (standard split)
- All 3 datasets × 3 budgets × 5 seeds
- Total: 7 lengths × 3 datasets × 3 budgets × 5 seeds = 315 experiments

Note on Distribution Shift:
- Longer sequences better capture the full attractor of chaotic systems
- We use fixed train_frac to ensure consistent train/test ratio
- This isolates the effect of data quantity from distribution shift

Output: JSON files with performance metrics at each data length
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
from pyreco.datasets import load as pyreco_load

# Lorenz/MG use pyreco_load, Santa Fe uses local_load
PYRECO_DATASETS = {'lorenz', 'mackeyglass'}

# Data lengths to test (log-spaced for better coverage)
# Note: santafe has fixed length ~10093, so longer lengths will be capped for it
DEFAULT_DATA_LENGTHS = [1000, 2000, 3000, 5000, 7000, 10000]

# Budget configurations (same as other experiments)
BUDGET_CONFIGS = {
    'small': {
        'pyreco': {'num_nodes': 100, 'density': 0.1, 'spec_rad': 0.9,
                   'leakage_rate': 0.4, 'fraction_input': 0.5, 'fraction_output': 1.0,
                   'activation': 'tanh', 'optimizer': 'ridge'},
        'lstm': {'hidden_size': 8, 'num_layers': 2, 'dropout': 0.3, 'learning_rate': 0.001}
    },
    'medium': {
        'pyreco': {'num_nodes': 300, 'density': 0.05, 'spec_rad': 0.9,
                   'leakage_rate': 0.5, 'fraction_input': 0.5, 'fraction_output': 1.0,
                   'activation': 'tanh', 'optimizer': 'ridge'},
        'lstm': {'hidden_size': 28, 'num_layers': 2, 'dropout': 0.3, 'learning_rate': 0.002}
    },
    'large': {
        'pyreco': {'num_nodes': 1000, 'density': 0.02, 'spec_rad': 0.95,
                   'leakage_rate': 0.6, 'fraction_input': 0.5, 'fraction_output': 1.0,
                   'activation': 'tanh', 'optimizer': 'ridge'},
        'lstm': {'hidden_size': 64, 'num_layers': 3, 'dropout': 0.4, 'learning_rate': 0.001}
    }
}


def load_best_configs(results_dir="results/final_v2"):
    """
    Load best configurations from single-step experiment results.
    Falls back to default configs if not found.
    """
    if not os.path.isabs(results_dir):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        results_dir = os.path.join(project_root, results_dir)

    pattern = f"{results_dir}/results_*.json"
    files = glob.glob(pattern)

    if not files:
        print(f"⚠️ No result files found in {results_dir}, using default configs")
        return None

    # Collect all results (supports both v1 and v2 JSON formats)
    records = []
    for file_path in files:
        with open(file_path, 'r') as f:
            data = json.load(f)

        metadata = data['metadata']
        # v2 format: metadata has budget_name
        # v1 format: budgets dict with keys
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

    # Find best config for each (dataset, budget, model_type) across train_fracs
    # Uses val_mse (not test_mse) to avoid test-set leakage
    best_configs = {}

    for record in records:
        dataset = record['dataset']
        budget = record['budget']
        model_type = record['model_type']

        key = (dataset, budget, model_type)
        if key not in best_configs:
            best_configs[key] = {'configs': [], 'mses': []}

        best_configs[key]['configs'].append(record['config'])
        best_configs[key]['mses'].append(record['val_mse'])

    # Average MSE and select best config
    final_configs = {}
    for key, data in best_configs.items():
        dataset, budget, model_type = key

        # Group by config and average MSE
        config_mse = {}
        for config, mse in zip(data['configs'], data['mses']):
            config_key = json.dumps(config, sort_keys=True)
            if config_key not in config_mse:
                config_mse[config_key] = []
            config_mse[config_key].append(mse)

        # Find config with lowest average MSE
        best_config_key = min(config_mse.keys(), key=lambda k: np.mean(config_mse[k]))
        best_config = json.loads(best_config_key)

        if dataset not in final_configs:
            final_configs[dataset] = {}
        if budget not in final_configs[dataset]:
            final_configs[dataset][budget] = {}

        final_configs[dataset][budget][model_type] = best_config

    return final_configs


def get_config(best_configs, dataset, budget, model_type):
    """Get config from best_configs or fall back to default."""
    if best_configs and dataset in best_configs:
        if budget in best_configs[dataset]:
            if model_type in best_configs[dataset][budget]:
                return best_configs[dataset][budget][model_type]

    # Fall back to default
    if model_type == 'pyreco_standard':
        return BUDGET_CONFIGS[budget]['pyreco']
    else:
        return BUDGET_CONFIGS[budget]['lstm']


def calculate_metrics(y_true, y_pred):
    """Calculate regression metrics."""
    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(mse))

    # NRMSE (normalized by std of true values)
    y_std = np.std(y_true)
    nrmse = rmse / y_std if y_std > 0 else float('inf')

    # R² score
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'nrmse': nrmse,
        'r2': r2
    }


def run_single_experiment(dataset, data_length, budget, seed, best_configs,
                          train_frac=0.7, n_in=100, output_dir="results/data_efficiency"):
    """
    Run data efficiency experiment for a single configuration.

    Args:
        dataset: Dataset name
        data_length: Total sequence length
        budget: Budget name (small/medium/large)
        seed: Random seed
        best_configs: Best configurations dict (or None for defaults)
        train_frac: Training fraction (default 0.7)
        n_in: Input window size
        output_dir: Output directory

    Returns:
        dict: Experiment results
    """
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset}, Length: {data_length}, Budget: {budget}, Seed: {seed}")
    print(f"{'='*60}")

    # Get configs
    pyreco_config = get_config(best_configs, dataset, budget, 'pyreco_standard')
    lstm_config = get_config(best_configs, dataset, budget, 'lstm')

    # Load data
    print("\n📊 Loading data...")
    try:
        set_seed(seed)
        load_func = pyreco_load if dataset.lower() in PYRECO_DATASETS else local_load
        X_train, y_train, X_val, y_val, X_test, y_test, scaler = load_func(
            dataset,
            n_samples=data_length,
            seed=seed,
            val_fraction=0.15,
            train_fraction=train_frac,
            n_in=n_in,
            n_out=1,
            standardize=True
        )
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return None

    print(f"  Train samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")

    # Check minimum data requirements
    min_samples_required = 50
    if len(X_train) < min_samples_required:
        print(f"⚠️ Insufficient training samples ({len(X_train)} < {min_samples_required})")
        return None

    results = {
        'metadata': {
            'dataset': dataset,
            'data_length': data_length,
            'train_frac': train_frac,
            'budget': budget,
            'seed': seed,
            'n_in': n_in,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'timestamp': datetime.now().isoformat()
        },
        'models': {}
    }

    # Train on train+val (consistent with run_final_v2.py)
    X_full = np.concatenate([X_train, X_val], axis=0)
    y_full = np.concatenate([y_train, y_val], axis=0)

    # Train and evaluate PyReCo
    print("\n🔧 Training PyReCo...")
    try:
        set_seed(seed)
        pyreco_model = PyReCoStandardModel(**pyreco_config, verbose=False)

        start_time = time.time()
        pyreco_model.fit(X_full, y_full)
        train_time = time.time() - start_time

        start_time = time.time()
        y_pred_pyreco = pyreco_model.predict(X_test)
        eval_time = time.time() - start_time

        y_pred_pyreco = np.asarray(y_pred_pyreco).reshape(y_test.shape)
        metrics = calculate_metrics(y_test, y_pred_pyreco)

        results['models']['pyreco_standard'] = {
            'config': pyreco_config,
            'train_time': train_time,
            'eval_time': eval_time,
            'metrics': metrics
        }

        print(f"  ✅ PyReCo: MSE={metrics['mse']:.6f}, R²={metrics['r2']:.4f}, Time={train_time:.2f}s")

    except Exception as e:
        print(f"  ❌ PyReCo failed: {e}")
        results['models']['pyreco_standard'] = {'error': str(e)}

    # Train and evaluate LSTM
    print("\n🔧 Training LSTM...")
    try:
        set_seed(seed)
        lstm_model = LSTMModel(
            hidden_size=lstm_config['hidden_size'],
            num_layers=lstm_config['num_layers'],
            dropout=lstm_config['dropout'],
            learning_rate=lstm_config['learning_rate'],
            epochs=100,
            batch_size=32,
            patience=20,
            verbose=False
        )

        start_time = time.time()
        lstm_model.fit(X_full, y_full)
        train_time = time.time() - start_time

        start_time = time.time()
        y_pred_lstm = lstm_model.predict(X_test)
        eval_time = time.time() - start_time

        y_pred_lstm = np.asarray(y_pred_lstm).reshape(y_test.shape)
        metrics = calculate_metrics(y_test, y_pred_lstm)

        results['models']['lstm'] = {
            'config': lstm_config,
            'train_time': train_time,
            'eval_time': eval_time,
            'metrics': metrics
        }

        print(f"  ✅ LSTM: MSE={metrics['mse']:.6f}, R²={metrics['r2']:.4f}, Time={train_time:.2f}s")

    except Exception as e:
        print(f"  ❌ LSTM failed: {e}")
        results['models']['lstm'] = {'error': str(e)}

    # Save results
    if not os.path.isabs(output_dir):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(project_root, output_dir)

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(
        output_dir,
        f"dataeff_{dataset}_len{data_length}_seed{seed}_{budget}.json"
    )

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Saved to {output_file}")

    return results


def run_all_experiments(datasets, data_lengths, budgets, seeds, train_frac=0.7, output_dir="results/data_efficiency", results_dir="results/final"):
    """Run all data efficiency experiments."""

    # Load best configs
    print("📚 Loading best configurations...")
    best_configs = load_best_configs(results_dir)

    # Calculate total experiments
    total = len(datasets) * len(data_lengths) * len(budgets) * len(seeds)
    print(f"\n🚀 Running {total} data efficiency experiments")
    print(f"   Datasets: {datasets}")
    print(f"   Data lengths: {data_lengths}")
    print(f"   Budgets: {budgets}")
    print(f"   Seeds: {seeds}")
    print(f"   Train fraction: {train_frac}")

    completed = 0
    failed = 0

    for dataset in datasets:
        for data_length in data_lengths:
            for budget in budgets:
                for seed in seeds:
                    try:
                        result = run_single_experiment(
                            dataset=dataset,
                            data_length=data_length,
                            budget=budget,
                            seed=seed,
                            best_configs=best_configs,
                            train_frac=train_frac,
                            output_dir=output_dir
                        )

                        if result:
                            completed += 1
                        else:
                            failed += 1

                    except Exception as e:
                        print(f"❌ Experiment failed: {e}")
                        failed += 1

                    print(f"\n📊 Progress: {completed + failed}/{total} "
                          f"(✅ {completed} / ❌ {failed})")

    print(f"\n{'='*60}")
    print(f"🎉 Data Efficiency Experiments Complete!")
    print(f"   Total: {total}")
    print(f"   Completed: {completed}")
    print(f"   Failed: {failed}")
    print(f"   Results saved to: {output_dir}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Run data efficiency experiments")
    parser.add_argument("--datasets", nargs="+",
                        default=["santafe", "lorenz", "mackeyglass"],
                        help="Datasets to test (santafe, lorenz, mackeyglass)")
    parser.add_argument("--lengths", nargs="+", type=int,
                        default=DEFAULT_DATA_LENGTHS,
                        help="Data lengths to test")
    parser.add_argument("--budgets", nargs="+",
                        default=["small", "medium", "large"],
                        help="Budget levels")
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[42, 43, 44, 45, 46],
                        help="Random seeds")
    parser.add_argument("--train-frac", type=float, default=0.7,
                        help="Training fraction")
    parser.add_argument("--output-dir", type=str,
                        default="results/data_efficiency",
                        help="Output directory")
    parser.add_argument("--results-dir", type=str,
                        default="results/final_v2",
                        help="Directory with main experiment results for best configs")
    parser.add_argument("--single", action="store_true",
                        help="Run single experiment for testing")

    args = parser.parse_args()

    if args.single:
        # Quick test with one configuration
        best_configs = load_best_configs(args.results_dir)
        run_single_experiment(
            dataset="lorenz",
            data_length=2000,
            budget="medium",
            seed=42,
            best_configs=best_configs,
            train_frac=args.train_frac,
            output_dir=args.output_dir
        )
    else:
        run_all_experiments(
            datasets=args.datasets,
            data_lengths=args.lengths,
            budgets=args.budgets,
            seeds=args.seeds,
            train_frac=args.train_frac,
            output_dir=args.output_dir,
            results_dir=args.results_dir
        )


if __name__ == "__main__":
    main()
