#!/usr/bin/env python3
"""
Hyperparameter Sensitivity Analysis across Train Ratios

Analyzes how optimal hyperparameter selection changes with train_frac
for both PyReCo and LSTM models, and how this affects performance.

Reads existing results from results/final/ (no new experiments needed).

Usage:
    python analysis/hyperparameter_sensitivity_analysis.py
"""

import json
import glob
import os
import sys
import numpy as np
from collections import defaultdict

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def load_all_results(results_dir="results/final"):
    """Load all result JSON files."""
    if not os.path.isabs(results_dir):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        results_dir = os.path.join(project_root, results_dir)

    files = sorted(glob.glob(f"{results_dir}/results_*.json"))
    print(f"Loading {len(files)} result files from {results_dir}")

    records = []
    for file_path in files:
        with open(file_path, 'r') as f:
            data = json.load(f)
        meta = data['metadata']
        for budget_name, budget_val in data['budgets'].items():
            for model_result in data['results'][budget_name]:
                records.append({
                    'dataset': meta['dataset'],
                    'seed': meta['seed'],
                    'train_frac': meta['train_frac'],
                    'budget': budget_name,
                    'model_type': model_result['model_type'],
                    'config': model_result['config'],
                    'test_mse': model_result['test_mse'],
                    'test_r2': model_result['test_r2'],
                    'final_train_time': model_result['final_train_time'],
                    'val_mse': model_result['val_mse'],
                })
    print(f"Loaded {len(records)} records")
    return records


def analyze_pyreco_hyperparams(records):
    """Analyze PyReCo hyperparameter changes across train_fracs."""
    print("\n" + "=" * 80)
    print("PYRECO HYPERPARAMETER SENSITIVITY ANALYSIS")
    print("=" * 80)

    pyreco = [r for r in records if r['model_type'] == 'pyreco_standard']
    datasets = sorted(set(r['dataset'] for r in pyreco))
    budgets = ['small', 'medium', 'large']
    train_fracs = sorted(set(r['train_frac'] for r in pyreco))
    hp_keys = ['spec_rad', 'leakage_rate', 'density', 'activation']

    for dataset in datasets:
        print(f"\n{'─' * 80}")
        print(f"Dataset: {dataset.upper()}")
        print(f"{'─' * 80}")

        for budget in budgets:
            subset = [r for r in pyreco
                      if r['dataset'] == dataset and r['budget'] == budget]
            if not subset:
                continue

            print(f"\n  Budget: {budget}")
            header = f"  {'train_frac':>10}"
            for hp in hp_keys:
                header += f"  {hp:>14}"
            header += f"  {'R² (mean)':>12}  {'MSE (mean)':>12}"
            print(header)
            print("  " + "-" * (len(header) - 2))

            for tf in train_fracs:
                tf_records = [r for r in subset if r['train_frac'] == tf]
                if not tf_records:
                    continue

                # Majority vote for config across seeds
                config_counts = defaultdict(int)
                for r in tf_records:
                    config_key = json.dumps(
                        {k: r['config'].get(k) for k in hp_keys},
                        sort_keys=True
                    )
                    config_counts[config_key] += 1

                majority_config = json.loads(
                    max(config_counts, key=config_counts.get)
                )
                n_agree = max(config_counts.values())
                n_total = len(tf_records)

                mean_r2 = np.mean([r['test_r2'] for r in tf_records])
                mean_mse = np.mean([r['test_mse'] for r in tf_records])

                row = f"  {tf:>10.1f}"
                for hp in hp_keys:
                    val = majority_config.get(hp, '-')
                    if isinstance(val, float):
                        row += f"  {val:>14.2f}"
                    else:
                        row += f"  {str(val):>14}"
                row += f"  {mean_r2:>12.6f}  {mean_mse:>12.6f}"
                if n_agree < n_total:
                    row += f"  ({n_agree}/{n_total} seeds agree)"
                print(row)


def analyze_lstm_hyperparams(records):
    """Analyze LSTM hyperparameter changes across train_fracs."""
    print("\n" + "=" * 80)
    print("LSTM HYPERPARAMETER SENSITIVITY ANALYSIS")
    print("=" * 80)

    lstm = [r for r in records if r['model_type'] == 'lstm']
    datasets = sorted(set(r['dataset'] for r in lstm))
    budgets = ['small', 'medium', 'large']
    train_fracs = sorted(set(r['train_frac'] for r in lstm))
    hp_keys = ['hidden_size', 'num_layers', 'dropout', 'learning_rate']

    for dataset in datasets:
        print(f"\n{'─' * 80}")
        print(f"Dataset: {dataset.upper()}")
        print(f"{'─' * 80}")

        for budget in budgets:
            subset = [r for r in lstm
                      if r['dataset'] == dataset and r['budget'] == budget]
            if not subset:
                continue

            print(f"\n  Budget: {budget}")
            header = f"  {'train_frac':>10}"
            for hp in hp_keys:
                header += f"  {hp:>14}"
            header += f"  {'R² (mean)':>12}  {'MSE (mean)':>12}"
            print(header)
            print("  " + "-" * (len(header) - 2))

            for tf in train_fracs:
                tf_records = [r for r in subset if r['train_frac'] == tf]
                if not tf_records:
                    continue

                config_counts = defaultdict(int)
                for r in tf_records:
                    config_key = json.dumps(
                        {k: r['config'].get(k) for k in hp_keys},
                        sort_keys=True
                    )
                    config_counts[config_key] += 1

                majority_config = json.loads(
                    max(config_counts, key=config_counts.get)
                )
                n_agree = max(config_counts.values())
                n_total = len(tf_records)

                mean_r2 = np.mean([r['test_r2'] for r in tf_records])
                mean_mse = np.mean([r['test_mse'] for r in tf_records])

                row = f"  {tf:>10.1f}"
                for hp in hp_keys:
                    val = majority_config.get(hp, '-')
                    if isinstance(val, float):
                        row += f"  {val:>14.3f}"
                    elif isinstance(val, int):
                        row += f"  {val:>14d}"
                    else:
                        row += f"  {str(val):>14}"
                row += f"  {mean_r2:>12.6f}  {mean_mse:>12.6f}"
                if n_agree < n_total:
                    row += f"  ({n_agree}/{n_total} seeds agree)"
                print(row)


def analyze_config_stability(records):
    """Measure how stable each hyperparameter is across train_fracs."""
    print("\n" + "=" * 80)
    print("HYPERPARAMETER STABILITY SUMMARY")
    print("=" * 80)
    print("(How many unique values does each HP take across train_fracs?)")

    for model_type, model_label in [('pyreco_standard', 'PyReCo'),
                                     ('lstm', 'LSTM')]:
        print(f"\n{'─' * 60}")
        print(f"Model: {model_label}")
        print(f"{'─' * 60}")

        model_records = [r for r in records if r['model_type'] == model_type]
        datasets = sorted(set(r['dataset'] for r in model_records))
        budgets = ['small', 'medium', 'large']
        train_fracs = sorted(set(r['train_frac'] for r in model_records))

        if model_type == 'pyreco_standard':
            hp_keys = ['spec_rad', 'leakage_rate', 'density', 'activation']
        else:
            hp_keys = ['hidden_size', 'num_layers', 'dropout', 'learning_rate']

        # Per dataset+budget: count unique HP values across train_fracs
        header = f"  {'Dataset':<15} {'Budget':<8}"
        for hp in hp_keys:
            header += f"  {hp:>14}"
        header += f"  {'Stability':>10}"
        print(header)
        print("  " + "-" * (len(header) - 2))

        total_changes = 0
        total_possible = 0

        for dataset in datasets:
            for budget in budgets:
                subset = [r for r in model_records
                          if r['dataset'] == dataset and r['budget'] == budget]
                if not subset:
                    continue

                # Get majority config per train_frac
                configs_by_tf = {}
                for tf in train_fracs:
                    tf_records = [r for r in subset if r['train_frac'] == tf]
                    if not tf_records:
                        continue
                    config_counts = defaultdict(int)
                    for r in tf_records:
                        config_key = json.dumps(
                            {k: r['config'].get(k) for k in hp_keys},
                            sort_keys=True
                        )
                        config_counts[config_key] += 1
                    majority_config = json.loads(
                        max(config_counts, key=config_counts.get)
                    )
                    configs_by_tf[tf] = majority_config

                if len(configs_by_tf) < 2:
                    continue

                row = f"  {dataset:<15} {budget:<8}"
                n_stable = 0
                for hp in hp_keys:
                    unique_vals = set(
                        str(c.get(hp)) for c in configs_by_tf.values()
                    )
                    n_unique = len(unique_vals)
                    if n_unique == 1:
                        row += f"  {'stable':>14}"
                        n_stable += 1
                    else:
                        vals_str = ','.join(sorted(unique_vals))
                        if len(vals_str) > 14:
                            row += f"  {n_unique:>12}vals"
                        else:
                            row += f"  {vals_str:>14}"
                    total_possible += 1
                    if n_unique == 1:
                        total_changes += 0
                    else:
                        total_changes += 1

                stability = n_stable / len(hp_keys) * 100
                row += f"  {stability:>9.0f}%"
                print(row)

        if total_possible > 0:
            overall = (1 - total_changes / total_possible) * 100
            print(f"\n  Overall stability: {overall:.1f}% "
                  f"({total_possible - total_changes}/{total_possible} "
                  f"HP-dataset-budget combinations unchanged)")


def analyze_performance_impact(records):
    """Analyze if config changes actually affect performance significantly."""
    print("\n" + "=" * 80)
    print("PERFORMANCE IMPACT OF HYPERPARAMETER CHANGES")
    print("=" * 80)
    print("(Comparing R² across train_fracs for each dataset × budget)")

    for model_type, model_label in [('pyreco_standard', 'PyReCo'),
                                     ('lstm', 'LSTM')]:
        print(f"\n{'─' * 60}")
        print(f"Model: {model_label}")
        print(f"{'─' * 60}")

        model_records = [r for r in records if r['model_type'] == model_type]
        datasets = sorted(set(r['dataset'] for r in model_records))
        budgets = ['small', 'medium', 'large']
        train_fracs = sorted(set(r['train_frac'] for r in model_records))

        for dataset in datasets:
            print(f"\n  Dataset: {dataset}")
            header = f"  {'train_frac':>10}"
            for budget in budgets:
                header += f"  {'R²_' + budget:>14}  {'±std':>8}"
            print(header)
            print("  " + "-" * (len(header) - 2))

            for tf in train_fracs:
                row = f"  {tf:>10.1f}"
                for budget in budgets:
                    tf_records = [
                        r for r in model_records
                        if r['dataset'] == dataset
                        and r['budget'] == budget
                        and r['train_frac'] == tf
                    ]
                    if tf_records:
                        r2_vals = [r['test_r2'] for r in tf_records]
                        mean_r2 = np.mean(r2_vals)
                        std_r2 = np.std(r2_vals)
                        row += f"  {mean_r2:>14.6f}  {std_r2:>8.6f}"
                    else:
                        row += f"  {'N/A':>14}  {'N/A':>8}"
                print(row)


def analyze_cross_seed_agreement(records):
    """Analyze how often seeds agree on the same config."""
    print("\n" + "=" * 80)
    print("CROSS-SEED CONFIG AGREEMENT")
    print("=" * 80)
    print("(Do different seeds select the same optimal hyperparameters?)")

    for model_type, model_label in [('pyreco_standard', 'PyReCo'),
                                     ('lstm', 'LSTM')]:
        print(f"\n{'─' * 60}")
        print(f"Model: {model_label}")
        print(f"{'─' * 60}")

        model_records = [r for r in records if r['model_type'] == model_type]
        datasets = sorted(set(r['dataset'] for r in model_records))
        budgets = ['small', 'medium', 'large']
        train_fracs = sorted(set(r['train_frac'] for r in model_records))

        if model_type == 'pyreco_standard':
            hp_keys = ['spec_rad', 'leakage_rate', 'density']
        else:
            hp_keys = ['hidden_size', 'num_layers', 'dropout', 'learning_rate']

        header = f"  {'Dataset':<15} {'Budget':<8} {'train_frac':>10} {'Agreement':>10} {'#Configs':>9}"
        print(header)
        print("  " + "-" * (len(header) - 2))

        agreement_rates = []
        for dataset in datasets:
            for budget in budgets:
                for tf in train_fracs:
                    tf_records = [
                        r for r in model_records
                        if r['dataset'] == dataset
                        and r['budget'] == budget
                        and r['train_frac'] == tf
                    ]
                    if len(tf_records) < 2:
                        continue

                    config_counts = defaultdict(int)
                    for r in tf_records:
                        config_key = json.dumps(
                            {k: r['config'].get(k) for k in hp_keys},
                            sort_keys=True
                        )
                        config_counts[config_key] += 1

                    max_agree = max(config_counts.values())
                    n_total = len(tf_records)
                    n_unique = len(config_counts)
                    rate = max_agree / n_total

                    agreement_rates.append(rate)

                    if rate < 1.0:  # Only show disagreements
                        print(f"  {dataset:<15} {budget:<8} {tf:>10.1f} "
                              f"{max_agree}/{n_total}{'':>5} {n_unique:>9}")

        if agreement_rates:
            avg_rate = np.mean(agreement_rates)
            full_agree = sum(1 for r in agreement_rates if r >= 1.0)
            print(f"\n  Average agreement rate: {avg_rate:.1%}")
            print(f"  Full agreement: {full_agree}/{len(agreement_rates)} "
                  f"({full_agree/len(agreement_rates):.1%})")


def main():
    records = load_all_results()

    analyze_pyreco_hyperparams(records)
    analyze_lstm_hyperparams(records)
    analyze_config_stability(records)
    analyze_performance_impact(records)
    analyze_cross_seed_agreement(records)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
