"""
Comprehensive analysis of PyReCo vs LSTM experimental results.

Analyzes 45 experiments across 3 datasets, 5 seeds, 3 train ratios, and 3 parameter budgets.
"""

import json
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict


def load_all_results(results_dir="results_final"):
    """Load all experimental result files."""
    pattern = f"{results_dir}/results_*.json"
    files = glob.glob(pattern)

    results = []
    for file_path in files:
        with open(file_path, 'r') as f:
            data = json.load(f)
            results.append({
                'file': Path(file_path).name,
                'data': data
            })

    return results


def extract_metrics(results):
    """Extract all metrics into a structured DataFrame."""
    records = []

    for result in results:
        metadata = result['data']['metadata']
        budgets = result['data']['budgets']

        for budget_name, budget_value in budgets.items():
            budget_results = result['data']['results'][budget_name]

            for model_result in budget_results:
                record = {
                    # Metadata
                    'dataset': metadata['dataset'],
                    'seed': metadata['seed'],
                    'train_frac': metadata['train_frac'],
                    'n_in': metadata['n_in'],

                    # Budget info
                    'budget': budget_name,
                    'budget_value': budget_value,

                    # Model info
                    'model_type': model_result['model_type'],
                    'trainable_params': model_result['param_info']['trainable'],
                    'total_params': model_result['param_info']['total'],

                    # Performance metrics
                    'test_mse': model_result['test_mse'],
                    'test_mae': model_result['test_mae'],
                    'test_r2': model_result['test_r2'],
                    'val_mse': model_result['val_mse'],

                    # Timing
                    'tune_time': model_result['tune_time'],
                    'final_train_time': model_result['final_train_time'],
                }

                records.append(record)

    return pd.DataFrame(records)


def compute_summary_statistics(df):
    """Compute summary statistics grouped by various factors."""

    # Group by dataset, budget, model_type
    summary = df.groupby(['dataset', 'budget', 'model_type']).agg({
        'test_mse': ['mean', 'std', 'min', 'max'],
        'test_mae': ['mean', 'std', 'min', 'max'],
        'test_r2': ['mean', 'std', 'min', 'max'],
        'tune_time': ['mean', 'std'],
        'final_train_time': ['mean', 'std'],
        'trainable_params': 'mean',
    }).round(6)

    return summary


def compare_models(df):
    """Compare PyReCo vs LSTM performance."""

    # Separate PyReCo and LSTM results
    pyreco_df = df[df['model_type'] == 'pyreco_standard']
    lstm_df = df[df['model_type'] == 'lstm']

    # Merge on matching conditions
    comparison = pyreco_df.merge(
        lstm_df,
        on=['dataset', 'seed', 'train_frac', 'budget'],
        suffixes=('_pyreco', '_lstm')
    )

    # Compute performance differences
    comparison['mse_improvement'] = (
        (comparison['test_mse_lstm'] - comparison['test_mse_pyreco'])
        / comparison['test_mse_lstm'] * 100
    )
    comparison['mae_improvement'] = (
        (comparison['test_mae_lstm'] - comparison['test_mae_pyreco'])
        / comparison['test_mae_lstm'] * 100
    )
    comparison['r2_improvement'] = (
        comparison['test_r2_pyreco'] - comparison['test_r2_lstm']
    ) * 100

    comparison['time_ratio'] = (
        comparison['final_train_time_pyreco'] / comparison['final_train_time_lstm']
    )

    return comparison


def analyze_by_dataset(df):
    """Analyze results separately for each dataset."""
    datasets = df['dataset'].unique()

    results = {}
    for dataset in datasets:
        dataset_df = df[df['dataset'] == dataset]

        results[dataset] = {
            'summary': dataset_df.groupby(['budget', 'model_type']).agg({
                'test_mse': ['mean', 'std'],
                'test_r2': ['mean', 'std'],
                'final_train_time': ['mean', 'std'],
            }).round(6),

            'best_configs': dataset_df.loc[
                dataset_df.groupby(['budget', 'model_type'])['test_mse'].idxmin()
            ][['budget', 'model_type', 'test_mse', 'test_r2', 'seed', 'train_frac']]
        }

    return results


def analyze_by_train_ratio(df):
    """Analyze how performance changes with training data size."""

    summary = df.groupby(['dataset', 'budget', 'train_frac', 'model_type']).agg({
        'test_mse': ['mean', 'std'],
        'test_r2': ['mean', 'std'],
    }).round(6)

    return summary


def generate_report(df, output_file="results_final/analysis_report.txt"):
    """Generate a comprehensive text report."""

    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("COMPREHENSIVE EXPERIMENTAL RESULTS ANALYSIS\n")
        f.write("PyReCo vs LSTM Performance Comparison\n")
        f.write("=" * 80 + "\n\n")

        # Overview
        f.write("1. EXPERIMENT OVERVIEW\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total experiments: {len(df) // 2}\n")
        f.write(f"Datasets: {', '.join(df['dataset'].unique())}\n")
        f.write(f"Seeds: {sorted(df['seed'].unique())}\n")
        f.write(f"Train ratios: {sorted(df['train_frac'].unique())}\n")
        f.write(f"Budgets: {', '.join(df['budget'].unique())}\n\n")

        # Overall comparison
        f.write("2. OVERALL PERFORMANCE COMPARISON\n")
        f.write("-" * 80 + "\n")
        summary = compute_summary_statistics(df)
        f.write(summary.to_string())
        f.write("\n\n")

        # Dataset-specific analysis
        f.write("3. DATASET-SPECIFIC ANALYSIS\n")
        f.write("-" * 80 + "\n")
        dataset_results = analyze_by_dataset(df)
        for dataset, data in dataset_results.items():
            f.write(f"\n{dataset.upper()} Dataset:\n")
            f.write(data['summary'].to_string())
            f.write("\n\nBest configurations:\n")
            f.write(data['best_configs'].to_string())
            f.write("\n")

        # Train ratio analysis
        f.write("\n4. TRAINING DATA SIZE ANALYSIS\n")
        f.write("-" * 80 + "\n")
        train_ratio_summary = analyze_by_train_ratio(df)
        f.write(train_ratio_summary.to_string())
        f.write("\n\n")

        # Model comparison
        f.write("5. PYRECO VS LSTM COMPARISON\n")
        f.write("-" * 80 + "\n")
        comparison = compare_models(df)

        f.write(f"\nAverage MSE improvement (PyReCo over LSTM): "
                f"{comparison['mse_improvement'].mean():.2f}%\n")
        f.write(f"Average MAE improvement (PyReCo over LSTM): "
                f"{comparison['mae_improvement'].mean():.2f}%\n")
        f.write(f"Average R² improvement (PyReCo over LSTM): "
                f"{comparison['r2_improvement'].mean():.2f}%\n")
        f.write(f"Average training time ratio (PyReCo/LSTM): "
                f"{comparison['time_ratio'].mean():.4f}x\n")

        # Win rates
        pyreco_wins = (comparison['test_mse_pyreco'] < comparison['test_mse_lstm']).sum()
        total = len(comparison)
        f.write(f"\nPyReCo win rate (lower MSE): {pyreco_wins}/{total} "
                f"({pyreco_wins/total*100:.1f}%)\n")

        f.write("\n" + "=" * 80 + "\n")

    print(f"✅ Report saved to: {output_file}")


def generate_csv_summary(df, output_file="results_final/results_summary.csv"):
    """Export results to CSV for further analysis."""
    df.to_csv(output_file, index=False)
    print(f"✅ CSV summary saved to: {output_file}")


def main():
    """Main analysis pipeline."""
    print("=" * 80)
    print("ANALYZING EXPERIMENTAL RESULTS")
    print("=" * 80)

    # Load results
    print("\n1. Loading result files...")
    results = load_all_results()
    print(f"   ✓ Loaded {len(results)} result files")

    # Extract metrics
    print("\n2. Extracting metrics...")
    df = extract_metrics(results)
    print(f"   ✓ Extracted {len(df)} model evaluations")
    print(f"   ✓ Datasets: {df['dataset'].unique()}")
    print(f"   ✓ Models per config: {len(df) // (len(df['dataset'].unique()) * 5 * 3 * 3)}")

    # Generate report
    print("\n3. Generating comprehensive report...")
    generate_report(df)

    # Export CSV
    print("\n4. Exporting CSV summary...")
    generate_csv_summary(df)

    # Quick summary statistics
    print("\n5. QUICK SUMMARY")
    print("-" * 80)

    comparison = compare_models(df)

    print(f"\nOverall Performance (averaged across all experiments):")
    print(f"  PyReCo average test MSE: {df[df['model_type']=='pyreco_standard']['test_mse'].mean():.6f}")
    print(f"  LSTM average test MSE:   {df[df['model_type']=='lstm']['test_mse'].mean():.6f}")
    print(f"  PyReCo average test R²:  {df[df['model_type']=='pyreco_standard']['test_r2'].mean():.6f}")
    print(f"  LSTM average test R²:    {df[df['model_type']=='lstm']['test_r2'].mean():.6f}")

    print(f"\nTraining Time:")
    print(f"  PyReCo average: {df[df['model_type']=='pyreco_standard']['final_train_time'].mean():.2f}s")
    print(f"  LSTM average:   {df[df['model_type']=='lstm']['final_train_time'].mean():.2f}s")

    pyreco_wins = (comparison['test_mse_pyreco'] < comparison['test_mse_lstm']).sum()
    total = len(comparison)
    print(f"\nWin Rate:")
    print(f"  PyReCo wins: {pyreco_wins}/{total} ({pyreco_wins/total*100:.1f}%)")
    print(f"  LSTM wins:   {total-pyreco_wins}/{total} ({(total-pyreco_wins)/total*100:.1f}%)")

    # Budget analysis
    print(f"\nPerformance by Budget:")
    for budget in ['small', 'medium', 'large']:
        budget_df = df[df['budget'] == budget]
        pyreco_mse = budget_df[budget_df['model_type']=='pyreco_standard']['test_mse'].mean()
        lstm_mse = budget_df[budget_df['model_type']=='lstm']['test_mse'].mean()
        print(f"  {budget.capitalize()} budget:")
        print(f"    PyReCo MSE: {pyreco_mse:.6f}")
        print(f"    LSTM MSE:   {lstm_mse:.6f}")
        print(f"    Winner: {'PyReCo' if pyreco_mse < lstm_mse else 'LSTM'}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
