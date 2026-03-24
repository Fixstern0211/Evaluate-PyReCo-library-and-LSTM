#!/usr/bin/env python3
"""
Generate comprehensive results tables from DATA EFFICIENCY experiments.
Data efficiency experiments test model performance across different data lengths.
Data source: results/data_efficiency/ (dataeff_*.json files)
Outputs both console tables and CSV files for analysis.
"""

import json
import os
from pathlib import Path
from collections import defaultdict
import csv

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "data_efficiency"
OUTPUT_DIR = PROJECT_ROOT / "results" / "tables" / "data_efficiency"


def load_all_results():
    """Load all data efficiency experiment results."""
    results = []
    for f in RESULTS_DIR.glob("dataeff_*.json"):
        try:
            with open(f) as fp:
                data = json.load(fp)
                results.append(data)
        except Exception as e:
            print(f"Error loading {f}: {e}")
    return results


def aggregate_by_dataset_budget(results):
    """Aggregate results by dataset and budget."""
    agg = defaultdict(lambda: defaultdict(lambda: {"pyreco": [], "lstm": []}))

    for r in results:
        meta = r["metadata"]
        dataset = meta["dataset"]
        budget = meta["budget"]
        data_len = meta["data_length"]

        for model_key, model_data in r["models"].items():
            if "pyreco" in model_key.lower():
                model_type = "pyreco"
            elif "lstm" in model_key.lower():
                model_type = "lstm"
            else:
                continue

            agg[dataset][budget][model_type].append({
                "data_length": data_len,
                "seed": meta["seed"],
                "mse": model_data["metrics"]["mse"],
                "r2": model_data["metrics"]["r2"],
                "train_time": model_data["train_time"]
            })

    return agg


def compute_stats(values):
    """Compute mean and std of a list of values."""
    if not values:
        return 0, 0
    mean = sum(values) / len(values)
    if len(values) > 1:
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        std = variance ** 0.5
    else:
        std = 0
    return mean, std


def print_summary_table(agg):
    """Print summary table by dataset and budget."""
    print("\n" + "=" * 100)
    print("SUMMARY TABLE: PyReCo vs LSTM by Dataset and Budget")
    print("=" * 100)

    header = f"{'Dataset':<12} {'Budget':<8} {'PyReCo MSE':<18} {'LSTM MSE':<18} {'PyReCo R²':<14} {'LSTM R²':<14} {'Winner':<10}"
    print(header)
    print("-" * 100)

    for dataset in ["lorenz", "mackeyglass", "santafe"]:
        for budget in ["small", "medium", "large"]:
            if dataset not in agg or budget not in agg[dataset]:
                continue

            pyreco_data = agg[dataset][budget]["pyreco"]
            lstm_data = agg[dataset][budget]["lstm"]

            if not pyreco_data or not lstm_data:
                continue

            pyreco_mse_mean, pyreco_mse_std = compute_stats([d["mse"] for d in pyreco_data])
            lstm_mse_mean, lstm_mse_std = compute_stats([d["mse"] for d in lstm_data])
            pyreco_r2_mean, pyreco_r2_std = compute_stats([d["r2"] for d in pyreco_data])
            lstm_r2_mean, lstm_r2_std = compute_stats([d["r2"] for d in lstm_data])

            winner = "PyReCo" if pyreco_r2_mean > lstm_r2_mean else "LSTM"

            print(f"{dataset:<12} {budget:<8} "
                  f"{pyreco_mse_mean:.6f}±{pyreco_mse_std:.4f}  "
                  f"{lstm_mse_mean:.6f}±{lstm_mse_std:.4f}  "
                  f"{pyreco_r2_mean:.4f}±{pyreco_r2_std:.3f}  "
                  f"{lstm_r2_mean:.4f}±{lstm_r2_std:.3f}  "
                  f"**{winner}**")
        print()


def print_training_time_table(agg):
    """Print training time comparison table."""
    print("\n" + "=" * 100)
    print("TRAINING TIME COMPARISON (seconds)")
    print("=" * 100)

    header = f"{'Dataset':<12} {'Budget':<8} {'PyReCo Time':<16} {'LSTM Time':<16} {'Speedup':<20}"
    print(header)
    print("-" * 100)

    for dataset in ["lorenz", "mackeyglass", "santafe"]:
        for budget in ["small", "medium", "large"]:
            if dataset not in agg or budget not in agg[dataset]:
                continue

            pyreco_data = agg[dataset][budget]["pyreco"]
            lstm_data = agg[dataset][budget]["lstm"]

            if not pyreco_data or not lstm_data:
                continue

            pyreco_time_mean, pyreco_time_std = compute_stats([d["train_time"] for d in pyreco_data])
            lstm_time_mean, lstm_time_std = compute_stats([d["train_time"] for d in lstm_data])

            if pyreco_time_mean < lstm_time_mean:
                speedup = f"PyReCo {lstm_time_mean/pyreco_time_mean:.1f}x faster"
            else:
                speedup = f"LSTM {pyreco_time_mean/lstm_time_mean:.1f}x faster"

            print(f"{dataset:<12} {budget:<8} "
                  f"{pyreco_time_mean:>6.2f}s±{pyreco_time_std:.2f}s   "
                  f"{lstm_time_mean:>6.2f}s±{lstm_time_std:.2f}s   "
                  f"{speedup}")
        print()


def print_data_length_table(agg):
    """Print performance by data length."""
    print("\n" + "=" * 120)
    print("PERFORMANCE BY DATA LENGTH (R² Score, Budget=Small)")
    print("=" * 120)

    # Collect all data lengths
    all_lengths = set()
    for dataset in agg:
        for budget in agg[dataset]:
            for model in agg[dataset][budget]:
                for d in agg[dataset][budget][model]:
                    all_lengths.add(d["data_length"])
    all_lengths = sorted(all_lengths)

    header = f"{'Dataset':<12} {'Model':<8} " + " ".join(f"{'Len=' + str(l):<12}" for l in all_lengths)
    print(header)
    print("-" * 120)

    for dataset in ["lorenz", "mackeyglass", "santafe"]:
        if dataset not in agg or "small" not in agg[dataset]:
            continue

        for model in ["pyreco", "lstm"]:
            data = agg[dataset]["small"][model]

            # Group by data length
            by_length = defaultdict(list)
            for d in data:
                by_length[d["data_length"]].append(d["r2"])

            row = f"{dataset:<12} {model.upper():<8} "
            for length in all_lengths:
                if length in by_length:
                    mean_r2 = sum(by_length[length]) / len(by_length[length])
                    row += f"{mean_r2:.4f}       "
                else:
                    row += f"{'N/A':<12} "
            print(row)
        print()


def print_winner_summary(agg):
    """Print winner summary by dataset."""
    print("\n" + "=" * 80)
    print("WINNER SUMMARY BY DATASET")
    print("=" * 80)

    for dataset in ["lorenz", "mackeyglass", "santafe"]:
        if dataset not in agg:
            continue

        pyreco_wins = 0
        lstm_wins = 0
        total = 0

        pyreco_r2_all = []
        lstm_r2_all = []

        for budget in agg[dataset]:
            pyreco_data = agg[dataset][budget]["pyreco"]
            lstm_data = agg[dataset][budget]["lstm"]

            pyreco_r2_all.extend([d["r2"] for d in pyreco_data])
            lstm_r2_all.extend([d["r2"] for d in lstm_data])

            pyreco_r2_mean = sum(d["r2"] for d in pyreco_data) / len(pyreco_data) if pyreco_data else 0
            lstm_r2_mean = sum(d["r2"] for d in lstm_data) / len(lstm_data) if lstm_data else 0

            if pyreco_r2_mean > lstm_r2_mean:
                pyreco_wins += 1
            else:
                lstm_wins += 1
            total += 1

        avg_pyreco = sum(pyreco_r2_all) / len(pyreco_r2_all) if pyreco_r2_all else 0
        avg_lstm = sum(lstm_r2_all) / len(lstm_r2_all) if lstm_r2_all else 0

        winner = "PyReCo" if pyreco_wins > lstm_wins else "LSTM"

        print(f"\n{dataset.upper()}")
        print(f"  PyReCo wins: {pyreco_wins}/{total} budgets")
        print(f"  LSTM wins:   {lstm_wins}/{total} budgets")
        print(f"  Overall Winner: **{winner}**")
        print(f"  Average R² - PyReCo: {avg_pyreco:.4f}, LSTM: {avg_lstm:.4f}")


def save_to_csv(agg):
    """Save results to CSV files."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Summary CSV
    summary_file = OUTPUT_DIR / "summary_by_dataset_budget.csv"
    with open(summary_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Dataset", "Budget", "PyReCo_MSE_Mean", "PyReCo_MSE_Std",
                        "LSTM_MSE_Mean", "LSTM_MSE_Std", "PyReCo_R2_Mean", "PyReCo_R2_Std",
                        "LSTM_R2_Mean", "LSTM_R2_Std", "PyReCo_Time_Mean", "LSTM_Time_Mean", "Winner"])

        for dataset in ["lorenz", "mackeyglass", "santafe"]:
            for budget in ["small", "medium", "large"]:
                if dataset not in agg or budget not in agg[dataset]:
                    continue

                pyreco_data = agg[dataset][budget]["pyreco"]
                lstm_data = agg[dataset][budget]["lstm"]

                if not pyreco_data or not lstm_data:
                    continue

                pyreco_mse_mean, pyreco_mse_std = compute_stats([d["mse"] for d in pyreco_data])
                lstm_mse_mean, lstm_mse_std = compute_stats([d["mse"] for d in lstm_data])
                pyreco_r2_mean, pyreco_r2_std = compute_stats([d["r2"] for d in pyreco_data])
                lstm_r2_mean, lstm_r2_std = compute_stats([d["r2"] for d in lstm_data])
                pyreco_time_mean, _ = compute_stats([d["train_time"] for d in pyreco_data])
                lstm_time_mean, _ = compute_stats([d["train_time"] for d in lstm_data])

                winner = "PyReCo" if pyreco_r2_mean > lstm_r2_mean else "LSTM"

                writer.writerow([dataset, budget,
                                f"{pyreco_mse_mean:.6f}", f"{pyreco_mse_std:.6f}",
                                f"{lstm_mse_mean:.6f}", f"{lstm_mse_std:.6f}",
                                f"{pyreco_r2_mean:.4f}", f"{pyreco_r2_std:.4f}",
                                f"{lstm_r2_mean:.4f}", f"{lstm_r2_std:.4f}",
                                f"{pyreco_time_mean:.2f}", f"{lstm_time_mean:.2f}",
                                winner])

    print(f"\nSaved summary to: {summary_file}")

    # Detailed CSV with all experiments
    detailed_file = OUTPUT_DIR / "detailed_results.csv"
    with open(detailed_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Dataset", "Budget", "Data_Length", "Seed", "Model", "MSE", "R2", "Train_Time"])

        for dataset in agg:
            for budget in agg[dataset]:
                for model in agg[dataset][budget]:
                    for d in agg[dataset][budget][model]:
                        writer.writerow([dataset, budget, d["data_length"], d["seed"],
                                        model.upper(), f"{d['mse']:.6f}", f"{d['r2']:.4f}",
                                        f"{d['train_time']:.2f}"])

    print(f"Saved detailed results to: {detailed_file}")


def print_markdown_table(agg):
    """Print Markdown formatted tables for documentation."""
    print("\n" + "=" * 80)
    print("MARKDOWN TABLES (Copy to documentation)")
    print("=" * 80)

    print("\n### Performance Summary (R² Score)\n")
    print("| Dataset | Budget | PyReCo R² | LSTM R² | Winner |")
    print("|---------|--------|-----------|---------|--------|")

    for dataset in ["lorenz", "mackeyglass", "santafe"]:
        for budget in ["small", "medium", "large"]:
            if dataset not in agg or budget not in agg[dataset]:
                continue

            pyreco_data = agg[dataset][budget]["pyreco"]
            lstm_data = agg[dataset][budget]["lstm"]

            if not pyreco_data or not lstm_data:
                continue

            pyreco_r2_mean, pyreco_r2_std = compute_stats([d["r2"] for d in pyreco_data])
            lstm_r2_mean, lstm_r2_std = compute_stats([d["r2"] for d in lstm_data])

            winner = "**PyReCo**" if pyreco_r2_mean > lstm_r2_mean else "**LSTM**"

            print(f"| {dataset} | {budget} | {pyreco_r2_mean:.4f} ± {pyreco_r2_std:.4f} | "
                  f"{lstm_r2_mean:.4f} ± {lstm_r2_std:.4f} | {winner} |")

    print("\n### Training Time Comparison\n")
    print("| Dataset | Budget | PyReCo (s) | LSTM (s) | Speedup |")
    print("|---------|--------|------------|----------|---------|")

    for dataset in ["lorenz", "mackeyglass", "santafe"]:
        for budget in ["small", "medium", "large"]:
            if dataset not in agg or budget not in agg[dataset]:
                continue

            pyreco_data = agg[dataset][budget]["pyreco"]
            lstm_data = agg[dataset][budget]["lstm"]

            if not pyreco_data or not lstm_data:
                continue

            pyreco_time_mean, _ = compute_stats([d["train_time"] for d in pyreco_data])
            lstm_time_mean, _ = compute_stats([d["train_time"] for d in lstm_data])

            if pyreco_time_mean < lstm_time_mean:
                speedup = f"PyReCo {lstm_time_mean/pyreco_time_mean:.1f}x faster"
            else:
                speedup = f"LSTM {pyreco_time_mean/lstm_time_mean:.1f}x faster"

            print(f"| {dataset} | {budget} | {pyreco_time_mean:.2f} | {lstm_time_mean:.2f} | {speedup} |")


def print_all_experiments_table(results):
    """Print table showing ALL individual experiments."""
    print("\n" + "=" * 150)
    print("ALL DATA EFFICIENCY EXPERIMENTS - DETAILED VIEW (270 experiments)")
    print("=" * 150)

    header = (f"{'Dataset':<12} {'Length':<8} {'Seed':<6} {'Budget':<8} {'Model':<12} "
              f"{'MSE':<14} {'R²':<10} {'Time(s)':<12}")
    print(header)
    print("-" * 150)

    # Collect all experiments
    all_exps = []
    for r in results:
        meta = r["metadata"]
        for model_key, model_data in r["models"].items():
            model_name = "PyReCo" if "pyreco" in model_key.lower() else "LSTM"
            all_exps.append({
                "dataset": meta["dataset"],
                "data_length": meta["data_length"],
                "seed": meta["seed"],
                "budget": meta["budget"],
                "model": model_name,
                "mse": model_data["metrics"]["mse"],
                "r2": model_data["metrics"]["r2"],
                "train_time": model_data["train_time"]
            })

    # Sort by dataset, data_length, budget, seed, model
    sorted_exps = sorted(all_exps,
                         key=lambda x: (x["dataset"], x["data_length"], x["budget"], x["seed"], x["model"]))

    prev_dataset = None
    prev_length = None

    for exp in sorted_exps:
        if exp["dataset"] != prev_dataset:
            if prev_dataset is not None:
                print()
            prev_dataset = exp["dataset"]
            prev_length = None

        if exp["data_length"] != prev_length:
            if prev_length is not None and exp["dataset"] == prev_dataset:
                print("-" * 150)
            prev_length = exp["data_length"]

        print(f"{exp['dataset']:<12} {exp['data_length']:<8} {exp['seed']:<6} {exp['budget']:<8} "
              f"{exp['model']:<12} {exp['mse']:<14.6f} {exp['r2']:<10.4f} {exp['train_time']:<12.2f}")


def print_detailed_by_data_length(results):
    """Print detailed performance breakdown by data length."""
    print("\n" + "=" * 140)
    print("PERFORMANCE BY DATA LENGTH AND BUDGET")
    print("=" * 140)

    # Organize data
    by_config = defaultdict(lambda: {"pyreco": {"mse": [], "r2": []}, "lstm": {"mse": [], "r2": []}})

    for r in results:
        meta = r["metadata"]
        key = (meta["dataset"], meta["data_length"], meta["budget"])
        for model_key, model_data in r["models"].items():
            model = "pyreco" if "pyreco" in model_key.lower() else "lstm"
            by_config[key][model]["mse"].append(model_data["metrics"]["mse"])
            by_config[key][model]["r2"].append(model_data["metrics"]["r2"])

    header = (f"{'Dataset':<12} {'Length':<8} {'Budget':<8} "
              f"{'PyReCo R² (±std)':<20} {'LSTM R² (±std)':<20} {'Winner':<10} {'Diff':<8}")
    print(header)
    print("-" * 140)

    for dataset in ["lorenz", "mackeyglass", "santafe"]:
        for length in [1000, 2000, 3000, 5000, 7000, 10000]:
            for budget in ["small", "medium", "large"]:
                key = (dataset, length, budget)
                if key not in by_config:
                    continue

                pyreco_r2, pyreco_std = compute_stats(by_config[key]["pyreco"]["r2"])
                lstm_r2, lstm_std = compute_stats(by_config[key]["lstm"]["r2"])

                winner = "PyReCo" if pyreco_r2 > lstm_r2 else "LSTM"
                diff = abs(pyreco_r2 - lstm_r2)

                print(f"{dataset:<12} {length:<8} {budget:<8} "
                      f"{pyreco_r2:.4f} ± {pyreco_std:.4f}      "
                      f"{lstm_r2:.4f} ± {lstm_std:.4f}      "
                      f"{winner:<10} {diff:.4f}")
            if budget == "large":
                print()


def main():
    print("=" * 80)
    print("DATA EFFICIENCY EXPERIMENTS TABLE GENERATOR")
    print("Testing model performance across different data lengths")
    print("=" * 80)

    print("\nLoading experiment results...")
    results = load_all_results()
    print(f"Loaded {len(results)} experiment files")

    if not results:
        print("No results found!")
        return

    # Count unique configurations
    datasets = set(r["metadata"]["dataset"] for r in results)
    lengths = set(r["metadata"]["data_length"] for r in results)
    budgets = set(r["metadata"]["budget"] for r in results)
    seeds = set(r["metadata"]["seed"] for r in results)

    print(f"\nDatasets: {sorted(datasets)}")
    print(f"Data lengths: {sorted(lengths)}")
    print(f"Budgets: {sorted(budgets)}")
    print(f"Seeds: {sorted(seeds)}")
    print(f"Total experiments: {len(results) * 2} (PyReCo + LSTM each)")

    agg = aggregate_by_dataset_budget(results)

    # Print all tables
    print_all_experiments_table(results)
    print_detailed_by_data_length(results)
    print_summary_table(agg)
    print_training_time_table(agg)
    print_data_length_table(agg)
    print_winner_summary(agg)
    print_markdown_table(agg)

    # Save to CSV
    save_to_csv(agg)

    print("\n" + "=" * 80)
    print("DATA EFFICIENCY TABLE GENERATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
