#!/usr/bin/env python3
"""
Generate comprehensive results tables from MULTI-STEP PREDICTION experiments.
Multi-step experiments test model prediction accuracy at different horizons (1, 5, 10, 20, 50 steps).
Data source: results/multi_step/ (multistep_*.json files)
Outputs both console tables and CSV files for analysis.
"""

import json
import os
from pathlib import Path
from collections import defaultdict
import csv

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "multi_step"
OUTPUT_DIR = PROJECT_ROOT / "results" / "tables" / "multi_step"


def load_all_results():
    """Load all multi-step experiment results."""
    results = []
    for f in RESULTS_DIR.glob("multistep_*.json"):
        try:
            with open(f) as fp:
                data = json.load(fp)
                data["_source_file"] = str(f)
                results.append(data)
        except Exception as e:
            print(f"Error loading {f}: {e}")
    return results


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


def print_all_experiments_table(results):
    """Print table showing ALL individual experiments."""
    print("\n" + "=" * 180)
    print("ALL MULTI-STEP PREDICTION EXPERIMENTS - DETAILED VIEW")
    print("=" * 180)

    header = (f"{'Dataset':<12} {'Seed':<6} {'Train%':<8} {'Budget':<8} {'Model':<10} "
              f"{'H=1 R²':<10} {'H=5 R²':<10} {'H=10 R²':<10} {'H=20 R²':<10} {'H=50 R²':<10} {'Time(s)':<12}")
    print(header)
    print("-" * 180)

    # Collect all experiments
    all_exps = []
    for r in results:
        meta = r["metadata"]
        horizons = meta.get("horizons", [1, 5, 10, 20, 50])

        for model_key, model_data in r.get("models", {}).items():
            model_name = "PyReCo" if "pyreco" in model_key.lower() else "LSTM"
            horizon_results = model_data.get("horizon_results", {})

            exp = {
                "dataset": meta.get("dataset", "unknown"),
                "seed": meta.get("seed", 0),
                "train_frac": meta.get("train_frac", 0),
                "budget": meta.get("budget", "unknown"),
                "model": model_name,
                "train_time": model_data.get("train_time", 0),
            }

            # Add R² for each horizon
            for h in horizons:
                h_str = str(h)
                if h_str in horizon_results:
                    exp[f"h{h}_r2"] = horizon_results[h_str].get("r2", 0)
                    exp[f"h{h}_mse"] = horizon_results[h_str].get("mse", 0)
                else:
                    exp[f"h{h}_r2"] = 0
                    exp[f"h{h}_mse"] = 0

            all_exps.append(exp)

    # Sort by dataset, budget, train_frac, seed, model
    sorted_exps = sorted(all_exps,
                         key=lambda x: (x["dataset"], x["budget"], x["train_frac"], x["seed"], x["model"]))

    prev_dataset = None
    prev_budget = None

    for exp in sorted_exps:
        if exp["dataset"] != prev_dataset:
            if prev_dataset is not None:
                print()
            prev_dataset = exp["dataset"]
            prev_budget = None

        if exp["budget"] != prev_budget:
            if prev_budget is not None and exp["dataset"] == prev_dataset:
                print("-" * 180)
            prev_budget = exp["budget"]

        print(f"{exp['dataset']:<12} {exp['seed']:<6} {exp['train_frac']:<8.1f} {exp['budget']:<8} "
              f"{exp['model']:<10} "
              f"{exp.get('h1_r2', 0):<10.4f} "
              f"{exp.get('h5_r2', 0):<10.4f} "
              f"{exp.get('h10_r2', 0):<10.4f} "
              f"{exp.get('h20_r2', 0):<10.4f} "
              f"{exp.get('h50_r2', 0):<10.4f} "
              f"{exp['train_time']:<12.2f}")


def print_summary_by_horizon(results):
    """Print summary table aggregated by dataset, budget, and horizon."""
    print("\n" + "=" * 160)
    print("SUMMARY BY DATASET, BUDGET, AND PREDICTION HORIZON (Mean ± Std across seeds)")
    print("=" * 160)

    # Group by dataset, budget, model, horizon
    grouped = defaultdict(lambda: {"r2": [], "mse": []})

    for r in results:
        meta = r["metadata"]
        horizons = meta.get("horizons", [1, 5, 10, 20, 50])

        for model_key, model_data in r.get("models", {}).items():
            model = "pyreco" if "pyreco" in model_key.lower() else "lstm"
            horizon_results = model_data.get("horizon_results", {})

            for h in horizons:
                h_str = str(h)
                if h_str in horizon_results:
                    key = (meta["dataset"], meta["budget"], model, h)
                    grouped[key]["r2"].append(horizon_results[h_str].get("r2", 0))
                    grouped[key]["mse"].append(horizon_results[h_str].get("mse", 0))

    header = (f"{'Dataset':<12} {'Budget':<8} {'Horizon':<8} "
              f"{'PyReCo R² (±std)':<22} {'LSTM R² (±std)':<22} {'Winner':<10} {'Diff':<8}")
    print(header)
    print("-" * 160)

    for dataset in ["lorenz", "mackeyglass", "santafe"]:
        for budget in ["small", "medium", "large"]:
            for horizon in [1, 5, 10, 20, 50]:
                pyreco_key = (dataset, budget, "pyreco", horizon)
                lstm_key = (dataset, budget, "lstm", horizon)

                if pyreco_key not in grouped or lstm_key not in grouped:
                    continue

                pyreco_r2, pyreco_std = compute_stats(grouped[pyreco_key]["r2"])
                lstm_r2, lstm_std = compute_stats(grouped[lstm_key]["r2"])

                winner = "PyReCo" if pyreco_r2 > lstm_r2 else "LSTM"
                diff = abs(pyreco_r2 - lstm_r2)

                print(f"{dataset:<12} {budget:<8} {horizon:<8} "
                      f"{pyreco_r2:.4f} ± {pyreco_std:.4f}        "
                      f"{lstm_r2:.4f} ± {lstm_std:.4f}        "
                      f"{winner:<10} {diff:.4f}")
            print()


def print_horizon_comparison_table(results):
    """Print table comparing performance across prediction horizons."""
    print("\n" + "=" * 140)
    print("PERFORMANCE DEGRADATION WITH PREDICTION HORIZON")
    print("=" * 140)

    # Group by dataset, budget, model
    grouped = defaultdict(lambda: {h: [] for h in [1, 5, 10, 20, 50]})

    for r in results:
        meta = r["metadata"]

        for model_key, model_data in r.get("models", {}).items():
            model = "PyReCo" if "pyreco" in model_key.lower() else "LSTM"
            horizon_results = model_data.get("horizon_results", {})

            key = (meta["dataset"], meta["budget"], model)
            for h in [1, 5, 10, 20, 50]:
                h_str = str(h)
                if h_str in horizon_results:
                    grouped[key][h].append(horizon_results[h_str].get("r2", 0))

    header = (f"{'Dataset':<12} {'Budget':<8} {'Model':<10} "
              f"{'H=1':<10} {'H=5':<10} {'H=10':<10} {'H=20':<10} {'H=50':<10} {'Decay%':<12}")
    print(header)
    print("-" * 140)

    for dataset in ["lorenz", "mackeyglass", "santafe"]:
        for budget in ["small", "medium", "large"]:
            for model in ["PyReCo", "LSTM"]:
                key = (dataset, budget, model)
                if key not in grouped:
                    continue

                r2_values = {}
                for h in [1, 5, 10, 20, 50]:
                    if grouped[key][h]:
                        r2_values[h] = sum(grouped[key][h]) / len(grouped[key][h])
                    else:
                        r2_values[h] = 0

                # Calculate decay from H=1 to H=50
                if r2_values.get(1, 0) > 0:
                    decay = (r2_values.get(1, 0) - r2_values.get(50, 0)) / r2_values.get(1, 0) * 100
                else:
                    decay = 0

                print(f"{dataset:<12} {budget:<8} {model:<10} "
                      f"{r2_values.get(1, 0):<10.4f} "
                      f"{r2_values.get(5, 0):<10.4f} "
                      f"{r2_values.get(10, 0):<10.4f} "
                      f"{r2_values.get(20, 0):<10.4f} "
                      f"{r2_values.get(50, 0):<10.4f} "
                      f"{decay:<12.1f}%")
            print()


def print_winner_by_horizon(results):
    """Print winner analysis for each prediction horizon."""
    print("\n" + "=" * 100)
    print("WINNER ANALYSIS BY PREDICTION HORIZON")
    print("=" * 100)

    # Group by dataset, budget, horizon
    grouped = defaultdict(lambda: {"pyreco": [], "lstm": []})

    for r in results:
        meta = r["metadata"]

        for model_key, model_data in r.get("models", {}).items():
            model = "pyreco" if "pyreco" in model_key.lower() else "lstm"
            horizon_results = model_data.get("horizon_results", {})

            for h in [1, 5, 10, 20, 50]:
                h_str = str(h)
                if h_str in horizon_results:
                    key = (meta["dataset"], h)
                    grouped[key][model].append(horizon_results[h_str].get("r2", 0))

    # Count wins by horizon
    horizon_wins = {h: {"pyreco": 0, "lstm": 0} for h in [1, 5, 10, 20, 50]}

    print(f"\n{'Horizon':<10} {'Dataset':<12} {'PyReCo R²':<14} {'LSTM R²':<14} {'Winner':<12}")
    print("-" * 100)

    for h in [1, 5, 10, 20, 50]:
        for dataset in ["lorenz", "mackeyglass", "santafe"]:
            key = (dataset, h)
            if key not in grouped:
                continue

            pyreco_r2 = sum(grouped[key]["pyreco"]) / len(grouped[key]["pyreco"]) if grouped[key]["pyreco"] else 0
            lstm_r2 = sum(grouped[key]["lstm"]) / len(grouped[key]["lstm"]) if grouped[key]["lstm"] else 0

            winner = "PyReCo" if pyreco_r2 > lstm_r2 else "LSTM"
            horizon_wins[h][winner.lower()] += 1

            print(f"{h:<10} {dataset:<12} {pyreco_r2:<14.4f} {lstm_r2:<14.4f} **{winner}**")
        print()

    # Summary
    print("\n" + "=" * 60)
    print("WINS BY HORIZON (across all datasets)")
    print("=" * 60)
    print(f"{'Horizon':<10} {'PyReCo Wins':<15} {'LSTM Wins':<15} {'Overall Winner':<15}")
    print("-" * 60)
    for h in [1, 5, 10, 20, 50]:
        overall = "PyReCo" if horizon_wins[h]["pyreco"] > horizon_wins[h]["lstm"] else "LSTM"
        print(f"{h:<10} {horizon_wins[h]['pyreco']:<15} {horizon_wins[h]['lstm']:<15} **{overall}**")


def save_to_csv(results):
    """Save results to CSV files."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Detailed CSV with all experiments
    detailed_file = OUTPUT_DIR / "multi_step_detailed.csv"
    with open(detailed_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Dataset", "Seed", "Train_Frac", "Budget", "Model", "Horizon",
                        "MSE", "R2", "MAE", "RMSE", "NRMSE"])

        for r in results:
            meta = r["metadata"]
            for model_key, model_data in r.get("models", {}).items():
                model_name = "PyReCo" if "pyreco" in model_key.lower() else "LSTM"
                horizon_results = model_data.get("horizon_results", {})

                for h_str, metrics in horizon_results.items():
                    writer.writerow([
                        meta.get("dataset", ""),
                        meta.get("seed", 0),
                        meta.get("train_frac", 0),
                        meta.get("budget", ""),
                        model_name,
                        int(h_str),
                        f"{metrics.get('mse', 0):.6f}",
                        f"{metrics.get('r2', 0):.4f}",
                        f"{metrics.get('mae', 0):.6f}",
                        f"{metrics.get('rmse', 0):.6f}",
                        f"{metrics.get('nrmse', 0):.6f}"
                    ])

    print(f"\nSaved detailed results to: {detailed_file}")

    # Summary CSV grouped by dataset, budget, horizon
    summary_file = OUTPUT_DIR / "multi_step_summary.csv"
    grouped = defaultdict(lambda: {"pyreco_r2": [], "lstm_r2": [], "pyreco_mse": [], "lstm_mse": []})

    for r in results:
        meta = r["metadata"]
        for model_key, model_data in r.get("models", {}).items():
            model = "pyreco" if "pyreco" in model_key.lower() else "lstm"
            horizon_results = model_data.get("horizon_results", {})

            for h_str, metrics in horizon_results.items():
                key = (meta["dataset"], meta["budget"], int(h_str))
                grouped[key][f"{model}_r2"].append(metrics.get("r2", 0))
                grouped[key][f"{model}_mse"].append(metrics.get("mse", 0))

    with open(summary_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Dataset", "Budget", "Horizon",
                        "PyReCo_R2_Mean", "PyReCo_R2_Std", "LSTM_R2_Mean", "LSTM_R2_Std",
                        "PyReCo_MSE_Mean", "LSTM_MSE_Mean", "Winner", "N"])

        for dataset in ["lorenz", "mackeyglass", "santafe"]:
            for budget in ["small", "medium", "large"]:
                for horizon in [1, 5, 10, 20, 50]:
                    key = (dataset, budget, horizon)
                    if key not in grouped:
                        continue

                    data = grouped[key]
                    pyreco_r2, pyreco_r2_std = compute_stats(data["pyreco_r2"])
                    lstm_r2, lstm_r2_std = compute_stats(data["lstm_r2"])
                    pyreco_mse, _ = compute_stats(data["pyreco_mse"])
                    lstm_mse, _ = compute_stats(data["lstm_mse"])

                    winner = "PyReCo" if pyreco_r2 > lstm_r2 else "LSTM"
                    n = len(data["pyreco_r2"])

                    writer.writerow([
                        dataset, budget, horizon,
                        f"{pyreco_r2:.4f}", f"{pyreco_r2_std:.4f}",
                        f"{lstm_r2:.4f}", f"{lstm_r2_std:.4f}",
                        f"{pyreco_mse:.6f}", f"{lstm_mse:.6f}",
                        winner, n
                    ])

    print(f"Saved summary to: {summary_file}")


def print_markdown_table(results):
    """Print Markdown formatted tables."""
    print("\n" + "=" * 80)
    print("MARKDOWN TABLES (Copy to documentation)")
    print("=" * 80)

    # Group data
    grouped = defaultdict(lambda: {"pyreco": [], "lstm": []})

    for r in results:
        meta = r["metadata"]
        for model_key, model_data in r.get("models", {}).items():
            model = "pyreco" if "pyreco" in model_key.lower() else "lstm"
            horizon_results = model_data.get("horizon_results", {})

            for h_str, metrics in horizon_results.items():
                key = (meta["dataset"], meta["budget"], int(h_str))
                grouped[key][model].append(metrics.get("r2", 0))

    print("\n### Multi-Step Prediction Results (R² Score)\n")
    print("| Dataset | Budget | Horizon | PyReCo R² | LSTM R² | Winner |")
    print("|---------|--------|---------|-----------|---------|--------|")

    for dataset in ["lorenz", "mackeyglass", "santafe"]:
        for budget in ["small", "medium", "large"]:
            for horizon in [1, 5, 10, 20, 50]:
                key = (dataset, budget, horizon)
                if key not in grouped:
                    continue

                pyreco_r2, pyreco_std = compute_stats(grouped[key]["pyreco"])
                lstm_r2, lstm_std = compute_stats(grouped[key]["lstm"])

                winner = "**PyReCo**" if pyreco_r2 > lstm_r2 else "**LSTM**"

                print(f"| {dataset} | {budget} | {horizon} | {pyreco_r2:.4f} ± {pyreco_std:.4f} | "
                      f"{lstm_r2:.4f} ± {lstm_std:.4f} | {winner} |")


def main():
    print("=" * 80)
    print("MULTI-STEP PREDICTION EXPERIMENTS TABLE GENERATOR")
    print("Testing model prediction accuracy at different horizons")
    print("=" * 80)

    print("\nLoading experiment results...")
    results = load_all_results()
    print(f"Loaded {len(results)} experiment files")

    if not results:
        print("No results found!")
        return

    # Count unique configurations
    datasets = set(r["metadata"]["dataset"] for r in results)
    budgets = set(r["metadata"]["budget"] for r in results)
    seeds = set(r["metadata"]["seed"] for r in results)
    train_fracs = set(r["metadata"]["train_frac"] for r in results)

    print(f"\nDatasets: {sorted(datasets)}")
    print(f"Budgets: {sorted(budgets)}")
    print(f"Seeds: {sorted(seeds)}")
    print(f"Train fractions: {sorted(train_fracs)}")
    print(f"Prediction horizons: [1, 5, 10, 20, 50]")
    print(f"Total experiments: {len(results)} files")

    # Print all tables
    print_all_experiments_table(results)
    print_summary_by_horizon(results)
    print_horizon_comparison_table(results)
    print_winner_by_horizon(results)
    print_markdown_table(results)

    # Save to CSV
    save_to_csv(results)

    print("\n" + "=" * 80)
    print("MULTI-STEP PREDICTION TABLE GENERATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
