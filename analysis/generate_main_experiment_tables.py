#!/usr/bin/env python3
"""
Generate comprehensive results tables from main experiments.
Main experiments compare PyReCo vs LSTM across different parameter budgets.
Data sources: results/final/ and results/backup_20251201_lstm_no_tuning/
"""

import json
import os
from pathlib import Path
from collections import defaultdict
import csv

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIRS = [
    PROJECT_ROOT / "results" / "final_with_lstm",  # Main experiments with both PyReCo and LSTM
]
OUTPUT_DIR = PROJECT_ROOT / "results" / "tables" / "main_experiments"


def load_all_results():
    """Load all main experiment results."""
    results = []
    for results_dir in RESULTS_DIRS:
        if not results_dir.exists():
            continue
        for f in results_dir.glob("results_*.json"):
            # Skip progress tracking files
            if "progress" in f.name:
                continue
            try:
                with open(f) as fp:
                    data = json.load(fp)
                    data["_source_file"] = str(f)
                    results.append(data)
            except Exception as e:
                print(f"Error loading {f}: {e}")
    return results


def extract_experiment_info(result):
    """Extract structured info from a single experiment result."""
    meta = result.get("metadata", {})
    experiments = []

    for budget, budget_results in result.get("results", {}).items():
        if not isinstance(budget_results, list):
            continue
        for model_result in budget_results:
            exp = {
                "dataset": meta.get("dataset", "unknown"),
                "seed": meta.get("seed", 0),
                "train_frac": meta.get("train_frac", 0),
                "budget": budget,
                "model_type": model_result.get("model_type", "unknown"),
                "test_mse": model_result.get("test_mse", 0),
                "test_r2": model_result.get("test_r2", 0),
                "test_mae": model_result.get("test_mae", 0),
                "trainable_params": model_result.get("param_info", {}).get("trainable", 0),
                "total_params": model_result.get("param_info", {}).get("total", 0),
                "tune_time": model_result.get("tune_time", 0),
                "train_time": model_result.get("final_train_time", 0),
                "inference_time_ms": model_result.get("inference_time_per_sample_ms", 0),
            }
            experiments.append(exp)

    return experiments


def aggregate_results(results):
    """Aggregate all experiments into a structured format."""
    all_experiments = []
    for result in results:
        experiments = extract_experiment_info(result)
        all_experiments.extend(experiments)
    return all_experiments


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


def print_all_experiments_table(experiments):
    """Print table showing ALL individual experiments."""
    print("\n" + "=" * 140)
    print("ALL MAIN EXPERIMENTS - DETAILED VIEW")
    print("=" * 140)

    header = (f"{'Dataset':<12} {'Seed':<6} {'Train%':<8} {'Budget':<8} {'Model':<18} "
              f"{'MSE':<14} {'R²':<10} {'Params':<10} {'Time(s)':<12}")
    print(header)
    print("-" * 140)

    # Sort by dataset, budget, train_frac, seed, model
    sorted_exps = sorted(experiments,
                         key=lambda x: (x["dataset"], x["budget"], x["train_frac"], x["seed"], x["model_type"]))

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
                print("-" * 140)
            prev_budget = exp["budget"]

        model_name = exp["model_type"].replace("_standard", "").replace("pyreco", "PyReCo").upper()
        if "lstm" in model_name.lower():
            model_name = "LSTM"
        elif "pyreco" in model_name.lower():
            model_name = "PyReCo"

        print(f"{exp['dataset']:<12} {exp['seed']:<6} {exp['train_frac']:<8.1f} {exp['budget']:<8} "
              f"{model_name:<18} {exp['test_mse']:<14.6f} {exp['test_r2']:<10.4f} "
              f"{exp['trainable_params']:<10} {exp['tune_time'] + exp['train_time']:<12.2f}")


def print_summary_by_dataset_budget(experiments):
    """Print summary table aggregated by dataset and budget."""
    print("\n" + "=" * 120)
    print("SUMMARY BY DATASET AND BUDGET (Mean ± Std across seeds and train fractions)")
    print("=" * 120)

    # Group by dataset, budget, model
    grouped = defaultdict(lambda: {"mse": [], "r2": [], "time": []})

    for exp in experiments:
        key = (exp["dataset"], exp["budget"], exp["model_type"])
        grouped[key]["mse"].append(exp["test_mse"])
        grouped[key]["r2"].append(exp["test_r2"])
        grouped[key]["time"].append(exp["tune_time"] + exp["train_time"])

    header = (f"{'Dataset':<12} {'Budget':<8} {'Model':<12} {'MSE (mean±std)':<22} "
              f"{'R² (mean±std)':<18} {'Time (mean)':<14} {'N':<6}")
    print(header)
    print("-" * 120)

    for dataset in ["lorenz", "mackeyglass", "santafe"]:
        for budget in ["small", "medium", "large"]:
            for model_type in ["pyreco_standard", "lstm"]:
                key = (dataset, budget, model_type)
                if key not in grouped:
                    continue

                data = grouped[key]
                mse_mean, mse_std = compute_stats(data["mse"])
                r2_mean, r2_std = compute_stats(data["r2"])
                time_mean, _ = compute_stats(data["time"])

                model_name = "PyReCo" if "pyreco" in model_type else "LSTM"

                print(f"{dataset:<12} {budget:<8} {model_name:<12} "
                      f"{mse_mean:.6f}±{mse_std:.4f}     "
                      f"{r2_mean:.4f}±{r2_std:.4f}   "
                      f"{time_mean:>10.2f}s     {len(data['mse']):<6}")
        print()


def print_winner_analysis(experiments):
    """Print analysis of which model wins in each condition."""
    print("\n" + "=" * 100)
    print("WINNER ANALYSIS BY DATASET AND BUDGET")
    print("=" * 100)

    # Group by dataset, budget
    grouped = defaultdict(lambda: {"pyreco": [], "lstm": []})

    for exp in experiments:
        key = (exp["dataset"], exp["budget"])
        if "pyreco" in exp["model_type"].lower():
            grouped[key]["pyreco"].append(exp["test_r2"])
        elif "lstm" in exp["model_type"].lower():
            grouped[key]["lstm"].append(exp["test_r2"])

    header = f"{'Dataset':<12} {'Budget':<8} {'PyReCo R²':<16} {'LSTM R²':<16} {'Winner':<12} {'Difference':<12}"
    print(header)
    print("-" * 100)

    dataset_wins = defaultdict(lambda: {"pyreco": 0, "lstm": 0})

    for dataset in ["lorenz", "mackeyglass", "santafe"]:
        for budget in ["small", "medium", "large"]:
            key = (dataset, budget)
            if key not in grouped:
                continue

            pyreco_r2 = sum(grouped[key]["pyreco"]) / len(grouped[key]["pyreco"]) if grouped[key]["pyreco"] else 0
            lstm_r2 = sum(grouped[key]["lstm"]) / len(grouped[key]["lstm"]) if grouped[key]["lstm"] else 0

            if pyreco_r2 > lstm_r2:
                winner = "**PyReCo**"
                dataset_wins[dataset]["pyreco"] += 1
            else:
                winner = "**LSTM**"
                dataset_wins[dataset]["lstm"] += 1

            diff = abs(pyreco_r2 - lstm_r2)

            print(f"{dataset:<12} {budget:<8} {pyreco_r2:.4f}           {lstm_r2:.4f}           "
                  f"{winner:<12} {diff:.4f}")
        print()

    print("\n" + "=" * 60)
    print("OVERALL WINNER BY DATASET")
    print("=" * 60)
    for dataset in ["lorenz", "mackeyglass", "santafe"]:
        wins = dataset_wins[dataset]
        overall = "PyReCo" if wins["pyreco"] > wins["lstm"] else "LSTM"
        print(f"{dataset:<12}: PyReCo wins {wins['pyreco']}/3, LSTM wins {wins['lstm']}/3 -> **{overall}**")


def print_training_efficiency_table(experiments):
    """Print training time and parameter efficiency comparison."""
    print("\n" + "=" * 130)
    print("TRAINING EFFICIENCY COMPARISON")
    print("=" * 130)

    # Group by dataset, budget, model
    grouped = defaultdict(lambda: {"time": [], "params": [], "r2": []})

    for exp in experiments:
        key = (exp["dataset"], exp["budget"], exp["model_type"])
        grouped[key]["time"].append(exp["tune_time"] + exp["train_time"])
        grouped[key]["params"].append(exp["trainable_params"])
        grouped[key]["r2"].append(exp["test_r2"])

    header = (f"{'Dataset':<12} {'Budget':<8} {'Model':<10} {'Params':<12} {'Time(s)':<12} "
              f"{'R²':<10} {'R²/Param':<14} {'Speedup':<20}")
    print(header)
    print("-" * 130)

    for dataset in ["lorenz", "mackeyglass", "santafe"]:
        for budget in ["small", "medium", "large"]:
            pyreco_key = (dataset, budget, "pyreco_standard")
            lstm_key = (dataset, budget, "lstm")

            if pyreco_key not in grouped or lstm_key not in grouped:
                continue

            pyreco_data = grouped[pyreco_key]
            lstm_data = grouped[lstm_key]

            pyreco_time = sum(pyreco_data["time"]) / len(pyreco_data["time"])
            lstm_time = sum(lstm_data["time"]) / len(lstm_data["time"])

            pyreco_params = sum(pyreco_data["params"]) / len(pyreco_data["params"])
            lstm_params = sum(lstm_data["params"]) / len(lstm_data["params"])

            pyreco_r2 = sum(pyreco_data["r2"]) / len(pyreco_data["r2"])
            lstm_r2 = sum(lstm_data["r2"]) / len(lstm_data["r2"])

            pyreco_eff = pyreco_r2 / pyreco_params * 1000 if pyreco_params > 0 else 0
            lstm_eff = lstm_r2 / lstm_params * 1000 if lstm_params > 0 else 0

            if pyreco_time < lstm_time:
                speedup = f"PyReCo {lstm_time/pyreco_time:.1f}x faster"
            else:
                speedup = f"LSTM {pyreco_time/lstm_time:.1f}x faster"

            print(f"{dataset:<12} {budget:<8} PyReCo    {int(pyreco_params):<12} {pyreco_time:<12.2f} "
                  f"{pyreco_r2:<10.4f} {pyreco_eff:<14.4f}")
            print(f"{'':<12} {'':<8} LSTM      {int(lstm_params):<12} {lstm_time:<12.2f} "
                  f"{lstm_r2:<10.4f} {lstm_eff:<14.4f} {speedup}")
            print()


def save_to_csv(experiments):
    """Save all results to CSV files."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Detailed CSV with all experiments
    detailed_file = OUTPUT_DIR / "main_experiments_detailed.csv"
    with open(detailed_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Dataset", "Seed", "Train_Frac", "Budget", "Model",
                        "MSE", "R2", "MAE", "Trainable_Params", "Total_Params",
                        "Tune_Time", "Train_Time", "Inference_Time_ms"])

        for exp in experiments:
            model_name = "PyReCo" if "pyreco" in exp["model_type"].lower() else "LSTM"
            writer.writerow([
                exp["dataset"], exp["seed"], exp["train_frac"], exp["budget"],
                model_name, f"{exp['test_mse']:.6f}", f"{exp['test_r2']:.4f}",
                f"{exp['test_mae']:.6f}", exp["trainable_params"], exp["total_params"],
                f"{exp['tune_time']:.2f}", f"{exp['train_time']:.2f}",
                f"{exp['inference_time_ms']:.4f}"
            ])

    print(f"\nSaved detailed results to: {detailed_file}")

    # Summary CSV
    summary_file = OUTPUT_DIR / "main_experiments_summary.csv"
    grouped = defaultdict(lambda: {"mse": [], "r2": [], "time": [], "params": []})

    for exp in experiments:
        key = (exp["dataset"], exp["budget"], exp["model_type"])
        grouped[key]["mse"].append(exp["test_mse"])
        grouped[key]["r2"].append(exp["test_r2"])
        grouped[key]["time"].append(exp["tune_time"] + exp["train_time"])
        grouped[key]["params"].append(exp["trainable_params"])

    with open(summary_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Dataset", "Budget", "Model", "MSE_Mean", "MSE_Std",
                        "R2_Mean", "R2_Std", "Time_Mean", "Params_Mean", "N", "Winner"])

        for dataset in ["lorenz", "mackeyglass", "santafe"]:
            for budget in ["small", "medium", "large"]:
                pyreco_key = (dataset, budget, "pyreco_standard")
                lstm_key = (dataset, budget, "lstm")

                pyreco_r2 = 0
                lstm_r2 = 0

                for model_key in [pyreco_key, lstm_key]:
                    if model_key not in grouped:
                        continue

                    data = grouped[model_key]
                    mse_mean, mse_std = compute_stats(data["mse"])
                    r2_mean, r2_std = compute_stats(data["r2"])
                    time_mean, _ = compute_stats(data["time"])
                    params_mean, _ = compute_stats(data["params"])

                    model_name = "PyReCo" if "pyreco" in model_key[2] else "LSTM"

                    if model_name == "PyReCo":
                        pyreco_r2 = r2_mean
                    else:
                        lstm_r2 = r2_mean

                    winner = ""
                    if model_name == "LSTM":
                        winner = "PyReCo" if pyreco_r2 > lstm_r2 else "LSTM"

                    writer.writerow([
                        dataset, budget, model_name,
                        f"{mse_mean:.6f}", f"{mse_std:.6f}",
                        f"{r2_mean:.4f}", f"{r2_std:.4f}",
                        f"{time_mean:.2f}", f"{params_mean:.0f}",
                        len(data["mse"]), winner
                    ])

    print(f"Saved summary to: {summary_file}")


def print_markdown_table(experiments):
    """Print Markdown formatted tables."""
    print("\n" + "=" * 80)
    print("MARKDOWN TABLES (Copy to documentation)")
    print("=" * 80)

    # Group data
    grouped = defaultdict(lambda: {"pyreco": {"r2": [], "mse": []}, "lstm": {"r2": [], "mse": []}})

    for exp in experiments:
        key = (exp["dataset"], exp["budget"])
        model = "pyreco" if "pyreco" in exp["model_type"].lower() else "lstm"
        grouped[key][model]["r2"].append(exp["test_r2"])
        grouped[key][model]["mse"].append(exp["test_mse"])

    print("\n### Main Experiment Results (R² Score)\n")
    print("| Dataset | Budget | PyReCo R² | LSTM R² | Winner |")
    print("|---------|--------|-----------|---------|--------|")

    for dataset in ["lorenz", "mackeyglass", "santafe"]:
        for budget in ["small", "medium", "large"]:
            key = (dataset, budget)
            if key not in grouped:
                continue

            pyreco_r2, pyreco_std = compute_stats(grouped[key]["pyreco"]["r2"])
            lstm_r2, lstm_std = compute_stats(grouped[key]["lstm"]["r2"])

            winner = "**PyReCo**" if pyreco_r2 > lstm_r2 else "**LSTM**"

            print(f"| {dataset} | {budget} | {pyreco_r2:.4f} ± {pyreco_std:.4f} | "
                  f"{lstm_r2:.4f} ± {lstm_std:.4f} | {winner} |")


def main():
    print("=" * 80)
    print("MAIN EXPERIMENTS TABLE GENERATOR")
    print("PyReCo vs LSTM Comparison across Parameter Budgets")
    print("=" * 80)

    print("\nLoading experiment results...")
    results = load_all_results()
    print(f"Loaded {len(results)} experiment files")

    if not results:
        print("No results found!")
        return

    experiments = aggregate_results(results)
    print(f"Total experiments: {len(experiments)}")

    # Count unique configurations
    datasets = set(e["dataset"] for e in experiments)
    budgets = set(e["budget"] for e in experiments)
    seeds = set(e["seed"] for e in experiments)
    train_fracs = set(e["train_frac"] for e in experiments)
    models = set(e["model_type"] for e in experiments)

    print(f"\nDatasets: {sorted(datasets)}")
    print(f"Budgets: {sorted(budgets)}")
    print(f"Seeds: {sorted(seeds)}")
    print(f"Train fractions: {sorted(train_fracs)}")
    print(f"Models: {sorted(models)}")

    # Print all tables
    print_all_experiments_table(experiments)
    print_summary_by_dataset_budget(experiments)
    print_winner_analysis(experiments)
    print_training_efficiency_table(experiments)
    print_markdown_table(experiments)

    # Save to CSV
    save_to_csv(experiments)

    print("\n" + "=" * 80)
    print("MAIN EXPERIMENTS TABLE GENERATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
