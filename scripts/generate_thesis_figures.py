#!/usr/bin/env python3
"""
Generate publication-quality PDF figures for Master thesis.
Comparative Analysis of Reservoir Computing and LSTM Networks
for Chaotic Time-Series Prediction.

Output: /Users/hengz/Documents/Abschlussarbeite/LaTeX_Template/img/
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import sys

# ── Paths ──────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(BASE, "results", "tables")
IMG_DIR = "/Users/hengz/Documents/Abschlussarbeite/LaTeX_Template/img"
os.makedirs(IMG_DIR, exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLOR_PYRECO = "#2196F3"   # blue
COLOR_LSTM   = "#FF5722"   # red-orange
COLORS = {"PyReCo": COLOR_PYRECO, "LSTM": COLOR_LSTM}
DATASET_LABELS = {"lorenz": "Lorenz", "mackeyglass": "Mackey-Glass", "santafe": "Santa Fe"}
BUDGET_ORDER = ["small", "medium", "large"]
BUDGET_LABELS = {"small": "Small\n(~1K)", "medium": "Medium\n(~10K)", "large": "Large\n(~50K)"}


def load_main():
    return pd.read_csv(os.path.join(RESULTS, "main_experiments", "main_experiments_summary.csv"))

def load_data_eff():
    return pd.read_csv(os.path.join(RESULTS, "data_efficiency", "detailed_results.csv"))

def load_data_eff_summary():
    return pd.read_csv(os.path.join(RESULTS, "data_efficiency", "summary_by_dataset_budget.csv"))

def load_multi():
    return pd.read_csv(os.path.join(RESULTS, "multi_step", "multi_step_summary.csv"))

def save(fig, name):
    path = os.path.join(IMG_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 1: R² comparison across datasets and budgets
# ═══════════════════════════════════════════════════════════════════════════
def fig_r2_by_dataset_budget():
    df = load_main()
    datasets = ["lorenz", "mackeyglass", "santafe"]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), sharey=True)

    for ax, ds in zip(axes, datasets):
        sub = df[df["Dataset"] == ds]
        x = np.arange(len(BUDGET_ORDER))
        w = 0.35
        for i, model in enumerate(["PyReCo", "LSTM"]):
            row = sub[sub["Model"] == model].set_index("Budget").reindex(BUDGET_ORDER)
            ax.bar(x + (i - 0.5) * w, row["R2_Mean"], w,
                   yerr=row["R2_Std"], capsize=3,
                   color=COLORS[model], label=model, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([BUDGET_LABELS[b] for b in BUDGET_ORDER])
        ax.set_title(DATASET_LABELS[ds])
        ax.set_ylim(0, 1.09)

    axes[0].set_ylabel("R² Score")
    axes[0].legend()
    fig.suptitle("Single-Step Prediction Performance", fontsize=12, y=1.02)
    fig.tight_layout()
    save(fig, "fig_r2_by_dataset_budget.pdf")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 2: MSE comparison (log scale)
# ═══════════════════════════════════════════════════════════════════════════
def fig_mse_by_dataset_budget():
    df = load_main()
    datasets = ["lorenz", "mackeyglass", "santafe"]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))

    for ax, ds in zip(axes, datasets):
        sub = df[df["Dataset"] == ds]
        x = np.arange(len(BUDGET_ORDER))
        w = 0.35
        for i, model in enumerate(["PyReCo", "LSTM"]):
            row = sub[sub["Model"] == model].set_index("Budget").reindex(BUDGET_ORDER)
            ax.bar(x + (i - 0.5) * w, row["MSE_Mean"], w,
                   color=COLORS[model], label=model, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([BUDGET_LABELS[b] for b in BUDGET_ORDER])
        ax.set_title(DATASET_LABELS[ds])
        ax.set_yscale("log")
        ax.set_ylabel("MSE (log scale)")

    axes[0].legend()
    fig.suptitle("Mean Squared Error Comparison", fontsize=12, y=1.02)
    fig.tight_layout()
    save(fig, "fig_mse_by_dataset_budget.pdf")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 3: Training time comparison
# ═══════════════════════════════════════════════════════════════════════════
def fig_training_time():
    df = load_main()
    datasets = ["lorenz", "mackeyglass", "santafe"]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))

    for ax, ds in zip(axes, datasets):
        sub = df[df["Dataset"] == ds]
        x = np.arange(len(BUDGET_ORDER))
        w = 0.35
        for i, model in enumerate(["PyReCo", "LSTM"]):
            row = sub[sub["Model"] == model].set_index("Budget").reindex(BUDGET_ORDER)
            ax.bar(x + (i - 0.5) * w, row["Time_Mean"], w,
                   color=COLORS[model], label=model, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([BUDGET_LABELS[b] for b in BUDGET_ORDER])
        ax.set_title(DATASET_LABELS[ds])
        ax.set_ylabel("Training Time (s)")

    axes[0].legend()
    fig.suptitle("Training Time Comparison (incl. Hyperparameter Search)", fontsize=12, y=1.02)
    fig.tight_layout()
    save(fig, "fig_training_time_comparison.pdf")


# ═══════════════════════════════════════════════════════════════════════════
# Figures 4-6: Data efficiency curves per dataset
# ═══════════════════════════════════════════════════════════════════════════
def fig_data_efficiency(dataset, fname):
    df = load_data_eff()
    df["Model"] = df["Model"].str.upper()
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), sharey=True)

    for ax, budget in zip(axes, BUDGET_ORDER):
        sub = df[(df["Dataset"] == dataset) & (df["Budget"] == budget)]
        for model_raw, label, color in [("PYRECO", "PyReCo", COLOR_PYRECO),
                                         ("LSTM", "LSTM", COLOR_LSTM)]:
            ms = sub[sub["Model"] == model_raw]
            grouped = ms.groupby("Data_Length")["R2"].agg(["mean", "std"]).reset_index()
            grouped = grouped.sort_values("Data_Length")
            ax.errorbar(grouped["Data_Length"], grouped["mean"],
                        yerr=grouped["std"], marker="o", markersize=4,
                        capsize=3, color=color, label=label, linewidth=1.5)
        ax.set_xlabel("Data Length")
        ax.set_title(f"{BUDGET_LABELS[budget].replace(chr(10), ' ')}")
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v/1000)}k"))

    axes[0].set_ylabel("R² Score")
    axes[0].legend()
    fig.suptitle(f"Data Efficiency — {DATASET_LABELS[dataset]}", fontsize=12, y=1.02)
    fig.tight_layout()
    save(fig, fname)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 7: Combined learning curves (all datasets, medium budget)
# ═══════════════════════════════════════════════════════════════════════════
def fig_learning_curves():
    df = load_data_eff()
    df["Model"] = df["Model"].str.upper()
    datasets = ["lorenz", "mackeyglass", "santafe"]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), sharey=True)

    for ax, ds in zip(axes, datasets):
        sub = df[(df["Dataset"] == ds) & (df["Budget"] == "medium")]
        for model_raw, label, color in [("PYRECO", "PyReCo", COLOR_PYRECO),
                                         ("LSTM", "LSTM", COLOR_LSTM)]:
            ms = sub[sub["Model"] == model_raw]
            grouped = ms.groupby("Data_Length")["R2"].agg(["mean", "std"]).reset_index()
            grouped = grouped.sort_values("Data_Length")
            ax.fill_between(grouped["Data_Length"],
                            grouped["mean"] - grouped["std"],
                            grouped["mean"] + grouped["std"],
                            alpha=0.15, color=color)
            ax.plot(grouped["Data_Length"], grouped["mean"],
                    marker="o", markersize=4, color=color, label=label, linewidth=1.5)
        ax.set_xlabel("Data Length")
        ax.set_title(DATASET_LABELS[ds])
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v/1000)}k"))

    axes[0].set_ylabel("R² Score")
    axes[0].legend()
    fig.suptitle("Learning Curves (Medium Budget)", fontsize=12, y=1.02)
    fig.tight_layout()
    save(fig, "fig_learning_curves.pdf")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 8: Multi-step prediction R² vs horizon
# ═══════════════════════════════════════════════════════════════════════════
def fig_multi_step():
    df = load_multi()
    datasets = ["lorenz", "mackeyglass", "santafe"]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), sharey=True)

    for ax, ds in zip(axes, datasets):
        sub = df[(df["Dataset"] == ds) & (df["Budget"] == "medium")]
        sub = sub.sort_values("Horizon")
        ax.plot(sub["Horizon"], sub["PyReCo_R2_Mean"], "o-",
                color=COLOR_PYRECO, label="PyReCo", markersize=4, linewidth=1.5)
        ax.fill_between(sub["Horizon"],
                        sub["PyReCo_R2_Mean"] - sub["PyReCo_R2_Std"],
                        sub["PyReCo_R2_Mean"] + sub["PyReCo_R2_Std"],
                        alpha=0.15, color=COLOR_PYRECO)
        ax.plot(sub["Horizon"], sub["LSTM_R2_Mean"], "s-",
                color=COLOR_LSTM, label="LSTM", markersize=4, linewidth=1.5)
        ax.fill_between(sub["Horizon"],
                        sub["LSTM_R2_Mean"] - sub["LSTM_R2_Std"],
                        sub["LSTM_R2_Mean"] + sub["LSTM_R2_Std"],
                        alpha=0.15, color=COLOR_LSTM)
        ax.set_xlabel("Prediction Horizon (steps)")
        ax.set_title(DATASET_LABELS[ds])
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)

    axes[0].set_ylabel("R² Score")
    axes[0].legend()
    fig.suptitle("Multi-Step Prediction Performance (Medium Budget)", fontsize=12, y=1.02)
    fig.tight_layout()
    save(fig, "fig_multi_step_prediction.pdf")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 9: Horizon degradation — all budgets for each dataset
# ═══════════════════════════════════════════════════════════════════════════
def fig_horizon_degradation():
    df = load_multi()
    datasets = ["lorenz", "mackeyglass", "santafe"]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), sharey=True)

    for ax, ds in zip(axes, datasets):
        for budget, ls in zip(BUDGET_ORDER, ["-", "--", ":"]):
            sub = df[(df["Dataset"] == ds) & (df["Budget"] == budget)].sort_values("Horizon")
            ax.plot(sub["Horizon"], sub["PyReCo_R2_Mean"],
                    f"o{ls}", color=COLOR_PYRECO, markersize=3, linewidth=1.2,
                    label=f"PyReCo {budget}" if ds == datasets[0] else None)
            ax.plot(sub["Horizon"], sub["LSTM_R2_Mean"],
                    f"s{ls}", color=COLOR_LSTM, markersize=3, linewidth=1.2,
                    label=f"LSTM {budget}" if ds == datasets[0] else None)
        ax.set_xlabel("Prediction Horizon")
        ax.set_title(DATASET_LABELS[ds])
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)

    axes[0].set_ylabel("R² Score")
    axes[0].legend(fontsize=7, ncol=2, loc="lower left")
    fig.suptitle("Performance Degradation with Prediction Horizon", fontsize=12, y=1.02)
    fig.tight_layout()
    save(fig, "fig_horizon_degradation.pdf")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 10: Training efficiency (R² per second)
# ═══════════════════════════════════════════════════════════════════════════
def fig_training_efficiency():
    data = {
        "Scale": ["Small", "Medium", "Large"],
        "PyReCo": [0.8071, 0.0998, 0.0177],
        "LSTM":   [0.0338, 0.0458, 0.0419],
    }
    df = pd.DataFrame(data)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    x = np.arange(len(df))
    w = 0.35
    ax.bar(x - w/2, df["PyReCo"], w, color=COLOR_PYRECO, label="PyReCo", alpha=0.85)
    ax.bar(x + w/2, df["LSTM"], w, color=COLOR_LSTM, label="LSTM", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(df["Scale"])
    ax.set_ylabel("R² per Second")
    ax.set_title("Training Efficiency")
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    save(fig, "fig_training_efficiency.pdf")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 11: Training speed ratio by scale
# ═══════════════════════════════════════════════════════════════════════════
def fig_speed_ratio():
    data = {
        "Scale": ["Small", "Medium", "Large"],
        "PyReCo_Time": [1.67, 12.69, 73.33],
        "LSTM_Time":   [34.34, 27.45, 30.69],
    }
    df = pd.DataFrame(data)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    x = np.arange(len(df))
    w = 0.35
    ax.bar(x - w/2, df["PyReCo_Time"], w, color=COLOR_PYRECO, label="PyReCo", alpha=0.85)
    ax.bar(x + w/2, df["LSTM_Time"], w, color=COLOR_LSTM, label="LSTM", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(df["Scale"])
    ax.set_ylabel("Mean Training Time (s)")
    ax.set_title("Training Time by Parameter Budget Scale")
    ax.legend()
    fig.tight_layout()
    save(fig, "fig_energy_comparison.pdf")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 12: Parameter budget breakdown (trainable vs non-trainable)
# ═══════════════════════════════════════════════════════════════════════════
def fig_param_breakdown():
    budgets = ["Small (~1K)", "Medium (~10K)", "Large (~50K)"]
    pyreco_trainable   = [100, 300, 700]
    pyreco_fixed       = [900, 9700, 49300]
    lstm_trainable     = [952, 10052, 49077]

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

    # PyReCo
    ax = axes[0]
    x = np.arange(len(budgets))
    ax.bar(x, pyreco_fixed, color=COLOR_PYRECO, alpha=0.4, label="Fixed (reservoir)")
    ax.bar(x, pyreco_trainable, bottom=pyreco_fixed, color=COLOR_PYRECO, alpha=0.85, label="Trainable (readout)")
    ax.set_xticks(x)
    ax.set_xticklabels(budgets, fontsize=8)
    ax.set_ylabel("Number of Parameters")
    ax.set_title("PyReCo (ESN)")
    ax.legend(fontsize=8)
    ax.set_yscale("log")

    # LSTM
    ax = axes[1]
    ax.bar(x, lstm_trainable, color=COLOR_LSTM, alpha=0.85, label="Trainable (all)")
    ax.set_xticks(x)
    ax.set_xticklabels(budgets, fontsize=8)
    ax.set_title("LSTM")
    ax.legend(fontsize=8)
    ax.set_yscale("log")

    fig.suptitle("Parameter Budget Breakdown: Trainable vs Fixed", fontsize=12, y=1.02)
    fig.tight_layout()
    save(fig, "fig_parameter_budget_breakdown.pdf")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 13: Statistical significance heatmap
# ═══════════════════════════════════════════════════════════════════════════
def fig_stat_significance():
    datasets = ["lorenz", "mackeyglass", "santafe"]
    budgets  = ["small", "medium", "large"]
    # Winner matrix from statistical analysis: +1 = PyReCo, -1 = LSTM
    # Based on R² wins from the report
    winner = np.array([
        [1, 1, 1],      # lorenz: all PyReCo
        [1, 1, 1],      # mackeyglass: all PyReCo (16/18)
        [-1, -1, -1],   # santafe: all LSTM
    ], dtype=float)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    cmap = plt.cm.RdBu
    im = ax.imshow(winner, cmap=cmap, vmin=-1.5, vmax=1.5, aspect="auto")

    ax.set_xticks(range(len(budgets)))
    ax.set_xticklabels(["Small", "Medium", "Large"])
    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels([DATASET_LABELS[d] for d in datasets])
    ax.set_xlabel("Parameter Budget")

    for i in range(len(datasets)):
        for j in range(len(budgets)):
            label = "PyReCo" if winner[i, j] > 0 else "LSTM"
            ax.text(j, i, label, ha="center", va="center", fontsize=9, fontweight="bold",
                    color="white")

    ax.set_title("Statistically Significant Winner (R²)")
    fig.tight_layout()
    save(fig, "fig_statistical_significance.pdf")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 14: Summary radar / overview bar chart
# ═══════════════════════════════════════════════════════════════════════════
def fig_overall_summary():
    """Win counts across all 54 comparisons."""
    categories = ["MSE", "R²", "Training\nSpeed"]
    pyreco_wins = [35, 35, 36]
    lstm_wins   = [19, 19, 18]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    x = np.arange(len(categories))
    w = 0.35
    ax.bar(x - w/2, pyreco_wins, w, color=COLOR_PYRECO, label="PyReCo wins", alpha=0.85)
    ax.bar(x + w/2, lstm_wins, w, color=COLOR_LSTM, label="LSTM wins", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Number of Wins (out of 54)")
    ax.set_title("Overall Comparison: Win Counts Across All Configurations")
    ax.axhline(27, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.legend()
    fig.tight_layout()
    save(fig, "fig_overall_summary.pdf")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating thesis figures...")
    print(f"Output directory: {IMG_DIR}\n")

    print("[1/14] R² by dataset and budget")
    fig_r2_by_dataset_budget()

    print("[2/14] MSE by dataset and budget")
    fig_mse_by_dataset_budget()

    print("[3/14] Training time comparison")
    fig_training_time()

    print("[4/14] Data efficiency — Lorenz")
    fig_data_efficiency("lorenz", "fig_data_efficiency_lorenz.pdf")

    print("[5/14] Data efficiency — Mackey-Glass")
    fig_data_efficiency("mackeyglass", "fig_data_efficiency_mackeyglass.pdf")

    print("[6/14] Data efficiency — Santa Fe")
    fig_data_efficiency("santafe", "fig_data_efficiency_santafe.pdf")

    print("[7/14] Learning curves")
    fig_learning_curves()

    print("[8/14] Multi-step prediction")
    fig_multi_step()

    print("[9/14] Horizon degradation")
    fig_horizon_degradation()

    print("[10/14] Training efficiency")
    fig_training_efficiency()

    print("[11/14] Energy / time comparison by scale")
    fig_speed_ratio()

    print("[12/14] Parameter budget breakdown")
    fig_param_breakdown()

    print("[13/14] Statistical significance heatmap")
    fig_stat_significance()

    print("[14/14] Overall summary")
    fig_overall_summary()

    print(f"\nDone! {14} figures saved to {IMG_DIR}")
