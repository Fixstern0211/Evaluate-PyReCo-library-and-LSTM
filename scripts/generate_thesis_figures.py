#!/usr/bin/env python3
"""
Generate publication-quality PDF figures for Master thesis (V2 data).
Comparative Analysis of Reservoir Computing and LSTM Networks
for Chaotic Time-Series Prediction.

Data sources:
  - results/tables/v2/  (from statistical_analysis_v2.py and statistical_analysis_multistep_v2.py)
  - results/final_v2/   (raw JSON, for detailed per-seed data)

Output: /Users/hengz/Documents/Abschlussarbeite/LaTeX_Template/img/
"""

import pandas as pd
import numpy as np
import json
import glob
import os
import re
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Paths ──────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLES_V2 = os.path.join(BASE, "results", "tables", "v2")
FINAL_V2 = os.path.join(BASE, "results", "final_v2")
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

COLOR_PYRECO = "#2196F3"
COLOR_LSTM   = "#FF5722"
COLORS = {"pyreco_standard": COLOR_PYRECO, "lstm": COLOR_LSTM,
           "PyReCo": COLOR_PYRECO, "LSTM": COLOR_LSTM}
DATASET_LABELS = {"lorenz": "Lorenz", "mackeyglass": "Mackey-Glass", "santafe": "Santa Fe"}
BUDGET_ORDER = ["small", "medium", "large"]
BUDGET_LABELS = {"small": "Small\n(~1K)", "medium": "Medium\n(~10K)", "large": "Large\n(~50K)"}


# ── Data Loading ───────────────────────────────────────────────────────────

def load_v2_raw():
    """Load all V2 raw results into a DataFrame."""
    records = []
    for fp in sorted(glob.glob(os.path.join(FINAL_V2, "results_*.json"))):
        with open(fp) as f:
            d = json.load(f)
        meta = d["metadata"]
        budget = meta["budget_name"]
        for entry in d["results"].get(budget, []):
            pi = entry.get("param_info", {})
            records.append({
                "dataset": meta["dataset"], "budget": budget,
                "seed": meta["seed"], "train_frac": meta["train_frac"],
                "model_type": entry["model_type"],
                "test_r2": entry.get("test_r2"),
                "test_mse": entry.get("test_mse"),
                "final_train_time": entry.get("final_train_time", 0),
                "trainable_params": pi.get("trainable"),
                "total_params": pi.get("total"),
            })
    return pd.DataFrame(records)


def load_summary_by_db():
    """Load summary by (dataset, budget) from CSV."""
    path = os.path.join(TABLES_V2, "summary_by_dataset_budget.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    # Fallback: compute from raw
    df = load_v2_raw()
    rows = []
    for (ds, budget, mt), g in df.groupby(["dataset", "budget", "model_type"]):
        rows.append({
            "dataset": ds, "budget": budget, "model_type": mt,
            "r2_mean": g["test_r2"].mean(), "r2_std": g["test_r2"].std(),
            "mse_mean": g["test_mse"].mean(), "mse_std": g["test_mse"].std(),
            "time_mean": g["final_train_time"].mean(), "time_std": g["final_train_time"].std(),
            "params_mean": g["trainable_params"].mean(),
        })
    return pd.DataFrame(rows)


def load_multistep_for_figures():
    """Load multi-step pivot table."""
    path = os.path.join(TABLES_V2, "multistep_for_figures.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def save(fig, name):
    path = os.path.join(IMG_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 1: R² comparison across datasets and budgets
# ═══════════════════════════════════════════════════════════════════════════
def fig_r2_by_dataset_budget():
    df = load_summary_by_db()
    datasets = ["lorenz", "mackeyglass", "santafe"]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), sharey=True)

    for ax, ds in zip(axes, datasets):
        sub = df[df["dataset"] == ds]
        x = np.arange(len(BUDGET_ORDER))
        w = 0.35
        for i, (mt, label) in enumerate([("pyreco_standard", "PyReCo"), ("lstm", "LSTM")]):
            row = sub[sub["model_type"] == mt].set_index("budget").reindex(BUDGET_ORDER)
            ax.bar(x + (i - 0.5) * w, row["r2_mean"], w,
                   yerr=row["r2_std"], capsize=3,
                   color=COLORS[mt], label=label, alpha=0.85)
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
    df = load_summary_by_db()
    datasets = ["lorenz", "mackeyglass", "santafe"]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))

    for ax, ds in zip(axes, datasets):
        sub = df[df["dataset"] == ds]
        x = np.arange(len(BUDGET_ORDER))
        w = 0.35
        for i, (mt, label) in enumerate([("pyreco_standard", "PyReCo"), ("lstm", "LSTM")]):
            row = sub[sub["model_type"] == mt].set_index("budget").reindex(BUDGET_ORDER)
            ax.bar(x + (i - 0.5) * w, row["mse_mean"], w,
                   yerr=row["mse_std"], capsize=3,
                   color=COLORS[mt], label=label, alpha=0.85)
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
    df = load_summary_by_db()
    datasets = ["lorenz", "mackeyglass", "santafe"]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))

    for ax, ds in zip(axes, datasets):
        sub = df[df["dataset"] == ds]
        x = np.arange(len(BUDGET_ORDER))
        w = 0.35
        for i, (mt, label) in enumerate([("pyreco_standard", "PyReCo"), ("lstm", "LSTM")]):
            row = sub[sub["model_type"] == mt].set_index("budget").reindex(BUDGET_ORDER)
            ax.bar(x + (i - 0.5) * w, row["time_mean"], w,
                   color=COLORS[mt], label=label, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([BUDGET_LABELS[b] for b in BUDGET_ORDER])
        ax.set_title(DATASET_LABELS[ds])
        ax.set_ylabel("Training Time (s)")

    axes[0].legend()
    fig.suptitle("Training Time Comparison", fontsize=12, y=1.02)
    fig.tight_layout()
    save(fig, "fig_training_time_comparison.pdf")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 4: Data efficiency — R² vs train_frac (from main experiments)
# ═══════════════════════════════════════════════════════════════════════════
def fig_data_efficiency():
    df = load_v2_raw()
    datasets = ["lorenz", "mackeyglass", "santafe"]
    fig, axes = plt.subplots(3, 3, figsize=(10, 9), sharey="row")

    for row_idx, ds in enumerate(datasets):
        for col_idx, budget in enumerate(BUDGET_ORDER):
            ax = axes[row_idx, col_idx]
            sub = df[(df["dataset"] == ds) & (df["budget"] == budget)]

            for mt, label, color in [("pyreco_standard", "PyReCo", COLOR_PYRECO),
                                      ("lstm", "LSTM", COLOR_LSTM)]:
                ms = sub[sub["model_type"] == mt]
                grouped = ms.groupby("train_frac")["test_r2"].agg(["mean", "std"]).reset_index()
                grouped = grouped.sort_values("train_frac")
                ax.errorbar(grouped["train_frac"], grouped["mean"],
                            yerr=grouped["std"], marker="o", markersize=4,
                            capsize=3, color=color, label=label, linewidth=1.5)

            ax.set_xlabel("Train Fraction")
            if col_idx == 0:
                ax.set_ylabel(f"{DATASET_LABELS[ds]}\nR² Score")
            if row_idx == 0:
                ax.set_title(f"{BUDGET_LABELS[budget].replace(chr(10), ' ')}")
            if row_idx == 0 and col_idx == 0:
                ax.legend()

    fig.suptitle("Data Efficiency: R² vs Training Fraction", fontsize=12, y=1.01)
    fig.tight_layout()
    save(fig, "fig_data_efficiency.pdf")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 5: Multi-step prediction R² vs horizon
# ═══════════════════════════════════════════════════════════════════════════
def fig_multi_step():
    df = load_multistep_for_figures()
    if df is None:
        print("  SKIP: no multi-step data")
        return
    datasets = ["lorenz", "mackeyglass", "santafe"]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), sharey=True)

    for ax, ds in zip(axes, datasets):
        sub = df[(df["dataset"] == ds) & (df["budget"] == "medium")].sort_values("horizon")
        if len(sub) == 0:
            continue
        ax.plot(sub["horizon"], sub["pyreco_r2_mean"], "o-",
                color=COLOR_PYRECO, label="PyReCo", markersize=4, linewidth=1.5)
        ax.fill_between(sub["horizon"],
                        sub["pyreco_r2_mean"] - sub["pyreco_r2_std"],
                        sub["pyreco_r2_mean"] + sub["pyreco_r2_std"],
                        alpha=0.15, color=COLOR_PYRECO)
        ax.plot(sub["horizon"], sub["lstm_r2_mean"], "s-",
                color=COLOR_LSTM, label="LSTM", markersize=4, linewidth=1.5)
        ax.fill_between(sub["horizon"],
                        sub["lstm_r2_mean"] - sub["lstm_r2_std"],
                        sub["lstm_r2_mean"] + sub["lstm_r2_std"],
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
# Figure 6: Horizon degradation — all budgets
# ═══════════════════════════════════════════════════════════════════════════
def fig_horizon_degradation():
    df = load_multistep_for_figures()
    if df is None:
        print("  SKIP: no multi-step data")
        return
    datasets = ["lorenz", "mackeyglass", "santafe"]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), sharey=True)

    for ax, ds in zip(axes, datasets):
        for budget, ls in zip(BUDGET_ORDER, ["-", "--", ":"]):
            sub = df[(df["dataset"] == ds) & (df["budget"] == budget)].sort_values("horizon")
            if len(sub) == 0:
                continue
            ax.plot(sub["horizon"], sub["pyreco_r2_mean"],
                    f"o{ls}", color=COLOR_PYRECO, markersize=3, linewidth=1.2,
                    label=f"PyReCo {budget}" if ds == datasets[0] else None)
            ax.plot(sub["horizon"], sub["lstm_r2_mean"],
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
# Figure 7: Parameter budget breakdown
# ═══════════════════════════════════════════════════════════════════════════
def fig_param_breakdown():
    df = load_v2_raw()
    # Parameter counts vary by dataset (d_in: lorenz=3, others=1).
    # Show per-dataset bars grouped by budget.
    datasets = sorted(df["dataset"].unique())
    avail = [b for b in BUDGET_ORDER if b in df["budget"].values]

    # Aggregate: median per (dataset, budget, model_type) across seeds/train_fracs
    param_info = df.groupby(["dataset", "budget", "model_type"]).agg(
        trainable=("trainable_params", "median"),
        total=("total_params", "median"),
    ).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    n_ds = len(datasets)
    n_budgets = len(avail)
    bar_width = 0.25
    x = np.arange(n_budgets)
    ds_colors = {datasets[i]: c for i, c in enumerate(["#2196F3", "#FF9800", "#4CAF50"])}

    # PyReCo
    ax = axes[0]
    for i, ds in enumerate(datasets):
        sub = param_info[(param_info["model_type"] == "pyreco_standard") & (param_info["dataset"] == ds)]
        sub = sub.set_index("budget").reindex(avail)
        fixed = (sub["total"] - sub["trainable"]).values
        trainable = sub["trainable"].values
        offset = (i - (n_ds - 1) / 2) * bar_width
        ax.bar(x + offset, fixed, bar_width * 0.9, color=ds_colors[ds], alpha=0.4)
        ax.bar(x + offset, trainable, bar_width * 0.9, bottom=fixed,
               color=ds_colors[ds], alpha=0.85, label=ds.capitalize())
    ax.set_xticks(x)
    ax.set_xticklabels([BUDGET_LABELS[b].replace("\n", " ") for b in avail], fontsize=8)
    ax.set_ylabel("Parameters"); ax.set_title("PyReCo (ESN)")
    ax.legend(fontsize=7); ax.set_yscale("log")

    # LSTM
    ax = axes[1]
    for i, ds in enumerate(datasets):
        sub = param_info[(param_info["model_type"] == "lstm") & (param_info["dataset"] == ds)]
        sub = sub.set_index("budget").reindex(avail)
        offset = (i - (n_ds - 1) / 2) * bar_width
        ax.bar(x + offset, sub["trainable"].values, bar_width * 0.9,
               color=ds_colors[ds], alpha=0.85, label=ds.capitalize())
    ax.set_xticks(x)
    ax.set_xticklabels([BUDGET_LABELS[b].replace("\n", " ") for b in avail], fontsize=8)
    ax.set_title("LSTM"); ax.legend(fontsize=7); ax.set_yscale("log")

    fig.suptitle("Parameter Budget Breakdown", fontsize=12, y=1.02)
    fig.tight_layout()
    save(fig, "fig_parameter_budget_breakdown.pdf")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 8: Winner heatmap
# ═══════════════════════════════════════════════════════════════════════════
def fig_winner_heatmap():
    df = load_v2_raw()
    datasets = ["lorenz", "mackeyglass", "santafe"]
    budgets = BUDGET_ORDER

    winner = np.zeros((len(datasets), len(budgets)))
    for i, ds in enumerate(datasets):
        for j, budget in enumerate(budgets):
            sub = df[(df["dataset"] == ds) & (df["budget"] == budget)]
            p_r2 = sub[sub["model_type"] == "pyreco_standard"]["test_r2"].mean()
            l_r2 = sub[sub["model_type"] == "lstm"]["test_r2"].mean()
            winner[i, j] = 1.0 if p_r2 > l_r2 else -1.0

    fig, ax = plt.subplots(figsize=(5, 3.5))
    im = ax.imshow(winner, cmap=plt.cm.RdBu, vmin=-1.5, vmax=1.5, aspect="auto")
    ax.set_xticks(range(len(budgets)))
    ax.set_xticklabels(["Small", "Medium", "Large"])
    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels([DATASET_LABELS[d] for d in datasets])
    ax.set_xlabel("Parameter Budget")

    for i in range(len(datasets)):
        for j in range(len(budgets)):
            label = "PyReCo" if winner[i, j] > 0 else "LSTM"
            ax.text(j, i, label, ha="center", va="center", fontsize=9,
                    fontweight="bold", color="white")

    ax.set_title("Winner by Mean R² Score")
    fig.tight_layout()
    save(fig, "fig_winner_heatmap.pdf")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 9: Overall win counts
# ═══════════════════════════════════════════════════════════════════════════
def fig_overall_summary():
    df = load_v2_raw()
    key_cols = ["dataset", "seed", "train_frac", "budget"]
    pyreco = df[df["model_type"] == "pyreco_standard"].set_index(key_cols)
    lstm = df[df["model_type"] == "lstm"].set_index(key_cols)
    common = pyreco.index.intersection(lstm.index)

    categories = ["MSE", "R²", "Training\nSpeed"]
    pyreco_wins, lstm_wins = [], []

    pw = (pyreco.loc[common, "test_mse"] < lstm.loc[common, "test_mse"]).sum()
    pyreco_wins.append(pw); lstm_wins.append(len(common) - pw)

    pw = (pyreco.loc[common, "test_r2"] > lstm.loc[common, "test_r2"]).sum()
    pyreco_wins.append(pw); lstm_wins.append(len(common) - pw)

    pw = (pyreco.loc[common, "final_train_time"] < lstm.loc[common, "final_train_time"]).sum()
    pyreco_wins.append(pw); lstm_wins.append(len(common) - pw)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    x = np.arange(len(categories))
    w = 0.35
    ax.bar(x - w/2, pyreco_wins, w, color=COLOR_PYRECO, label="PyReCo wins", alpha=0.85)
    ax.bar(x + w/2, lstm_wins, w, color=COLOR_LSTM, label="LSTM wins", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(categories)
    ax.set_ylabel(f"Wins (out of {len(common)})")
    ax.set_title("Overall Comparison: Win Counts")
    ax.axhline(len(common) / 2, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.legend()
    fig.tight_layout()
    save(fig, "fig_overall_summary.pdf")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating thesis figures (V2 data)...")
    print(f"Output directory: {IMG_DIR}\n")

    print("[1/9] R² by dataset and budget")
    fig_r2_by_dataset_budget()

    print("[2/9] MSE by dataset and budget")
    fig_mse_by_dataset_budget()

    print("[3/9] Training time comparison")
    fig_training_time()

    print("[4/9] Data efficiency (R² vs train_frac)")
    fig_data_efficiency()

    print("[5/9] Multi-step prediction")
    fig_multi_step()

    print("[6/9] Horizon degradation")
    fig_horizon_degradation()

    print("[7/9] Parameter budget breakdown")
    fig_param_breakdown()

    print("[8/9] Winner heatmap")
    fig_winner_heatmap()

    print("[9/9] Overall summary")
    fig_overall_summary()

    print(f"\nDone! 9 figures saved to {IMG_DIR}")
