#!/usr/bin/env python3
"""
Statistical Analysis for V2 Multi-Step Prediction Experiments.

Loads data from results/multi_step_v2/, performs paired comparisons
at each prediction horizon.

Usage:
    python analysis/statistical_analysis_multistep_v2.py
"""

import json
import glob
import os
import re
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy import stats

try:
    from statsmodels.stats.multitest import multipletests
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


def load_multistep_results(results_dir="results/multi_step_v2"):
    """Load all multi-step results into a flat DataFrame."""
    records = []
    for fp in sorted(glob.glob(os.path.join(results_dir, "multistep_*.json"))):
        with open(fp) as f:
            d = json.load(f)
        meta = d["metadata"]
        for model_type, mdata in d.get("models", {}).items():
            if "error" in mdata:
                continue
            for h_str, metrics in mdata.get("horizon_results", {}).items():
                records.append({
                    "dataset": meta["dataset"],
                    "budget": meta["budget"],
                    "seed": meta["seed"],
                    "train_frac": meta["train_frac"],
                    "model_type": model_type,
                    "horizon": int(h_str),
                    "mse": metrics.get("mse"),
                    "r2": metrics.get("r2"),
                    "mae": metrics.get("mae"),
                    "nrmse": metrics.get("nrmse"),
                    "train_time": mdata.get("train_time", 0),
                    "eval_time": mdata.get("eval_time", 0),
                })
    return pd.DataFrame(records)


def paired_ttest_with_ci(g1, g2, confidence=0.95):
    g1, g2 = np.array(g1, dtype=float), np.array(g2, dtype=float)
    diff = g1 - g2
    n = len(diff)
    if n < 2:
        return None
    t_stat, p_value = stats.ttest_rel(g1, g2)
    mean_diff = np.mean(diff)
    se = stats.sem(diff)
    t_crit = stats.t.ppf((1 + confidence) / 2, n - 1)
    return {
        "mean_diff": mean_diff, "ci_lower": mean_diff - t_crit * se,
        "ci_upper": mean_diff + t_crit * se, "t_stat": t_stat,
        "p_value": p_value, "n": n,
    }


def cohens_d(g1, g2):
    diff = np.array(g1, dtype=float) - np.array(g2, dtype=float)
    n = len(diff)
    if n < 2:
        return 0, "undefined"
    sd = np.std(diff, ddof=1)
    if sd == 0:
        return 0, "undefined"
    d = np.mean(diff) / sd
    abs_d = abs(d)
    interp = "negligible" if abs_d < 0.2 else "small" if abs_d < 0.5 else "medium" if abs_d < 0.8 else "large"
    return d, interp


def analyze(df, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    horizons = sorted(df["horizon"].unique())

    # === Summary table: mean±std per (dataset, budget, horizon, model) ===
    summary_rows = []
    for (ds, budget, horizon, mt), g in df.groupby(
        ["dataset", "budget", "horizon", "model_type"]
    ):
        summary_rows.append({
            "dataset": ds, "budget": budget, "horizon": horizon,
            "model_type": mt, "n": len(g),
            "r2_mean": g["r2"].mean(), "r2_std": g["r2"].std(),
            "mse_mean": g["mse"].mean(), "mse_std": g["mse"].std(),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "multistep_summary.csv", index=False)

    # === Paired comparison per (dataset, budget, train_frac, horizon) ===
    comparisons = []
    for (ds, budget, tf, horizon), gdf in df.groupby(
        ["dataset", "budget", "train_frac", "horizon"]
    ):
        pyreco = gdf[gdf["model_type"] == "pyreco_standard"].sort_values("seed")
        lstm = gdf[gdf["model_type"] == "lstm"].sort_values("seed")
        common = sorted(set(pyreco["seed"]) & set(lstm["seed"]))
        if len(common) < 2:
            continue
        p_r2 = pyreco[pyreco["seed"].isin(common)]["r2"].values
        l_r2 = lstm[lstm["seed"].isin(common)]["r2"].values
        p_mse = pyreco[pyreco["seed"].isin(common)]["mse"].values
        l_mse = lstm[lstm["seed"].isin(common)]["mse"].values

        ttest_r2 = paired_ttest_with_ci(p_r2, l_r2)
        d_r2, interp_r2 = cohens_d(p_r2, l_r2)
        ttest_mse = paired_ttest_with_ci(p_mse, l_mse)

        comparisons.append({
            "dataset": ds, "budget": budget, "train_frac": tf,
            "horizon": horizon, "n_pairs": len(common),
            "pyreco_r2_mean": np.mean(p_r2), "lstm_r2_mean": np.mean(l_r2),
            "r2_diff": np.mean(p_r2) - np.mean(l_r2),
            "r2_p": ttest_r2["p_value"] if ttest_r2 else None,
            "r2_d": d_r2, "r2_effect": interp_r2,
            "pyreco_mse_mean": np.mean(p_mse), "lstm_mse_mean": np.mean(l_mse),
            "mse_p": ttest_mse["p_value"] if ttest_mse else None,
            "r2_winner": "PyReCo" if np.mean(p_r2) > np.mean(l_r2) else "LSTM",
        })

    comp_df = pd.DataFrame(comparisons)
    comp_df.to_csv(output_dir / "multistep_comparisons.csv", index=False)

    # === Pivot table for figures: per (dataset, budget, horizon) ===
    fig_rows = []
    for (ds, budget, horizon), gdf in df.groupby(["dataset", "budget", "horizon"]):
        pyreco = gdf[gdf["model_type"] == "pyreco_standard"]
        lstm = gdf[gdf["model_type"] == "lstm"]
        fig_rows.append({
            "dataset": ds, "budget": budget, "horizon": horizon,
            "pyreco_r2_mean": pyreco["r2"].mean(), "pyreco_r2_std": pyreco["r2"].std(),
            "lstm_r2_mean": lstm["r2"].mean(), "lstm_r2_std": lstm["r2"].std(),
            "pyreco_mse_mean": pyreco["mse"].mean(), "pyreco_mse_std": pyreco["mse"].std(),
            "lstm_mse_mean": lstm["mse"].mean(), "lstm_mse_std": lstm["mse"].std(),
        })
    fig_df = pd.DataFrame(fig_rows)
    fig_df.to_csv(output_dir / "multistep_for_figures.csv", index=False)

    # === Report ===
    lines = []
    lines.append("=" * 80)
    lines.append("V2 MULTI-STEP STATISTICAL ANALYSIS")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 80)

    lines.append(f"\nRecords: {len(df)}")
    lines.append(f"Datasets: {sorted(df['dataset'].unique())}")
    lines.append(f"Budgets: {sorted(df['budget'].unique())}")
    lines.append(f"Horizons: {horizons}")
    lines.append(f"Comparisons: {len(comp_df)}")

    # Win counts per horizon
    lines.append(f"\n{'='*80}")
    lines.append("WIN COUNTS BY HORIZON (R²)")
    lines.append("=" * 80)
    for h in horizons:
        hdf = comp_df[comp_df["horizon"] == h]
        if len(hdf) == 0:
            continue
        p_wins = (hdf["r2_winner"] == "PyReCo").sum()
        l_wins = (hdf["r2_winner"] == "LSTM").sum()
        sig = hdf[hdf["r2_p"].notna() & (hdf["r2_p"] < 0.05)]
        lines.append(f"  h={h:>3}: PyReCo {p_wins}/{len(hdf)}, LSTM {l_wins}/{len(hdf)}, significant: {len(sig)}/{len(hdf)}")

    # Per dataset summary
    for ds in sorted(df["dataset"].unique()):
        lines.append(f"\n{'='*80}")
        lines.append(f"DATASET: {ds.upper()}")
        lines.append("=" * 80)
        for budget in ["small", "medium", "large"]:
            bdf = comp_df[(comp_df["dataset"] == ds) & (comp_df["budget"] == budget)]
            if len(bdf) == 0:
                continue
            lines.append(f"\n  {budget.upper()} Budget:")
            for h in horizons:
                hdf = bdf[bdf["horizon"] == h]
                if len(hdf) == 0:
                    continue
                p_wins = (hdf["r2_winner"] == "PyReCo").sum()
                l_wins = (hdf["r2_winner"] == "LSTM").sum()
                avg_diff = hdf["r2_diff"].mean()
                lines.append(f"    h={h:>3}: PyReCo {p_wins}/{len(hdf)}, LSTM {l_wins}/{len(hdf)}, mean ΔR²={avg_diff:+.4f}")

    report = "\n".join(lines)
    print(report)

    with open(output_dir / "multistep_report_v2.txt", "w") as f:
        f.write(report)

    # JSON summary
    json_summary = {
        "generated": datetime.now().isoformat(),
        "n_records": len(df),
        "n_comparisons": len(comp_df),
        "win_counts_by_horizon": {},
    }
    for h in horizons:
        hdf = comp_df[comp_df["horizon"] == h]
        json_summary["win_counts_by_horizon"][int(h)] = {
            "pyreco": int((hdf["r2_winner"] == "PyReCo").sum()),
            "lstm": int((hdf["r2_winner"] == "LSTM").sum()),
            "total": len(hdf),
        }
    with open(output_dir / "multistep_summary.json", "w") as f:
        json.dump(json_summary, f, indent=2)

    return comp_df, summary_df


def main():
    parser = argparse.ArgumentParser(description="V2 Multi-Step Statistical Analysis")
    parser.add_argument("--results-dir", default="results/multi_step_v2")
    parser.add_argument("--output-dir", default="results/tables/v2")
    args = parser.parse_args()

    print("Loading multi-step data...")
    df = load_multistep_results(args.results_dir)
    if len(df) == 0:
        print("No data found!")
        return
    print(f"  {len(df)} records loaded")

    analyze(df, args.output_dir)
    print(f"\nOutputs saved to {args.output_dir}")


if __name__ == "__main__":
    main()
