#!/usr/bin/env python3
"""
Statistical Analysis for V2 Experiments (PyReCo budget-matched vs LSTM reused).

Loads data directly from results/final_v2/ (each JSON contains both pyreco + lstm).

Statistical methods:
1. Paired t-test with 95% CI
2. Wilcoxon signed-rank test
3. Shapiro-Wilk normality test on differences
4. Cohen's d_z effect size with CI
5. Holm-Bonferroni multiple comparison correction

Usage:
    python analysis/statistical_analysis_v2.py
    python analysis/statistical_analysis_v2.py --results-dir results/final_v2
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
    print("Warning: statsmodels not installed. Multiple comparison correction unavailable.")


# ============================================================================
# Statistical Test Functions (reused from statistical_analysis.py)
# ============================================================================

def paired_ttest_with_ci(group1, group2, confidence=0.95):
    """Paired t-test with confidence interval."""
    group1 = np.array(group1, dtype=float)
    group2 = np.array(group2, dtype=float)
    diff = group1 - group2
    n = len(diff)

    if n < 2:
        return None

    t_stat, p_value = stats.ttest_rel(group1, group2)
    mean_diff = np.mean(diff)
    se_diff = stats.sem(diff)
    df = n - 1
    t_crit = stats.t.ppf((1 + confidence) / 2, df)
    ci_lower = mean_diff - t_crit * se_diff
    ci_upper = mean_diff + t_crit * se_diff

    return {
        "mean1": np.mean(group1),
        "std1": np.std(group1, ddof=1),
        "mean2": np.mean(group2),
        "std2": np.std(group2, ddof=1),
        "mean_diff": mean_diff,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "t_stat": t_stat,
        "p_value": p_value,
        "df": df,
        "n": n,
    }


def wilcoxon_test(group1, group2):
    """Wilcoxon signed-rank test (non-parametric)."""
    n = len(group1)
    try:
        w_stat, p_value = stats.wilcoxon(group1, group2, zero_method="wilcox")
        return {
            "statistic": w_stat,
            "p_value": p_value,
            "success": True,
            "underpowered": n <= 5,
        }
    except Exception as e:
        return {"statistic": None, "p_value": None, "success": False, "error": str(e)}


def shapiro_wilk_test(data):
    """Normality test using Shapiro-Wilk."""
    n = len(data)
    if n < 3:
        return {"statistic": None, "p_value": None, "is_normal": None, "low_power": True}
    stat, p_value = stats.shapiro(data)
    return {
        "statistic": stat,
        "p_value": p_value,
        "is_normal": p_value > 0.05,
        "low_power": n <= 7,
    }


def cohens_d_with_ci(group1, group2, confidence=0.95):
    """Cohen's d_z effect size for paired samples with CI."""
    diff = np.array(group1, dtype=float) - np.array(group2, dtype=float)
    n = len(diff)

    if n < 2:
        return 0, 0, 0, "undefined"

    sd_diff = np.std(diff, ddof=1)
    if sd_diff == 0:
        return 0, 0, 0, "undefined"

    d = np.mean(diff) / sd_diff
    se_d = np.sqrt(1.0 / n + d**2 / (2.0 * n))
    z_crit = stats.norm.ppf((1 + confidence) / 2)
    ci_lower = d - z_crit * se_d
    ci_upper = d + z_crit * se_d

    abs_d = abs(d)
    if abs_d < 0.2:
        interpretation = "negligible"
    elif abs_d < 0.5:
        interpretation = "small"
    elif abs_d < 0.8:
        interpretation = "medium"
    else:
        interpretation = "large"

    return d, ci_lower, ci_upper, interpretation


# ============================================================================
# Data Loading
# ============================================================================

def load_v2_results(results_dir: str = "results/final_v2") -> pd.DataFrame:
    """Load all results from V2 directory (each file has pyreco + lstm)."""
    records = []
    pattern = os.path.join(results_dir, "results_*.json")

    for fpath in sorted(glob.glob(pattern)):
        fname = os.path.basename(fpath)
        m = re.match(
            r"results_(\w+?)_(small|medium|large)_seed(\d+)_train([\d.]+)\.json",
            fname,
        )
        if not m:
            continue

        with open(fpath) as f:
            data = json.load(f)

        metadata = data["metadata"]
        dataset = metadata["dataset"]
        seed = metadata["seed"]
        train_frac = metadata["train_frac"]

        for budget_name, entries in data["results"].items():
            for entry in entries:
                param_info = entry.get("param_info", {})
                records.append({
                    "dataset": dataset,
                    "budget": budget_name,
                    "seed": seed,
                    "train_frac": train_frac,
                    "model_type": entry["model_type"],
                    "test_mse": entry.get("test_mse"),
                    "test_r2": entry.get("test_r2"),
                    "val_mse": entry.get("val_mse"),
                    "final_train_time": entry.get("final_train_time", 0),
                    "trainable_params": param_info.get("trainable"),
                    "total_params": param_info.get("total"),
                    "inference_time_per_sample_ms": entry.get(
                        "inference_time_per_sample_ms"
                    ),
                })

    return pd.DataFrame(records)


# ============================================================================
# Analysis
# ============================================================================

def perform_pairwise_comparison(df: pd.DataFrame) -> list[dict]:
    """Paired comparison: PyReCo vs LSTM per (dataset, budget, train_frac)."""
    comparisons = []

    for (dataset, budget, train_frac), group_df in df.groupby(
        ["dataset", "budget", "train_frac"]
    ):
        pyreco = group_df[group_df["model_type"] == "pyreco_standard"].sort_values("seed")
        lstm = group_df[group_df["model_type"] == "lstm"].sort_values("seed")

        common_seeds = sorted(set(pyreco["seed"]) & set(lstm["seed"]))
        if len(common_seeds) < 2:
            continue

        pyreco = pyreco[pyreco["seed"].isin(common_seeds)]
        lstm = lstm[lstm["seed"].isin(common_seeds)]

        comparison = {
            "dataset": dataset,
            "budget": budget,
            "train_frac": train_frac,
            "n_pairs": len(common_seeds),
            "seeds": common_seeds,
        }

        for metric, lower_better in [
            ("test_mse", True),
            ("test_r2", False),
            ("final_train_time", True),
        ]:
            p_vals = pyreco[metric].values
            l_vals = lstm[metric].values
            diff = p_vals - l_vals

            comparison[metric] = {
                "pyreco_mean": float(np.mean(p_vals)),
                "pyreco_std": float(np.std(p_vals, ddof=1)),
                "lstm_mean": float(np.mean(l_vals)),
                "lstm_std": float(np.std(l_vals, ddof=1)),
                "ttest": paired_ttest_with_ci(p_vals, l_vals),
                "wilcoxon": wilcoxon_test(p_vals, l_vals),
                "effect_size": cohens_d_with_ci(p_vals, l_vals),
                "shapiro": shapiro_wilk_test(diff),
                "lower_better": lower_better,
            }

        comparisons.append(comparison)

    # Holm-Bonferroni correction
    if HAS_STATSMODELS and comparisons:
        for metric in ["test_mse", "test_r2", "final_train_time"]:
            p_values = []
            indices = []
            for i, comp in enumerate(comparisons):
                if comp[metric]["ttest"]:
                    p_values.append(comp[metric]["ttest"]["p_value"])
                    indices.append(i)
            if p_values:
                _, p_corrected, _, _ = multipletests(p_values, alpha=0.05, method="holm")
                for j, idx in enumerate(indices):
                    comparisons[idx][metric]["ttest"]["p_corrected"] = float(p_corrected[j])

    return comparisons


def generate_summary_tables(df: pd.DataFrame, output_dir: Path):
    """Generate summary CSV tables."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Per (dataset, budget): mean±std across seeds and train_fracs
    summary_rows = []
    for (dataset, budget, model_type), g in df.groupby(
        ["dataset", "budget", "model_type"]
    ):
        summary_rows.append({
            "dataset": dataset,
            "budget": budget,
            "model_type": model_type,
            "n": len(g),
            "r2_mean": g["test_r2"].mean(),
            "r2_std": g["test_r2"].std(),
            "mse_mean": g["test_mse"].mean(),
            "mse_std": g["test_mse"].std(),
            "time_mean": g["final_train_time"].mean(),
            "time_std": g["final_train_time"].std(),
            "params_mean": g["trainable_params"].mean(),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "summary_by_dataset_budget.csv", index=False)

    # Per (dataset, budget, train_frac): mean±std across 5 seeds
    detail_rows = []
    for (dataset, budget, train_frac, model_type), g in df.groupby(
        ["dataset", "budget", "train_frac", "model_type"]
    ):
        detail_rows.append({
            "dataset": dataset,
            "budget": budget,
            "train_frac": train_frac,
            "model_type": model_type,
            "n_seeds": len(g),
            "r2_mean": g["test_r2"].mean(),
            "r2_std": g["test_r2"].std(),
            "mse_mean": g["test_mse"].mean(),
            "mse_std": g["test_mse"].std(),
            "time_mean": g["final_train_time"].mean(),
            "time_std": g["final_train_time"].std(),
        })

    detail_df = pd.DataFrame(detail_rows)
    detail_df.to_csv(output_dir / "summary_by_dataset_budget_tf.csv", index=False)

    return summary_df, detail_df


def generate_report(df: pd.DataFrame, comparisons: list[dict]) -> str:
    """Generate terminal report."""
    lines = []
    lines.append("=" * 80)
    lines.append("V2 STATISTICAL ANALYSIS REPORT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 80)

    # Data summary
    n_pyreco = len(df[df["model_type"] == "pyreco_standard"])
    n_lstm = len(df[df["model_type"] == "lstm"])
    lines.append(f"\n## Data Summary")
    lines.append(f"  PyReCo records (V2): {n_pyreco}")
    lines.append(f"  LSTM records (V1):   {n_lstm}")
    lines.append(f"  Total:               {len(df)}")
    lines.append(f"  Datasets:  {sorted(df['dataset'].unique())}")
    lines.append(f"  Budgets:   {sorted(df['budget'].unique())}")
    lines.append(f"  Train fracs: {sorted(df['train_frac'].unique())}")
    lines.append(f"  Seeds:     {sorted(df['seed'].unique())}")
    lines.append(f"  Comparisons: {len(comparisons)}")

    lines.append(f"\n## Methods")
    lines.append("  - Paired t-test with 95% CI")
    lines.append("  - Cohen's d_z (paired) with 95% CI")
    lines.append("  - Shapiro-Wilk normality on differences")
    lines.append("  - Wilcoxon signed-rank (non-parametric)")
    if HAS_STATSMODELS:
        lines.append("  - Holm-Bonferroni correction")
    lines.append("  - n=5 seeds per condition; Wilcoxon min p=0.0625")

    # Per dataset, per budget
    for dataset in sorted(df["dataset"].unique()):
        lines.append(f"\n{'=' * 80}")
        lines.append(f"DATASET: {dataset.upper()}")
        lines.append("=" * 80)

        ds_comps = [c for c in comparisons if c["dataset"] == dataset]

        for budget in ["small", "medium", "large"]:
            budget_comps = [c for c in ds_comps if c["budget"] == budget]
            if not budget_comps:
                continue

            lines.append(f"\n### {budget.upper()} Budget")
            lines.append("-" * 60)

            for comp in sorted(budget_comps, key=lambda c: c["train_frac"]):
                tf = comp["train_frac"]
                n = comp["n_pairs"]
                lines.append(f"\n  Train Fraction: {tf} (n={n} pairs)")

                for metric, label, fmt, unit in [
                    ("test_mse", "MSE", ".6e", ""),
                    ("test_r2", "R²", ".6f", ""),
                    ("final_train_time", "Train Time", ".2f", "s"),
                ]:
                    m = comp[metric]
                    lower_better = m["lower_better"]
                    better_str = "lower is better" if lower_better else "higher is better"
                    ttest = m["ttest"]
                    d_val, d_lo, d_hi, d_interp = m["effect_size"]
                    shapiro = m.get("shapiro", {})
                    wilcoxon = m.get("wilcoxon", {})

                    lines.append(f"\n    [{label}] ({better_str})")
                    lines.append(f"      PyReCo: {m['pyreco_mean']:{fmt}}{unit} +/- {m['pyreco_std']:{fmt}}{unit}")
                    lines.append(f"      LSTM:   {m['lstm_mean']:{fmt}}{unit} +/- {m['lstm_std']:{fmt}}{unit}")

                    if ttest:
                        p = ttest["p_value"]
                        p_corr = ttest.get("p_corrected", p)
                        sig = _sig_stars(p)
                        sig_corr = _sig_stars(p_corr)

                        lines.append(f"      Mean Diff: {ttest['mean_diff']:{fmt}} [95% CI: {ttest['ci_lower']:{fmt}}, {ttest['ci_upper']:{fmt}}]")
                        lines.append(f"      t({ttest['df']}) = {ttest['t_stat']:.3f}, p = {p:.4f} {sig}")
                        if p_corr != p:
                            lines.append(f"      p (Holm-Bonferroni) = {p_corr:.4f} {sig_corr}")
                        lines.append(f"      Cohen's d_z = {d_val:.3f} [{d_lo:.3f}, {d_hi:.3f}] ({d_interp})")

                    if shapiro and shapiro.get("p_value") is not None:
                        normal_str = "normal" if shapiro["is_normal"] else "non-normal"
                        lp = " [low power]" if shapiro.get("low_power") else ""
                        lines.append(f"      Shapiro-Wilk: W={shapiro['statistic']:.4f}, p={shapiro['p_value']:.4f} ({normal_str}){lp}")

                    if wilcoxon and wilcoxon.get("success"):
                        w_sig = _sig_stars(wilcoxon["p_value"])
                        up = " [underpowered, min p=0.0625]" if wilcoxon.get("underpowered") else ""
                        lines.append(f"      Wilcoxon: W={wilcoxon['statistic']:.1f}, p={wilcoxon['p_value']:.4f} {w_sig}{up}")

                    # Winner
                    if lower_better:
                        winner = "PyReCo" if m["pyreco_mean"] < m["lstm_mean"] else "LSTM"
                        if metric == "final_train_time":
                            ratio = m["lstm_mean"] / max(m["pyreco_mean"], 1e-9)
                            if ratio > 1:
                                lines.append(f"      -> PyReCo is {ratio:.1f}x faster")
                            else:
                                lines.append(f"      -> LSTM is {1/ratio:.1f}x faster")
                        else:
                            lines.append(f"      -> Winner: {winner}")
                    else:
                        winner = "PyReCo" if m["pyreco_mean"] > m["lstm_mean"] else "LSTM"
                        lines.append(f"      -> Winner: {winner}")

    # Overall summary
    lines.append(f"\n{'=' * 80}")
    lines.append("OVERALL SUMMARY")
    lines.append("=" * 80)

    total = len(comparisons)
    wins = {"mse": {"PyReCo": 0, "LSTM": 0}, "r2": {"PyReCo": 0, "LSTM": 0}, "time": {"PyReCo": 0, "LSTM": 0}}

    for comp in comparisons:
        m_mse = comp["test_mse"]
        m_r2 = comp["test_r2"]
        m_time = comp["final_train_time"]

        wins["mse"]["PyReCo" if m_mse["pyreco_mean"] < m_mse["lstm_mean"] else "LSTM"] += 1
        wins["r2"]["PyReCo" if m_r2["pyreco_mean"] > m_r2["lstm_mean"] else "LSTM"] += 1
        wins["time"]["PyReCo" if m_time["pyreco_mean"] < m_time["lstm_mean"] else "LSTM"] += 1

    lines.append(f"\n  Total comparisons: {total}")
    lines.append(f"  MSE wins:   PyReCo {wins['mse']['PyReCo']}/{total}, LSTM {wins['mse']['LSTM']}/{total}")
    lines.append(f"  R^2 wins:   PyReCo {wins['r2']['PyReCo']}/{total}, LSTM {wins['r2']['LSTM']}/{total}")
    lines.append(f"  Time wins:  PyReCo {wins['time']['PyReCo']}/{total}, LSTM {wins['time']['LSTM']}/{total}")

    # By dataset
    lines.append(f"\n  Winners by Dataset (R^2):")
    for dataset in sorted(df["dataset"].unique()):
        ds_comps = [c for c in comparisons if c["dataset"] == dataset]
        if not ds_comps:
            continue
        p_wins = sum(1 for c in ds_comps if c["test_r2"]["pyreco_mean"] > c["test_r2"]["lstm_mean"])
        winner = "PyReCo" if p_wins > len(ds_comps) / 2 else "LSTM"
        lines.append(f"    {dataset}: {winner} (PyReCo {p_wins}/{len(ds_comps)})")

    # By budget
    lines.append(f"\n  Winners by Budget (R^2):")
    for budget in ["small", "medium", "large"]:
        b_comps = [c for c in comparisons if c["budget"] == budget]
        if not b_comps:
            continue
        p_wins = sum(1 for c in b_comps if c["test_r2"]["pyreco_mean"] > c["test_r2"]["lstm_mean"])
        winner = "PyReCo" if p_wins > len(b_comps) / 2 else "LSTM"
        lines.append(f"    {budget}: {winner} (PyReCo {p_wins}/{len(b_comps)})")

    # Significant results summary
    lines.append(f"\n  Statistically significant R^2 differences (p < 0.05, uncorrected):")
    sig_comps = [
        c for c in comparisons
        if c["test_r2"]["ttest"] and c["test_r2"]["ttest"]["p_value"] < 0.05
    ]
    if sig_comps:
        for c in sig_comps:
            p = c["test_r2"]["ttest"]["p_value"]
            diff = c["test_r2"]["pyreco_mean"] - c["test_r2"]["lstm_mean"]
            winner = "PyReCo" if diff > 0 else "LSTM"
            lines.append(
                f"    {c['dataset']}/{c['budget']}/tf={c['train_frac']}: "
                f"p={p:.4f}, diff={diff:+.6f} ({winner} better)"
            )
    else:
        lines.append("    (none)")

    lines.append(f"\n{'=' * 80}")
    lines.append("Legend: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
    lines.append("=" * 80)

    return "\n".join(lines)


def _sig_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return "ns"


def save_json_summary(comparisons: list[dict], output_path: Path):
    """Save machine-readable summary."""
    summary = {
        "generated": datetime.now().isoformat(),
        "n_comparisons": len(comparisons),
        "comparisons": [],
    }

    for comp in comparisons:
        entry = {
            "dataset": comp["dataset"],
            "budget": comp["budget"],
            "train_frac": comp["train_frac"],
            "n_pairs": comp["n_pairs"],
        }
        for metric in ["test_mse", "test_r2", "final_train_time"]:
            m = comp[metric]
            ttest = m["ttest"]
            d_val, d_lo, d_hi, d_interp = m["effect_size"]
            entry[metric] = {
                "pyreco_mean": m["pyreco_mean"],
                "pyreco_std": m["pyreco_std"],
                "lstm_mean": m["lstm_mean"],
                "lstm_std": m["lstm_std"],
                "p_value": ttest["p_value"] if ttest else None,
                "p_corrected": ttest.get("p_corrected") if ttest else None,
                "cohens_d": d_val,
                "cohens_d_ci": [d_lo, d_hi],
                "effect_interpretation": d_interp,
            }
        summary["comparisons"].append(entry)

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="V2 Statistical Analysis")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/final_v2",
        help="Directory with V2 JSON result files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/tables/v2",
        help="Output directory",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("V2 STATISTICAL ANALYSIS")
    print("=" * 60)

    print(f"\nLoading data from {args.results_dir}...")
    df = load_v2_results(args.results_dir)

    if len(df) == 0:
        print("ERROR: No data found!")
        return

    # Filter to paired data only (both PyReCo and LSTM present)
    key_cols = ["dataset", "budget", "seed", "train_frac"]
    df_pyreco = df[df["model_type"] == "pyreco_standard"]
    df_lstm = df[df["model_type"] == "lstm"]
    pyreco_keys = set(df_pyreco[key_cols].apply(tuple, axis=1))
    lstm_keys = set(df_lstm[key_cols].apply(tuple, axis=1))
    paired_keys = pyreco_keys & lstm_keys

    df_paired = df[df[key_cols].apply(tuple, axis=1).isin(paired_keys)]
    print(f"  Total records: {len(df)}")
    print(f"  Paired records: {len(df_paired)} ({len(paired_keys)} conditions)")

    # Filter NaN
    df_paired = df_paired[df_paired["test_mse"].notna() & df_paired["test_r2"].notna()]
    print(f"  Valid records: {len(df_paired)}")

    # Pairwise comparisons
    print("\nPerforming pairwise comparisons...")
    comparisons = perform_pairwise_comparison(df_paired)
    print(f"  Comparisons: {len(comparisons)}")

    # Summary tables
    print("\nGenerating summary tables...")
    summary_df, detail_df = generate_summary_tables(df_paired, output_dir)

    # Report
    print("\nGenerating report...")
    report = generate_report(df_paired, comparisons)
    print("\n" + report)

    # Save outputs
    report_path = output_dir / "statistical_report_v2.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")

    # Comparison CSV
    comp_records = []
    for comp in comparisons:
        rec = {
            "dataset": comp["dataset"],
            "budget": comp["budget"],
            "train_frac": comp["train_frac"],
            "n_pairs": comp["n_pairs"],
        }
        for metric in ["test_mse", "test_r2", "final_train_time"]:
            m = comp[metric]
            ttest = m["ttest"]
            d_val, _, _, d_interp = m["effect_size"]
            rec[f"{metric}_pyreco"] = m["pyreco_mean"]
            rec[f"{metric}_lstm"] = m["lstm_mean"]
            rec[f"{metric}_p"] = ttest["p_value"] if ttest else None
            rec[f"{metric}_p_corr"] = ttest.get("p_corrected") if ttest else None
            rec[f"{metric}_d"] = d_val
            rec[f"{metric}_effect"] = d_interp
        comp_records.append(rec)

    comp_csv_path = output_dir / "statistical_comparisons_v2.csv"
    pd.DataFrame(comp_records).to_csv(comp_csv_path, index=False)
    print(f"Comparisons CSV saved to: {comp_csv_path}")

    # JSON summary
    json_path = output_dir / "statistical_summary.json"
    save_json_summary(comparisons, json_path)
    print(f"JSON summary saved to: {json_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
