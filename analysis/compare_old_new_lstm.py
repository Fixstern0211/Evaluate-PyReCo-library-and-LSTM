"""
Compare old (results/final/) vs new (results/final_v2/) LSTM results.

Old format: one file per (dataset, seed, train_frac), contains 3 budgets (small/medium/large).
New format: one file per (dataset, budget_name, seed, train_frac).

Match by (dataset, budget_name, seed, train_frac), extract LSTM entries, compare.
"""

import json
import glob
import os
import re
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OLD_DIR = os.path.join(BASE_DIR, "results", "final")
NEW_DIR = os.path.join(BASE_DIR, "results", "final_v2")


def load_old_results():
    """Load old results: one file has 3 budgets. Returns dict keyed by (dataset, budget, seed, train_frac)."""
    records = {}
    pattern = os.path.join(OLD_DIR, "results_*_seed*_train*_*.json")
    for fpath in sorted(glob.glob(pattern)):
        fname = os.path.basename(fpath)
        # e.g. results_lorenz_seed42_train0.5_20260219_034216.json
        m = re.match(r"results_(\w+)_seed(\d+)_train([\d.]+)_\d+_\d+\.json", fname)
        if not m:
            continue
        dataset, seed, train_frac = m.group(1), int(m.group(2)), float(m.group(3))

        with open(fpath) as f:
            data = json.load(f)

        for budget_name, entries in data.get("results", {}).items():
            lstm_entry = None
            for entry in entries:
                if entry.get("model_type") == "lstm":
                    lstm_entry = entry
                    break
            if lstm_entry:
                key = (dataset, budget_name, seed, train_frac)
                records[key] = lstm_entry
    return records


def load_new_results():
    """Load new results: one file per budget. Returns dict keyed by (dataset, budget, seed, train_frac)."""
    records = {}
    pattern = os.path.join(NEW_DIR, "results_*_seed*_train*.json")
    for fpath in sorted(glob.glob(pattern)):
        fname = os.path.basename(fpath)
        # e.g. results_lorenz_small_seed42_train0.5.json
        m = re.match(r"results_(\w+?)_(small|medium|large)_seed(\d+)_train([\d.]+)\.json", fname)
        if not m:
            continue
        dataset, budget_name = m.group(1), m.group(2)
        seed, train_frac = int(m.group(3)), float(m.group(4))

        with open(fpath) as f:
            data = json.load(f)

        for bname, entries in data.get("results", {}).items():
            lstm_entry = None
            for entry in entries:
                if entry.get("model_type") == "lstm":
                    lstm_entry = entry
                    break
            if lstm_entry:
                key = (dataset, bname, seed, train_frac)
                records[key] = lstm_entry
    return records


def main():
    old = load_old_results()
    new = load_new_results()

    # Find matching keys
    common_keys = sorted(set(old.keys()) & set(new.keys()))
    print(f"Old LSTM entries: {len(old)}")
    print(f"New LSTM entries: {len(new)}")
    print(f"Matched pairs:   {len(common_keys)}")
    print()

    if not common_keys:
        print("No matching pairs found!")
        sys.exit(1)

    # Step 2: Per-pair comparison
    print("=" * 120)
    print(f"{'Dataset':<12} {'Budget':<8} {'Seed':<6} {'TF':<5} | "
          f"{'Old h':<6} {'Old params':<11} {'Old R²':<12} {'Old MSE':<12} | "
          f"{'New h':<6} {'New params':<11} {'New R²':<12} {'New MSE':<12} | "
          f"{'ΔR²':<12} {'MSE ratio':<10}")
    print("-" * 120)

    pairs = []
    for key in common_keys:
        ds, bn, seed, tf = key
        o, n = old[key], new[key]
        o_r2 = o.get("test_r2", float("nan"))
        n_r2 = n.get("test_r2", float("nan"))
        o_mse = o.get("test_mse", float("nan"))
        n_mse = n.get("test_mse", float("nan"))
        o_h = o.get("config", {}).get("hidden_size", "?")
        n_h = n.get("config", {}).get("hidden_size", "?")
        o_params = o.get("param_info", {}).get("trainable", "?")
        n_params = n.get("param_info", {}).get("trainable", "?")

        delta_r2 = n_r2 - o_r2
        mse_ratio = n_mse / o_mse if o_mse != 0 else float("nan")

        pairs.append({
            "key": key, "ds": ds, "bn": bn, "seed": seed, "tf": tf,
            "old_r2": o_r2, "new_r2": n_r2, "old_mse": o_mse, "new_mse": n_mse,
            "delta_r2": delta_r2, "mse_ratio": mse_ratio,
            "old_h": o_h, "new_h": n_h, "old_params": o_params, "new_params": n_params,
        })

        print(f"{ds:<12} {bn:<8} {seed:<6} {tf:<5.1f} | "
              f"{str(o_h):<6} {str(o_params):<11} {o_r2:<12.8f} {o_mse:<12.6e} | "
              f"{str(n_h):<6} {str(n_params):<11} {n_r2:<12.8f} {n_mse:<12.6e} | "
              f"{delta_r2:<+12.8f} {mse_ratio:<10.4f}")

    print()

    # Step 3: Aggregate by (dataset, budget)
    print("=" * 100)
    print("AGGREGATE BY (dataset, budget)")
    print("=" * 100)
    print(f"{'Dataset':<12} {'Budget':<8} {'N':<4} | "
          f"{'Mean|ΔR²|':<12} {'Max|ΔR²|':<12} {'New wins':<10} {'Old wins':<10} | "
          f"{'Pearson r':<10} {'t-test p':<10}")
    print("-" * 100)

    groups = defaultdict(list)
    for p in pairs:
        groups[(p["ds"], p["bn"])].append(p)

    for (ds, bn), items in sorted(groups.items()):
        old_r2s = np.array([p["old_r2"] for p in items])
        new_r2s = np.array([p["new_r2"] for p in items])
        deltas = np.array([p["delta_r2"] for p in items])

        mean_abs_delta = np.mean(np.abs(deltas))
        max_abs_delta = np.max(np.abs(deltas))
        new_wins = np.sum(deltas > 0)
        old_wins = np.sum(deltas < 0)

        if len(items) >= 3:
            r, _ = stats.pearsonr(old_r2s, new_r2s)
            t_stat, p_val = stats.ttest_rel(old_r2s, new_r2s)
        else:
            r = float("nan")
            p_val = float("nan")

        print(f"{ds:<12} {bn:<8} {len(items):<4} | "
              f"{mean_abs_delta:<12.8f} {max_abs_delta:<12.8f} {new_wins:<10} {old_wins:<10} | "
              f"{r:<10.6f} {p_val:<10.4f}")

    print()

    # Step 4: Stratify by train_frac
    print("=" * 100)
    print("STRATIFIED BY train_frac")
    print("=" * 100)

    tf_groups = defaultdict(list)
    for p in pairs:
        tf_groups[p["tf"]].append(p)

    print(f"{'TF':<6} {'N':<4} | {'Mean|ΔR²|':<12} {'Max|ΔR²|':<12} {'Mean R²(old)':<14} {'Mean R²(new)':<14} | "
          f"{'Pearson r':<10} {'t-test p':<10} {'New wins':<10} {'Old wins':<10}")
    print("-" * 100)

    for tf in sorted(tf_groups.keys()):
        items = tf_groups[tf]
        old_r2s = np.array([p["old_r2"] for p in items])
        new_r2s = np.array([p["new_r2"] for p in items])
        deltas = np.array([p["delta_r2"] for p in items])

        mean_abs_delta = np.mean(np.abs(deltas))
        max_abs_delta = np.max(np.abs(deltas))
        new_wins = np.sum(deltas > 0)
        old_wins = np.sum(deltas < 0)

        if len(items) >= 3:
            r, _ = stats.pearsonr(old_r2s, new_r2s)
            t_stat, p_val = stats.ttest_rel(old_r2s, new_r2s)
        else:
            r = float("nan")
            p_val = float("nan")

        print(f"{tf:<6.1f} {len(items):<4} | "
              f"{mean_abs_delta:<12.8f} {max_abs_delta:<12.8f} {np.mean(old_r2s):<14.8f} {np.mean(new_r2s):<14.8f} | "
              f"{r:<10.6f} {p_val:<10.4f} {new_wins:<10} {old_wins:<10}")

    # Also stratify: tf >= 0.5 vs tf < 0.5
    print()
    for label, cond in [("tf < 0.5 (underfitting regime)", lambda p: p["tf"] < 0.5),
                         ("tf >= 0.5 (adequate data)", lambda p: p["tf"] >= 0.5)]:
        items = [p for p in pairs if cond(p)]
        if not items:
            continue
        old_r2s = np.array([p["old_r2"] for p in items])
        new_r2s = np.array([p["new_r2"] for p in items])
        deltas = np.array([p["delta_r2"] for p in items])

        mean_abs_delta = np.mean(np.abs(deltas))
        max_abs_delta = np.max(np.abs(deltas))
        new_wins = int(np.sum(deltas > 0))
        old_wins = int(np.sum(deltas < 0))

        if len(items) >= 3:
            r, _ = stats.pearsonr(old_r2s, new_r2s)
            t_stat, p_val = stats.ttest_rel(old_r2s, new_r2s)
        else:
            r = float("nan")
            p_val = float("nan")

        print(f"  {label}: N={len(items)}")
        print(f"    Mean|ΔR²| = {mean_abs_delta:.8f},  Max|ΔR²| = {max_abs_delta:.8f}")
        print(f"    Pearson r = {r:.6f},  t-test p = {p_val:.4f}")
        print(f"    New wins: {new_wins}, Old wins: {old_wins}")
        print()

    # Step 5: Conclusion
    print("=" * 100)
    print("CONCLUSION")
    print("=" * 100)
    tf_high = [p for p in pairs if p["tf"] >= 0.5]
    if tf_high:
        deltas_high = np.array([p["delta_r2"] for p in tf_high])
        mean_abs = np.mean(np.abs(deltas_high))
        max_abs = np.max(np.abs(deltas_high))
        if len(tf_high) >= 3:
            _, p_val = stats.ttest_rel(
                [p["old_r2"] for p in tf_high],
                [p["new_r2"] for p in tf_high]
            )
        else:
            p_val = float("nan")

        reuse_ok = mean_abs < 0.001 and (np.isnan(p_val) or p_val > 0.05)
        print(f"For tf >= 0.5 ({len(tf_high)} pairs):")
        print(f"  Mean|ΔR²| = {mean_abs:.8f} {'< 0.001 ✓' if mean_abs < 0.001 else '>= 0.001 ✗'}")
        print(f"  Max|ΔR²|  = {max_abs:.8f}")
        print(f"  t-test p  = {p_val:.4f} {'> 0.05 ✓' if (np.isnan(p_val) or p_val > 0.05) else '<= 0.05 ✗'}")
        print()
        if reuse_ok:
            print("  → RECOMMENDATION: Old LSTM data can be reused. Differences are negligible.")
        else:
            print("  → RECOMMENDATION: Differences are non-negligible. Re-run LSTM with new config.")
    else:
        print("No tf >= 0.5 pairs available for conclusion.")


if __name__ == "__main__":
    main()
