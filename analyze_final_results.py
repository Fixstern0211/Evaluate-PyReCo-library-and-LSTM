#!/usr/bin/env python3
"""Comprehensive analysis of final experiment results."""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = Path("/Users/hengz/Evaluate-PyReCo-library-and-LSTM/results/final")

def load_all_results():
    """Load all JSON result files."""
    records = []
    for f in sorted(RESULTS_DIR.glob("results_*.json")):
        with open(f) as fh:
            data = json.load(fh)
        meta = data["metadata"]
        dataset = meta["dataset"]
        seed = meta["seed"]
        train_frac = meta["train_frac"]

        for budget_name, budget_results in data["results"].items():
            for entry in budget_results:
                rec = {
                    "file": f.name,
                    "dataset": dataset,
                    "seed": seed,
                    "train_frac": train_frac,
                    "budget": budget_name,
                    "budget_value": data["budgets"][budget_name],
                    "model_type": entry["model_type"],
                    "test_r2": entry["test_r2"],
                    "test_mse": entry["test_mse"],
                    "test_mae": entry.get("test_mae"),
                    "val_mse": entry.get("val_mse"),
                    "tune_time": entry["tune_time"],
                    "final_train_time": entry["final_train_time"],
                    "inference_time_total": entry.get("inference_time_total"),
                    "n_test_samples": entry.get("n_test_samples"),
                }
                # Extract config details
                cfg = entry["config"]
                if entry["model_type"] == "lstm":
                    rec["learning_rate"] = cfg.get("learning_rate")
                    rec["dropout"] = cfg.get("dropout")
                    rec["num_layers"] = cfg.get("num_layers")
                    rec["hidden_size"] = cfg.get("hidden_size")
                else:
                    rec["spec_rad"] = cfg.get("spec_rad")
                    rec["leakage_rate"] = cfg.get("leakage_rate")
                    rec["density"] = cfg.get("density")
                    rec["num_nodes"] = cfg.get("num_nodes")
                    rec["activation"] = cfg.get("activation")
                    rec["fraction_input"] = cfg.get("fraction_input")

                # Param info
                pinfo = entry.get("param_info", {})
                rec["trainable_params"] = pinfo.get("trainable")
                rec["total_params"] = pinfo.get("total")

                records.append(rec)
    return records

def print_separator(title):
    print(f"\n{'='*90}")
    print(f"  {title}")
    print(f"{'='*90}")

def analyze_all(records):
    # Basic stats
    files = set(r["file"] for r in records)
    datasets = set(r["dataset"] for r in records)
    seeds = sorted(set(r["seed"] for r in records))
    train_fracs = sorted(set(r["train_frac"] for r in records))
    budgets_order = ["small", "medium", "large"]

    print_separator("OVERVIEW")
    print(f"Total JSON files loaded: {len(files)}")
    print(f"Datasets: {datasets}")
    print(f"Seeds: {seeds}")
    print(f"Train fractions: {train_fracs}")
    print(f"Total records: {len(records)}")

    # =========================================================================
    # TABLE 1: Full results table
    # =========================================================================
    print_separator("TABLE 1: ALL RESULTS (sorted by seed, train_frac, budget)")

    header = f"{'Seed':>4} {'TrFr':>5} {'Budget':>6} {'Model':>16} {'R2':>8} {'MSE':>10} {'TuneT(s)':>10} {'TrainT(s)':>10}"
    print(header)
    print("-" * len(header))

    for seed in seeds:
        for tf in train_fracs:
            for bud in budgets_order:
                for r in records:
                    if r["seed"] == seed and r["train_frac"] == tf and r["budget"] == bud:
                        model = r["model_type"]
                        r2 = r["test_r2"]
                        mse = r["test_mse"]
                        tt = r["tune_time"]
                        ft = r["final_train_time"]
                        print(f"{seed:>4} {tf:>5.1f} {bud:>6} {model:>16} {r2:>8.4f} {mse:>10.4f} {tt:>10.1f} {ft:>10.2f}")
            print()  # blank line between train_fracs

    # =========================================================================
    # Q1: Does R2 improve from small -> medium -> large for each model?
    # =========================================================================
    print_separator("Q1: R2 TREND small -> medium -> large (per model, averaged across seeds)")

    for model in ["pyreco_standard", "lstm"]:
        print(f"\n--- {model.upper()} ---")
        header = f"{'TrFr':>5} | {'small':>8} {'medium':>8} {'large':>8} | {'s->m':>7} {'m->l':>7} {'s->l':>7} | {'Monotone?':>10}"
        print(header)
        print("-" * len(header))
        for tf in train_fracs:
            avgs = {}
            for bud in budgets_order:
                vals = [r["test_r2"] for r in records
                        if r["model_type"] == model and r["train_frac"] == tf and r["budget"] == bud]
                avgs[bud] = sum(vals) / len(vals) if vals else float('nan')

            s_to_m = avgs["medium"] - avgs["small"]
            m_to_l = avgs["large"] - avgs["medium"]
            s_to_l = avgs["large"] - avgs["small"]
            monotone = "YES" if s_to_m > 0 and m_to_l > 0 else "NO"

            print(f"{tf:>5.1f} | {avgs['small']:>8.4f} {avgs['medium']:>8.4f} {avgs['large']:>8.4f} | {s_to_m:>+7.4f} {m_to_l:>+7.4f} {s_to_l:>+7.4f} | {monotone:>10}")

    # =========================================================================
    # Q2: Does R2 improve with increasing train_ratio?
    # =========================================================================
    print_separator("Q2: R2 vs TRAIN RATIO (averaged across seeds)")

    for model in ["pyreco_standard", "lstm"]:
        print(f"\n--- {model.upper()} ---")
        header = f"{'Budget':>6} |" + "".join(f" {tf:>6.1f}" for tf in train_fracs) + " | Trend"
        print(header)
        print("-" * len(header))
        for bud in budgets_order:
            vals_by_tf = []
            line = f"{bud:>6} |"
            for tf in train_fracs:
                v = [r["test_r2"] for r in records
                     if r["model_type"] == model and r["train_frac"] == tf and r["budget"] == bud]
                avg = sum(v) / len(v) if v else float('nan')
                vals_by_tf.append(avg)
                line += f" {avg:>6.4f}"

            # Check if monotonically increasing
            increasing = all(vals_by_tf[i] <= vals_by_tf[i+1] for i in range(len(vals_by_tf)-1))
            overall_up = vals_by_tf[-1] > vals_by_tf[0] if vals_by_tf else False
            trend = "MONO UP" if increasing else ("UP overall" if overall_up else "NOT UP")
            line += f" | {trend}"
            print(line)

    # =========================================================================
    # Q3: LSTM hyperparameter consistency
    # =========================================================================
    print_separator("Q3: LSTM CHOSEN HYPERPARAMETERS ACROSS SEEDS")

    for bud in budgets_order:
        print(f"\n--- Budget: {bud.upper()} ---")
        header = f"{'Seed':>4} {'TrFr':>5} {'LR':>8} {'Drop':>6} {'Layers':>6} {'Hidden':>6} {'R2':>8} {'ValMSE':>12}"
        print(header)
        print("-" * len(header))
        for seed in seeds:
            for tf in train_fracs:
                for r in records:
                    if (r["model_type"] == "lstm" and r["seed"] == seed
                        and r["train_frac"] == tf and r["budget"] == bud):
                        print(f"{seed:>4} {tf:>5.1f} {r['learning_rate']:>8.4f} {r['dropout']:>6.2f} {r['num_layers']:>6} {r['hidden_size']:>6} {r['test_r2']:>8.4f} {r['val_mse']:>12.6e}")

    # Summary of LSTM hyperparameter distributions per budget
    print(f"\n--- LSTM Hyperparameter Distribution Summary ---")
    for bud in budgets_order:
        lstm_recs = [r for r in records if r["model_type"] == "lstm" and r["budget"] == bud]
        lrs = [r["learning_rate"] for r in lstm_recs]
        drops = [r["dropout"] for r in lstm_recs]
        layers = [r["num_layers"] for r in lstm_recs]
        hiddens = [r["hidden_size"] for r in lstm_recs]

        print(f"\n  Budget={bud} (n={len(lstm_recs)}):")
        print(f"    learning_rate : unique={sorted(set(lrs))}, most_common={max(set(lrs), key=lrs.count)}")
        print(f"    dropout       : unique={sorted(set(drops))}, most_common={max(set(drops), key=drops.count)}")
        print(f"    num_layers    : unique={sorted(set(layers))}, most_common={max(set(layers), key=layers.count)}")
        print(f"    hidden_size   : min={min(hiddens)}, max={max(hiddens)}, mean={sum(hiddens)/len(hiddens):.1f}")

    # =========================================================================
    # Q4: PyReCo chosen hyperparameters
    # =========================================================================
    print_separator("Q4: PYRECO CHOSEN HYPERPARAMETERS ACROSS SEEDS")

    for bud in budgets_order:
        print(f"\n--- Budget: {bud.upper()} ---")
        header = f"{'Seed':>4} {'TrFr':>5} {'Nodes':>6} {'SpRad':>6} {'Leak':>5} {'Dens':>6} {'R2':>8} {'ValMSE':>12}"
        print(header)
        print("-" * len(header))
        for seed in seeds:
            for tf in train_fracs:
                for r in records:
                    if (r["model_type"] == "pyreco_standard" and r["seed"] == seed
                        and r["train_frac"] == tf and r["budget"] == bud):
                        print(f"{seed:>4} {tf:>5.1f} {r['num_nodes']:>6} {r['spec_rad']:>6.2f} {r['leakage_rate']:>5.2f} {r['density']:>6.3f} {r['test_r2']:>8.4f} {r['val_mse']:>12.6e}")

    # Summary
    print(f"\n--- PyReCo Hyperparameter Distribution Summary ---")
    for bud in budgets_order:
        precs = [r for r in records if r["model_type"] == "pyreco_standard" and r["budget"] == bud]
        srs = [r["spec_rad"] for r in precs]
        lks = [r["leakage_rate"] for r in precs]
        dens = [r["density"] for r in precs]
        nodes = [r["num_nodes"] for r in precs]

        print(f"\n  Budget={bud} (n={len(precs)}):")
        print(f"    num_nodes    : unique={sorted(set(nodes))}")
        print(f"    spec_rad     : unique={sorted(set(srs))}, most_common={max(set(srs), key=srs.count)}")
        print(f"    leakage_rate : unique={sorted(set(lks))}, most_common={max(set(lks), key=lks.count)}")
        print(f"    density      : unique={sorted(set(dens))}, most_common={max(set(dens), key=dens.count)}")

    # =========================================================================
    # Q5: Anomalies
    # =========================================================================
    print_separator("Q5: ANOMALY DETECTION")

    # Check for R2 dropping with more budget
    print("\n--- Cases where R2 drops from small->medium or medium->large ---")
    anomaly_count = 0
    for model in ["pyreco_standard", "lstm"]:
        for seed in seeds:
            for tf in train_fracs:
                r2_by_bud = {}
                for bud in budgets_order:
                    vals = [r["test_r2"] for r in records
                            if r["model_type"] == model and r["seed"] == seed
                            and r["train_frac"] == tf and r["budget"] == bud]
                    if vals:
                        r2_by_bud[bud] = vals[0]

                if len(r2_by_bud) == 3:
                    if r2_by_bud["medium"] < r2_by_bud["small"]:
                        print(f"  DROP s->m: {model}, seed={seed}, tf={tf}: "
                              f"small={r2_by_bud['small']:.4f} -> medium={r2_by_bud['medium']:.4f}")
                        anomaly_count += 1
                    if r2_by_bud["large"] < r2_by_bud["medium"]:
                        print(f"  DROP m->l: {model}, seed={seed}, tf={tf}: "
                              f"medium={r2_by_bud['medium']:.4f} -> large={r2_by_bud['large']:.4f}")
                        anomaly_count += 1

    if anomaly_count == 0:
        print("  None found - R2 always improves or stays same with larger budget.")
    else:
        print(f"\n  Total anomalies: {anomaly_count}")

    # Check for R2 dropping with more training data
    print("\n--- Cases where R2 drops with more training data (adjacent fractions) ---")
    anomaly_count2 = 0
    for model in ["pyreco_standard", "lstm"]:
        for seed in seeds:
            for bud in budgets_order:
                prev_r2 = None
                prev_tf = None
                for tf in train_fracs:
                    vals = [r["test_r2"] for r in records
                            if r["model_type"] == model and r["seed"] == seed
                            and r["train_frac"] == tf and r["budget"] == bud]
                    if vals:
                        curr_r2 = vals[0]
                        if prev_r2 is not None and curr_r2 < prev_r2 - 0.01:  # threshold to ignore noise
                            print(f"  DROP: {model}, seed={seed}, budget={bud}: "
                                  f"tf={prev_tf}(R2={prev_r2:.4f}) -> tf={tf}(R2={curr_r2:.4f}) "
                                  f"delta={curr_r2-prev_r2:+.4f}")
                            anomaly_count2 += 1
                        prev_r2 = curr_r2
                        prev_tf = tf

    if anomaly_count2 == 0:
        print("  None found (with >0.01 threshold).")
    else:
        print(f"\n  Total anomalies: {anomaly_count2}")

    # Negative R2 values
    print("\n--- Negative R2 values ---")
    neg_count = 0
    for r in records:
        if r["test_r2"] < 0:
            print(f"  NEGATIVE R2: {r['model_type']}, seed={r['seed']}, tf={r['train_frac']}, "
                  f"budget={r['budget']}, R2={r['test_r2']:.4f}")
            neg_count += 1
    if neg_count == 0:
        print("  None found.")

    # Very large val_mse vs test_mse discrepancy
    print("\n--- Large val_mse vs test_mse discrepancy (ratio > 100) ---")
    disc_count = 0
    for r in records:
        if r["val_mse"] and r["val_mse"] > 0:
            ratio = r["test_mse"] / r["val_mse"]
            if ratio > 100:
                print(f"  DISCREPANCY: {r['model_type']}, seed={r['seed']}, tf={r['train_frac']}, "
                      f"budget={r['budget']}: val_mse={r['val_mse']:.6e}, test_mse={r['test_mse']:.4f}, "
                      f"ratio={ratio:.0f}x")
                disc_count += 1
    if disc_count == 0:
        print("  None found.")
    else:
        print(f"  Total: {disc_count}")

    # =========================================================================
    # Q6: PyReCo vs LSTM comparison
    # =========================================================================
    print_separator("Q6: PYRECO vs LSTM HEAD-TO-HEAD COMPARISON")

    print("\n--- Average R2 by Budget Level (across all seeds and train_fracs) ---")
    header = f"{'Budget':>6} {'PyReCo_R2':>10} {'LSTM_R2':>10} {'Winner':>10} {'Delta':>8}"
    print(header)
    print("-" * len(header))

    for bud in budgets_order:
        pyreco_r2s = [r["test_r2"] for r in records if r["model_type"] == "pyreco_standard" and r["budget"] == bud]
        lstm_r2s = [r["test_r2"] for r in records if r["model_type"] == "lstm" and r["budget"] == bud]

        p_avg = sum(pyreco_r2s) / len(pyreco_r2s) if pyreco_r2s else 0
        l_avg = sum(lstm_r2s) / len(lstm_r2s) if lstm_r2s else 0
        winner = "PyReCo" if p_avg > l_avg else "LSTM"
        delta = p_avg - l_avg
        print(f"{bud:>6} {p_avg:>10.4f} {l_avg:>10.4f} {winner:>10} {delta:>+8.4f}")

    # Per-experiment win counts
    print("\n--- Win/Loss/Tie Counts (per individual experiment) ---")
    header = f"{'Budget':>6} {'PyReCo Wins':>12} {'LSTM Wins':>10} {'Ties':>6} {'Total':>6}"
    print(header)
    print("-" * len(header))

    for bud in budgets_order:
        pw, lw, ties = 0, 0, 0
        for seed in seeds:
            for tf in train_fracs:
                p_r2 = [r["test_r2"] for r in records
                        if r["model_type"] == "pyreco_standard" and r["seed"] == seed
                        and r["train_frac"] == tf and r["budget"] == bud]
                l_r2 = [r["test_r2"] for r in records
                        if r["model_type"] == "lstm" and r["seed"] == seed
                        and r["train_frac"] == tf and r["budget"] == bud]
                if p_r2 and l_r2:
                    if p_r2[0] > l_r2[0]:
                        pw += 1
                    elif l_r2[0] > p_r2[0]:
                        lw += 1
                    else:
                        ties += 1
        total = pw + lw + ties
        print(f"{bud:>6} {pw:>12} {lw:>10} {ties:>6} {total:>6}")

    # Detailed comparison table (avg across seeds for each train_frac x budget)
    print("\n--- Detailed R2 Comparison: PyReCo vs LSTM (avg across seeds) ---")
    header = f"{'TrFr':>5} {'Budget':>6} {'PyReCo':>8} {'LSTM':>8} {'Winner':>8} {'Delta':>8}"
    print(header)
    print("-" * len(header))
    for tf in train_fracs:
        for bud in budgets_order:
            p_vals = [r["test_r2"] for r in records
                      if r["model_type"] == "pyreco_standard" and r["train_frac"] == tf and r["budget"] == bud]
            l_vals = [r["test_r2"] for r in records
                      if r["model_type"] == "lstm" and r["train_frac"] == tf and r["budget"] == bud]
            p_avg = sum(p_vals) / len(p_vals) if p_vals else 0
            l_avg = sum(l_vals) / len(l_vals) if l_vals else 0
            winner = "PyReCo" if p_avg > l_avg else "LSTM"
            delta = p_avg - l_avg
            print(f"{tf:>5.1f} {bud:>6} {p_avg:>8.4f} {l_avg:>8.4f} {winner:>8} {delta:>+8.4f}")
        print()

    # =========================================================================
    # Q7: Training time patterns
    # =========================================================================
    print_separator("Q7: TRAINING TIME PATTERNS")

    print("\n--- Average Tune Time (hyperparameter search) by Budget ---")
    header = f"{'Budget':>6} {'PyReCo(s)':>12} {'LSTM(s)':>10} {'Ratio(L/P)':>12}"
    print(header)
    print("-" * len(header))
    for bud in budgets_order:
        p_times = [r["tune_time"] for r in records if r["model_type"] == "pyreco_standard" and r["budget"] == bud]
        l_times = [r["tune_time"] for r in records if r["model_type"] == "lstm" and r["budget"] == bud]
        p_avg = sum(p_times) / len(p_times) if p_times else 0
        l_avg = sum(l_times) / len(l_times) if l_times else 0
        ratio = l_avg / p_avg if p_avg > 0 else float('inf')
        print(f"{bud:>6} {p_avg:>12.1f} {l_avg:>10.1f} {ratio:>12.1f}x")

    print("\n--- Average Final Train Time by Budget ---")
    header = f"{'Budget':>6} {'PyReCo(s)':>12} {'LSTM(s)':>10} {'Ratio(L/P)':>12}"
    print(header)
    print("-" * len(header))
    for bud in budgets_order:
        p_times = [r["final_train_time"] for r in records if r["model_type"] == "pyreco_standard" and r["budget"] == bud]
        l_times = [r["final_train_time"] for r in records if r["model_type"] == "lstm" and r["budget"] == bud]
        p_avg = sum(p_times) / len(p_times) if p_times else 0
        l_avg = sum(l_times) / len(l_times) if l_times else 0
        ratio = l_avg / p_avg if p_avg > 0 else float('inf')
        print(f"{bud:>6} {p_avg:>12.2f} {l_avg:>10.2f} {ratio:>12.1f}x")

    print("\n--- Average Total Time (tune + final_train) by Budget ---")
    header = f"{'Budget':>6} {'PyReCo(s)':>12} {'LSTM(s)':>10} {'Ratio(L/P)':>12}"
    print(header)
    print("-" * len(header))
    for bud in budgets_order:
        p_times = [r["tune_time"] + r["final_train_time"] for r in records if r["model_type"] == "pyreco_standard" and r["budget"] == bud]
        l_times = [r["tune_time"] + r["final_train_time"] for r in records if r["model_type"] == "lstm" and r["budget"] == bud]
        p_avg = sum(p_times) / len(p_times) if p_times else 0
        l_avg = sum(l_times) / len(l_times) if l_times else 0
        ratio = l_avg / p_avg if p_avg > 0 else float('inf')
        print(f"{bud:>6} {p_avg:>12.1f} {l_avg:>10.1f} {ratio:>12.1f}x")

    # Time vs train_frac
    print("\n--- Total Time (tune+train) by Train Fraction ---")
    for model in ["pyreco_standard", "lstm"]:
        print(f"\n  {model.upper()}:")
        header = f"  {'TrFr':>5} |" + "".join(f" {bud:>10}" for bud in budgets_order)
        print(header)
        print("  " + "-" * (len(header)-2))
        for tf in train_fracs:
            line = f"  {tf:>5.1f} |"
            for bud in budgets_order:
                times = [r["tune_time"] + r["final_train_time"] for r in records
                         if r["model_type"] == model and r["train_frac"] == tf and r["budget"] == bud]
                avg = sum(times) / len(times) if times else 0
                line += f" {avg:>10.1f}"
            print(line)

    # =========================================================================
    # OVERALL SUMMARY
    # =========================================================================
    print_separator("OVERALL SUMMARY")

    # Aggregate R2
    all_pyreco_r2 = [r["test_r2"] for r in records if r["model_type"] == "pyreco_standard"]
    all_lstm_r2 = [r["test_r2"] for r in records if r["model_type"] == "lstm"]

    print(f"\nOverall Average R2:")
    print(f"  PyReCo: {sum(all_pyreco_r2)/len(all_pyreco_r2):.4f} (min={min(all_pyreco_r2):.4f}, max={max(all_pyreco_r2):.4f})")
    print(f"  LSTM:   {sum(all_lstm_r2)/len(all_lstm_r2):.4f} (min={min(all_lstm_r2):.4f}, max={max(all_lstm_r2):.4f})")

    # Win counts total
    total_pw, total_lw = 0, 0
    for seed in seeds:
        for tf in train_fracs:
            for bud in budgets_order:
                p_r2 = [r["test_r2"] for r in records
                        if r["model_type"] == "pyreco_standard" and r["seed"] == seed
                        and r["train_frac"] == tf and r["budget"] == bud]
                l_r2 = [r["test_r2"] for r in records
                        if r["model_type"] == "lstm" and r["seed"] == seed
                        and r["train_frac"] == tf and r["budget"] == bud]
                if p_r2 and l_r2:
                    if p_r2[0] > l_r2[0]:
                        total_pw += 1
                    elif l_r2[0] > p_r2[0]:
                        total_lw += 1

    total_comparisons = total_pw + total_lw
    print(f"\nHead-to-Head Wins (across all {total_comparisons} comparisons):")
    print(f"  PyReCo wins: {total_pw} ({100*total_pw/total_comparisons:.1f}%)")
    print(f"  LSTM wins:   {total_lw} ({100*total_lw/total_comparisons:.1f}%)")

    # Best and worst R2 for each model
    print(f"\nBest R2 per model:")
    best_p = max(records, key=lambda r: r["test_r2"] if r["model_type"] == "pyreco_standard" else -999)
    best_l = max(records, key=lambda r: r["test_r2"] if r["model_type"] == "lstm" else -999)
    # Actually filter properly
    best_p = max([r for r in records if r["model_type"] == "pyreco_standard"], key=lambda r: r["test_r2"])
    best_l = max([r for r in records if r["model_type"] == "lstm"], key=lambda r: r["test_r2"])
    print(f"  PyReCo best: R2={best_p['test_r2']:.4f} (seed={best_p['seed']}, tf={best_p['train_frac']}, budget={best_p['budget']})")
    print(f"  LSTM best:   R2={best_l['test_r2']:.4f} (seed={best_l['seed']}, tf={best_l['train_frac']}, budget={best_l['budget']})")

    worst_p = min([r for r in records if r["model_type"] == "pyreco_standard"], key=lambda r: r["test_r2"])
    worst_l = min([r for r in records if r["model_type"] == "lstm"], key=lambda r: r["test_r2"])
    print(f"  PyReCo worst: R2={worst_p['test_r2']:.4f} (seed={worst_p['seed']}, tf={worst_p['train_frac']}, budget={worst_p['budget']})")
    print(f"  LSTM worst:   R2={worst_l['test_r2']:.4f} (seed={worst_l['seed']}, tf={worst_l['train_frac']}, budget={worst_l['budget']})")


if __name__ == "__main__":
    records = load_all_results()
    if not records:
        print("ERROR: No records loaded!")
        sys.exit(1)
    analyze_all(records)
