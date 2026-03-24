#!/usr/bin/env python3
"""
Check for overlapping (duplicate) parameter combinations in merged pretuning results.

For each dataset and budget, identifies parameter combos that appear more than once
in all_results, and compares their MSE and R2 values.
"""

import json
from collections import defaultdict

DATASETS = ["lorenz", "mackeyglass", "santafe"]
BASE_PATH = "/Users/hengz/Evaluate-PyReCo-library-and-LSTM/results/pretuning"

# The params keys that define a unique combination
PARAM_KEYS = ["spec_rad", "leakage_rate", "density", "fraction_input"]


def params_to_key(params):
    """Create a hashable key from the relevant parameter values."""
    return tuple(params[k] for k in PARAM_KEYS)


def main():
    total_duplicates = 0

    for dataset in DATASETS:
        filepath = f"{BASE_PATH}/pretuning_{dataset}_merged.json"
        try:
            with open(filepath) as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"WARNING: File not found: {filepath}")
            continue

        print("=" * 80)
        print(f"DATASET: {dataset.upper()}")
        print(f"File: {filepath}")
        print("=" * 80)

        for budget in data["budgets"]:
            budget_name = budget["budget_name"]
            budget_value = budget["budget_value"]
            num_nodes = budget["num_nodes"]
            all_results = budget["all_results"]
            sources = budget.get("sources", {})

            print(f"\n  Budget: {budget_name} (value={budget_value}, "
                  f"num_nodes={num_nodes}, n_results={len(all_results)})")
            print(f"  Sources: original={sources.get('original', '?')}, "
                  f"supplementary={sources.get('supplementary', '?')}")

            # Group results by parameter combination
            groups = defaultdict(list)
            for i, result in enumerate(all_results):
                key = params_to_key(result["params"])
                groups[key].append((i, result))

            # Find duplicates
            duplicates = {k: v for k, v in groups.items() if len(v) > 1}

            if not duplicates:
                print("  --> No duplicate parameter combinations found.")
                continue

            print(f"  --> Found {len(duplicates)} duplicate parameter combination(s):")
            total_duplicates += len(duplicates)

            for combo_idx, (key, entries) in enumerate(sorted(duplicates.items()), 1):
                print(f"\n    Duplicate #{combo_idx}:")
                print(f"      Parameters: spec_rad={key[0]}, leakage_rate={key[1]}, "
                      f"density={key[2]}, fraction_input={key[3]}")
                print(f"      Appears {len(entries)} times (indices: "
                      f"{', '.join(str(e[0]) for e in entries)})")

                # Compare all pairs
                for a_idx in range(len(entries)):
                    for b_idx in range(a_idx + 1, len(entries)):
                        idx_a, res_a = entries[a_idx]
                        idx_b, res_b = entries[b_idx]

                        mse_a = res_a["cv_mean"]
                        mse_b = res_b["cv_mean"]
                        r2_a = res_a["cv_r2_mean"]
                        r2_b = res_b["cv_r2_mean"]

                        print(f"\n      Pair: index {idx_a} vs index {idx_b}")
                        print(f"        Entry A (idx {idx_a}): "
                              f"cv_mean(MSE)={mse_a:.10e}, cv_r2_mean(R2)={r2_a:.10f}")
                        print(f"        Entry B (idx {idx_b}): "
                              f"cv_mean(MSE)={mse_b:.10e}, cv_r2_mean(R2)={r2_b:.10f}")

                        if mse_a == mse_b and r2_a == r2_b:
                            print("        --> EXACT MATCH (identical MSE and R2)")
                        else:
                            mse_diff = abs(mse_a - mse_b)
                            r2_diff = abs(r2_a - r2_b)
                            # Relative difference for MSE
                            mse_rel = mse_diff / max(abs(mse_a), abs(mse_b), 1e-30) * 100
                            print(f"        --> DIFFERENT VALUES:")
                            print(f"           MSE  diff = {mse_diff:.10e} "
                                  f"(relative: {mse_rel:.4f}%)")
                            print(f"           R2   diff = {r2_diff:.10e}")

                            # Also compare fold-level scores if available
                            folds_a = res_a.get("fold_mse_scores", [])
                            folds_b = res_b.get("fold_mse_scores", [])
                            if folds_a and folds_b and folds_a == folds_b:
                                print("           (But fold-level scores are identical)")
                            elif folds_a and folds_b:
                                n_matching = sum(
                                    1 for a, b in zip(folds_a, folds_b) if a == b
                                )
                                print(f"           Fold MSE scores: "
                                      f"{n_matching}/{len(folds_a)} folds match")

    print("\n" + "=" * 80)
    print(f"SUMMARY: {total_duplicates} total duplicate parameter combinations "
          f"found across all datasets and budgets.")
    print("=" * 80)


if __name__ == "__main__":
    main()
