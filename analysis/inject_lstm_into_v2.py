#!/usr/bin/env python3
"""
Inject V1 LSTM results into V2 JSON files.

For each V1 file (results/final/results_{dataset}_seed{seed}_train{tf}_{ts}.json),
extract the LSTM entry under each budget (small/medium/large) and append it to
the corresponding V2 file (results/final_v2/results_{dataset}_{budget}_seed{seed}_train{tf}.json).

Skips if the V2 file already contains an LSTM entry or doesn't exist.
"""

import json
import glob
import os
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
V1_DIR = BASE_DIR / "results" / "final"
V2_DIR = BASE_DIR / "results" / "final_v2"


def main():
    v1_pattern = str(V1_DIR / "results_*_seed*_train*_*.json")
    v1_files = sorted(glob.glob(v1_pattern))

    print(f"Found {len(v1_files)} V1 files")

    injected = 0
    skipped_exists = 0
    skipped_no_v2 = 0
    skipped_no_lstm = 0

    for v1_path in v1_files:
        fname = os.path.basename(v1_path)
        m = re.match(r"results_(\w+)_seed(\d+)_train([\d.]+)_\d+_\d+\.json", fname)
        if not m:
            continue

        dataset, seed, train_frac = m.group(1), int(m.group(2)), m.group(3)

        with open(v1_path) as f:
            v1_data = json.load(f)

        for budget_name, entries in v1_data.get("results", {}).items():
            # Find LSTM entry
            lstm_entry = None
            for entry in entries:
                if entry.get("model_type") == "lstm":
                    lstm_entry = entry
                    break

            if lstm_entry is None:
                skipped_no_lstm += 1
                continue

            # Target V2 file
            v2_fname = f"results_{dataset}_{budget_name}_seed{seed}_train{train_frac}.json"
            v2_path = V2_DIR / v2_fname

            if not v2_path.exists():
                skipped_no_v2 += 1
                continue

            # Load V2 file
            with open(v2_path) as f:
                v2_data = json.load(f)

            # Check if LSTM already exists
            existing_models = [
                e["model_type"] for e in v2_data["results"].get(budget_name, [])
            ]
            if "lstm" in existing_models:
                skipped_exists += 1
                continue

            # Inject
            v2_data["results"][budget_name].append(lstm_entry)

            with open(v2_path, "w") as f:
                json.dump(v2_data, f, indent=2)

            injected += 1
            print(f"  + {v2_fname}")

    print(f"\nDone:")
    print(f"  Injected:          {injected}")
    print(f"  Skipped (exists):  {skipped_exists}")
    print(f"  Skipped (no V2):   {skipped_no_v2}")
    print(f"  Skipped (no LSTM): {skipped_no_lstm}")


if __name__ == "__main__":
    main()
