#!/usr/bin/env python3
"""Build final_v3 by preserving ESN results from final_v2 and fully rerunning LSTM.

This script reads existing JSON files from ``results/final_v2`` and writes new
JSON files to ``results/final_v3`` with:

- the original ESN entry copied as-is
- a newly generated LSTM entry obtained by rerunning the full 32-combination
  LSTM grid search followed by final retraining on train+val

The goal is to produce a new result directory with a protocol-consistent LSTM
side, without modifying the existing final_v2 files.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_SOURCE_DIR = DEFAULT_REPO_ROOT / "results" / "final_v2"
DEFAULT_OUTPUT_DIR = DEFAULT_REPO_ROOT / "results" / "final_v3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build final_v3 by rerunning only the LSTM side with full-grid tuning."
    )
    parser.add_argument("--repo-root", default=str(DEFAULT_REPO_ROOT))
    parser.add_argument("--source-dir", default=str(DEFAULT_SOURCE_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--dataset", choices=["lorenz", "mackeyglass", "santafe"])
    parser.add_argument("--budget", choices=["small", "medium", "large"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--train-frac", type=float)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def import_repo_modules(repo_root: str):
    repo_root = os.path.abspath(repo_root)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    import experiments.run_final_v2 as rf
    from models.lstm_model import LSTMModel, tune_lstm_hyperparameters

    return rf, LSTMModel, tune_lstm_hyperparameters


def read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def iter_source_files(
    source_dir: Path,
    dataset: Optional[str],
    budget: Optional[str],
    seed: Optional[int],
    train_frac: Optional[float],
) -> List[Path]:
    files = sorted(source_dir.glob("results_*.json"))
    out: List[Path] = []
    for path in files:
        try:
            data = read_json(path)
            meta = data["metadata"]
        except Exception:
            continue
        if dataset and meta.get("dataset") != dataset:
            continue
        if budget and meta.get("budget_name") != budget:
            continue
        if seed is not None and meta.get("seed") != seed:
            continue
        if train_frac is not None and float(meta.get("train_frac")) != float(train_frac):
            continue
        out.append(path)
    return out


def has_complete_v3(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        data = read_json(path)
        meta = data["metadata"]
        budget_name = meta["budget_name"]
        model_types = {row["model_type"] for row in data["results"].get(budget_name, [])}
        return {"pyreco_standard", "lstm"}.issubset(model_types) and meta.get(
            "lstm_rerun_mode"
        ) == "full-grid"
    except Exception:
        return False


def cleanup_torch() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def compute_inline_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_pred = np.asarray(y_pred).reshape(y_true.shape)
    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    return {"mse": mse, "mae": mae, "r2": r2}


def rerun_lstm_full_grid(
    *,
    rf: Any,
    LSTMModel: Any,
    tune_lstm_hyperparameters: Any,
    dataset: str,
    budget_name: str,
    train_frac: float,
    seed: int,
    device: str,
    verbose: bool,
) -> Dict[str, Any]:
    budget = rf.BUDGETS[budget_name]
    d_in = 3 if dataset == "lorenz" else 1
    d_out = d_in

    rf.set_seed(seed)
    load_func = rf.pyreco_load if dataset in rf.PYRECO_DATASETS else rf.local_load
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = load_func(
        dataset,
        n_samples=5000,
        seed=seed,
        train_fraction=train_frac,
        val_fraction=0.15,
        n_in=100,
        n_out=1,
        standardize=True,
    )

    layer_map = rf.lstm_layer_hidden_map(budget, d_in, d_out, [1, 2])

    lstm_device = device
    if lstm_device == "auto":
        lstm_device = "cpu"
        try:
            import torch

            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                lstm_device = "mps"
            elif torch.cuda.is_available():
                lstm_device = "cuda"
        except Exception:
            pass

    rf.set_seed(seed)
    t_tune_start = time.time()
    tune_results = tune_lstm_hyperparameters(
        X_train,
        y_train,
        X_val,
        y_val,
        param_grid=rf.LSTM_PARAM_GRID,
        lstm_device=lstm_device,
        verbose=verbose,
        layer_hidden_map=layer_map,
    )
    tune_time = time.time() - t_tune_start

    best_params = tune_results["best_params"]
    best_val_mse = tune_results["best_score"]
    best_epoch = None
    best_combo_train_time = None
    for row in tune_results.get("all_results", []):
        if row["params"] == best_params:
            best_epoch = row.get("best_epoch")
            best_combo_train_time = row.get("train_time")
            break

    del tune_results
    cleanup_torch()

    X_full, y_full = rf._make_trainval_windows(dataset, seed, train_frac, scaler)

    rf.set_seed(seed)
    final_lstm = LSTMModel(**best_params, device=lstm_device, verbose=verbose)
    if best_epoch and best_epoch > 0:
        final_lstm.epochs = int(best_epoch)

    green = rf.GreenMetricsTracker()
    green.start()

    t0 = time.time()
    final_lstm.fit(X_full, y_full)
    final_train_time = time.time() - t0

    green_metrics = green.stop()

    test_pred = final_lstm.predict(X_test)
    test_metrics = compute_inline_metrics(y_test, test_pred)

    _ = final_lstm.predict(X_test[:10])
    t0 = time.time()
    _ = final_lstm.predict(X_test)
    inference_time = time.time() - t0
    per_sample_ms = inference_time / len(X_test) * 1000

    h = best_params["hidden_size"]
    nl = best_params.get("num_layers", 1)
    lstm_pi = rf.lstm_total_params(h, d_in, d_out, nl)

    out = {
        "model_type": "lstm",
        "config": best_params,
        "param_info": {
            "trainable": lstm_pi["total"],
            "total": lstm_pi["total"],
            "lstm_layers": lstm_pi["lstm_params"],
            "output": lstm_pi["fc_params"],
        },
        "tune_time": tune_time,
        "best_combo_train_time": float(best_combo_train_time)
        if best_combo_train_time is not None
        else None,
        "final_train_time": final_train_time,
        "inference_time_total": inference_time,
        "inference_time_per_sample_ms": per_sample_ms,
        "n_test_samples": len(X_test),
        "val_mse": best_val_mse,
        "test_mse": test_metrics["mse"],
        "test_mae": test_metrics["mae"],
        "test_r2": test_metrics["r2"],
        "best_epoch": best_epoch,
        "green_metrics": green_metrics,
        "rerun_note": "full-grid rerun for final_v3",
        "rerun_timestamp": datetime.now().isoformat(),
        "resolved_device": lstm_device,
    }

    del final_lstm
    cleanup_torch()
    return out


def build_v3_payload(source_data: Dict[str, Any], new_lstm: Dict[str, Any]) -> Dict[str, Any]:
    payload = deepcopy(source_data)
    meta = payload["metadata"]
    budget_name = meta["budget_name"]
    old_rows = payload["results"].get(budget_name, [])
    esn_rows = [row for row in old_rows if row.get("model_type") != "lstm"]
    esn_rows.append(new_lstm)
    payload["results"][budget_name] = esn_rows

    meta["timestamp"] = datetime.now().isoformat()
    meta["source_result_dir"] = "final_v2"
    meta["lstm_rerun_mode"] = "full-grid"
    meta["lstm_rerun_timestamp"] = datetime.now().isoformat()
    env = meta.get("environment", {})
    env["final_v3_lstm_device"] = new_lstm.get("resolved_device")
    payload["metadata"]["environment"] = env
    return payload


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rf, LSTMModel, tune_lstm_hyperparameters = import_repo_modules(args.repo_root)

    files = iter_source_files(
        source_dir, args.dataset, args.budget, args.seed, args.train_frac
    )
    if args.limit:
        files = files[: args.limit]

    tasks: List[Path] = []
    for source_path in files:
        target_path = output_dir / source_path.name
        if args.resume and has_complete_v3(target_path):
            continue
        tasks.append(source_path)

    print(f"Source dir: {source_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Tasks to run: {len(tasks)}")
    print(f"Device: {args.device}")

    if args.dry_run:
        for path in tasks:
            print(f"  {path.name}")
        return

    total = len(tasks)
    start_all = time.time()
    for idx, source_path in enumerate(tasks, start=1):
        source_data = read_json(source_path)
        meta = source_data["metadata"]
        dataset = meta["dataset"]
        budget_name = meta["budget_name"]
        train_frac = float(meta["train_frac"])
        seed = int(meta["seed"])

        print("\n" + "=" * 72)
        print(
            f"[{idx}/{total}] {dataset} | {budget_name} | train_frac={train_frac} | seed={seed}"
        )
        print("=" * 72)

        t0 = time.time()
        new_lstm = rerun_lstm_full_grid(
            rf=rf,
            LSTMModel=LSTMModel,
            tune_lstm_hyperparameters=tune_lstm_hyperparameters,
            dataset=dataset,
            budget_name=budget_name,
            train_frac=train_frac,
            seed=seed,
            device=args.device,
            verbose=args.verbose,
        )
        payload = build_v3_payload(source_data, new_lstm)
        target_path = output_dir / source_path.name
        with open(target_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

        elapsed = time.time() - t0
        total_elapsed = time.time() - start_all
        mean_per_case = total_elapsed / idx
        remaining = mean_per_case * (total - idx)

        print(
            f"Saved {target_path.name} | R²={new_lstm['test_r2']:.6f} "
            f"MSE={new_lstm['test_mse']:.6f} | tune={new_lstm['tune_time']:.0f}s "
            f"retrain={new_lstm['final_train_time']:.1f}s"
        )
        print(
            f"Elapsed {elapsed/60:.1f} min | Total {total_elapsed/3600:.2f} h | "
            f"ETA {remaining/3600:.2f} h"
        )

    print("\n" + "=" * 72)
    print(
        f"DONE: wrote {total} files to {output_dir} in {(time.time() - start_all)/3600:.2f} h"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
