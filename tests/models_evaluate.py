import argparse
import os
import json
import math
from datetime import datetime
from timeit import main
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.utils.load_dataset import load_data, set_seed
from src.utils import plot, process_datasets
from src.utils.node_number import best_num_nodes_and_fraction_out, compute_readout_F_from_budget
from src.utils import train_pyreco_model
# ===【注释掉】不再使用PyReCo自带的cross_val（因为它会打乱时序）
# from itertools import product
# from pyreco.cross_validation import cross_val
# from pyreco.models import ReservoirComputer as RC


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", type=int, default=5000)
    ap.add_argument("--train-frac", type=float, default=0.6)
    ap.add_argument("--n-in", type=int, default=100)
    ap.add_argument("--budget", type=int, default=10000)
    ap.add_argument("--num-nodes", type=int, default=800)
    ap.add_argument("--spec-rad", type=float, default=0.95)
    ap.add_argument("--leakage", type=float, default=0.3)
    ap.add_argument("--density", type=float, default=0.05)
    ap.add_argument("--ridge-alpha", type=float, default=1e-3)
    ap.add_argument("--discard-transients", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=str, default="results_advanced")

    # closed-loop specific
    ap.add_argument("--horizon", type=int, default=200, help="Closed-loop rollout steps")

    args = ap.parse_args()

    # 1) Data
    # 1) Lorenz（默认参数，长度 5000）
    data, meta = load_data("lorenz", length=args.length, seed=args.seed)

    # 2) Lorenz（自定义参数）
    # data, meta = load_data("lorenz", length=8000, rho=30.0, sigma=10.0, beta=8.0/3.0, x0=(1.0, 1.0, 1.0))

    # # 3) Mackey–Glass（指定 tau 与 seed）
    # data, meta = load_data("mackeyglass", length=10000, tau=17, seed=42)

    # # 4) Santa Fe（随机窗口裁剪 2000 点，并加少量噪声）
    # data, meta = load_data("santafe", length=2000, random_window=True, seed=7, noise=1e-3)

    Dout = meta["Dout"]

    # 2) Split
    train, test, split = process_datasets.split_datasets(data, args.train_frac)

    # ===【改动】使用时序CV时不需要预先分割train/val
    # 原代码：手动分割85/15
    # n_tr = int(0.85 * len(train))
    # series_train = train[:n_tr]
    # series_val = train[n_tr:]
    # print(f"shape of train: {series_train.shape}, val: {series_val.shape}, test: {test.shape}")

    # 新代码：使用全部train，CV函数会自动时序分割
    print(f"shape of train: {train.shape}, test: {test.shape}")

    # 3) Standardize
    scX = StandardScaler().fit(train)   # 仅用训练集估计均值/方差
    series_train = scX.transform(train)
    # ===【改动】不再需要series_val
    # series_val = scX.transform(series_val)
    series_test  = scX.transform(test)
    print(f"shape of standardized train: {series_train.shape}, test: {series_test.shape}")

    # 4) Windows (teacher forcing, 1-step)
    Xtr, Ytr = process_datasets.sliding_window(series_train, args.n_in, 1)
    # ===【改动】不再需要Xva, Yva
    # Xva, Yva = process_datasets.sliding_window(series_val, args.n_in, 1)
    Xte, Yte = process_datasets.sliding_window(series_test, args.n_in, 1)
    print(f"Train/Test window sizes: {Xtr.shape}/{Xte.shape}")


    # 5) Budget -> fraction_output
    Ftarget = compute_readout_F_from_budget(args.budget, Dout)
    chosen_num_nodes, chosen_fraction_out, chosen_F_real = best_num_nodes_and_fraction_out(Ftarget, [args.num_nodes])
    print(f"Chosen num_nodes: {chosen_num_nodes}, fraction_output: {chosen_fraction_out:.6f}, F_real: {chosen_F_real} (target was {Ftarget})")


    # 6) Build & fit model with Time Series CV
    grid = {
    "spec_rad":       [0.8, 1.0, 1.2],
    "leakage_rate":   [0.2, 0.4, 0.5],
    "density":        [0.5, 0.6, 0.8],
    # "alpha":          [1e-4, 1e-2, 1e0],   # 可选，前提是库里有 RidgeSK
    # 如需同步搜索输入/输出比例也行：
    # "fraction_input":  [0.5, 0.75, 1.0],
    # "fraction_output": [0.2, 0.28, 0.35],
    # "num_nodes":       [800, 1200, 1600],
    }

    # ===【核心改动】使用时序交叉验证函数
    # 原代码：使用train_rc，需要手动传入Xva, Yva
    # best_params, best_metrics, final_model = train_pyreco.train_rc(
    #     Xtr, Ytr, Xva, Yva,
    #     ...
    # )

    # 新代码：使用train_rc_timeseries_cv，不需要验证集
    best_params, best_metrics, final_model = train_pyreco_model.train_rc_timeseries_cv(
        Xtr, Ytr,  # ===【改动】只传训练数据，不需要Xva, Yva
        chosen_num_nodes=chosen_num_nodes,
        default_spec_rad=1.0,
        default_leakage=0.3,
        default_density=0.1,
        default_activation="tanh",
        default_fraction_input=0.5,
        fraction_output=chosen_fraction_out,
        default_alpha=None,        # 不设则用 "ridge" 默认；或给 1.0 作基线
        grid=grid,
        n_splits=5,                # ===【新增】时序CV的折数，可调整为3-7
        score="mse",               # 或 "r2"
    )

    print("\n" + "="*100)
    print("Best params:", best_params)
    print("Best CV metrics:", best_metrics)
    print("="*100 + "\n")


    # 6.5) 评估测试集
    Ypred_test = final_model.predict(Xte)


    # 7) Mode-specific experiments
    # stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    # meta = {
    #     "timestamp": stamp,
    #     "mode": args.mode,
    #     "dataset": args.dataset,
    #     "train_frac": args.train_frac,
    #     "n_in": args.n_in,
    #     "budget": args.budget,
    #     "F_target": Ftarget,
    #     "num_nodes": args.num_nodes,
    #     "fraction_output": round(chosen_fraction_out, 6),
    #     "spec_rad": args.spec_rad,
    #     "leakage": args.leakage,
    #     "density": args.density,
    #     "ridge_alpha": args.ridge_alpha,
    #     "seed": args.seed,
    # }



if __name__ == "__main__":
    main()