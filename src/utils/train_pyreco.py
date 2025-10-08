from itertools import product
import numpy as np
from pyreco.models import ReservoirComputer as RC
from pyreco.optimizers import RidgeSK
import time


def train_rc(
    Xtr, Ytr, Xva, Yva,
    *,
    chosen_num_nodes: int,
    fraction_output: float,
    # 显式默认值（若 grid 未覆盖就用它们）
    default_spec_rad: float = 1.0,
    default_leakage: float = 0.3,
    default_density: float = 0.1,
    default_activation: str = "tanh",
    default_fraction_input: float = 0.5,
    default_alpha: float | None = None,   # 若提供且 RidgeSK 可用，则启用
    grid: dict,
    score: str = "mse",                   # 或 "r2"
):
    metric_names = ['mse', 'mae', 'r2']
    if score not in metric_names:
        raise ValueError(f"score 必须在 {metric_names} 中，当前为 {score!r}")

    # 要遍历的键及其候选值（grid 有啥就扫啥）
    keys = list(grid.keys())
    vals = [grid[k] for k in keys]
    if not keys:
        # 没有 grid 就用默认值跑 1 次
        keys, vals = [], [()]

    maximize = (score == "r2")
    best_score = -float("inf") if maximize else float("inf")
    best_params, best_metrics, final_model = None, None, None

    def _mk_optimizer(alpha_val):
        # 优先使用传入 alpha 或 grid 提供的 alpha；否则用 "ridge"
        if RidgeSK is not None and alpha_val is not None:
            return RidgeSK(alpha=float(alpha_val))
        return "ridge"

    for combo in product(*vals):
        # 先用默认值
        spec_rad     = default_spec_rad
        leakage_rate = default_leakage
        density      = default_density
        activation   = default_activation
        fraction_in  = default_fraction_input
        alpha_val    = default_alpha

        start_time = time.time()
        print(f"time: {start_time} with trying combo: {combo}")

        # 用 grid 覆盖（若提供）
        for k, v in zip(keys, combo):
            if   k == "spec_rad":        spec_rad     = float(v)
            elif k == "leakage_rate":    leakage_rate = float(v)
            elif k == "density":         density      = float(v)
            elif k == "activation":      activation   = str(v)
            elif k == "fraction_input":  fraction_in  = float(v)
            elif k == "alpha":           alpha_val    = float(v)
            elif k == "fraction_output": fraction_output = float(v)  # 允许在 grid 动态调
            elif k == "num_nodes":       chosen_num_nodes = int(v)    # 允许在 grid 动态调
            else:
                print(f"[warn] 未识别的超参键：{k}（已忽略）")

        optimizer = _mk_optimizer(alpha_val)

        # —— 显式构造 RC（不使用 **params）——
        model = RC(
            num_nodes=chosen_num_nodes,
            density=density,
            activation=activation,
            leakage_rate=leakage_rate,
            spec_rad=spec_rad,
            fraction_input=fraction_in,
            fraction_output=fraction_output,
            optimizer=optimizer,
        )

        # 训练 + 验证评分
        model.fit(Xtr, Ytr)
        vals_list = model.evaluate(x=Xva, y=Yva, metrics=metric_names)
        metrics = dict(zip(metric_names, vals_list))

        print(metrics)

        cur = metrics[score]
        is_better = (cur > best_score) if maximize else (cur < best_score)
        if is_better:
            best_score = cur
            best_metrics = metrics
            best_params = dict(
                num_nodes=chosen_num_nodes,
                density=density,
                activation=activation,
                leakage_rate=leakage_rate,
                spec_rad=spec_rad,
                fraction_input=fraction_in,
                fraction_output=fraction_output,
                optimizer=("RidgeSK" if (RidgeSK and alpha_val is not None) else "ridge"),
                alpha=alpha_val,
            )

    if best_params is None:
        raise RuntimeError("No valid parameter combination found.")

    # —— 用 train+val 重训最终模型（同样显式传参）——
    Xfull = np.concatenate([Xtr, Xva], axis=0)
    Yfull = np.concatenate([Ytr, Yva], axis=0)
    final_model = RC(
        num_nodes=best_params["num_nodes"],
        density=best_params["density"],
        activation=best_params["activation"],
        leakage_rate=best_params["leakage_rate"],
        spec_rad=best_params["spec_rad"],
        fraction_input=best_params["fraction_input"],
        fraction_output=best_params["fraction_output"],
        optimizer=_mk_optimizer(best_params.get("alpha")),
    )
    final_model.fit(Xfull, Yfull)
    return best_params, best_metrics, final_model
