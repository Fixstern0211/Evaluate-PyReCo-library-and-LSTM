from itertools import product
import numpy as np
from pyreco.models import ReservoirComputer as RC
from pyreco.optimizers import RidgeSK
from datetime import datetime
from typing import List, Tuple


def train_rc(
    Xtr, Ytr, Xva, Yva,
    *,
    cv_splits: int = 0,
    chosen_num_nodes: int,
    fraction_output: float,
    # 显式默认值（若 grid 未覆盖就用它们）
    default_spec_rad: float = 1.0,
    default_leakage: float = 0.3,
    default_density: float = 0.1,
    default_activation: str = "tanh",
    default_fraction_input: float = 0.5,
    default_alpha: float = 1.0,   # 若提供且 RidgeSK 可用，则启用
    grid: dict,
    score: str = "mse",                   # 或 "r2"
):
    metric_names = ['mse', 'mae', 'r2']
    if score not in metric_names:
        raise ValueError(f"score must be in {metric_names}, current is: {score!r}")

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

        
        nowtime = datetime.now().strftime("%Y%m%d-%H%M%S")
        print(f"time: {nowtime} with trying combo: {combo}")

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


from typing import List, Tuple
import numpy as np
from itertools import product
from datetime import datetime

try:
    from sklearn.linear_model import Ridge as RidgeSK
except Exception:
    RidgeSK = None

from pyreco.models import ReservoirComputer as RC
from process_datasets import timeseries_cv_split


def train_rc_timeseries_cv(
    Xtr, Ytr,  # ===【改动】全部训练数据（不需要预先分割train/val）
    *,
    chosen_num_nodes: int,
    fraction_output: float,
    # 显式默认值
    default_spec_rad: float = 1.0,
    default_leakage: float = 0.3,
    default_density: float = 0.1,
    default_activation: str = "tanh",
    default_fraction_input: float = 0.5,
    default_alpha: float = 1.0,
    grid: dict,
    n_splits: int = 5,  # ===【新增】时序CV的折数
    score: str = "mse",
):
    """
    使用时序交叉验证进行超参数搜索的RC训练函数

    与原train_rc的主要区别：
    1. 不需要传入Xva, Yva（验证集）- 函数内部会自动进行时序分割
    2. 使用时序CV在Xtr上评估每个超参数组合（保持时间顺序）
    3. 避免PyReCo自带cross_val的随机打乱问题

    参数：
        Xtr, Ytr: 全部训练数据
        chosen_num_nodes: 储备池节点数
        fraction_output: 输出连接比例
        default_*: 默认超参数值
        grid: 要搜索的超参数网格
        n_splits: 时序CV的折数（默认5）
        score: 评估指标 ('mse', 'mae', 'r2')

    返回：
        best_params: 最佳超参数字典
        best_metrics: 最佳CV评分（包含cv_mean, cv_std, cv_scores）
        final_model: 用最佳参数在全部Xtr上训练的最终模型
    """
    metric_names = ['mse', 'mae', 'r2']
    if score not in metric_names:
        raise ValueError(f"score must be in {metric_names}, current is: {score!r}")

    keys = list(grid.keys())
    vals = [grid[k] for k in keys]
    if not keys:
        keys, vals = [], [()]

    maximize = (score == "r2")
    best_score = -float("inf") if maximize else float("inf")
    best_params, best_metrics, final_model = None, None, None

    def _mk_optimizer(alpha_val):
        if RidgeSK is not None and alpha_val is not None:
            return RidgeSK(alpha=float(alpha_val))
        return "ridge"

    # ===【核心改动】遍历grid，用时序CV评估每个组合
    for combo in product(*vals):
        # 构建超参数
        spec_rad     = default_spec_rad
        leakage_rate = default_leakage
        density      = default_density
        activation   = default_activation
        fraction_in  = default_fraction_input
        alpha_val    = default_alpha

        nowtime = datetime.now().strftime("%Y%m%d-%H%M%S")
        print(f"[{nowtime}] Trying combo: {combo}")

        for k, v in zip(keys, combo):
            if   k == "spec_rad":        spec_rad     = float(v)
            elif k == "leakage_rate":    leakage_rate = float(v)
            elif k == "density":         density      = float(v)
            elif k == "activation":      activation   = str(v)
            elif k == "fraction_input":  fraction_in  = float(v)
            elif k == "alpha":           alpha_val    = float(v)
            elif k == "fraction_output": fraction_output = float(v)
            elif k == "num_nodes":       chosen_num_nodes = int(v)
            else:
                print(f"[warn] 未识别的超参键：{k}（已忽略）")

        optimizer = _mk_optimizer(alpha_val)

        # ===【新增】时序交叉验证评估
        cv_scores = []  # 存储每个fold的得分
        splits = timeseries_cv_split(Xtr, Ytr, n_splits)

        for fold_idx, (X_train, y_train, X_val, y_val) in enumerate(splits):
            # 为每个fold创建新模型（避免状态污染）
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

            model.fit(X_train, y_train)
            vals_list = model.evaluate(x=X_val, y=y_val, metrics=metric_names)
            metrics = dict(zip(metric_names, vals_list))
            fold_score = metrics[score]
            cv_scores.append(fold_score)

            print(f"  Fold {fold_idx+1}/{len(splits)}: {score}={fold_score:.6f}")

        # 计算CV的平均分和标准差
        mean_cv_score = np.mean(cv_scores)
        std_cv_score = np.std(cv_scores)
        print(f"  CV Mean {score}: {mean_cv_score:.6f} (±{std_cv_score:.6f})")

        # 判断是否为最佳
        is_better = (mean_cv_score > best_score) if maximize else (mean_cv_score < best_score)
        if is_better:
            best_score = mean_cv_score
            best_metrics = {
                'cv_mean': mean_cv_score,
                'cv_std': std_cv_score,
                'cv_scores': cv_scores,  # 保留每个fold的得分
            }
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

    print(f"\n{'='*60}")
    print(f"最佳超参数: {best_params}")
    print(f"最佳CV得分: {best_metrics['cv_mean']:.6f} (±{best_metrics['cv_std']:.6f})")
    print(f"{'='*60}\n")

    # ===【保持不变】用全部训练数据重训最终模型
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
    final_model.fit(Xtr, Ytr)

    return best_params, best_metrics, final_model
