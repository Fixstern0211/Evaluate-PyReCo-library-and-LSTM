#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Advanced experiments using pyReCo.custom_models (layer-by-layer assembly).
- Modes:
    1) closed_loop: train teacher-forced 1-step RC, then do iterative multi-step rollouts.
    2) pruning: train RC, then prune nodes to smaller sizes and evaluate perf-size curve.
- Datasets: Lorenz / Mackey-Glass / Santa Fe (reservoirpy.datasets)
- Train/test split (rolling-origin), standardization
- Trainable-parameter budget alignment via fraction_output

Example:
  # Closed-loop 200 steps on Lorenz
  python run_advanced_with_custom_models.py --mode closed_loop --dataset lorenz --horizon 200 --budget 10000 --n-in 100

  # Pruning curve on Santa Fe
  python run_advanced_with_custom_models.py --mode pruning --dataset santafe --budget 10000 --n-in 100 --prune-ratio 0.5
"""

import argparse
import json
import math
import os
import random
from datetime import datetime

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# ---- pyReCo (low-level) ----
from pyreco.custom_models import RC as CustomRC
from pyreco.layers import InputLayer, RandomReservoirLayer, ReadoutLayer

# optional pruning (if available)
try:
    from pyreco.pruning import prune_model
    HAS_PRUNING = True
except Exception:
    HAS_PRUNING = False

from reservoirpy import datasets as rds


# Build custom rc model
def build_custom_rc(n_in, n_features, n_out, num_nodes, fraction_output,
             spec_rad=0.95, leakage_rate=0.3, density=0.05,
             activation="tanh", ridge_alpha=1e-3, discard_transients=20):
    """
    Assemble an RC model using the custom_models API.

    ===[FIX] input_shape should be (n_in, n_features) or (None, n_features)

    In PyReCo's custom_models, InputLayer.input_shape format is:
      - input_shape[0]: number of time steps (n_timesteps), can be None for variable length
      - input_shape[1]: feature dimension (n_features / n_states)

    The actual input data shape is (batch, n_timesteps, n_features)

    Parameters:
        n_in: input window length (number of time steps)
        n_features: number of features per time step
        n_out: output dimension
        num_nodes: number of reservoir nodes
        fraction_output: fraction of output connections
        others: RC hyperparameters
    """
    m = CustomRC()
    # input_shape is (n_timesteps, n_features), not including batch dimension
    m.add(InputLayer(input_shape=(n_in, n_features)))
    m.add(RandomReservoirLayer(nodes=num_nodes, density=density,
                               activation=activation,
                               leakage_rate=leakage_rate, spec_rad=spec_rad))
    m.add(ReadoutLayer(output_shape=(None, n_out), fraction_out=fraction_output))
    m.compile(optimizer="ridge",
              metrics=["mse"], discard_transients=discard_transients)
    return m

def iterative_closed_loop(model, init_window, horizon):
    """
    model: trained teacher-forced model (1-step)
    init_window: [1, n_in, D] last window from test set
    returns rollout [horizon, D]
    """
    n_in = init_window.shape[1]
    D = init_window.shape[-1]
    window = init_window.copy()  # [1, n_in, D]
    outs = []
    for _ in range(horizon):
        y1 = model.predict(window)  # [1, 1, D] if trained with n_out=1
        outs.append(y1[0, 0])
        # shift window: drop oldest, append prediction
        new_frame = y1
        window = np.concatenate([window[:, 1:, :], new_frame], axis=1)
    return np.stack(outs, axis=0)  # [H, D]

def save_json(outdir, fname, obj):
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, fname), "w") as f:
        json.dump(obj, f, indent=2)

