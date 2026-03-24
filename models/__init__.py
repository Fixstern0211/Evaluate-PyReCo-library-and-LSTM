"""
Models package - Unified model interface

This package provides a unified interface for all models,
ensuring fair comparison between PyReCo and LSTM.

Available models:
- BaseTimeSeriesModel: Abstract base class
- LSTMModel: PyTorch-based LSTM
- PyReCoStandardModel: PyReCo with standard API
- PyReCoCustomModel: PyReCo with custom layer API
- PyReCoMultiLayerModel: Multi-layer PyReCo model

Helper functions:
- compare_models: Batch compare multiple models
- tune_lstm_hyperparameters: LSTM hyperparameter search
- tune_pyreco_hyperparameters: PyReCo hyperparameter search
- tune_pyreco_with_cv: PyReCo tuning with time series CV
"""

from .base_model import BaseTimeSeriesModel, compare_models
from .lstm_model import LSTMModel, tune_lstm_hyperparameters
from .pyreco_wrapper import (
    PyReCoStandardModel,
    tune_pyreco_hyperparameters,
    tune_pyreco_with_cv
)
from .pyreco_custom_wrapper import (
    PyReCoCustomModel,
    PyReCoMultiLayerModel
)

__all__ = [
    # Base class and utilities
    'BaseTimeSeriesModel',
    'compare_models',

    # LSTM models
    'LSTMModel',
    'tune_lstm_hyperparameters',

    # PyReCo models
    'PyReCoStandardModel',
    'PyReCoCustomModel',
    'PyReCoMultiLayerModel',

    # Hyperparameter tuning
    'tune_pyreco_hyperparameters',
    'tune_pyreco_with_cv',
]
