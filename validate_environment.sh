#!/bin/bash
set -e

echo "🔍 Validating PyReCo vs LSTM Environment"
echo "======================================="

# Check if we're in the right conda environment
if [[ "$CONDA_DEFAULT_ENV" != "pyreco-lstm-evaluation" ]]; then
    echo "❌ Please activate the environment first: conda activate pyreco-lstm-evaluation"
    exit 1
fi

echo "📋 Environment: $CONDA_DEFAULT_ENV ✅"

# Check Python version
PYTHON_VERSION=$(python --version 2>&1 | cut -d' ' -f2)
echo "🐍 Python version: $PYTHON_VERSION"

if [[ "$PYTHON_VERSION" != "3.11.9" ]]; then
    echo "⚠️  Warning: Expected Python 3.11.9, got $PYTHON_VERSION"
fi

# Test core imports
echo ""
echo "📦 Testing core package imports..."

python -c "
import sys
import numpy as np
import scipy
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sklearn
import torch
import pyreco
import reservoirpy

print(f'✅ NumPy {np.__version__}')
print(f'✅ SciPy {scipy.__version__}')
print(f'✅ Pandas {pd.__version__}')
print(f'✅ Matplotlib {matplotlib.__version__}')
print(f'✅ Scikit-learn {sklearn.__version__}')
print(f'✅ PyTorch {torch.__version__}')
pyreco_version = getattr(pyreco, \"__version__\", \"unknown\")
print(f'✅ PyReCo {pyreco_version}')
print(f'✅ ReservoirPy {reservoirpy.__version__}')
"

if [ $? -eq 0 ]; then
    echo ""
    echo "🧪 Running quick functionality test..."

python -c "
import numpy as np
try:
    from pyreco.datasets import load as pyreco_load
except ImportError:
    pyreco_load = None

if pyreco_load is not None:
    # Test new datasets interface
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = pyreco_load(
        'lorenz', n_samples=500, seed=42, n_in=10, n_out=1,
        train_fraction=0.6, val_fraction=0.15, standardize=True
    )

    print(f'✅ Dataset loading: Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}')
    print(f'✅ Data standardization: mean≈{X_train.mean():.3f}, std≈{X_train.std():.3f}')
else:
    print('⚠️ PyReCo datasets.load unavailable; skipping dataset interface test')
    X_train = np.random.randn(100, 10, 3)
    y_train = np.random.randn(100, 1, 3)

# Test basic model creation
from models.pyreco_wrapper import PyReCoStandardModel
model = PyReCoStandardModel(num_nodes=50)
print('✅ PyReCo model creation')

# Test LSTM model
from models.lstm_model import LSTMModel
lstm = LSTMModel(hidden_size=16, num_layers=1)
print('✅ LSTM model creation')
"

    if [ $? -eq 0 ]; then
        echo ""
        echo "🎉 Environment validation PASSED!"
        echo "✅ All packages installed correctly"
        echo "✅ Core functionality working"
        echo "✅ Ready for experiments!"
    else
        echo ""
        echo "❌ Functionality test FAILED!"
        exit 1
    fi
else
    echo ""
    echo "❌ Package import test FAILED!"
    exit 1
fi
