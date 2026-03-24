# Project Structure

## Overview
This project evaluates PyReCo (Reservoir Computing) library against LSTM models on chaotic time series prediction tasks.

## Directory Structure

```
Evaluate-PyReCo-library-and-LSTM/
│
├── experiments/              # Experiment scripts
│   ├── run_final_experiments.py      # Main experiment runner (45 experiments)
│   ├── test_model_scaling.py         # Single experiment testing
│   └── monitor_experiments.py        # Experiment monitoring utility
│
├── scripts/                  # Analysis and utility scripts
│   ├── analyze_final_results.py      # Results analysis (generates reports)
│   ├── analyze_comprehensive_results.py
│   └── analyze_results.py
│
├── src/                      # Source code
│   └── utils/
│       ├── load_dataset.py           # Dataset loading utilities
│       ├── process_datasets.py       # Dataset processing
│       └── train_lstm.py             # LSTM training utilities
│
├── models/                   # Model definitions
│   ├── pyreco_wrapper.py             # PyReCo model wrapper
│   └── lstm_model.py                 # LSTM model implementation
│
├── results/                  # All experimental results
│   ├── final/                        # Final experiment results (45 experiments)
│   │   ├── results_lorenz_*.json     # Lorenz dataset results (15)
│   │   ├── results_mackeyglass_*.json # Mackey-Glass results (15)
│   │   ├── results_santafe_*.json    # Santa Fe results (15)
│   │   ├── analysis_report.txt       # Comprehensive analysis report
│   │   ├── results_summary.csv       # CSV summary for analysis
│   │   └── experiment_progress.json  # Experiment tracking
│   ├── pretuning/                    # Hyperparameter pretuning results
│   └── archived/                     # Old/archived results
│
├── tests/                    # Test scripts
│   ├── test_fairness.py              # Fairness testing
│   ├── test_consistency.py           # Reproducibility tests
│   ├── test_gpu_acceleration.py      # GPU performance tests
│   └── ...
│
├── docs/                     # Documentation
│   ├── EXPERIMENTAL_RESULTS_SPECIFICATION.md
│   ├── LITERATURE_SUPPORT_TUNING_STRATEGY.md
│   └── archived/                     # Old documentation
│
├── logs/                     # Log files
│
├── archive/                  # Deprecated/old files
│   └── backup/                       # Old backup directory
│
├── .venv/                    # Virtual environment
├── .git/                     # Git repository
├── .gitignore
└── README.md                 # Project readme

```

## Experimental Results

### Summary (45 experiments completed)
- **Datasets**: Lorenz, Mackey-Glass, Santa Fe
- **Seeds**: 42, 43, 44, 45, 46 (5 seeds)
- **Train Ratios**: 0.5, 0.7, 0.9 (3 ratios)
- **Parameter Budgets**: Small (1K), Medium (10K), Large (30K)
- **Total Runtime**: ~19 hours
- **PyReCo Win Rate**: 67.4% (91/135 comparisons)

### Key Findings
1. **PyReCo excels at small parameter budgets** (2x better MSE)
2. **LSTM catches up with more parameters** (medium/large budgets)
3. **Dataset-dependent performance**:
   - Lorenz: PyReCo dominates (~1000x better)
   - Mackey-Glass: PyReCo wins consistently
   - Santa Fe: LSTM performs better
4. **Training time**: PyReCo is 2.4x faster on average

## Usage

### Running Experiments
```bash
# Full experiment suite
python experiments/run_final_experiments.py

# Single experiment test
python experiments/test_model_scaling.py

# Monitor running experiments
python experiments/monitor_experiments.py
```

### Analyzing Results
```bash
# Analyze all results
python scripts/analyze_final_results.py

# View report
cat results/final/analysis_report.txt

# Load CSV for custom analysis
# results/final/results_summary.csv
```

## Contributing to PyReCo

This project also includes a contribution to the PyReCo open-source library:
- **Issue #71**: Implementation of chaotic datasets (Lorenz, Mackey-Glass)
- **Location**: Separate PyReCo repository
- **Status**: Pull Request submitted and under review

