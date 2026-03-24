# PyReCo vs LSTM Evaluation Framework

A comprehensive, statistically rigorous comparison of Reservoir Computing (PyReCo) and LSTM for chaotic time series prediction. This project serves as the experimental framework for a Master's thesis evaluating the two architectures under controlled, fair conditions.

## Research Objective

Compare **Echo State Networks** (via the PyReCo library) and **LSTM networks** (via PyTorch) on chaotic time series forecasting, controlling for:
- **Parameter budget**: Both models are constrained to the same number of parameters (1K / 10K / 50K)
- **Hyperparameter tuning**: Both models undergo grid search optimization
- **Data splits**: Identical train/validation/test splits across all experiments
- **Evaluation metrics**: Same metrics (MSE, MAE, R²) computed by a shared base class

## Datasets

| Dataset | Type | Origin | Length | Characteristics |
|---------|------|--------|--------|-----------------|
| **Lorenz** | Synthetic | 3D ODE system | 5000 | Deterministic chaos, noise-free, butterfly attractor |
| **Mackey-Glass** | Synthetic | Delay differential equation (τ=17) | 5000 | Infinite-dimensional chaos, long-range dependencies |
| **Santa Fe** | Real-world | NH₃ laser intensity measurements | ~1000 | Measurement noise, non-stationarity, small dataset |

## Model Architectures

### PyReCo (Echo State Network)
- Reservoir with sparse random connectivity + ridge regression readout
- Hyperparameters tuned via **two-phase approach**: 5-fold CV pretuning on broad grid → dataset-specific narrowed grid (4–6 combinations)
- Key parameters: spectral radius, leakage rate, density, input fraction

### LSTM
- Vanilla LSTM (Hochreiter & Schmidhuber 1997, with forget gate from Gers et al. 2000)
- Architecture: LSTM layer(s) → last timestep → fully connected output
- **40-combination grid search**: num_layers (1,2) × learning_rate (0.0005–0.01) × dropout (0.0–0.3)
- hidden_size computed per num_layers to match parameter budget
- Training: Adam optimizer, validation-based early stopping with best model checkpoint

## Experiment Design

### Main Experiments (90 runs)
- 3 datasets × 5 random seeds (42–46) × 6 train ratios (0.2, 0.3, 0.5, 0.6, 0.7, 0.8)
- Each run tests 3 parameter scales (Small=1K, Medium=10K, Large=50K)
- Both models tuned per run → best config retrained on train+val → evaluated on test set

### Additional Experiments

| Experiment | Description | Count |
|------------|-------------|-------|
| **Pre-tuning** | 5-fold CV grid search for PyReCo optimal parameters per dataset | 3 datasets × 3 budgets |
| **Data efficiency** | Performance vs. data length (1K–10K samples) | 270 |
| **Multi-step prediction** | Free-run prediction at horizons 1, 5, 10, 20, 50 steps | 270 |
| **Inference benchmark** | Single-sample and batch latency measurement | 12 |
| **Green computing** | Energy consumption, carbon footprint, peak memory | 9 |
| **Statistical analysis** | Paired t-tests, Wilcoxon signed-rank, Cohen's d | Post-processing |

## Project Structure

```
├── models/                          # Model implementations
│   ├── base_model.py                #   Abstract base class (shared metrics)
│   ├── pyreco_wrapper.py            #   PyReCo Standard/Custom wrappers
│   └── lstm_model.py                #   LSTM with val-based early stopping + checkpoint
│
├── experiments/                     # Experiment scripts
│   ├── test_model_scaling.py        #   Core: single experiment (dataset × seed × ratio)
│   ├── run_final_experiments.py     #   Runner: all 90 main experiments
│   ├── run_data_efficiency_experiments.py
│   ├── run_multi_step_experiments.py
│   ├── run_optimized_pretuning.py   #   PyReCo 5-fold CV pretuning
│   ├── test_inference_benchmark.py  #   Inference latency benchmark
│   └── test_autoregressive.py       #   Free-run autoregressive evaluation
│
├── analysis/                        # Post-processing analysis
│   ├── statistical_analysis.py      #   t-tests, Wilcoxon, Cohen's d
│   ├── generate_main_experiment_tables.py
│   ├── generate_data_efficiency_tables.py
│   ├── generate_multi_step_tables.py
│   ├── green_computing_analysis.py
│   └── decision_guide_generator.py
│
├── scripts/                         # Thesis-specific scripts
│   └── generate_thesis_figures.py   #   Publication-quality PDF figures
│
├── src/utils/                       # Shared utilities
│   ├── load_dataset.py              #   Lorenz, Mackey-Glass, Santa Fe loaders
│   ├── process_datasets.py          #   Windowing, train/val/test split
│   └── evaluation.py                #   Multi-step evaluation helpers
│
├── results/                         # Experiment outputs (JSON)
│   ├── final/                       #   Main experiments (current run)
│   ├── pretuning/                   #   Pre-tuning results
│   ├── data_efficiency/             #   Data efficiency experiments
│   ├── multi_step/                  #   Multi-step prediction experiments
│   ├── inference_benchmark/         #   Inference latency results
│   ├── green_metrics/               #   Energy/carbon metrics
│   └── tables/                      #   Generated CSV summary tables
│
├── docs/                            # Documentation
│   ├── LITERATURE_SUPPORT_TUNING_STRATEGY.md  # Literature references & tuning rationale
│   ├── EVALUATION_PROTOCOL.md                 # Evaluation methodology
│   ├── UPDATED_EXPERIMENTS_PLAN.md            # Full experiment roadmap
│   ├── EXPERIMENTAL_RESULTS_SPECIFICATION.md  # JSON output format spec
│   ├── DECISION_GUIDE.md                      # Model selection guide
│   └── RESULTS_SUMMARY.md                     # Results overview
│
├── environment.yml                  # Conda environment specification
├── requirements-lock.txt            # Pinned package versions
└── setup_environment.sh             # One-step environment setup
```

## Quick Start

```bash
# 1. Setup environment
chmod +x setup_environment.sh validate_environment.sh
./setup_environment.sh
./validate_environment.sh

# 2. Run a quick test (single dataset, single seed, ~1 hour)
cd experiments
python run_final_experiments.py --quick

# 3. Run full experiments (~90 hours)
nohup python -u run_final_experiments.py > ../logs/exp_final.log 2>&1 &
```

## Key Design Decisions

- **Parameter budget matching**: Models are compared at equal total parameter counts, not equal architecture. LSTM hidden_size is computed to match the budget for each num_layers.
- **Two-phase training**: Phase 1 (grid search) selects best hyperparameters using validation data. Phase 2 retrains on combined train+val data for final evaluation on test set.
- **PyReCo pretuning**: ESNs are highly sensitive to hyperparameters (Jiang et al. 2022). A broad 5-fold CV pretuning narrows the search space per dataset before main experiments.
- **LSTM early stopping**: Validation-based early stopping with best model checkpoint prevents overfitting and ensures fair comparison with PyReCo's closed-form ridge regression.

## Requirements

- Python 3.11+
- PyTorch 2.0+
- PyReCo 1.1.0
- See `environment.yml` for full dependencies

## Documentation

- [Literature Support & Tuning Strategy](docs/LITERATURE_SUPPORT_TUNING_STRATEGY.md) — ESN/LSTM literature, hyperparameter rationale
- [Evaluation Protocol](docs/EVALUATION_PROTOCOL.md) — Metrics, prediction modes, methodology
- [Experiment Plan](docs/UPDATED_EXPERIMENTS_PLAN.md) — Full project roadmap
- [Results Specification](docs/EXPERIMENTAL_RESULTS_SPECIFICATION.md) — JSON output format
