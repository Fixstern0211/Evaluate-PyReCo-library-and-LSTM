# PyReCo vs LSTM Evaluation Framework

A statistically rigorous comparison of Reservoir Computing (PyReCo) and LSTM for chaotic time series prediction. Experimental framework for a Master's thesis.

## Research Objective

Compare **Echo State Networks** (via PyReCo) and **LSTM networks** (via PyTorch) on chaotic time series forecasting under controlled conditions:
- **Parameter budget**: Both models constrained to the same total parameters (Small=1K, Medium=10K, Large=50K)
- **Hyperparameter tuning**: Grid search optimization for both models
- **Data splits**: Identical train/validation/test splits per seed
- **Evaluation metrics**: MSE, MAE, R², NRMSE via shared base class

## Datasets

| Dataset | Type | Dimensions | Length | Characteristics |
|---------|------|-----------|--------|-----------------|
| **Lorenz** | Synthetic 3D ODE | 3 | 5000 | Deterministic chaos, butterfly attractor |
| **Mackey-Glass** | Synthetic DDE (τ=17) | 1 | 5000 | Infinite-dimensional chaos |
| **Santa Fe** | Real-world laser | 1 | ~1000 | Measurement noise, small dataset |

## Model Architectures

### PyReCo (Echo State Network)
- Sparse random reservoir + ridge regression readout
- Two-phase tuning: 5-fold CV pretuning → narrowed grid search per (dataset, budget)
- Key parameters: spectral radius, leakage rate, density, input fraction, reservoir size N (dynamically computed from budget)

### LSTM
- Vanilla LSTM (Hochreiter & Schmidhuber 1997)
- Architecture: LSTM layer(s) → last timestep → fully connected output
- 40-combination grid: num_layers × learning_rate × dropout
- hidden_size computed to match parameter budget per num_layers
- Adam optimizer, validation-based early stopping with checkpoint

## Experiment Design (V2)

V2 experiments fix the budget-matching issue from V1 where PyReCo's `fraction_input` parameter caused under-utilization of the parameter budget. In V2, reservoir size N is dynamically computed to match the target budget exactly.

### Main Experiments (270 runs)
- 3 datasets × 3 budgets × 5 seeds (42–46) × 6 train fractions (0.2–0.8)
- Both models tuned per run → best config retrained on train+val → evaluated on test set
- Results: `results/final_v2/`

### Multi-Step Prediction (270 runs)
- Free-run autoregressive prediction at horizons 1, 5, 10, 20, 50
- Per-seed best configs from main experiments, retrained on train+val
- Results: `results/multi_step_v2/`

### Green Computing (18 conditions)
- Energy consumption (CodeCarbon), CO₂ emissions, peak memory, timing
- 3 datasets × 3 budgets × 2 models
- Results: `results/green_metrics_v2/`

### Statistical Analysis
- Paired t-tests with 95% CI, Cohen's d effect size
- Holm-Bonferroni correction for multiple comparisons
- Results: `results/tables/v2/`

## Project Structure

```
├── models/                              # Model implementations
│   ├── base_model.py                    #   Abstract base (shared metrics)
│   ├── pyreco_wrapper.py                #   PyReCo Standard/Custom wrappers
│   └── lstm_model.py                    #   LSTM with early stopping + checkpoint
│
├── experiments/                         # Experiment scripts
│   ├── run_final_v2.py                  #   V2 main experiments (270 runs)
│   ├── run_multi_step_v2.py             #   V2 multi-step prediction
│   ├── run_pretuning_v2.py              #   V2 PyReCo pretuning (5-fold CV)
│   ├── measure_green_metrics_v2.py      #   Energy/carbon measurement
│   ├── test_model_scaling.py            #   Core single-experiment logic
│   └── optimized_grids.py              #   Pretuning-derived search grids
│
├── analysis/                            # Post-processing
│   ├── statistical_analysis_v2.py       #   Single-step: t-tests, Cohen's d
│   ├── statistical_analysis_multistep_v2.py  # Multi-step stats + Holm correction
│   ├── inject_lstm_into_v2.py           #   Inject V1 LSTM results into V2 files
│   └── experiment_redesign_v2.md        #   V2 design doc + impact analysis
│
├── scripts/                             # Thesis output
│   └── generate_thesis_figures.py       #   Publication-quality PDF figures
│
├── src/utils/                           # Shared utilities
│   ├── load_dataset.py                  #   Dataset loaders + windowing
│   ├── process_datasets.py              #   Train/val/test splitting
│   └── evaluation.py                    #   Multi-step evaluation helpers
│
├── results/                             # Experiment outputs (JSON)
│   ├── final_v2/                        #   V2 main experiments
│   ├── pretuning_v2/                    #   V2 pretuning results
│   ├── multi_step_v2/                   #   V2 multi-step prediction
│   ├── green_metrics_v2/                #   Energy/carbon metrics
│   └── tables/v2/                       #   Statistical analysis CSVs
│
├── docs/                                # Documentation
├── requirements.txt                     # Python dependencies
└── thesis/                              # LaTeX thesis source (separate)
```

## Quick Start

```bash
# 1. Create virtual environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Run pretuning (required before main experiments)
python experiments/run_pretuning_v2.py

# 3. Run main experiments
python experiments/run_final_v2.py

# 4. Run multi-step prediction
python experiments/run_multi_step_v2.py

# 5. Run statistical analysis + generate figures
python analysis/statistical_analysis_v2.py
python analysis/statistical_analysis_multistep_v2.py
python scripts/generate_thesis_figures.py
```

## Requirements

- Python 3.11+
- PyTorch 2.0+
- PyReCo >= 1.1
- See `requirements.txt` for full dependencies
