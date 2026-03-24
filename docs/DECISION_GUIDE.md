# PyReCo vs LSTM Model Selection Decision Guide

> Auto-generated on 2026-01-26 23:17
> Based on experimental results analysis

## Executive Summary

- **Dataset Win Rate**: PyReCo 1 / LSTM 2
- **Training Speed**: PyReCo is 23-1x faster than LSTM
- **Memory Efficiency**: LSTM uses only 0.1% of PyReCo's memory

## Quick Decision Table

| Scenario | Recommendation | Confidence |
|----------|----------------|------------|
| Chaotic dynamical systems (mackeyglass) | **PyReCo** | 🟢 high |
| Real-world measurement data (santafe) | **LSTM** | 🟢 high |
| Parameter budget < 10K and training speed matters | **PyReCo** | 🟢 high |
| Memory-constrained environment (e.g., embedded devices) | **LSTM** | 🟢 high |
| Small-scale tasks with energy efficiency priority | **PyReCo** | 🟡 medium |
| Rapid prototyping / initial experiments | **PyReCo** | 🟢 high |
| Production deployment requiring interpretability | **Requires evaluation** | 🔴 low |

## Analysis by Dataset

### LORENZ

- **Recommendation**: 🥈 **LSTM**
- **PyReCo R²**: 0.8995
- **LSTM R²**: 0.9479
- **Training Speedup**: 1.1x (PyReCo faster)
- **Confidence**: medium

### MACKEYGLASS

- **Recommendation**: 🏆 **PyReCo**
- **PyReCo R²**: 0.9877
- **LSTM R²**: 0.9576
- **Training Speedup**: 1.3x (PyReCo faster)
- **Confidence**: medium

### SANTAFE

- **Recommendation**: 🥈 **LSTM**
- **PyReCo R²**: 0.7658
- **LSTM R²**: 0.9603
- **Training Speedup**: 1.2x (PyReCo faster)
- **Confidence**: high

## Analysis by Parameter Scale

| Scale | Params | Winner | PyReCo R² | LSTM R² | Speedup |
|-------|--------|--------|-----------|---------|---------|
| small | ~1K | LSTM | 0.8983 | 0.9284 | 22.9x |
| medium | ~10K | LSTM | 0.8786 | 0.9655 | 2.4x |
| large | ~50K | LSTM | 0.8762 | 0.9720 | 0.5x |

## Green Computing Metrics

| Scale | PyReCo Memory | LSTM Memory | PyReCo Energy | LSTM Energy |
|-------|---------------|-------------|---------------|-------------|
| small | 498 MB | 56 MB | 0.0010 kWh | 0.0045 kWh |
| medium | 1472 MB | 4 MB | 0.0076 kWh | 0.0043 kWh |
| large | 3424 MB | 4 MB | 0.0412 kWh | 0.0043 kWh |

### Green Computing Recommendations

- **Memory-constrained**: Choose LSTM (lower and stable memory footprint)
- **Small-scale, low energy**: Choose PyReCo (lower energy consumption)
- **Large-scale, low energy**: Choose LSTM (lower energy consumption)

## Decision Rules Explained

### Rule 1: Chaotic dynamical systems (mackeyglass)

- **Recommendation**: PyReCo
- **Reason**: Reservoir computing excels at short-term prediction of chaotic systems
- **Confidence**: high

### Rule 2: Real-world measurement data (santafe)

- **Recommendation**: LSTM
- **Reason**: LSTM handles noise and complex patterns better
- **Confidence**: high

### Rule 3: Parameter budget < 10K and training speed matters

- **Recommendation**: PyReCo
- **Reason**: PyReCo trains 23x faster at small scale
- **Confidence**: high

### Rule 4: Memory-constrained environment (e.g., embedded devices)

- **Recommendation**: LSTM
- **Reason**: LSTM has lower and more stable memory footprint
- **Confidence**: high

### Rule 5: Small-scale tasks with energy efficiency priority

- **Recommendation**: PyReCo
- **Reason**: PyReCo consumes less energy at small scale
- **Confidence**: medium

### Rule 6: Rapid prototyping / initial experiments

- **Recommendation**: PyReCo
- **Reason**: Fast training enables quick iteration
- **Confidence**: high

### Rule 7: Production deployment requiring interpretability

- **Recommendation**: Requires evaluation
- **Reason**: Both models are black-boxes; test on specific use case
- **Confidence**: low

## Conclusion

### When to Use PyReCo
- ✅ Chaotic dynamical system prediction (Lorenz, Mackey-Glass)
- ✅ Rapid prototyping and experiment iteration
- ✅ Small parameter budget (<10K)
- ✅ Training speed-sensitive applications

### When to Use LSTM
- ✅ Real-world measurement data (e.g., Santa Fe)
- ✅ Memory-constrained environments
- ✅ Energy efficiency at large scale
- ✅ Need for mature ecosystem support

### General Recommendations
1. **Start with PyReCo for quick validation** - Fast training enables rapid iteration
2. **Choose based on data type** - PyReCo for chaotic systems, test both for real data
3. **Consider resource constraints** - LSTM for memory limits, PyReCo for time limits

---
*This guide was auto-generated based on experiments on 3 datasets*