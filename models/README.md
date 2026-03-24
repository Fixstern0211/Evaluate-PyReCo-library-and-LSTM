# Models Module - User Guide

## 📖 What is an Abstract Base Class?

**Simple Understanding**: An abstract base class is like a "standard contract" that specifies what functions all models must provide.

**Analogy**:
- Abstract Base Class = Car Driving Manual (specifies: must have steering wheel, gas pedal, brake)
- PyReCo Model = Tesla (implements in its own way: electric accelerator)
- LSTM Model = Audi (implements in its own way: fuel accelerator)
- You = Driver (no matter what car you drive, the operation is the same!)

## 🎯 Why Use Abstract Base Class?

### Problem: Without Unified Interface

```python
# Training PyReCo
pyreco_model = ReservoirComputer(...)
pyreco_model.fit(X, y)
pyreco_pred = pyreco_model.predict(X_test)
pyreco_mse = pyreco_model.evaluate(X_test, y_test, metrics=['mse'])

# Training LSTM - completely different interface!
lstm_model = MyLSTM(...)
lstm_model.train(X, y, epochs=100)  # ❌ Different method name
lstm_pred = lstm_model.forward(X_test)  # ❌ Different method name
lstm_mse = compute_mse(y_test, lstm_pred)  # ❌ Manual calculation needed

# Add another model? Need to write a different set of code again...
```

### Solution: Use Abstract Base Class

```python
from models import BaseTimeSeriesModel, compare_models

# All models inherit from the same base class
class PyReCoModel(BaseTimeSeriesModel):
    def fit(self, X, y): ...
    def predict(self, X): ...

class LSTMModel(BaseTimeSeriesModel):
    def fit(self, X, y): ...
    def predict(self, X): ...

# Now you can use a unified way to handle them!
models = [PyReCoModel(...), LSTMModel(...)]
results = compare_models(models, X_train, y_train, X_test, y_test)
```

**Advantages**:
✅ Unified Interface - All models use the same method names
✅ Consistent Evaluation - Automatically uses PyReCo's metrics (fair comparison)
✅ Clean Code - One function handles all models
✅ Easy Extension - Adding new models only requires inheriting the base class

## 🚀 Quick Start

### 1. Import Base Class

```python
from models.base_model import BaseTimeSeriesModel
import time
```

### 2. Create Your Model Class (Inherit from Base)

```python
class MyModel(BaseTimeSeriesModel):
    def __init__(self):
        # Call parent class initialization
        super().__init__(
            name="My Model",
            config={'param1': value1}
        )
        # Your model initialization code
        self.model = ...

    def fit(self, X_train, y_train):
        """Must implement: training method"""
        start = time.time()

        # Your training code
        self.model.fit(X_train, y_train)

        # Update state (important!)
        self.training_time = time.time() - start
        self.is_trained = True

    def predict(self, X):
        """Must implement: prediction method"""
        start = time.time()

        # Your prediction code
        predictions = self.model.predict(X)

        # Record time
        self.prediction_time = time.time() - start
        return predictions
```

### 3. Use Your Model

```python
# Create model
model = MyModel()

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate (automatically uses PyReCo's metrics)
results = model.evaluate(X_test, y_test, metrics=['mse', 'mae', 'r2'])
print(results)  # {'mse': 0.123, 'mae': 0.234, 'r2': 0.89}
```

### 4. Batch Compare Multiple Models

```python
from models.base_model import compare_models

models = [
    PyReCoModel(...),
    LSTMModel(...),
    MyModel(...),
]

results = compare_models(
    models=models,
    X_train=X_train, y_train=y_train,
    X_test=X_test, y_test=y_test
)

# Results are automatically printed and returned as dictionary
# {'PyReCo': {'mse': 0.1, ...}, 'LSTM': {'mse': 0.2, ...}, ...}
```

## 📚 Complete Example

See the implemented models for reference:

```bash
# PyReCo wrapper implementation
models/pyreco_wrapper.py

# LSTM model implementation
models/lstm_model.py

# Run model scaling comparison
python experiments/test_model_scaling.py --dataset lorenz
```

These implementations demonstrate:
- How to create a PyReCo wrapper (see `pyreco_wrapper.py`)
- How to create an LSTM model (see `lstm_model.py`)
- How to use compare_models for batch processing
- How to compare results

## 🔑 Core Concepts

### Methods You Must Implement

| Method | Description | Return Value |
|--------|-------------|--------------|
| `fit(X_train, y_train)` | Train the model | None |
| `predict(X)` | Make predictions | np.ndarray |

### Already Implemented Methods (Ready to Use)

| Method | Description | Return Value |
|--------|-------------|--------------|
| `evaluate(X, y, metrics)` | Evaluate model (uses PyReCo's metrics) | Dict[str, float] |
| `get_info()` | Get model information | Dict[str, Any] |

### Automatically Tracked Attributes

| Attribute | Description |
|-----------|-------------|
| `name` | Model name |
| `config` | Model configuration |
| `is_trained` | Whether model is trained |
| `training_time` | Training time (seconds) |
| `prediction_time` | Prediction time (seconds) |

## ⚠️ Important Notes

1. **Must call parent class initialization**
   ```python
   def __init__(self):
       super().__init__(name="Model Name", config={...})  # Required!
   ```

2. **Must set state after training**
   ```python
   def fit(self, X, y):
       # Training code...
       self.is_trained = True  # Must set!
   ```

3. **evaluate() method uses PyReCo's metrics**
   - Ensures all models use the same evaluation standards
   - No need to implement yourself, just inherit

## 📊 Supported Evaluation Metrics

| Metric | Description | Source |
|--------|-------------|--------|
| `mse` | Mean Squared Error | pyreco.metrics.mse |
| `mae` | Mean Absolute Error | pyreco.metrics.mae |
| `r2` | R² Score | pyreco.metrics.r2 |
| `rmse` | Root Mean Squared Error | sqrt(MSE) |

Usage example:
```python
results = model.evaluate(X_test, y_test, metrics=['mse', 'mae', 'r2', 'rmse'])
```

## 🎓 Next Steps

1. ✅ Understand abstract base class concept (Done)
2. ✅ Run example code (`examples/example_base_model_usage.py`)
3. 📝 Create PyReCo wrapper (`models/pyreco_wrapper.py`)
4. 🤖 Create LSTM model (`models/lstm_model.py`)
5. 🚀 Start your experiments!

## 💡 FAQ

### Q1: Do I have to use an abstract base class?
A: Not mandatory, but strongly recommended. Using abstract base class provides:
- Fair comparison (unified evaluation method)
- Cleaner code (unified interface)
- Easy extension (simple to add new models)

### Q2: What if my model interface is very special?
A: Adapt it inside the wrapper. For example:
```python
class MySpecialModel(BaseTimeSeriesModel):
    def fit(self, X, y):
        # Call your special interface internally
        self.model.special_train_method(X, y, special_param=123)
        self.is_trained = True
```

### Q3: Can I add custom evaluation metrics?
A: Yes, override the evaluate() method:
```python
def evaluate(self, X, y, metrics=None):
    # First call parent's evaluate
    results = super().evaluate(X, y, metrics)

    # Add custom metric
    y_pred = self.predict(X)
    results['custom_metric'] = my_custom_function(y, y_pred)

    return results
```

## 📞 Need Help?

If you have questions, please check:
- Model implementations: `models/pyreco_wrapper.py`, `models/lstm_model.py`
- Experiment plan: `docs/UPDATED_EXPERIMENTS_PLAN.md`
- Quick start guide: `QUICK_START.md`

---

**Last Updated**: January 2026
**Version**: 1.1 (Updated references)
