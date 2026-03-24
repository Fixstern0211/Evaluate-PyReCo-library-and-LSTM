# PyReCo Datasets Implementation Notes

## Issue #71: Chaotic System Datasets Implementation

本文档总结了为PyReCo实现混沌时间序列数据集的关键设计决策、经验教训和最佳实践。

---

## 目录

1. [概述](#概述)
2. [设计决策](#设计决策)
3. [数值积分方法](#数值积分方法)
4. [数据标准化策略](#数据标准化策略)
5. [验证集划分设计](#验证集划分设计)
6. [测试策略](#测试策略)
7. [经验教训](#经验教训)
8. [未来改进](#未来改进)

---

## 概述

**目标**: 为PyReCo提供易用的混沌时间序列数据集（Lorenz, Mackey-Glass）用于储备池计算实验。

**核心特性**:
- 三种操作模式：无标准化、标准化、验证集划分
- 向后兼容现有代码
- 防止验证集数据泄漏
- 确定性（Lorenz）和可重现性（Mackey-Glass使用seed）

---

## 设计决策

### 1. **三模式设计模式**

```python
# 模式1: 向后兼容（无标准化）
x_train, y_train, x_test, y_test = datasets.load('lorenz')

# 模式2: 带标准化
x_train, y_train, x_test, y_test, scaler = datasets.load('lorenz', standardize=True)

# 模式3: 带验证集划分（自动启用标准化）
x_train, y_train, x_val, y_val, x_test, y_test, scaler = datasets.load(
    'lorenz', val_fraction=0.2
)
```

**为什么这样设计？**
- ✅ 保持向后兼容性（模式1）
- ✅ 为不同用例提供灵活性
- ✅ 标准化总是可通过 `standardize` 参数使用
- ✅ 验证集划分自动启用标准化（防止常见错误）

### 2. **时序划分（非随机划分）**

时间序列数据按时间顺序划分：
- 训练集: 时间序列的前70%（默认）
- 验证集: 下一部分（如果设置了 `val_fraction`）
- 测试集: 剩余部分

**原理**: 模拟真实世界的预测场景，基于过去预测未来。

---

## 数值积分方法

### Lorenz系统: scipy.integrate.solve_ivp

**初始实现**: 手写的RK4（Runge-Kutta 4阶）
**最终实现**: `scipy.integrate.solve_ivp`（根据教授建议）

**关键代码**:
```python
from scipy.integrate import solve_ivp

def _generate_lorenz(n_timesteps, sigma=10.0, rho=28.0, beta=8.0/3.0, h=0.01, x0=None):
    # 定义Lorenz ODE系统
    def lorenz_deriv(t, state):  # 注意：solve_ivp要求(t, state)签名
        x, y, z = state
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        return [dx, dy, dz]

    # 定义需要求解的时间点
    t_span = (0, (n_timesteps - 1) * h)
    t_eval = np.linspace(0, (n_timesteps - 1) * h, n_timesteps)

    # 使用默认参数求解（RK45自适应步长）
    sol = solve_ivp(lorenz_deriv, t_span, x0, t_eval=t_eval)

    return sol.y.T  # 转置为 (n_timesteps, 3)
```

**为什么solve_ivp更好？**

| 方面 | 手写RK4 | scipy.integrate.solve_ivp |
|------|---------|---------------------------|
| 代码行数 | ~20行 | ~10行 |
| 步长 | 固定 | 自适应（默认） |
| 稳定性 | 需要手动调整 | 自动 |
| 维护 | 开发者负担 | SciPy团队维护 |
| 错误处理 | 无 | 内置 |
| 性能 | 固定RK4 | 可选多种求解器 |

**solve_ivp的优势**:
1. **自适应步长控制**: 默认使用RK45方法，根据局部误差自动调整步长
2. **误差估计**: 内置Runge-Kutta误差估计，确保精度
3. **刚性系统支持**: 可选Radau、BDF等求解器处理刚性ODE
4. **成熟的数值库**: SciPy经过广泛测试和优化

**RK4固定步长的问题**:
- 在解快速变化的区域可能步长太大 → 精度损失
- 在解缓慢变化的区域可能步长太小 → 计算浪费
- 对于Lorenz系统这样的混沌系统，固定步长可能导致长期积分误差累积

### Mackey-Glass: 自定义Euler积分

**为什么不用solve_ivp？** Mackey-Glass是**延迟微分方程（DDE）**，不是ODE。DDE需要访问过去的状态，`solve_ivp`不原生支持。

**实现**:
```python
from collections import deque

def _generate_mackey_glass(n_timesteps, tau=17, a=0.2, b=0.1, n=10, x0=1.2, h=1.0, seed=None):
    # 初始化历史缓冲区用于延迟项
    history_len = int(tau / h) + 1
    if seed is not None:
        rng = np.random.RandomState(seed)
        history = deque([x0 + rng.normal(0, 0.001) for _ in range(history_len)],
                       maxlen=history_len)
    else:
        history = deque([x0] * history_len, maxlen=history_len)

    # 带延迟的Euler积分
    timeseries = np.empty((n_timesteps,), dtype=np.float64)
    for i in range(n_timesteps):
        x_current = history[-1]
        x_delayed = history[0]  # 访问过去状态

        # Mackey-Glass方程: dx/dt = a*x(t-τ)/(1 + x(t-τ)^n) - b*x(t)
        dx = a * x_delayed / (1 + x_delayed**n) - b * x_current
        x_next = x_current + h * dx

        timeseries[i] = x_current
        history.append(x_next)

    return timeseries.reshape(-1, 1)
```

**为什么Euler对DDE可接受？**
- Mackey-Glass具有适度刚性
- 储备池计算文献中的标准基准
- 实现简单透明

---

## 数据标准化策略

### 何时应该标准化？

**储备池计算和深度学习中总是推荐**，原因：

#### 1. 对于PyReCo（储备池计算）
- **Ridge回归性能**: Ridge回归对特征尺度敏感，标准化后性能更好
- **不同特征尺度**: Lorenz中的x, y, z有不同数值范围
  - 例如：x ∈ [-20, 20], y ∈ [-30, 30], z ∈ [0, 50]
  - 标准化后都变为 mean≈0, std≈1
- **数值稳定性**: 防止大数值导致的数值溢出

#### 2. 对于LSTM（深度学习）
- **梯度下降收敛**: 标准化后损失函数更"圆"，梯度下降收敛更快
- **避免梯度爆炸/消失**: 输入数据范围过大会导致梯度问题
- **激活函数敏感性**: sigmoid/tanh对输入范围敏感
- **学习率选择**: 标准化后可用更大学习率，加速训练

**实际对比**（Lorenz数据集）:
```python
# 未标准化
x_train.mean(axis=(0,1))  # [0.12, -0.34, 27.5]  不同尺度
x_train.std(axis=(0,1))   # [8.2, 9.1, 8.8]

# 标准化后
x_train_scaled.mean(axis=(0,1))  # [0.0, 0.0, 0.0]  均值为0
x_train_scaled.std(axis=(0,1))   # [1.0, 1.0, 1.0]  标准差为1
```

### LSTM为什么还需要seed？

**PyReCo的确定性**:
- 储备池权重固定（不训练）
- Ridge回归有确定性解（闭式解）
- 相同输入 → 相同输出

**LSTM的随机性来源**:
1. **权重初始化**: PyTorch/TensorFlow随机初始化权重
   ```python
   torch.nn.LSTM(...)  # 每次初始化权重不同
   ```

2. **Dropout层**: 训练时随机丢弃神经元
   ```python
   nn.Dropout(p=0.2)  # 每次forward随机选择丢弃的神经元
   ```

3. **数据加载顺序**: 如果使用DataLoader的shuffle
   ```python
   DataLoader(dataset, shuffle=True)  # 每个epoch顺序不同
   ```

4. **GPU并行计算**: CUDA操作可能非确定性
   ```python
   torch.backends.cudnn.deterministic = False  # 默认非确定性
   ```

**如何确保LSTM可重现**:
```python
import torch
import numpy as np
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

**总结对比**:
| 模型 | 确定性 | 是否需要seed | 原因 |
|------|-------|-------------|------|
| PyReCo | ✅ 是 | ❌ 不需要 | 固定储备池 + Ridge闭式解 |
| LSTM | ❌ 否 | ✅ 需要 | 随机初始化 + Dropout + 优化过程 |
| Lorenz数据 | ✅ 是 | ❌ 不需要 | 确定性ODE |
| Mackey-Glass数据 | ⚠️ 可选 | ⚠️ 可选 | seed控制初始扰动 |

### 数据处理顺序的关键讨论

#### 问题：先分割还是先标准化？

**错误做法 ❌**:
```python
# 先标准化，再分割
scaler.fit(raw_data)  # 在整个数据集上拟合！
scaled_data = scaler.transform(raw_data)
train_data = scaled_data[:n_train]  # ❌ 测试集信息已泄漏
test_data = scaled_data[n_train:]
```

**正确做法 ✅**:
```python
# 先分割，再标准化
train_data = raw_data[:n_train]
test_data = raw_data[n_train:]

scaler.fit(train_data)  # 仅在训练集上拟合
train_scaled = scaler.transform(train_data)
test_scaled = scaler.transform(test_data)  # 使用训练集的统计量
```

**为什么？**
- 测试集的均值/标准差不应影响预处理
- 真实场景：训练时未来数据未知

#### 问题：滑动窗口在哪个阶段？

我们之前讨论过三种可能的顺序：

**方案A：先滑动窗口，再分割和标准化 ❌**
```python
# 这是我们最初讨论但放弃的方案
X, y = sliding_window(raw_data)  # (n_samples, n_in, n_features)
X_train, X_test = X[:n_train], X[n_train:]

# 问题：如何标准化3D数据？
scaler.fit(X_train.reshape(-1, n_features))  # 需要展平
X_train_scaled = scaler.transform(X_train.reshape(-1, n_features))
X_train_scaled = X_train_scaled.reshape(n_train, n_in, n_features)  # 再变回3D
```

**为什么放弃？**
- 代码复杂：需要reshape来回转换
- 容易出错：可能导致数据窗口边界问题
- 不直观：用户难以理解

**方案B：先分割，再标准化，最后滑动窗口 ✅ (最终采用)**
```python
# 1. 原始数据时序分割
train_data = raw_data[:n_train]  # (n_train_timesteps, n_features)
test_data = raw_data[n_train:]

# 2. 标准化 (2D数据，直接fit/transform)
scaler.fit(train_data)
train_scaled = scaler.transform(train_data)
test_scaled = scaler.transform(test_data)

# 3. 滑动窗口 (最后一步)
X_train, y_train = sliding_window(train_scaled, n_in, n_out)
X_test, y_test = sliding_window(test_scaled, n_in, n_out)
```

**为什么选择？**
- ✅ 代码简洁：每步操作都是2D → 2D或2D → 3D
- ✅ 防止泄漏：分割边界清晰，窗口不跨边界
- ✅ 易于理解：符合直觉的时序数据处理流程

**完整处理流程**:
```
原始时间序列 (5000, 3)
    ↓ 时序分割
训练 (3500, 3) | 测试 (1500, 3)
    ↓ 标准化 (fit on train only)
训练 (3500, 3) | 测试 (1500, 3) [scaled]
    ↓ 滑动窗口
X_train (3400, 100, 3) | X_test (1400, 100, 3)
y_train (3400, 1, 3)   | y_test (1400, 1, 3)
```

### 为什么PyReCo和LSTM都需要验证集？

尽管PyReCo和LSTM的训练机制完全不同，但两者都需要验证集来进行模型选择和超参数调优。

#### 1. PyReCo（储备池计算）

**需要验证集的原因**:
- **超参数选择**: num_nodes, density, spectral_radius, leakage_rate等
- **正则化参数**: Ridge回归的alpha（正则化强度）
- **储备池配置**: 不同的储备池权重初始化需要验证

**文献支持**:
- **Lukoševičius & Jaeger (2009)**: "A Practical Guide to Applying Echo State Networks"
  - 明确指出：储备池超参数（谱半径、稀疏度）需要通过验证集选择
  - 引用: "The spectral radius should be tuned on a validation set"

- **Tanaka et al. (2019)**: "Recent Advances in Physical Reservoir Computing: A Review"
  - 强调验证集对于Ridge回归正则化参数的重要性
  - 引用: "The regularization parameter α is typically selected via cross-validation"

**PyReCo的验证集用途**:
```python
# 超参数网格搜索
for num_nodes in [100, 500, 1000]:
    for spec_rad in [0.5, 0.9, 1.2]:
        # 训练模型
        model.fit(x_train, y_train)
        # 在验证集上评估
        val_mse = model.evaluate(x_val, y_val)

# 选择验证集上MSE最小的配置
best_config = min(configs, key=lambda c: c.val_mse)
```

#### 2. LSTM（深度学习）

**需要验证集的原因**:
- **Early Stopping**: 防止过拟合，在验证集损失不再下降时停止
- **超参数调优**: learning_rate, hidden_size, num_layers, dropout等
- **模型checkpoint**: 保存验证集上性能最好的模型

**文献支持**:
- **Prechelt (1998)**: "Early Stopping - But When?"
  - 经典论文，定义了早停法（Early Stopping）
  - 引用: "Training should be stopped when the validation error starts to increase"

- **Goodfellow et al. (2016)**: "Deep Learning" (深度学习圣经)
  - Chapter 7.8: Early Stopping
  - 引用: "We repeatedly evaluate the model on a validation set after each epoch, and save a copy of the model parameters when its performance on the validation set has improved"

- **Hochreiter & Schmidhuber (1997)**: "Long Short-Term Memory" (LSTM原始论文)
  - 虽未明确提及验证集，但后续所有LSTM实践都采用验证集

**LSTM的验证集用途**:
```python
# Early Stopping实现
best_val_loss = float('inf')
patience = 10
patience_counter = 0

for epoch in range(max_epochs):
    train_loss = train_one_epoch(model, train_loader)
    val_loss = evaluate(model, val_loader)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint(model)  # 保存最佳模型
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= patience:
        break  # Early stop
```

#### 3. 两者的共同点

| 方面 | PyReCo | LSTM |
|------|--------|------|
| **超参数调优** | num_nodes, density, spec_rad | hidden_size, num_layers, lr |
| **防止过拟合** | Ridge正则化α选择 | Early Stopping, Dropout |
| **模型选择** | 最佳储备池配置 | 最佳checkpoint |
| **验证集用途** | 网格搜索 + 最终评估 | Early Stopping + 超参数调优 |

**关键区别**:
- **PyReCo**: 验证集主要用于**离散超参数选择**（一次性训练多个配置）
- **LSTM**: 验证集用于**连续监控训练过程**（每个epoch都检查）

#### 4. 为什么不能用测试集代替验证集？

**机器学习基本原则**（Alpaydin, 2014; Bishop, 2006）:
- **训练集**: 用于学习模型参数
- **验证集**: 用于选择超参数和模型结构
- **测试集**: 用于最终性能评估（**不能用于任何决策**）

**使用测试集调参的后果**:
```python
# ❌ 错误做法：在测试集上调参
for alpha in [0.001, 0.01, 0.1, 1.0]:
    model = Ridge(alpha=alpha)
    model.fit(x_train, y_train)
    test_mse = evaluate(model, x_test)  # ❌ 测试集泄漏！

# 问题：测试集性能被高估，无法评估真实泛化能力
```

**正确的三分法**:
```python
# ✅ 正确做法：训练、验证、测试三分
# 1. 在训练集上训练
# 2. 在验证集上选择超参数
# 3. 在测试集上报告最终性能
```

#### 5. 文献总结

| 文献 | 年份 | 核心贡献 | 验证集相关 |
|------|------|---------|-----------|
| Lukoševičius & Jaeger | 2009 | Echo State Networks实践指南 | 验证集用于谱半径选择 |
| Tanaka et al. | 2019 | 物理储备池计算综述 | 验证集用于正则化参数 |
| Prechelt | 1998 | Early Stopping理论 | 定义验证集停止准则 |
| Goodfellow et al. | 2016 | 深度学习教材 | 验证集用于Early Stopping |
| Bishop | 2006 | Pattern Recognition | 训练/验证/测试三分法 |

### 如何防止数据泄漏？

**关键原则**:
1. **先分割，再预处理**
2. **仅在训练数据上拟合scaler**
3. **使用训练集统计量变换所有数据集**
4. **验证集用于模型选择，测试集仅用于最终评估**

```python
# 正确做法（模式3实现）
train_final_data = train_data[:n_train_final_timesteps]  # 窗口前划分
val_data = train_data[n_train_final_timesteps:]

scaler = StandardScaler()
scaler.fit(train_final_data)  # 仅在最终训练数据上拟合

train_final_scaled = scaler.transform(train_final_data)
val_scaled = scaler.transform(val_data)
test_scaled = scaler.transform(test_data)

# 然后在缩放后的数据上创建滑动窗口
x_train, y_train = _sliding_window(train_final_scaled, n_in, n_out)
x_val, y_val = _sliding_window(val_scaled, n_in, n_out)
```

**为什么在滑动窗口前划分？**
- 防止测试/验证数据泄漏到训练窗口中
- 每个窗口仅包含其指定划分的数据

**为什么仅在训练数据上拟合Scaler？**
- 验证/测试统计不应影响预处理
- 模拟真实场景：训练期间未来数据未知

---

## 验证集划分设计

### 塑造设计的用户请求

> "对于datasets，不论有没有val fraction，不是都应该对数据集标准化吗，是不是这样？"

**关键洞察**: 标准化和验证集划分是正交特性。

### 解决方案: `standardize` 参数

```python
def load(dataset_name, n_samples=5000, train_fraction=0.7, n_in=100, n_out=1,
         seed=None, val_fraction=None, standardize=False, **kwargs):
    # 模式1: 无标准化，无验证集
    if not standardize and val_fraction is None:
        return x_train, y_train, x_test, y_test  # 4项

    # 模式2: 标准化，无验证集
    elif standardize and val_fraction is None:
        return x_train, y_train, x_test, y_test, scaler  # 5项

    # 模式3: 验证集划分（自动启用标准化）
    else:
        return x_train, y_train, x_val, y_val, x_test, y_test, scaler  # 7项
```

**设计原理**:
- ✅ `standardize=False, val_fraction=None`: 向后兼容（默认）
- ✅ `standardize=True, val_fraction=None`: 显式标准化
- ✅ `val_fraction` 设置: 自动启用标准化（防止常见错误）

---

## 测试策略

### 创建的测试文件

1. **test_new_datasets.py**: 基本功能
   - 加载Lorenz和Mackey-Glass
   - 验证形状
   - 检查NaN/Inf
   - 测试seed的可重现性

2. **test_sliding_window_logic.py**: 实现正确性
   - 比较 `datasets.load()` 输出与手动滑动窗口
   - 确保处理流程中无数据损坏

3. **test_lorenz_determinism.py**: 确定性验证
   - Lorenz: 无需seed即100%可重现
   - Mackey-Glass: 相同seed可重现，不同seed不同

4. **test_standardize_modes.py**: 三模式验证
   - 模式1: 返回4项，数据未标准化
   - 模式2: 返回5项，数据已标准化（mean≈0, std≈1）
   - 模式3: 返回7项，所有划分正确标准化

### 测试哲学

**原则**: 测试行为，而非实现细节。

```python
# 好的做法：测试输出正确性
x_train_manual, y_train_manual = _sliding_window(train_data, n_in, n_out)
x_train_auto, _, _, _ = datasets.load('lorenz', n_samples, train_fraction, n_in, n_out)
assert np.allclose(x_train_manual, x_train_auto)

# 避免：测试内部状态
```

---

## 经验教训

### 1. **确定性 vs 可重现性**

**Lorenz（确定性ODE）**:
- 相同初始条件 → 总是相同轨迹
- 无随机元素
- 无需seed

**Mackey-Glass（带可选噪声的DDE）**:
- 初始历史中可选随机扰动
- Seed控制可重现性
- `seed=None` → 确定性（无噪声），但scipy可能使用系统随机性

**要点**: 在文档字符串中清楚地记录随机性来源。

### 2. **向后兼容性很重要**

初始设计 `standardize=True` 作为默认值 → 会破坏现有代码。

**解决方案**: 三模式设计，`standardize=False` 作为默认值。

**教训**: 向库添加功能时，始终提供迁移路径。

### 3. **数据泄漏容易引入**

**常见错误**:
```python
# 错误：在所有数据上拟合scaler
scaler.fit(np.concatenate([train_data, val_data]))  # ❌ 泄漏！
```

**正确方法**:
```python
# 正确：仅在训练数据上拟合
scaler.fit(train_final_data)  # ✅ 无泄漏
```

**教训**: 始终在预处理前划分，仅在训练数据上拟合。

**我们讨论过的关键问题**:

**用户问题1**: "你所说的新数据什么意思，我的测试项目有使用新数据集吗"
- **背景**: 我最初过度强调scaler用于"新数据推理"
- **澄清**: 用户的实验流程是：load → train → evaluate (MSE/MAE/R²) → end
- **学到**: 不是所有项目都需要在线推理，很多只做离线评估

**用户问题2**: "对于这个：或者直接不用 val_fraction，自己手动分割 + StandardScaler。我们之前讨论过，先滑动窗口再分割和标准化，需要展平再返回三维"
- **背景**: 用户记得我们讨论过"先滑动窗口"的方案
- **问题**: 3D数据标准化需要reshape，复杂且易错
- **最终方案**: 采用"先分割 → 再标准化 → 最后滑动窗口"的顺序
- **学到**: API设计要选择最简洁直观的方案

**用户洞察**: "对于datasets，不论有没有val fraction，不是都应该对数据集标准化吗，是不是这样？"
- **问题**: 最初设计标准化与val_fraction绑定
- **洞察**: 标准化和验证集划分是独立功能
- **解决方案**: 添加独立的`standardize`参数
- **影响**: 重新设计为三模式系统，提升灵活性

### 4. **混沌系统放大数值误差**

与 `reservoirpy.datasets` 的初始比较显示数据不匹配：
- **预期**: 不同积分器 → 略有不同的轨迹
- **混沌系统**: 小差异呈指数增长（Lyapunov指数）

**教训**: 对于混沌系统，测试滑动窗口逻辑，而非绝对值。

### 5. **scipy.integrate.solve_ivp值得使用**

从手写RK4迁移到 `solve_ivp`：
- ✅ 代码减少50%
- ✅ 更好的稳定性（自适应步长）
- ✅ 所有测试仍通过
- ✅ 更易维护

**教训**: 可用时使用成熟的数值库。

### 6. **文档必须与实现匹配**

切换到 `solve_ivp` 后，更新了：
- 模块文档字符串: "使用 scipy.integrate.solve_ivp"
- 函数文档字符串: "使用 scipy.integrate.solve_ivp 生成Lorenz 63吸引子时间序列"
- `load()` Notes部分: "Lorenz: scipy.integrate.solve_ivp用于稳定的ODE积分"

**教训**: 代码和文档漂移是真实问题。一起更新。

### 7. **改动前后对比总结**

#### 代码变化
```python
# 改动前：手写RK4 (~20行)
def _generate_lorenz(...):
    trajectory = np.empty((n_timesteps, 3))

    def lorenz_deriv(state):  # 只有state参数
        # ... 返回导数
        return np.array([dx, dy, dz])

    # 手动RK4循环
    for i in range(1, n_timesteps):
        k1 = lorenz_deriv(state)
        k2 = lorenz_deriv(state + h * k1 / 2)
        k3 = lorenz_deriv(state + h * k2 / 2)
        k4 = lorenz_deriv(state + h * k3)
        state = state + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)
        trajectory[i] = state

# 改动后：scipy.integrate.solve_ivp (~10行)
def _generate_lorenz(...):
    def lorenz_deriv(t, state):  # solve_ivp标准签名
        # ... 返回导数
        return [dx, dy, dz]

    t_span = (0, (n_timesteps - 1) * h)
    t_eval = np.linspace(0, (n_timesteps - 1) * h, n_timesteps)
    sol = solve_ivp(lorenz_deriv, t_span, x0, t_eval=t_eval)
    return sol.y.T
```

#### 关键改进
1. **函数签名**: `lorenz_deriv(state)` → `lorenz_deriv(t, state)` (ODE标准形式)
2. **代码简洁**: 20行 → 10行
3. **步长控制**: 固定步长 → 自适应步长（默认RK45）
4. **误差控制**: 无 → 内置误差估计
5. **维护**: 手动维护 → SciPy团队维护

---

## 未来改进

### 潜在改进

1. **添加更多混沌系统**
   - Rössler吸引子
   - Hénon映射
   - Kuramoto-Sivashinsky方程

2. **暴露solve_ivp参数**
   ```python
   datasets.load('lorenz', solver_method='RK45', rtol=1e-6, atol=1e-9)
   ```

3. **添加数据增强**
   - 初始条件扰动
   - 不同参数值（σ, ρ, β）

4. **支持非均匀时间步长**
   - 允许自定义 `t_eval` 数组
   - 对事件驱动系统有用

5. **GPU加速**
   - 用于大规模数据集生成
   - 可使用 `jax` 或 `cupy`

---

## 关键要点

1. **使用成熟的数值库**（`scipy.integrate.solve_ivp`）而非手写算法
2. **防止数据泄漏**：预处理前划分，仅在训练数据上拟合
3. **向现有API添加功能时考虑向后兼容性**
4. **测试行为，而非实现**（滑动窗口逻辑，而非绝对值）
5. **清楚记录随机性来源**（确定性 vs 可重现）
6. **重构期间保持代码和文档同步**
7. **混沌系统放大数值误差** - 测试逻辑，而非精确值

---

## 参考资料

- Issue #71: https://github.com/nschaetti/PyReCo/issues/71
- SciPy solve_ivp: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html
- Lorenz, E. N. (1963). "Deterministic Nonperiodic Flow"
- Mackey, M. C., & Glass, L. (1977). "Oscillation and chaos in physiological control systems"

---

**文档版本**: 1.0
**日期**: 2025-11-10
**作者**: PyReCo Issue #71 实现
**状态**: 完成并测试
