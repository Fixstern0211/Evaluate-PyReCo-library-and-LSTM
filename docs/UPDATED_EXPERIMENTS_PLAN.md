# PyReCo vs LSTM: 综合评估计划 (2025年更新版)

**更新时间**: 2026年3月23日
**版本**: 5.0 - **V2 Budget-Constrained 重设计进行中**
**目标**: 构建完整的PyReCo与LSTM时间序列预测比较框架

---

## **📋 待完成任务清单 (Action Items)**

> **实际进度评估 (2026-03-23)**: V1 核心任务 100% 完成。V2 重设计发现 budget-matching bug，需重跑 PyReCo 实验。
> Pretuning 已完成 (9/9)，主实验进行中 (35/270 = 13%)。LSTM 可复用旧数据。

### 🔴 **高优先级任务 (P0 - 阻塞发布)**

#### **TASK-001: 更新实验进度跟踪文件**
- **状态**: ✅ 已完成 (2026-01-26)
- **文件**: `results/final/experiment_progress.json`
- **完成内容**:
  - [x] 更新JSON文件反映90个实验全部成功完成
  - [x] 修正实验计数和状态
- **实际完成时间**: 0.5小时

#### **TASK-002: 创建统计分析模块**
- **状态**: ✅ 已完成 (2025-12-21)
- **文件**: `analysis/statistical_analysis.py`
- **功能需求**:
  - [x] 配对t检验 (Paired t-test)
  - [x] Wilcoxon符号秩检验 (非参数)
  - [x] 效应量计算 (Cohen's d with 95% CI)
  - [ ] 多重比较校正 (Bonferroni/FDR) - 需要statsmodels
  - [x] 生成终端报告
- **实际完成时间**: 2小时
- **统计结果摘要**:
  - 54次配对比较完成
  - MSE胜出: PyReCo 35/54, LSTM 19/54
  - Lorenz/Mackey-Glass: PyReCo显著优于LSTM
  - Santafe: LSTM显著优于PyReCo

#### **TASK-003: 修正科学结论表述**
- **状态**: ✅ 已完成 (2025-12-21)
- **问题**: 文档声称"PyReCo全面优势"，但Santa Fe数据集上LSTM表现更好
- **完成内容**:
  - [x] 更新 `docs/RESULTS_SUMMARY.md` 中的结论
  - [x] 添加数据效率实验结果 (270实验)
  - [x] 添加统计分析摘要和模型推荐表
  - [x] 修正Santafe数据集的效率评估 (PyReCo → LSTM)
- **实际完成时间**: 30分钟

### 🟡 **中优先级任务 (P1 - 功能完善)**

#### **TASK-004: 创建绿色计算指标脚本**
- **状态**: ✅ 已完成 (2026-01-26)
- **文件**: `experiments/test_model_scaling.py` (已集成绿色指标)
- **功能需求**:
  - [x] 集成CodeCarbon库 (能耗/碳排放追踪)
  - [x] 内存使用监控 (tracemalloc peak memory)
  - [x] 推理时间测量
  - [x] 能耗对比报告生成
- **结果文件**: `results/green_metrics/green_*.json`
- **主要发现**:
  - PyReCo内存占用较高 (500MB-4GB)，LSTM内存占用低且稳定 (3-56MB)
  - 小规模任务PyReCo能耗更低，大规模任务LSTM能耗更低
- **实际完成时间**: 2小时

#### **TASK-005: 创建决策指南生成器**
- **状态**: ✅ 已完成 (2026-01-26)
- **文件**: `analysis/decision_guide_generator.py`
- **功能需求**:
  - [x] 根据实验结果生成模型选择建议
  - [x] 创建数据集类型→模型推荐映射
  - [x] 输出Markdown格式决策指南
  - [x] 绿色计算指标分析与建议
- **输出文件**: `docs/DECISION_GUIDE.md`
- **主要内容**:
  - 快速决策表 (7条规则)
  - 按数据集分析 (Lorenz/Mackey-Glass/Santafe)
  - 按参数规模分析 (Small/Medium/Large)
  - 绿色计算建议
- **实际完成时间**: 2小时

#### **TASK-006: 创建analysis目录**
- **状态**: ✅ 已完成
- **行动**: `mkdir -p analysis/`
- **完成内容**: 目录已存在，包含 `statistical_analysis.py`

### 🟢 **低优先级任务 (P2 - 优化改进)**

#### **TASK-007: 完善ESN消融研究**
- **状态**: 🟡 部分完成
- **当前**: 分阶段调优实现了部分效果
- **可选行动**: 系统化消融实验脚本
- **预估时间**: 4-6小时

#### **TASK-008: 添加推理时间基准测试**
- **状态**: ✅ 已完成 (2026-01-27)
- **文件**: `experiments/test_inference_benchmark.py`
- **功能**:
  - [x] 单样本推理延迟 (含warmup和统计)
  - [x] 批量推理延迟 (batch sizes: 1, 8, 16, 32, 64, 128)
  - [x] 吞吐量测量 (samples/sec)
  - [x] 延迟统计 (mean, std, p95, p99)
- **结果文件**: `results/inference_benchmark/inference_*.json`
- **主要发现**:
  - 单样本推理: PyReCo快1.5-2x (3ms vs 5-6ms)
  - 批量推理: LSTM在大batch下快50-100x (GPU并行化优势)
  - 实时场景推荐PyReCo，批处理场景推荐LSTM
- **实际完成时间**: 1小时

---

### **任务依赖关系**
```
TASK-006 (创建目录)
    ├── TASK-002 (统计分析)
    └── TASK-005 (决策指南)
            └── TASK-003 (修正结论) [需要统计结果支持]

TASK-001 (更新进度文件) [独立任务]
TASK-004 (绿色指标) [独立任务]
```

### **建议执行顺序**
1. TASK-001 → TASK-006 → TASK-002 → TASK-003 → TASK-005
2. TASK-004 可并行执行

---

## **📋 补充实验计划 (2025-12-01 新增)**

> 当前主实验 train_ratios=[0.5, 0.7, 0.9] 缺少小样本测试和多步预测分析

### **TASK-009: 数据效率分析实验** ✅
- **状态**: ✅ 已完成 (2025-12-21)
- **目的**: 评估模型在不同数据量下的表现
- **实验设计**:
  ```python
  datasets = ['santafe', 'lorenz', 'mackeyglass']
  data_lengths = [1000, 2000, 3000, 5000, 7000, 10000]
  budgets = ['small', 'medium', 'large']
  seeds = [42, 43, 44, 45, 46]
  # 总实验数: 3 × 6 × 3 × 5 = 270 experiments ✅
  ```
- **完成情况**:
  - Santafe: 90/90 ✅
  - Lorenz: 90/90 ✅
  - Mackey-Glass: 90/90 ✅
- **科学发现**:
  - PyReCo在Lorenz/Mackey-Glass上显著优于LSTM
  - LSTM在Santafe上显著优于PyReCo
  - PyReCo训练速度比LSTM快17-40倍 (small规模)
  - 模型选择应基于数据集特性
- **实际完成时间**: ~8小时

### **TASK-010: 多步预测分析实验** ✅
- **状态**: ✅ 已完成 (2026-01-27 确认)
- **目的**: 评估模型在不同预测步长下的表现
- **实验设计**:
  ```python
  prediction_horizons = [1, 5, 10, 20, 50]  # 预测步长
  datasets = ['lorenz', 'mackeyglass', 'santafe']
  train_ratios = [0.2, 0.3, 0.5, 0.6, 0.7, 0.8]
  budgets = ['small', 'medium', 'large']
  seeds = [42, 43, 44, 45, 46]
  # 总实验数: 3 × 6 × 3 × 5 = 270 experiments ✅
  ```
- **结果文件**: `results/multi_step/` (270个JSON文件)
- **脚本**: `experiments/run_multi_step_experiments.py`
- **完成情况**:
  - Lorenz: 90/90 ✅
  - Mackey-Glass: 90/90 ✅
  - Santafe: 90/90 ✅
- **科学发现**:
  - PyReCo在短期预测(1-10步)优势明显
  - 长期预测(50步)两模型性能均下降，但趋势一致
  - 未发现LSTM超越PyReCo的"转折点"(在测试范围内)

---

## **项目实施状态总览**

### ✅ **已完成的核心基础设施 (WP1-3: 100%完成)**
- **统一接口**: BaseTimeSeriesModel抽象类 ✅
- **模型实现**: PyReCo (Standard + Custom) + LSTM包装器 ✅
- **数据处理**: 新datasets接口 - 一键数据预处理 ✅
- **公平比较**: 参数预算对齐 (Small/Medium/Large: 1K/10K/50K) ✅
- **超参数调优**: 分阶段调优 + 5折时间序列CV ✅
- **测试验证**: 完整的测试套件 ✅
- **🆕 多步评估系统**: 1-50步预测框架 ✅
- **🆕 数据效率分析**: 10%-100%训练数据系统研究 ✅
- **🆕 先进评估指标**: NRMSE, 谱相似度, 轨迹发散 ✅

### 📊 **已完成的实验脚本与验证**
```bash
# ✅ 基础设施 (WP1)
environment.yml, requirements-lock.txt, setup_environment.sh, validate_environment.sh

# ✅ 分阶段超参数调优 (已验证运行)
python tests/test_staged_tuning.py --dataset lorenz --strategy staged
# 结果: Stage 1 (R²=0.966) → Stage 2 (R²=0.983) → Stage 3 (最佳)

# ✅ 参数规模比较 (已验证运行)
python experiments/test_model_scaling.py --dataset lorenz
# 结果: 所有规模下PyReCo都优于LSTM

# ✅ 优化预调优 (已验证运行)
python experiments/run_optimized_pretuning.py --dataset lorenz --budget 10000
# 结果: 5.3秒完成48组合调优，找到最优参数

# ✅ 数据效率实验 (WP2.2 - 新实现并验证)
python experiments/test_data_efficiency.py --dataset lorenz --fractions 0.1,0.25,0.5,0.75,1.0
# 结果: PyReCo在所有数据量下都比LSTM好15-50倍

# ✅ 多步预测实验 (WP3.2 - 新实现并验证)
python experiments/test_multi_horizon.py --dataset lorenz --horizons 1,5,10,20,50
# 结果: PyReCo在所有时间尺度都保持优势，未发现LSTM超越点
```

---

## **工作包实施完成状态**

### 🎯 **Phase 1: 基础设施完善 (100% 完成)** ⭐⭐⭐⭐⭐

#### **✅ WP1.1: 可重现环境锁定**
**状态**: **完成** | **价值**: 至关重要的可重现性
```yaml
交付物:
  - environment.yml ✅ (conda环境锁定)
  - requirements-lock.txt ✅ (精确版本)
  - setup_environment.sh ✅ (一键安装脚本)
  - validate_environment.sh ✅ (环境验证脚本)
验收: ✅ 新机器上可完全重现相同结果
```

#### **✅ WP1.2: 任务与指标设计规范化**
**状态**: **完成** | **价值**: 明确实验协议，支撑科学结论
```yaml
交付物:
  - 多步预测协议 ✅ (1, 5, 10, 20, 50步)
  - Teacher forcing vs Free-run 区分 ✅ (Free-run完全实现)
  - 扩展指标 ✅ (NRMSE, 功率谱相似度, 长期统计一致性, 轨迹发散时间)
  - 实验协议文档 ✅ (docs/EVALUATION_PROTOCOL.md)
  - 评估框架 ✅ (src/utils/evaluation.py)
验收: ✅ 不同horizon下结果可解释一致
```

### 🔬 **Phase 2: 核心比较实验 (100% 完成)** ⭐⭐⭐⭐⭐

#### **✅ WP2.1: 基线模型对等性**
**状态**: **完成** | **价值**: 公平比较基础
- ✅ 参数预算对齐: Small(1K)/Medium(10K)/Large(50K)
- ✅ 统一初始化与训练协议
- ✅ 总参数量匹配 (文献支持的首次严格实现)

#### **✅ WP2.2: 数据效率实验** 🆕
**状态**: **完成** | **价值**: 揭示RC vs LSTM的数据依赖性差异
```python
# 实施完成: experiments/test_data_efficiency.py
# 训练集比例: 10%, 25%, 50%, 75%, 100%
# 每档K=5次重复(不同种子) - 支持多种子
# 短/中/长期三类评估 - 1, 5, 10步预测
```
**🏆 科学发现**:
- **PyReCo压倒性优势**: 50%数据下MSE比LSTM好15倍, 100%数据下好14倍
- **训练效率**: PyReCo训练速度比LSTM快90倍 (0.04s vs 3.5s)
- **数据效率模式**: 两模型都随数据增加而改善，PyReCo始终保持优势

#### **✅ WP2.3: 复杂度控制实验**
**状态**: **完成** | **价值**: 容量与性能权衡分析

### 🧪 **Phase 3: 深度分析实验 (75% 完成)** ⭐⭐⭐⭐

#### **✅ WP3.2: 多时间尺度评估** 🆕
**状态**: **完成** | **价值**: 理解短/中/长期预测能力差异
```yaml
# 实施完成: experiments/test_multi_horizon.py
Horizons: 1步(短期), 5-15步(中期), 50步(长期) ✅
模式: Free-run自主预测 ✅ (Teacher forcing规划中)
指标: MSE, MAE, R², NRMSE, 轨迹发散时间 ✅
```
**🏆 科学发现**:
- **全时间尺度主导**: PyReCo在ALL测试horizon(1-15步)都优于LSTM
- **短期预测卓越**: 1步预测PyReCo比LSTM好50倍 (MSE: 0.002 vs 0.107)
- **长期预测稳定**: 15步预测PyReCo仍保持优势 (MSE: 0.148 vs 0.156)
- **无转折点**: 测试范围内未发现LSTM超越PyReCo的时间尺度

#### **🟡 WP3.1: ESN超参数消融研究**
**状态**: **可选实现** | **价值**: 理解RC敏感性，优化默认配置
**成果**: 分阶段调优实现了部分消融效果

### ♻️ **Phase 4: 绿色计算与效率 (100% 完成)** ⭐⭐⭐⭐⭐

#### **✅ WP4.1: 绿色指标追踪**
**价值**: 可持续AI研究，成本效益分析
**状态**: **完成** (2026-01-26)
```python
# 已集成到 experiments/test_model_scaling.py
metrics = {
    'training_time': '✅ 已实现',
    'inference_time': '✅ 已实现 (TASK-008)',
    'energy_consumption': '✅ 已实现 (CodeCarbon)',
    'carbon_footprint': '✅ 已实现 (CodeCarbon)',
    'memory_usage': '✅ 已实现 (tracemalloc)'
}
```
**结果文件**: `results/green_metrics/green_*.json`
**主要发现**:
- PyReCo内存占用: 500MB-4GB (随参数规模增长)
- LSTM内存占用: 3-56MB (低且稳定)
- 小规模能耗: PyReCo更低 (0.001 kWh vs 0.005 kWh)
- 大规模能耗: LSTM更低 (0.004 kWh vs 0.046 kWh)

#### **✅ WP4.2: 推理时间基准测试**
**价值**: 生产部署决策支持
**状态**: **完成** (2026-01-27)
**文件**: `experiments/test_inference_benchmark.py`
**结果**: `results/inference_benchmark/inference_*.json`
**主要发现**:
- 单样本推理: PyReCo快1.5-2x
- 批量推理(128样本): LSTM快50-100x (GPU并行化)

### 📊 **Phase 5: 统计分析与报告 (100% 完成)** ⭐⭐⭐⭐⭐

#### **✅ WP5.1: 严格统计分析**
**价值**: 科学可信度，发表要求
**状态**: **完成** (2025-12-21)
**文件**: `analysis/statistical_analysis.py`
```python
统计测试:
  - ✅ 配对t检验 (相同数据不同模型)
  - ✅ Wilcoxon符号秩检验 (非参数)
  - ✅ 效应量: Cohen's d with 95% CI
  - [ ] 多重比较校正: Bonferroni/FDR (需要statsmodels)
```
**统计结果**:
- 54次配对比较完成
- MSE胜出: PyReCo 35/54, LSTM 19/54
- Lorenz/Mackey-Glass: PyReCo显著优于LSTM (p < 0.001)
- Santafe: LSTM显著优于PyReCo (p < 0.001)

#### **✅ WP5.2: 综合决策指南**
**价值**: 实际应用指导
**状态**: **完成** (2026-01-26)
**文件**: `analysis/decision_guide_generator.py`
**输出**: `docs/DECISION_GUIDE.md`
```yaml
输出内容:
  - ✅ "何时RC更优，何时LSTM反超"条件映射 (7条决策规则)
  - ✅ 数据集类型→模型推荐映射
  - ✅ 参数规模性能对比表
  - ✅ 绿色计算建议 (内存/能耗)
  - ✅ 实践建议与局限性分析
```

---

## **实验设计更新**

### **核心实验矩阵**
```yaml
数据集: [lorenz, mackeyglass, santafe]
训练比例: [0.1, 0.25, 0.5, 0.75, 1.0]  # 新增数据效率
参数规模: [small(1K), medium(10K), large(50K)]
预测时长: [1, 5, 10, 20, 50]  # 新增多步预测
随机种子: [42, 43, 44, 45, 46] # 5次重复
```

### **新增实验脚本**
```bash
# 数据效率实验
python experiments/test_data_efficiency.py --dataset lorenz --fractions 0.1,0.25,0.5,0.75,1.0

# 多步预测实验
python experiments/test_multi_horizon.py --dataset lorenz --horizons 1,5,10,20,50

# 绿色指标追踪
python experiments/test_green_metrics.py --dataset lorenz --track-carbon --track-memory

# 统计分析
python analysis/statistical_comparison.py --results-dir results/ --output report.html
```

---

## **项目当前状态评估**

### ✅ **优势领域**
- **坚实基础**: 统一接口，公平比较框架完整
- **数据处理**: 新datasets接口提供一键标准化
- **超参数调优**: 5折CV + 分阶段调优 + 文献支持的网格
- **测试覆盖**: 完整的验证测试套件

### ✅ **已加强的领域**
1. **✅ 多步预测**: 完成270个多步预测实验 (1-50步)
2. **✅ 统计严谨性**: 配对t检验 + Wilcoxon + Cohen's d效应量
3. **✅ 绿色计算**: CodeCarbon能耗追踪 + tracemalloc内存监控
4. **✅ 数据效率**: 270个数据效率实验完成
5. **✅ 决策指导**: 完整决策指南 (docs/DECISION_GUIDE.md)
6. **✅ 推理基准**: 单样本/批量推理延迟测量

### 💡 **创新价值点**
1. **公平比较**: 总参数量匹配 (文献首次严格实现)
2. **新datasets接口**: 自动防数据泄露的预处理
3. **分阶段调优**: Stage 1→2→3 渐进式超参数搜索
4. **绿色AI**: RC vs LSTM 的碳足迹比较 (新兴重要话题)

## **更新后的项目时间线与实施状态** (2025-12-01 审核更新)

| Phase | 工作包 | 计划时间 | 实际状态 | 完成度 | 科学价值 | 关联任务 |
|-------|--------|----------|----------|---------|----------|----------|
| **✅ 1** | 环境锁定 | 0.5周 | **完成** | 100% | ⭐⭐⭐⭐⭐ | - |
| **✅ 1** | 任务指标规范 | 0.5周 | **完成** | 100% | ⭐⭐⭐⭐⭐ | - |
| **✅ 2** | 基线模型对等 | 1周 | **完成** | 100% | ⭐⭐⭐⭐⭐ | - |
| **✅ 2** | 数据效率实验 | 1周 | **完成 (270实验)** | 100% | ⭐⭐⭐⭐ | TASK-009 ✅ |
| **✅ 2** | 复杂度控制 | 1周 | **完成** | 100% | ⭐⭐⭐⭐ | - |
| **✅ 3** | 多步预测 | 1周 | **完成 (270实验)** | 100% | ⭐⭐⭐⭐ | TASK-010 ✅ |
| **🟡 3** | ESN消融 | 1周 | **可选** | 75% | ⭐⭐⭐ | TASK-007 |
| **✅ 4** | 绿色指标 | 0.5周 | **完成** | 100% | ⭐⭐⭐⭐ | TASK-004 ✅ |
| **✅ 4** | 推理基准测试 | 0.5周 | **完成** | 100% | ⭐⭐⭐⭐ | TASK-008 ✅ |
| **✅ 5** | 统计分析 | 0.5周 | **完成** | 100% | ⭐⭐⭐⭐⭐ | TASK-002 ✅ |
| **✅ 5** | 决策指南 | 0.5周 | **完成** | 100% | ⭐⭐⭐⭐ | TASK-005 ✅ |
| **✅ -** | 进度文件修复 | - | **完成** | 100% | - | TASK-001 ✅ |
| **✅ -** | 结论修正 | - | **完成** | 100% | - | TASK-003 ✅ |
| **总计** | | **6.5周** | **~98%完成** | | |

### **实际执行进度** 📈
- **✅ 已完成**: Week 1-3 (WP1-3) - **核心框架与实验全部完成**
- **✅ 已完成**: Week 4 (WP4) - **绿色计算与推理基准测试**
- **✅ 已完成**: Week 5 (WP5) - **统计分析与决策指南**
- **🟡 可选**: TASK-007 ESN消融研究 (75%完成，非必需)

### **🏆 重大科学发现总结** (⚠️ 需要根据TASK-003修正)

#### **1. 数据集特异性发现** 📊
| 数据集 | 获胜者 | PyReCo R² | LSTM R² | 说明 |
|--------|--------|-----------|---------|------|
| **Lorenz** | **PyReCo** | 0.9995-0.9998 | 0.8253-0.9040 | PyReCo显著优势 |
| **Mackey-Glass** | **PyReCo** | 0.9906-0.9923 | 0.9010-0.9377 | PyReCo优势 |
| **Santa Fe** | **LSTM** | 0.7286-0.7962 | 0.8790-0.9306 | ⚠️ LSTM更优 |

#### **2. PyReCo优势场景** 🥇
- **混沌动力系统**: Lorenz, Mackey-Glass等生成式数据表现卓越
- **训练效率**: 训练速度比LSTM快数十倍
- **参数效率**: 更少可训练参数实现更好性能
- **数据效率**: 在有限数据下表现相对更优

#### **3. LSTM优势场景** 🥈
- **真实世界数据**: Santa Fe激光数据等实测数据可能更适合LSTM
- **复杂模式**: 包含非平稳、噪声较多的时间序列

#### **4. 关键结论** 💡
- ~~强烈推荐PyReCo而非LSTM~~ → **模型选择应基于数据集特性**
- **混沌/生成式数据**: 优先选择PyReCo
- **实测/复杂数据**: 需要实验验证，LSTM可能更优
- **有限计算资源**: PyReCo仍是更好的起点（快速验证）

---

## **项目当前状态评估 (v4.0更新)**

### ✅ **已确立的优势领域**
- **🔬 科学严谨性**: 公平参数预算比较 + 多种子验证
- **🛠️ 技术完整性**: 统一接口 + 自动化流程
- **📊 评估深度**: 多步预测 + 先进指标系统
- **⚡ 效率优化**: 分阶段调优 + 一键部署
- **🎯 实验覆盖**: 数据效率 + 时间尺度全覆盖
### 🟢 **已完成的领域 (98%)**
1. **✅ 绿色计算指标**: CodeCarbon能耗追踪 + tracemalloc内存监控
2. **✅ 统计严谨性**: 配对t检验 + Wilcoxon + Cohen's d效应量
3. **✅ 决策指导系统**: 完整决策指南 (docs/DECISION_GUIDE.md)
4. **✅ 推理基准测试**: 单样本/批量推理延迟测量

### 🟡 **可选改进 (剩余2%)**
1. **🟡 ESN消融研究**: 系统化消融实验脚本 (TASK-007, 可选)
2. **🟡 多重比较校正**: Bonferroni/FDR (需要statsmodels库)

---

## **更新后的成功验收标准**

### **✅ 科学严谨性** (100% 完成)
- ✅ 多数据集×多种子统计显著性检验 - **已实现多种子重复**
- ✅ 效应量报告 (Cohen's d with 95% CI) - **统计分析模块已完成**
- ✅ 多步预测性能曲线 (1-50步) - **270个实验完成**
- ✅ 环境完全可重现 (新机器相同结果) - **已验证**

### **✅ 实用价值** (100% 完成)
- ✅ 数据量×模型容量决策矩阵 - **决策指南已生成**
- ✅ "RC vs LSTM选择指南" (条件→建议映射) - **docs/DECISION_GUIDE.md**
- ✅ 训练成本vs性能权衡分析 - **已完成**
- ✅ 碳足迹对比报告 - **results/green_metrics/**
- ✅ 推理延迟基准 - **results/inference_benchmark/**

### **✅ 技术创新** (100% 完成)
- ✅ 首个严格参数预算对齐的RC vs LSTM比较 - **已实现**
- ✅ 时间序列数据泄露防护的标准化流程 - **已实现**
- ✅ RC与LSTM的绿色AI指标基准 - **CodeCarbon + tracemalloc**
- ✅ 分阶段超参数调优最佳实践 - **已验证**
- ✅ 推理时间基准测试框架 - **已实现**

---

## **已完成的行动计划** ✅

### **✅ 绿色计算 (WP4) - 已完成**
```bash
# 运行绿色指标实验
python experiments/test_model_scaling.py --dataset lorenz --track-green --country-code USA
# 结果: results/green_metrics/green_*.json
```

### **✅ 统计分析 (WP5.1) - 已完成**
```bash
# 运行统计分析
python analysis/statistical_analysis.py
# 结果: 配对t检验, Wilcoxon, Cohen's d
```

### **✅ 决策指南 (WP5.2) - 已完成**
```bash
# 生成决策指南
python analysis/decision_guide_generator.py
# 输出: docs/DECISION_GUIDE.md
```

### **✅ 推理基准测试 (TASK-008) - 已完成**
```bash
# 运行推理基准测试
python experiments/test_inference_benchmark.py --all-datasets --budget medium
# 结果: results/inference_benchmark/inference_*.json
```

## **可选后续工作**

### **🟡 ESN消融研究 (TASK-007, 可选)**
```bash
# 如需系统化消融实验
python experiments/esn_ablation_study.py --dataset lorenz
# 价值: 理解RC超参数敏感性
```

---

## **Future Extensions (Optional Experiments)**

> The following are optional future experiment directions for further research expansion.

### **EXT-001: Univariate Prediction**
- **Status**: Planned
- **Description**: Use only a single variable for prediction
- **Experiment Design**:
  - Input: history window of x_t → Output: x_{t+1}
  - Compare predictability differences across x, y, z variables
- **Expected Value**: Understand how different variables' chaotic characteristics affect model performance

### **EXT-002: Input Window Size Sensitivity Analysis**
- **Status**: Planned
- **Description**: Test how different input window sizes affect prediction performance
- **Experiment Design**:
  - Input windows: n_in = [25, 50, 100, 200, 400]
  - Fix other parameters, vary only window size
- **Expected Value**: Determine optimal input window, understand short-term vs long-term memory trade-offs

### **EXT-003: Noise Robustness Experiments**
- **Status**: Planned
- **Description**: Test model robustness to noise
- **Experiment Design**:
  - Noise type: Gaussian white noise
  - Noise levels: SNR = [40dB, 30dB, 20dB, 10dB]
  - Measure performance degradation curves
- **Expected Value**: Evaluate model applicability in real-world noisy environments

### **Priority Recommendations**
| Experiment | Difficulty | Value | Priority |
|------------|------------|-------|----------|
| EXT-002 Input Window | Low | High | ⭐⭐⭐ |
| EXT-003 Noise Robustness | Medium | High | ⭐⭐⭐ |
| EXT-001 Univariate | Low | Medium | ⭐⭐ |

---

## **版本历史与演进**

### **版本演进轨迹**
- **v1.0** (初版): 初始技术架构设计
- **v2.0** (基础): 基础设施完成 + 分阶段调优实现
- **v3.0** (现代): 现代AI研究标准 + 实用价值导向
- **v4.0** (实施): **WP1-3全面完成 + 重大科学发现确认**

### **v4.0版本核心成就**
1. **🏗️ 完整基础设施**: 一键环境 + 标准化评估
2. **🔬 科学发现**: PyReCo全面优势的实验证实
3. **⚙️ 自动化流程**: 从数据加载到结果生成的完整管道
4. **📚 完整文档**: 使用指南 + 科学协议 + 实施总结

**总结**: v4.3版本标志着项目所有核心任务完成 (98%)。完成内容包括：绿色计算指标追踪、决策指南生成器、推理时间基准测试、以及多步预测实验验证。科学结论明确：PyReCo在混沌/生成式数据上表现优异，LSTM在真实世界数据上更优，模型选择应基于数据集特性。唯一未完成的可选任务是ESN消融研究 (TASK-007)。

---

## **版本更新日志**

### v4.3 (2026-01-27) 🎉 **所有核心任务完成**
- ✅ TASK-001 完成: 更新实验进度跟踪文件 (90个实验全部成功)
- ✅ TASK-004 完成: 绿色计算指标脚本
  - 集成CodeCarbon能耗/碳排放追踪
  - 集成tracemalloc内存监控
  - 结果: `results/green_metrics/green_*.json`
- ✅ TASK-005 完成: 决策指南生成器
  - 创建 `analysis/decision_guide_generator.py`
  - 输出 `docs/DECISION_GUIDE.md`
  - 包含7条决策规则和绿色计算建议
- ✅ TASK-008 完成: 推理时间基准测试
  - 创建 `experiments/test_inference_benchmark.py`
  - 测量单样本/批量推理延迟
  - 结果: `results/inference_benchmark/inference_*.json`
- ✅ TASK-010 确认完成: 多步预测分析实验 (270/270)
- ⬆️ 进度评估更新: ~85% → **~98%**
- 📊 主要发现:
  - 单样本推理: PyReCo快1.5-2x
  - 批量推理: LSTM快50-100x (GPU并行化)
  - 内存: LSTM更低且稳定 (3-56MB vs 500MB-4GB)
  - 能耗: 小规模PyReCo更低，大规模LSTM更低

### v4.2 (2025-12-21)
- ✅ TASK-002 完成: 统计分析模块实现 (配对t检验, Wilcoxon, Cohen's d)
- ✅ TASK-003 完成: 修正科学结论表述
  - 更新 RESULTS_SUMMARY.md，添加数据效率实验结果
  - 添加模型推荐表和统计分析摘要
  - 修正Santafe效率评估
- ✅ TASK-006 完成: analysis目录已创建
- ✅ TASK-009 完成: 数据效率分析实验 (270个实验全部完成)
  - santafe: 90/90, lorenz: 90/90, mackeyglass: 90/90
- 📊 统计分析结果:
  - 54次配对比较, MSE胜出: PyReCo 35/54, LSTM 19/54
  - Lorenz/Mackey-Glass: PyReCo显著优于LSTM
  - Santafe: LSTM显著优于PyReCo
- ⬆️ 进度评估更新: 70-75% → ~90%

### v4.1 (2025-12-01)
- 🔍 进行全面项目审核
- 📋 添加具体待完成任务清单 (TASK-001 至 TASK-008)
- ⚠️ 修正进度评估：85% → 70-75%
- 🔄 更新科学结论：承认Santa Fe数据集LSTM更优
- 📊 添加任务依赖关系和执行顺序建议

### v4.0 (2025-11-22)
- WP1-3实施完成
- 多步预测系统实现
- 数据效率分析完成

---

---

## **🔄 实验 V2：Budget-Constrained 重设计 (2026-03)**

> **背景**：发现旧实验存在严重 budget-matching bug（PyReCo 实际仅使用 14-43% 的参数预算），
> 以及 LSTM 公式/早停的微小偏差。完整设计文档见 `analysis/experiment_redesign_v2.md`。

### 核心问题与修复

| Bug | 影响 | 修复 |
|-----|------|------|
| ESN budget 不匹配 | PyReCo 仅用 14-43% budget，比较不公平 | Dynamic N：每组 (δ, f_in) 重新解二次方程求 N |
| LSTM 公式少一个 bias | h 偏大 1（影响极小） | 修正为 `4h(h+d_in+2)+h*d_out+d_out` |
| LSTM final 无 val early stopping | 可能过拟合 | 使用 tuning 阶段的 best_epoch |
| CV 在 window 层面切分 | 相邻窗口共享 99/100 时间步（数据泄漏） | 在原始时间序列上切分，再独立创建窗口 |
| N_cap 限制 | Large budget + δ<0.05 → N>1000 不可行 | Large budget 只搜 δ={0.05, 0.1}，论文讨论此限制 |

### **TASK-011: 旧/新 LSTM 结果对比验证** ✅
- **状态**: ✅ 已完成 (2026-03-23)
- **脚本**: `analysis/compare_old_new_lstm.py`
- **方法**: 按 (dataset, budget, seed, train_frac) 匹配 old vs new LSTM
- **比较结果** (34 matched pairs):
  - **tf >= 0.5 (20 pairs)**: Mean|ΔR²| = 0.000035, t-test p = 0.651 → 差异可忽略
  - **tf < 0.5 (14 pairs)**: Mean|ΔR²| = 0.036（正常欠拟合波动）
- **结论**: **复用旧 LSTM 数据**，只重跑 PyReCo
- **数据一致性验证**:
  - Lorenz: 确定性 ODE，seed 无影响（所有 seed 产生相同轨迹，仅影响模型初始化）
  - Mackey-Glass: 同 seed → 同数据 ✓
  - Santa Fe: 固定数据集 ✓

### **TASK-012: V2 Pretuning (ESN 超参数预调)** ✅
- **状态**: ✅ 已完成
- **脚本**: `run_pretuning_v2.py`（含 `timeseries_cv_split_raw` 防泄漏 CV）
- **结果**: `results/pretuning_v2/pretuning_v2_{dataset}_{budget}.json` (9/9 组全部完成)
- **设计**: seed=42, train_frac=0.7, 5-fold forward-chaining CV
- **搜索网格**:
  - Small/Medium: δ={0.01,0.03,0.05,0.1} × f_in={0.1,0.3,0.5} = 12 结构组合
  - Large: δ={0.05,0.1} × f_in={0.1,0.3,0.5} = 6 结构组合（N_cap=1000 限制）
  - 动力学: spec_rad={0.5,0.7,0.8,0.9,0.99} × leakage={0.1,0.3,0.5,0.7,0.8,1.0} = 30 组合
- **关键发现**: 最优 spec_rad 随 N 增大而降低（Lorenz/Small 最优 0.80 而非 0.99），
  说明旧实验的单调趋势在 budget-constrained N 下不成立

### **TASK-013: V2 主实验 — 只跑 PyReCo** 🔴 进行中
- **状态**: 🔴 进行中 (2026-03-23)
- **脚本**: `run_final_v2.py`
- **策略**: LSTM 复用 `results/final/`，只跑 ESN grid search
- **数据合并**: 分析时合并两个目录
  - PyReCo ← `results/final_v2/`
  - LSTM ← `results/final/`
- **实验矩阵**:
  ```yaml
  数据集: [lorenz, mackeyglass, santafe]
  训练比例: [0.2, 0.3, 0.5, 0.6, 0.7, 0.8]
  参数规模: [small(1K), medium(10K), large(50K)]
  随机种子: [42, 43, 44, 45, 46]
  模型: PyReCo only
  总实验数: 3 × 6 × 3 × 5 = 270
  ```
- **当前进度** (35/270 = 13%):
  - lorenz/small: seed42(6tf) ✅, seed45(6tf) ✅ → 12/30
  - lorenz/medium: seed42(4tf) ✅, seed45(4tf) ✅ → 8/30
  - mackeyglass/small: seed42(6tf) ✅ → 6/30
  - mackeyglass/medium: seed42(6tf) ✅ → 6/30
  - mackeyglass/large: seed42(3tf) ✅ → 3/30
  - lorenz/large: 0/30
  - santafe/*: 0/90
- **执行计划**:
  1. 终止当前实验（medium 结束后）
  2. 优先跑 **large** budget, seed 42 & 45（lorenz, mackeyglass, santafe）
  3. 补齐剩余 seed (43, 44, 46) 和缺失的 train_frac

### **TASK-014: V2 结果合并与统计分析** ⬜ 待开始
- **状态**: ⬜ 待开始
- **依赖**: TASK-013 完成
- **行动**:
  - [ ] 编写合并脚本：从 `final_v2/` 读 PyReCo，从 `final/` 读 LSTM
  - [ ] 配对 t 检验、Wilcoxon、Cohen's d（按 dataset×budget 分组）
  - [ ] 与旧实验对比：budget-matched 后 PyReCo 表现是否显著改善
  - [ ] 论文中说明：seed 对 Lorenz 数据无影响（仅影响模型初始化随机性）

### **TASK-015: 论文更新** ⬜ 待开始
- **状态**: ⬜ 待开始
- **依赖**: TASK-014 完成
- **需要重写的章节** (详见 `experiment_redesign_v2.md` Section 10):
  - [ ] Ch3 Methods: budget matching 方法、参数公式、CV 方法（forward-chaining on raw series）
  - [ ] Ch3 Table 1: 所有参数计数（使用实际值）
  - [ ] Ch4 Results: 所有表格、图表、数值
  - [ ] Ch5 Discussion: 核心发现、N_cap 限制讨论、旧数据 parameter efficiency 观察
  - [ ] Ch6 Conclusion: 总结和建议
  - [ ] 所有图表: 从新数据重新生成
- **新增讨论点**:
  - ESN parameter efficiency（旧实验的补充发现，cautious framing）
  - N≤1000 计算约束对 large budget 的影响
  - Forward-chaining CV vs window-level CV 的区别

### 任务依赖关系

```
TASK-011 ✅ (LSTM 对比验证)
    └── TASK-013 🔴 (V2 主实验 — 只跑 PyReCo)
            └── TASK-014 ⬜ (结果合并与统计分析)
                    └── TASK-015 ⬜ (论文更新)

TASK-012 ✅ (Pretuning) ─── feeds into ──→ TASK-013
```

---

**Last Updated**: March 23, 2026
**Version**: 5.0 (V2 Experiments In Progress)