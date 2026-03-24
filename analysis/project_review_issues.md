# Project Review Issues — 严格评审发现的问题

**评审日期**: 2026-03-17
**总评分**: B+ / 82分
**修订日期**: 2026-03-20

---

## 严重问题（影响结论可信度）

### Issue 1: 参数预算对比不公平 — 核心设计缺陷
**状态**: [x] 已解决

论文题目明确限定 "Matched total Parameter Budgets"。Ch3 已包含完整参数表（按数据集分列）、trainable 匹配后果分析表（Table trainable_matching）、论证段落。Ch5 讨论实验结果中的不对称影响。Ch5 Limitations 第5条提及。

**修改记录**:
- Ch3 Table 1: 重写为按 Lorenz/MG+SF 分行，使用实际实验数据
- Ch3: 新增 Table trainable_matching + 论证段落（从 Ch5 移入）
- Ch4 L47: 100 → 300 trainable (Lorenz)
- Ch5 L43-47: 按数据集分列实际倍数差
- Ch2/Ch4/Ch6: "3-5%" / "1-10%" → "1-21%"

---

### Issue 2: 仅 3 个数据集，全为混沌/动力学系统
**状态**: [x] 已解决（Ch5 Limitation #3 已覆盖）

论文题目限定 "Chaotic Time-Series Prediction"，3 个数据集（2 合成 + 1 真实）是该领域的标准 benchmark。Ch5 Limitation #3 明确说明推广需额外验证。

---

### Issue 3: 仅对比 Vanilla LSTM
**状态**: [x] 已解决（Ch5 Limitation #1 已覆盖）

Ch5 Limitation #1 明确列出：未评估 GRU、Transformer 等。Ch6 Future Work 建议扩展到其他架构。

---

## 中等问题

### Issue 4: 超参数搜索不对称
**状态**: [x] 已解决（Ch5 Limitation #7 已覆盖）

Ch5 Limitation #7 详细说明了 36 vs 9 的不对称及其原因（ESN 超参数敏感性更高），并指出通过分开报告 tuning cost 使不对称透明化。

---

### Issue 5: n=5 Seeds 统计功效不足
**状态**: [x] 已解决

**修改记录**:
- Ch3 Statistical Analysis: 新增 "Statistical power considerations" 段落，含功效数据和 Wilcoxon 排除论证
- Ch5 Limitations: 新增第8条，量化功效限制
- 添加 bib entry: boneau1960effects, holm1979simple

---

### Issue 6: Multi-step 评估的分布偏移
**状态**: [x] 已解决

**修改记录**:
- Ch5 Limitations: 新增第9条，说明单步训练 vs 自回归评估的分布偏移

---

## 次要问题

### Issue 7: LSTM 训练中 shuffle=True
**状态**: [x] 已解决

**修改记录**:
- Ch5 Limitations: 新增第10条，说明 mini-batch shuffling 和 hidden state 重置

---

### Issue 8: Early stopping 给 LSTM 隐性优势
**状态**: [x] 已解决

**修改记录**:
- Ch5 Limitations: 新增第11条，说明 LSTM early stopping vs PyReCo ridge regression 的正则化不对称

---

### Issue 9: 没有正式的单元测试
**状态**: [x] 不适用（代码工程问题，非论文内容）

---

## 额外结构修复

### Abstract "sustainability metrics" 不准确
**状态**: [x] 已解决
- 改为 "computational efficiency (training time, inference latency)"
- 移除 Wilcoxon 提及

### Ch1 Objectives 提到 Wilcoxon
**状态**: [x] 已解决
- 改为 "paired t-tests with Holm-Bonferroni correction and Cohen's d_z"

### Ch3 Statistical Methods 仍列 Wilcoxon 为主要方法
**状态**: [x] 已解决
- 移除 Wilcoxon 段落，新增 Holm-Bonferroni 和统计功效段落
- Cohen's d → Cohen's d_z (配对公式)

---

## 论文叙述结构问题

### Issue 10: Ch2 "Green computing" gap 承诺未兑现
**状态**: [x] 已解决

Ch2 L295 Gaps in Literature 写 "Environmental cost metrics (training time, **energy consumption, memory usage**) are rarely reported"，但 Ch4 Results 只报告了 training time 和 inference time，没有 energy consumption 和 memory usage。Abstract 已修正，但 Ch2 的 gap 声明仍在。

**需要**: 修改 Ch2 L295 的措辞，与实际交付的指标一致。

---

### Issue 11: Ch4 Results 子节顺序不自然
**状态**: [x] 已解决 — 交换 Multi-Step 和 Data Efficiency 位置

当前: Single-Step → Data Efficiency → Multi-Step → Training Efficiency → Statistics

更自然的叙事流: Single-Step → Multi-Step → Data Efficiency → Training Efficiency → Statistics

当前顺序在读者还没看完预测能力全貌（单步+多步）时就跳到数据量分析，打断了"模型预测能力"的叙事线。

**需要**: 交换 Data Efficiency 和 Multi-Step 的位置。

---

### Issue 12: Ch5 "Monotonic HP Trends" 位置突兀
**状态**: [x] 已解决 — 数据和表格移入 Ch4，理论解读留在 Ch5

Ch5 结构: Key Findings → Parameter Budget → **Monotonic HP Trends** → Efficiency → Limitations → Literature

Monotonic HP Trends 是技术性发现（ESN 超参数在测试范围内单调），夹在宏观讨论之间，打断"结果解读→实用性"的叙事流。

**建议**: 移到 Ch4 Results 作为 HP sensitivity 子节，或移到 Appendix。

---

### Issue 13: Ch2 Related Work / Gaps in Literature 位置
**状态**: [ ] 未解决（低优先级）

Ch2 末尾的 "Gaps in the Literature" 实质上是论证研究动机，功能上属于 Introduction。读者从 Ch1 的 motivation 到 Ch2 末尾才看到具体 gaps，中间隔了 5000 词理论。

**建议**: 可不改（德国高校论文标准结构），但如想更流畅可将 4 条 gaps 移入 Ch1。
