# Feature Selection Module

本模块实现了Lightweight IDS论文中提出的四种特征选择方法,用于降低特征维度并提升轻量级分类器的性能。

## 特征选择方法

### 1. Forward Selection (前向选择)
从空集开始,逐步添加最能提升模型性能的特征。

**优点:**
- 适合特征数量较多的情况
- 可以找到最优的特征子集
- 避免过拟合

**参数:**
- `max_features`: 最大选择特征数
- `min_improvement`: 最小提升阈值

### 2. Backward Elimination (后向消除)
从全集开始,逐步移除对模型性能影响最小的特征。

**优点:**
- 考虑特征之间的交互作用
- 适合特征数量适中的情况

**参数:**
- `min_features`: 最小保留特征数
- `max_drop`: 允许的最大性能下降

### 3. Correlation-based Selection (相关性选择)
选择与目标高度相关但彼此低相关的特征,减少冗余。

**优点:**
- 计算效率高
- 减少特征冗余
- 不需要训练模型

**参数:**
- `threshold`: 特征间相关性阈值
- `max_features`: 最大选择特征数

### 4. Importance-based Selection (重要性选择)
基于树模型的特征重要性选择最重要的特征。

**优点:**
- 直观易懂
- 计算速度快
- 适合树模型

**参数:**
- `top_k`: 选择前k个重要特征
- `threshold`: 重要性阈值

## 使用方法

### 1. 在Pipeline中使用

#### 使用单一方法
```bash
# Forward Selection
python pipeline.py --feature-selection forward --max-features 20

# Backward Elimination
python pipeline.py --feature-selection backward --max-features 15

# Correlation-based Selection
python pipeline.py --feature-selection correlation --max-features 20

# Importance-based Selection
python pipeline.py --feature-selection importance --max-features 20
```

#### 比较所有方法
```bash
python pipeline.py --feature-selection compare --max-features 20
```

#### 禁用特征选择
```bash
python pipeline.py  # 不添加--feature-selection参数
```

### 2. 独立评估脚本

评估和比较不同特征选择方法:

```bash
python evaluate_feature_selection.py \
    --dataset ../lab-ids-anta-main/Dataset/encoded_features_2017.csv \
    --max-features 20 \
    --output-dir ./results/feature_selection_eval
```

输出包括:
- `evaluation_results.csv`: 详细评估结果
- `feature_selection_details.json`: 特征选择详情
- `feature_selection_comparison.png`: 方法比较图
- `features_vs_performance.png`: 特征数量vs性能图

### 3. 在Python代码中使用

```python
from feature_selection import ForwardSelection
from sklearn.linear_model import SGDClassifier

# 创建分类器
clf = SGDClassifier(loss='log_loss', max_iter=1000, random_state=42)

# 创建特征选择器
selector = ForwardSelection(
    clf,
    max_features=20,
    min_improvement=0.001,
    cv=3,
    verbose=True
)

# 拟合并转换数据
X_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

print(f"Selected features: {selector.selected_features_}")
print(f"Number of features: {len(selector.selected_features_)}")
```

### 4. 比较多种方法

```python
from feature_selection import compare_feature_selection_methods
from sklearn.linear_model import SGDClassifier

clf = SGDClassifier(loss='log_loss', max_iter=1000, random_state=42)

results = compare_feature_selection_methods(
    X_train,
    y_train,
    clf,
    methods=['forward', 'backward', 'correlation', 'importance'],
    cv=3,
    verbose=True
)

# 查看结果
for method, result in results.items():
    print(f"{method}: {result['n_features']} features, score={result['score']:.4f}")

# 使用最佳方法
best_method = max(results.items(), key=lambda x: x[1]['score'])
best_selector = best_method[1]['selector']
X_selected = best_selector.transform(X_train)
```

## 配置文件

使用 `config_example.json` 配置特征选择参数:

```json
{
  "feature_selection": {
    "enabled": true,
    "method": "forward",
    "max_features": 20,
    "cv_folds": 3,
    "methods_config": {
      "forward": {
        "max_features": 20,
        "min_improvement": 0.001
      },
      "backward": {
        "min_features": 10,
        "max_drop": 0.01
      },
      "correlation": {
        "threshold": 0.7,
        "max_features": 20
      },
      "importance": {
        "top_k": 20
      }
    }
  }
}
```

## 完整示例

### 示例1: 完整Pipeline with特征选择

```bash
# 运行完整pipeline,使用Forward Selection选择20个特征
python pipeline.py \
    --dataset ../lab-ids-anta-main/Dataset/encoded_features_2017.csv \
    --feature-selection forward \
    --max-features 20 \
    --epsilon 0.3 \
    --epochs 15 \
    --output-dir ./results/with_fs
```

### 示例2: 评估特征选择方法

```bash
# 评估所有特征选择方法
python evaluate_feature_selection.py \
    --dataset ../lab-ids-anta-main/Dataset/encoded_features_2017.csv \
    --max-features 20
```

### 示例3: 比较有无特征选择的效果

```bash
# 不使用特征选择
python pipeline.py --output-dir ./results/no_fs

# 使用特征选择
python pipeline.py \
    --feature-selection forward \
    --max-features 20 \
    --output-dir ./results/with_fs

# 比较results/no_fs/metrics.csv和results/with_fs/metrics.csv
```

## 输出文件

使用特征选择后,会在输出目录生成以下文件:

1. **feature_selection.json**: 特征选择信息
   ```json
   {
     "method": "forward",
     "n_features_original": 78,
     "n_features_selected": 20,
     "selected_features": [0, 5, 10, 15, ...]
   }
   ```

2. **metrics.csv**: 模型评估指标(使用选择后的特征)

3. **X_train_adv.npy / X_test_adv.npy**: 对抗样本(已应用特征选择)

## 性能优化建议

1. **数据集大小**:
   - 小数据集(<10k样本): 使用Forward或Backward
   - 大数据集(>100k样本): 使用Correlation或Importance

2. **特征数量**:
   - 少量特征(<50): 使用Backward
   - 大量特征(>100): 使用Forward或Correlation

3. **计算资源**:
   - 有限计算资源: 使用Correlation或Importance
   - 充足计算资源: 使用Forward或Backward获得最佳效果

4. **交叉验证折数**:
   - 快速实验: `--fs-cv 3`
   - 精确评估: `--fs-cv 5` 或 `--fs-cv 10`

## API文档

### ForwardSelection

```python
ForwardSelection(
    estimator,              # sklearn分类器
    max_features=None,      # 最大特征数
    min_improvement=0.001,  # 最小提升阈值
    scoring='accuracy',     # 评分指标
    cv=3,                   # 交叉验证折数
    verbose=False           # 是否打印详细信息
)
```

### BackwardElimination

```python
BackwardElimination(
    estimator,              # sklearn分类器
    min_features=5,         # 最小特征数
    max_drop=0.01,          # 最大性能下降
    scoring='accuracy',     # 评分指标
    cv=3,                   # 交叉验证折数
    verbose=False           # 是否打印详细信息
)
```

### CorrelationBasedSelection

```python
CorrelationBasedSelection(
    estimator=None,         # sklearn分类器(可选)
    threshold=0.7,          # 相关性阈值
    max_features=None,      # 最大特征数
    scoring='accuracy',     # 评分指标
    cv=3,                   # 交叉验证折数
    verbose=False           # 是否打印详细信息
)
```

### ImportanceBasedSelection

```python
ImportanceBasedSelection(
    estimator,              # 带feature_importances_的分类器
    top_k=None,             # 选择前k个特征
    threshold=None,         # 重要性阈值
    scoring='accuracy',     # 评分指标
    cv=3,                   # 交叉验证折数
    verbose=False           # 是否打印详细信息
)
```

## 常见问题

### Q1: 哪种方法最好?
A: 没有绝对最好的方法,建议使用 `--feature-selection compare` 比较所有方法。

### Q2: 特征选择会提高性能吗?
A: 通常会提高性能并减少过拟合,尤其是在高维数据上。运行评估脚本验证效果。

### Q3: 特征选择需要多长时间?
A: 取决于方法和数据集大小:
- Correlation: 秒级
- Importance: 分钟级
- Forward/Backward: 可能需要10-30分钟

### Q4: 如何选择max_features?
A: 建议从原始特征数的25%-50%开始,如78个特征可选择20-40个。

### Q5: 特征选择后精度下降?
A: 可能原因:
- max_features设置过小,尝试增加
- cv折数过少,增加到5或10
- 数据集不适合该方法,尝试其他方法

## 参考文献

本模块基于以下论文实现:
- SSRN-id4378339: Lightweight IDS with Feature Selection
- 使用轻量级分类器(SGDClassifier, RidgeClassifier)
- 支持对抗训练场景

## 下一步

特征选择模块已完成移植,接下来可以:

1. **测试特征选择**: 运行评估脚本验证效果
2. **集成OWC-SAWN**: 实现对抗网络生成器和判别器
3. **MAB选择器**: 集成Multi-Armed Bandit动态分类器选择
4. **完整实验**: 运行端到端的对抗训练+特征选择+动态选择pipeline

运行测试:
```bash
# 测试特征选择模块
python feature_selection.py

# 评估不同方法
python evaluate_feature_selection.py \
    --dataset ../lab-ids-anta-main/Dataset/encoded_features_2017.csv

# 运行带特征选择的pipeline
python pipeline.py --feature-selection compare
```
