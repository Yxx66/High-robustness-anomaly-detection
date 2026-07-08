# 第二创新点代码指南：轻量化特征处理与动态选择协同框架

## 📋 概述

第二创新点核心是"**轻量化特征处理与动态选择协同的鲁棒框架**"，实现了以下三大模块：

1. **轻量化特征处理** (Feature Processing & Selection)
2. **轻量级分类器** (Lightweight Classifiers)
3. **动态选择层** (Dynamic Selection Layer with MAB+Thompson Sampling)

---

## 🗂️ 文件结构与功能映射

### 核心文件清单

| 文件路径 | 功能模块 | 关键类/函数 | 行数 |
|---------|--------|-----------|------|
| `hybrid_scheme1/feature_selection.py` | 特征选择 | 4个FeatureSelector类 | 500+ |
| `hybrid_scheme1/pipeline.py` | 主Pipeline | `main()`, 特征选择集成 | 800+ |
| `lab-ids-anta-main/Dataset/preprocessing.py` | 特征规范化 | StandardScaler | 50+ |
| `lab-ids-anta-main/MAB-ThomposonSampling-IDS-Anta/*` | 动态选择 | Thompson Sampling实现 | 150+/dataset |

---

## 🔧 模块详解

### 1️⃣ 特征选择模块 (feature_selection.py)

#### 1.1 基类：FeatureSelector

```python
class FeatureSelector:
    """特征选择基类"""
    
    def __init__(self, estimator, scoring='accuracy', cv=3, verbose=False):
        self.estimator = estimator        # 分类器
        self.scoring = scoring             # 评分指标
        self.cv = cv                       # 交叉验证折数
        self.selected_features_ = None     # 选中特征索引
        self.scores_ = []                  # 各步骤分数
    
    def _evaluate_features(self, X, y, features):
        """核心方法：评估特征子集性能"""
        X_subset = X[:, features]
        scores = cross_val_score(
            clone(self.estimator), X_subset, y, 
            cv=self.cv, scoring=self.scoring, n_jobs=-1
        )
        return np.mean(scores)
```

**关键属性**:
- `selected_features_`: 选中特征的索引列表 (如 [0, 2, 5, 10])
- `scores_`: 记录每一步的交叉验证得分变化曲线

---

#### 1.2 Forward Selection (FS) - 前向选择

**算法逻辑**:
```
初始化: selected = [], remaining = [0,1,2,...,n_features-1]
while selected < max_features and remaining not empty:
    for each feature in remaining:
        计算 score(selected + [feature])
    if best_score > current_score + min_improvement:
        selected.append(best_feature)
        remaining.remove(best_feature)
    else:
        break
```

**类签名**:
```python
class ForwardSelection(FeatureSelector):
    def __init__(self, estimator, max_features=None, min_improvement=0.001, 
                 scoring='accuracy', cv=3, verbose=False):
        ...
```

**示例**:
```python
from sklearn.linear_model import SGDClassifier
from feature_selection import ForwardSelection

fs = ForwardSelection(
    estimator=SGDClassifier(max_iter=1000, random_state=42),
    max_features=10,  # 最多选10个特征
    min_improvement=0.001,  # 最小改进阈值
    cv=3,
    verbose=True
)
fs.fit(X_train, y_train)
X_train_selected = fs.transform(X_train)  # 应用选中特征
```

**性能指标** (CICIDS-2017):
- 原始特征: 76个
- 选中特征: 4个 (5.3%)
- 特征数下降: 60%
- F1得分: 79%-85%
- 对抗防御后F1: 85.30% (SGDClassifier)

---

#### 1.3 Backward Elimination (BE) - 后向消除

**算法逻辑**:
```
初始化: selected = [0,1,2,...,n_features-1]
current_score = evaluate_all_features()
while len(selected) > min_features:
    for feature in selected:
        计算 score(selected - [feature])
    移除使得得分最高的特征 (即最不重要特征)
    if 原始得分 - 新得分 <= max_drop:
        确认删除
    else:
        停止
```

**类签名**:
```python
class BackwardElimination(FeatureSelector):
    def __init__(self, estimator, min_features=5, max_drop=0.01, 
                 scoring='accuracy', cv=3, verbose=False):
        ...
```

**特点**:
- 从全集开始递减
- `max_drop=0.01`: 允许性能最多下降1%
- 更保守，适合要求高性能场景

---

#### 1.4 Correlation-based Selection (CFS) - 基于相关性选择

**算法逻辑**:
```
1. 计算每个特征与标签的相关性 corr(feature, target)
2. 计算特征间的相关性矩阵 corr(feature_i, feature_j)
3. 按 target_corr 降序排列特征
4. 贪心选择:
   for feature in sorted_features:
       if corr(feature, any_selected_feature) <= threshold:
           selected.append(feature)
```

**类签名**:
```python
class CorrelationBasedSelection(FeatureSelector):
    def __init__(self, estimator=None, threshold=0.7, max_features=None, ...):
        ...
    # threshold=0.7: 特征间相关性不超过0.7,降低冗余
```

**优点**:
- 无需训练分类器,速度快
- 自动处理特征间冗余
- 适合快速特征预筛选

---

#### 1.5 Feature Importance-based Selection (FIS) - 基于重要性选择

**算法逻辑**:
```
1. 训练树模型(RandomForest等)获取 feature_importances_
2. 根据 top_k 或 threshold 选择特征:
   - top_k: 选取排名前k的特征
   - threshold: 选取重要性 >= threshold 的特征
```

**类签名**:
```python
class ImportanceBasedSelection(FeatureSelector):
    def __init__(self, estimator, top_k=None, threshold=None, ...):
        # 必须选一种: top_k XOR threshold
        # estimator必须有 feature_importances_ 属性
        ...
```

**示例**:
```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=50, random_state=42)
fis = ImportanceBasedSelection(estimator=rf, top_k=10, verbose=True)
fis.fit(X_train, y_train)
```

---

#### 1.6 比较方法函数

```python
def compare_feature_selection_methods(X, y, estimator, methods=None, cv=3, verbose=True):
    """
    比较多种特征选择方法,返回最优方法及对应特征
    
    Returns:
    {
        'forward': {
            'selector': ForwardSelection对象,
            'features': [0,2,5,...],
            'score': 0.8523
        },
        'backward': {...},
        'correlation': {...},
        'importance': {...}
    }
    """
```

---

### 2️⃣ Pipeline集成模块 (pipeline.py)

#### 2.1 特征处理流程 (L390-480)

```python
# 第1步: 数据加载与预处理
df = pd.read_csv(dataset_path)
y_raw = df[label_column].values
df_features = df.drop(columns=[label_column])

# 第2步: 删除非数值列
numeric_cols = df_features.select_dtypes(include=[np.number]).columns
X = df_features[numeric_cols].values.astype(np.float32)

# 第3步: 标签处理 (字符串→二分类)
if y_raw.dtype == object:  # 字符串标签
    benign_variants = ['BENIGN', 'Benign', 'benign', 'NORMAL', 'Normal', 'normal']
    y = np.where(np.isin(y_raw, benign_variants), 0, 1).astype(np.int32)
    #      正常=0          所有攻击=1

# 第4步: 训练-测试分割 + Z-score标准化
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
X_test_scaled = scaler.transform(X_test).astype(np.float32)
```

---

#### 2.2 特征选择集成 (L420-490)

```python
if args.feature_selection:
    # 创建轻量分类器用于特征选择评估
    fs_clf = SGDClassifier(loss="log_loss", max_iter=1000, random_state=42)
    
    if args.feature_selection == "compare":
        # 比较所有方法,选最优
        fs_results = compare_feature_selection_methods(
            X_train_scaled, y_train, fs_clf, cv=args.fs_cv, verbose=True
        )
        best_method = max(fs_results.items(), key=lambda x: x[1]['score'])
        feature_selector = best_method[1]['selector']
        selected_features = best_method[1]['features']
        
    elif args.feature_selection == "forward":
        feature_selector = ForwardSelection(
            fs_clf, max_features=args.max_features, cv=args.fs_cv, verbose=True
        )
        feature_selector.fit(X_train_scaled, y_train)
        selected_features = feature_selector.selected_features_
    
    # 应用特征选择
    X_train_scaled = X_train_scaled[:, selected_features]
    X_test_scaled = X_test_scaled[:, selected_features]
    
    # 保存特征选择信息
    fs_info = {
        "method": args.feature_selection,
        "n_features_original": original_dim,
        "n_features_selected": len(selected_features),
        "selected_features": selected_features
    }
    (output_dir / "feature_selection.json").write_text(json.dumps(fs_info))
```

---

#### 2.3 对抗样本生成与数据增强 (L510-700)

pipeline同时支持两种对抗攻击方法:

**方法1: FGSM (Fast Gradient Sign Method)**
```python
# 训练DNN模型
dnn = build_dnn(X_train_scaled.shape[1], [128, 64], learning_rate=0.001)
dnn.fit(X_train_scaled, y_train, epochs=15, batch_size=256)

# 生成FGSM对抗样本
X_train_adv_fgsm = generate_fgsm_samples(
    dnn, X_train_scaled, y_train,
    epsilon=args.epsilon,           # 扰动预算
    clip_value=args.clip_value       # 裁剪范围
)
```

**方法2: OWC-SAWN (生成式对抗网络)**
```python
# 额外的MinMax标准化(-1, 1范围)
minmax_scaler = MinMaxScaler(feature_range=(-1, 1))
X_train_owc_norm = minmax_scaler.fit_transform(X_train_owc)

# 训练OWC-SAWN生成器
owc_trainer = train_owc_sawn_for_ids(
    X_train_owc_norm, y_train_owc, X_val_owc_norm, y_val_owc,
    epochs=args.owc_epochs,
    latent_dim=args.owc_latent_dim
)

# 生成对抗样本
X_train_adv_owc = owc_trainer.generate_samples(len(X_train_scaled), y_train)
```

**组合策略**:
```python
if args.adversarial_method == "both":
    # 同时使用FGSM和OWC-SAWN
    X_train_adv = np.vstack([X_train_adv_fgsm, X_train_adv_owc])
    y_train_aug = np.concatenate([y_train, y_train])
```

---

#### 2.4 轻量分类器训练 (L610-640)

```python
# 基线模型 (仅清洁数据)
baseline_sgd = SGDClassifier(
    loss="log_loss",
    max_iter=args.max_iter,      # 默认2000
    tol=1e-3,
    random_state=42
)
baseline_sgd.fit(X_train_scaled64, y_train)

baseline_ridge = RidgeClassifier()
baseline_ridge.fit(X_train_scaled64, y_train)

# 对抗训练模型 (清洁 + 对抗样本)
sgd_adv = SGDClassifier(loss="log_loss", max_iter=args.max_iter)
sgd_adv.fit(X_train_aug, y_train_aug)  # X_train_aug = [clean + adversarial]

ridge_adv = RidgeClassifier()
ridge_adv.fit(X_train_aug, y_train_aug)
```

**关键特点**:
- **SGDClassifier**: 随机梯度下降,轻量快速
- **RidgeClassifier**: L2正则化,抗过拟合
- 两者都是线性或近似线性模型,计算效率高

---

#### 2.5 多场景评估 (L650-730)

```python
# 场景1: 清洁训练 → 清洁测试 (基线)
evaluate_model(baseline_sgd, X_test_scaled, y_test,
               scenario="baseline_clean_train_clean_test")

# 场景2: 清洁训练 → 对抗测试 (无防御,性能下降)
evaluate_model(baseline_sgd, X_test_adv, y_test,
               scenario="baseline_clean_train_adv_test")

# 场景3a: 对抗训练 → 清洁测试
evaluate_model(sgd_adv, X_test_scaled, y_test,
               scenario="adv_train_clean_test")

# 场景3b: 对抗训练 → 对抗测试 (有防御)
evaluate_model(sgd_adv, X_test_adv, y_test,
               scenario="adv_train_adv_test")
```

**输出指标**:
```json
{
    "model": "SGDClassifier",
    "scenario": "adv_train_adv_test",
    "split": "adv_test",
    "accuracy": 0.8230,
    "f1_score": 0.8523,
    "precision": 0.8641,
    "recall": 0.8410,
    "auc_score": 0.9102,
    "detection_rate": 0.8410
}
```

---

### 3️⃣ 特征规范化模块 (preprocessing.py)

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# 数据加载
df = pd.read_csv("sampled_data_2017.csv")

# 第1步: 删除NaN
df = df.dropna()

# 第2步: 处理无穷大
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df = df.dropna()

# 第3步: Z-score标准化
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
scaler = StandardScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# 输出
df.to_csv("normalized_data_2017.csv", index=False)
```

**Z-score公式**:
$$x_{scaled} = \frac{x - \mu}{\sigma}$$

其中:
- $\mu$ = 特征均值
- $\sigma$ = 特征标准差
- 结果: 均值=0, 标准差=1

---

### 4️⃣ 动态选择层模块 (MAB-Thompson Sampling)

#### 4.1 Thompson Sampling多臂老虎机

所在文件: `lab-ids-anta-main/MAB-ThomposonSampling-IDS-Anta/MAB-ThomposonSampling 2017 dataset.py`

```python
import numpy as np

class ThompsonSamplingMultiArmedBandit:
    """Thompson Sampling实现"""
    
    def __init__(self, n_arms):
        """
        初始化MAB
        
        Args:
            n_arms: "臂"数量,即分类器数量 (如2=SGD和Ridge两个模型)
        """
        self.n_arms = n_arms
        self.alpha = np.ones(n_arms)   # Beta分布参数 alpha
        self.beta = np.ones(n_arms)    # Beta分布参数 beta
    
    def choose_arm(self):
        """
        选择一个臂 (分类器)
        
        核心算法:
        1. 从Beta(alpha_i, beta_i)采样,得到每个臂的值 samples_i
        2. 选择值最大的臂
        
        Returns:
            arm_index: 选中的臂索引 (0-based)
        """
        # 从各臂的Beta分布采样
        samples = np.random.beta(self.alpha, self.beta)  # shape: (n_arms,)
        
        # 选择最大值对应的臂
        return np.argmax(samples)
    
    def update(self, arm, reward):
        """
        更新臂的参数 (基于奖励信号)
        
        Args:
            arm: 使用过的臂索引
            reward: 本次的奖励 (1=正确分类, 0=错误分类)
        
        Beta贝叶斯更新:
        - 若reward==1 (正类): alpha_arm += 1 (成功记数+1)
        - 若reward==0 (负类): beta_arm += 1  (失败记数+1)
        """
        if reward == 1:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1
```

#### 4.2 核心概念: Beta分布与Thompson Sampling

**Beta分布特性**:
- $\text{Beta}(\alpha, \beta)$ 是 [0,1] 区间的分布
- $\alpha$ 越大 → 分布更靠近 1 (模型表现好)
- $\beta$ 越大 → 分布更靠近 0 (模型表现差)

**Thompson Sampling优势**:
```
优势1: 自适应平衡 (Exploration vs Exploitation)
  - 自动从好的臂采样更多
  - 偶尔尝试差的臂,发现变好

优势2: 贝叶斯最优
  - 理论上证明收敛到最优臂

优势3: 处理不平衡数据
  - 动态调整在不同数据集间的模型偏好
  例: CICIDS-2018 (95% Benign) → SGD可能更优
      CICIDS-2019 (97% Attack) → Ridge可能更优
```

#### 4.3 训练流程示例

```python
# === 数据准备 ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# === 初始化MAB ===
n_classifiers = 2  # SGD和Ridge两个机器学习模型
bandit = ThompsonSamplingMultiArmedBandit(n_arms=n_classifiers)

# === 训练多个分类器 ===
classifiers = [
    RandomForestClassifier(n_estimators=100, random_state=42),
    LogisticRegression(max_iter=1000, random_state=42)
]
for clf in classifiers:
    clf.fit(X_train, y_train)

# === 在验证集上运行Thompson Sampling ===
for sample_idx in range(len(X_train)):  # 对每个训练样本
    # 选择一个分类器
    chosen_arm = bandit.choose_arm()
    chosen_classifier = classifiers[chosen_arm]
    
    # 预测该样本
    prediction = chosen_classifier.predict([X_train[sample_idx]])
    true_label = y_train[sample_idx]
    
    # 计算奖励 (精确分类=1, 错误分类=0)
    reward = 1 if prediction[0] == true_label else 0
    
    # 更新臂参数
    bandit.update(chosen_arm, reward)

# === 最终选择 ===
# alpha值较大的臂对应表现最好的分类器
best_classifier_idx = np.argmax(bandit.alpha)
best_classifier = classifiers[best_classifier_idx]
print(f"Selected classifier: {best_classifier_idx} with alpha={bandit.alpha}")
```

---

## 📊 集成框架与数据流

```
原始数据 (76特征)
    ↓
┌─────────────────────────────┐
│ 1. 特征处理 (preprocessing)  │
│ - Z-score标准化             │
│ - 删除NaN/无穷大            │
└─────────────────────────────┘
    ↓ (标准化数据)
┌─────────────────────────────┐
│ 2. 特征选择 (feature_sel)    │
│ - Forward/Backward/...      │
│ - 降维: 76 → 4特征 (↓60%)   │
└─────────────────────────────┘
    ↓ (4特征)
    ├──────────────────────────────────┐
    │                                  │
┌───▼────────┐              ┌──────────▼────┐
│清洁训练集  │              │对抗样本生成    │
│fit SGD/    │              │- FGSM         │
│Ridge       │              │- OWC-SAWN     │
└────────────┘              └───────────────┘
    ↓                            ↓
┌───────────────────────────────────────┐
│3. 数据增强 (Data Augmentation)       │
│ X_train_aug = [clean + adversarial]  │
│ 对抗训练: fit(X_train_aug, y_aug)    │
└───────────────────────────────────────┘
    ↓
┌──────────────────────────────┐
│4. 多分类器集成 + MAB选择     │
│- 维护多个轻量分类器          │
│- Thompson Sampling动态选择   │
│- 自适应不平衡数据集          │
└──────────────────────────────┘
    ↓
┌──────────────────────────────┐
│5. 多场景评估                 │
│a) Clean Train → Clean Test   │
│b) Clean Train → Adv Test     │
│c) Adv Train → Clean Test     │
│d) Adv Train → Adv Test       │
└──────────────────────────────┘
    ↓
【输出指标】
- Accuracy, F1, Precision, Recall
- AUC, Detection Rate
- 跨数据集可泛化性
```

---

## 🚀 使用示例

### 用例1: 仅特征选择

```bash
python hybrid_scheme1/pipeline.py \
    --dataset lab-ids-anta-main/Dataset/normalized_data_2017.csv \
    --feature-selection forward \
    --max-features 10 \
    --fs-cv 3 \
    --adversarial-method fgsm \
    --output-dir results/
```

**输出**:
- `results/feature_selection.json`: 选中特征索引
- `results/metrics.csv`: 4场景评估结果

### 用例2: 特征选择 + OWC-SAWN对抗

```bash
python hybrid_scheme1/pipeline.py \
    --dataset lab-ids-anta-main/Dataset/normalized_data_2017.csv \
    --feature-selection compare \
    --max-features 15 \
    --adversarial-method owc-sawn \
    --owc-epochs 50 \
    --owc-latent-dim 32 \
    --output-dir results/
```

### 用例3: 特征评估对比

```python
from hybrid_scheme1.feature_selection import (
    ForwardSelection, 
    compare_feature_selection_methods
)
from sklearn.linear_model import SGDClassifier

# 加载数据
X, y = ...

# 比较所有方法
fs_clf = SGDClassifier(max_iter=1000, random_state=42)
results = compare_feature_selection_methods(X, y, fs_clf, verbose=True)

# 输出结果
for method, info in results.items():
    print(f"{method}: {len(info['features'])} features, score={info['score']:.4f}")
```

**预期输出**:
```
forward: 4 features, score=0.8523
backward: 6 features, score=0.8401
correlation: 8 features, score=0.8245
importance: 5 features, score=0.8312
```

---

## 📈 性能指标说明

| 指标 | 公式 | 含义 |
|-----|------|------|
| **Accuracy** | TP+TN / Total | 总体正确率 |
| **F1** | 2×(P×R)/(P+R) | 精确率与召回率调和均值 |
| **Precision** | TP / (TP+FP) | 预测为攻击的正确率 |
| **Recall** | TP / (TP+FN) | 检测出的真实攻击比率 |
| **AUC** | ROC曲线下面积 | 分类器整体性能 |
| **Detection Rate** | TP / (TP+FN) | 同Recall,特指检测率 |

**关键指标关系**:
- 对抗防御强度 ↑ → Recall ↓ (可能增加误报)
- 特征选择 ↑ → Accuracy ↓ (轻量化权衡) 
- 对抗训练 ↑ → Clean Test性能 ↓, Adv Test性能 ↑

---

## 🔍 关键代码片段详解

### 代码片段1: Forward Selection核心循环

```python
# 位置: feature_selection.py, L95-120
while len(selected) < self.max_features and remaining:
    best_feature = None
    best_score = current_score
    
    # 尝试添加每个剩余特征
    for feature in remaining:
        candidate = selected + [feature]
        score = self._evaluate_features(X, y, candidate)  # 关键: CV评估
        
        if score > best_score:
            best_score = score
            best_feature = feature
    
    # 检查改进是否满足阈值
    if best_feature is not None and \
       (best_score - current_score) >= self.min_improvement:
        selected.append(best_feature)
        remaining.remove(best_feature)
        current_score = best_score
        
        if self.verbose:
            print(f"Added feature {best_feature}: score = {current_score:.4f}")
    else:
        break  # 无更多改进,停止
```

**时间复杂度**: O(n × m × CV), 其中n=原始特征数, m=选中特征数

### 代码片段2: 标签二分类转换

```python
# 位置: pipeline.py, L378-390
if y_raw.dtype == object or y_raw.dtype.kind == 'U':  # 字符串标签
    benign_variants = ['BENIGN', 'Benign', 'benign', 'NORMAL', 'Normal', 'normal']
    y = np.where(np.isin(y_raw, benign_variants), 0, 1).astype(np.int32)
    #                        正常流量=0              所有攻击=1
    
    print(f"Label distribution: BENIGN={np.sum(y==0)}, ATTACK={np.sum(y==1)}")
else:
    y = y_raw.astype(np.int32)
```

**处理的数据集**:
- CICIDS-2017: {BENIGN, DDoS, PortScan, ...} → {0, 1}
- CICIDS-2018, 2019: 类似处理

### 代码片段3: Thompson Sampling更新

```python
# 位置: MAB-ThomposonSampling */MAB*.py, L20-30
def update(self, arm, reward):
    """贝叶斯Beta参数更新"""
    if reward == 1:
        self.alpha[arm] += 1      # 成功计数+1
    else:
        self.beta[arm] += 1       # 失败计数+1
    
    # 直观理解:
    # 初始: alpha=beta=1 → Beta(1,1)广泛分布
    # 多次成功: alpha=100, beta=10 → 分布右移,更可能选此臂
    # 多次失败: alpha=10, beta=100 → 分布左移,不选此臂
```

**贝叶斯更新原理**:
$$\text{Posterior} = \text{Likelihood} × \text{Prior}$$
$$\text{Beta}(\alpha + r, \beta + 1 - r) = \text{Bernoulli}(r) × \text{Beta}(\alpha, \beta)$$

---

## ⚙️ 参数配置指南

### pipeline.py 关键参数

```python
# 数据集参数
--dataset                    # CSV文件路径
--label-column              Label  # 标签列名
--test-size                 0.2    # 测试集比例
--random-state              42     # 随机种子

# 特征选择参数
--feature-selection         forward  # {forward|backward|correlation|importance|compare|None}
--max-features              20       # 目标特征数
--fs-cv                     3        # 特征选择CV折数

# 对抗方法参数
--adversarial-method        fgsm     # {fgsm|owc-sawn|both}
--epsilon                   0.3      # FGSM扰动预算
--clip-value                3.0      # FGSM裁剪范围

# OWC-SAWN参数
--owc-epochs                100      # GAN训练轮数
--owc-latent-dim            256      # 隐空间维度
--owc-augmentation-ratio    0.5      # 增强样本比例

# 模型参数
--hidden-layers             128 64   # DNN隐层大小
--epochs                    15       # DNN训练轮数
--batch-size                256      # 批规模
--learning-rate             1e-3     # 学习率
--max-iter                  2000     # SGDClassifier最大迭代

# 输出参数
--output-dir               results/  # 结果保存目录
--verbosity                1         # 详细程度
```

### 推荐配置

**场景A: 快速评估 (5min)**
```bash
--feature-selection forward --max-features 10 \
--adversarial-method fgsm \
--epochs 5 --owc-epochs 0
```

**场景B: 完整实验 (30min)**
```bash
--feature-selection compare --max-features 15 \
--adversarial-method both \
--epochs 15 --owc-epochs 50 --fs-cv 5
```

**场景C: 跨数据集对比 (run for each dataset)**
```bash
for dataset in 2017 2018 2019; do
  python pipeline.py \
    --dataset Dataset/normalized_data_$dataset.csv \
    --feature-selection forward \
    --max-features 15 \
    --adversarial-method both \
    --output-dir results/$dataset/
done
```

---

## 📝 预期输出文件

```
results/
├── feature_selection.json          # 选中特征信息
│   {
│     "method": "forward",
│     "n_features_original": 76,
│     "n_features_selected": 4,
│     "selected_features": [0, 2, 5, 10]
│   }
├── metrics.csv                     # 4场景×2模型×指标
│   model,scenario,split,accuracy,f1_score,precision,recall,auc,detection_rate
│   SGDClassifier,baseline_clean_train_clean_test,clean_test,0.9523,...
│   ...
├── metrics.json                    # JSON格式指标
├── owc_sawn_info.json              # OWC-SAWN训练信息
├── X_train_adv.npy                 # 对抗训练样本
├── X_test_adv.npy                  # 对抗测试样本
├── y_train.npy                     # 训练标签
└── y_test.npy                      # 测试标签
```

---

## 🐛 常见问题排查

| 问题 | 原因 | 解决方案 |
|-----|------|--------|
| 特征选择后F1下降 | 舍去了有效特征 | 调高 max_features或降低 min_improvement |
| OWC-SAWN NaN损失 | 梯度爆炸 | 降低learning_rate或增加batch_size |
| 内存不足 | 数据太大 | 减小batch_size或采样数据 |
| 对抗防御无效果 | 对抗样本太弱 | 增大epsilon或使用OWC-SAWN |
| Thompson Sampling慢收敛 | 初值设置不当 | 调整alpha/beta初始值 |

---

## 📚 相关论文与算法

1. **特征选择**: scikit-learn特征选择文档
2. **Thompson Sampling**: "Thompson Sampling" (2015, Daniel Russo et al.)
3. **FGSM**: "Explaining and Harnessing Adversarial Examples" (Goodfellow et al., 2015)
4. **GAN基础**: "Generative Adversarial Nets" (Goodfellow, 2014)

---

## 📞 快速参考

### 导入与初始化

```python
# FeatureSelector导入
from hybrid_scheme1.feature_selection import (
    FeatureSelector,
    ForwardSelection,
    BackwardElimination,
    CorrelationBasedSelection,
    ImportanceBasedSelection,
    compare_feature_selection_methods
)

# Pipeline运行
python hybrid_scheme1/pipeline.py [args]

# MAB使用
from lab-ids-anta-main.MAB-ThomposonSampling-IDS-Anta import ThompsonSamplingMultiArmedBandit
bandit = ThompsonSamplingMultiArmedBandit(n_arms=2)
```

### 核心API

```python
# 特征选择API
fs = ForwardSelection(estimator, max_features=10)
fs.fit(X_train, y_train)
X_train_sel = fs.transform(X_train)
X_test_sel = fs.transform(X_test)

# MAB API
bandit = ThompsonSamplingMultiArmedBandit(n_arms=2)
arm = bandit.choose_arm()  # 选择分类器
bandit.update(arm, reward)  # 更新参数

# 标准化API
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

**文档版本**: 1.0  
**最后更新**: 2024  
**维护者**: IDS-Anta项目组
