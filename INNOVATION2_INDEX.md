# 第二创新点代码文档索引

> **快速导航**: 本文件帮助快速定位第二创新点"轻量化特征处理与动态选择协同框架"的核心代码与文档

---

## 📚 文档清单

### 1. 本索引文件
- **文件**: `INNOVATION2_INDEX.md` (本文)
- **内容**: 文档导航、快速查询
- **用途**: 新开发者上手指引
- **阅读时间**: 5分钟

### 2. 快速参考手册
- **文件**: [`INNOVATION2_QUICKREF.md`](INNOVATION2_QUICKREF.md)
- **内容**: 
  - 🎯 核心概览表 (模块对应表)
  - 🔄 数据流动图 (Mermaid可视化)
  - 📐 算法对比表 (4种特征选择)
  - 📊 性能基准表 (CICIDS-2017)
  - 🚀 一行命令速查
- **用途**: 快速理解框架、参数调优
- **阅读时间**: 15分钟

### 3. 详细代码指南
- **文件**: [`INNOVATION2_CODE_GUIDE.md`](INNOVATION2_CODE_GUIDE.md)
- **内容** (5000+字):
  - 📋 概述 (3大模块)
  - 🗂️ 文件结构与映射
  - 🔧 **模块详解**
    - 1️⃣ 特征选择 (4种方法: FS/BE/CFS/FIS)
    - 2️⃣ Pipeline集成 (L390-750)
    - 3️⃣ 特征规范化 (Z-score)
    - 4️⃣ Thompson Sampling (MAB)
  - 📊 集成框架与数据流
  - 🚀 使用示例 (3个用例)
  - ⚙️ 参数配置指南
  - 📝 预期输出文件
  - 🐛 常见问题排查
- **用途**: 代码开发、算法理解、问题排查
- **阅读时间**: 45分钟

---

## 🗂️ 源代码文件导航

### 核心代码文件

#### 1. 特征选择模块
```
📄 feature_selection.py (500+ 行)
├── FeatureSelector (基类, L20-70)
│   ├── __init__: scoring, cv, selected_features_
│   ├── fit(): 抽象方法
│   └── _evaluate_features(): 核心评估函数 (L60-75)
│
├── ForwardSelection (L85-130)
│   ├── fit(): 贪心添加特征
│   └── 特性: 快速, 常用
│
├── BackwardElimination (L155-200)
│   ├── fit(): 递减删除特征
│   └── 特性: 保守, 高精度
│
├── CorrelationBasedSelection (L225-300)
│   ├── fit(): 相关性消除冗余
│   └── 特性: 无需训练, 最快
│
├── ImportanceBasedSelection (L325-400)
│   ├── fit(): 树模型重要性筛选
│   └── 特性: 需特定模型, 准确
│
└── compare_feature_selection_methods() (L425-480)
    └── 返回: {method: {selector, features, score}}
```

**快速查询**: 
- "我要10个特征" → ForwardSelection (line 95)
- "特征太多" → CorrelationBasedSelection (line 250)
- "需要最优" → compare_feature_selection_methods (line 430)

---

#### 2. Pipeline集成
```
📄 pipeline.py (800+ 行)
├── build_dnn() (L45-58)
│   └── 构建DNN用于FGSM生成
│
├── generate_fgsm_samples() (L61-82)
│   ├── 输入: model, X, y, epsilon
│   └── 输出: X_adv_fgsm
│
├── train_owc_sawn_for_ids() (L83-180)
│   ├── 输入: X_train, y_train, X_val, y_val
│   └── 输出: trainer (包含生成器)
│
├── evaluate_model() (L181-227)
│   ├── 输入: classifier, X_test, y_test, scenario
│   └── 输出: 指标字典 (F1, AUC, etc.)
│
├── parse_args() (L228-360)
│   ├── 数据集参数
│   ├── 特征选择参数 (L315-328)
│   ├── 对抗方法参数 (L334-359)
│   └── 模型参数 (L273-347)
│
└── main() (L359-770)
    ├── 数据加载与预处理 (L390-410)
    ├── 特征选择 (L420-490)
    ├── FGSM生成 (L508-535)
    ├── OWC-SAWN训练 (L540-575)
    ├── 数据增强 (L610-615)
    ├── 轻量分类器训练 (L620-640)
    ├── 多场景评估 (L650-730)
    └── 结果保存 (L740-770)
```

**快速查询**:
- "如何做特征选择" → main() L420 (line 420)
- "如何生成对抗样本" → main() L508-575 (line 508)
- "如何训练分类器" → main() L620 (line 620)
- "如何评估模型" → evaluate_model() (line 181)

---

#### 3. 特征规范化
```
📄 lab-ids-anta-main/Dataset/preprocessing.py (50+ 行)
├── pd.read_csv() → df
├── df.dropna() → 删除空值
├── df.replace([np.inf, -np.inf], np.nan) → 处理无穷大
├── StandardScaler().fit_transform() → Z-score标准化
│   ├── 公式: x' = (x - μ) / σ
│   └── 结果: μ=0, σ=1
└── df.to_csv() → 保存标准化数据
```

**快速查询**:
- "如何标准化数据" → StandardScaler() (line 18)
- "数据预处理步骤" → preprocessing.py (line 1-50)

---

#### 4. Thompson Sampling
```
📄 lab-ids-anta-main/MAB-ThomposonSampling-IDS-Anta/
    MAB-ThomposonSampling 201X dataset.py (150+ 行/dataset)

├── ThompsonSamplingMultiArmedBandit (L1-35)
│   ├── __init__(n_arms)
│   │   ├── self.alpha = ones(n_arms)
│   │   └── self.beta = ones(n_arms)
│   │
│   ├── choose_arm() (L15-20)
│   │   ├── samples = np.random.beta(alpha, beta)
│   │   └── return argmax(samples)
│   │
│   └── update(arm, reward) (L22-28)
│       ├── if reward==1: alpha[arm] += 1
│       └── else: beta[arm] += 1
│
├── 分类器定义
│   ├── RandomForestClassifier (L40-60)
│   ├── LogisticRegression (L70-90)
│   └── [其他分类器]
│
└── 训练与评估
    ├── for i in range(len(X_train)):
    │   ├── arm = bandit.choose_arm()
    │   ├── prediction = classifier[arm].predict(X_train[i])
    │   ├── reward = (prediction == y_train[i])
    │   └── bandit.update(arm, reward)
    │
    └── 最终: best_arm = argmax(alpha)
```

**快速查询**:
- "Thompson Sampling怎么工作" → choose_arm() (line 18)
- "如何更新MAB参数" → update() (line 26)
- "如何选择最优分类器" → argmax(alpha) (line 140)

---

## 🔄 核心流程速查

### 流程1: 特征选择

```
1. 加载数据
   data = pd.read_csv('data.csv')
   X, y = data.drop('Label', axis=1), data['Label']

2. 创建分类器用于评估
   fs_clf = SGDClassifier(max_iter=1000)

3. 创建特征选择器
   selector = ForwardSelection(fs_clf, max_features=10, cv=3)

4. 拟合并应用
   selector.fit(X_train, y_train)
   X_train_selected = selector.transform(X_train)
   X_test_selected = selector.transform(X_test)

5. 输出结果
   print(selector.selected_features_)  # [0, 2, 5, 10, ...]
```

**文件**: pipeline.py L420-490  
**输出**: feature_selection.json

---

### 流程2: 对抗样本生成

#### 方法A: FGSM
```
1. 构造及训练DNN
   dnn = build_dnn(input_dim, [128, 64], lr=0.001)
   dnn.fit(X_train_scaled, y_train, epochs=15)

2. 生成FGSM样本
   X_train_adv = generate_fgsm_samples(
       dnn, X_train_scaled, y_train,
       epsilon=0.3, clip_value=3.0
   )
   X_test_adv = generate_fgsm_samples(
       dnn, X_test_scaled, y_test,
       epsilon=0.3, clip_value=3.0
   )
```

**文件**: pipeline.py L508-535  
**时间**: ~2 min

#### 方法B: OWC-SAWN
```
1. MinMax标准化到[-1, 1]
   minmax_scaler = MinMaxScaler((-1, 1))
   X_train_norm = minmax_scaler.fit_transform(X_train)

2. 训练OWC-SAWN
   owc_trainer = train_owc_sawn_for_ids(
       X_train_norm, y_train, X_val_norm, y_val_norm,
       epochs=50, latent_dim=256
   )

3. 生成对抗样本
   X_train_adv_norm = owc_trainer.generate_samples(
       len(X_train), y_train
   )
   X_train_adv = minmax_scaler.inverse_transform(X_train_adv_norm)
```

**文件**: pipeline.py L540-600  
**时间**: ~20 min

---

### 流程3: 轻量分类器训练

```
1. 定义基线模型 (清洁数据)
   baseline_sgd = SGDClassifier(max_iter=2000)
   baseline_sgd.fit(X_train_scaled, y_train)

2. 定义对抗训练模型
   sgd_adv = SGDClassifier(max_iter=2000)
   X_train_aug = np.vstack([X_train_scaled, X_train_adv])
   y_train_aug = np.concatenate([y_train, y_train])
   sgd_adv.fit(X_train_aug, y_train_aug)

3. 评估
   metrics_baseline = evaluate_model(
       baseline_sgd, X_test_scaled, y_test,
       scenario="baseline_clean_train_clean_test"
   )
   metrics_adv = evaluate_model(
       sgd_adv, X_test_adv, y_test,
       scenario="adv_train_adv_test"
   )
```

**文件**: pipeline.py L613-640  
**输出**: metrics.csv

---

### 流程4: Thompson Sampling动态选择

```
1. 初始化MAB
   bandit = ThompsonSamplingMultiArmedBandit(n_arms=2)
   classifiers = [SGDClassifier(), RidgeClassifier()]

2. 在验证集上训练MAB
   for i in range(len(X_val)):
       arm = bandit.choose_arm()  # 选择分类器
       pred = classifiers[arm].predict([X_val[i]])
       reward = (pred[0] == y_val[i]) ? 1 : 0
       bandit.update(arm, reward)

3. 选择最优分类器
   best_arm = np.argmax(bandit.alpha)
   best_classifier = classifiers[best_arm]
```

**文件**: MAB-*/MAB*.py L1-150  
**特点**: 自动、自适应、不需手工调参

---

## 📊 输出文件格式

### 特征选择结果
```json
// feature_selection.json
{
  "method": "forward",
  "n_features_original": 76,
  "n_features_selected": 4,
  "selected_features": [0, 2, 5, 10]
}
```

### 性能指标
```csv
// metrics.csv (头行)
model,scenario,split,accuracy,f1_score,precision,recall,auc_score,detection_rate
SGDClassifier,baseline_clean_train_clean_test,clean_test,0.9523,0.8521,0.8641,0.8410,0.9521,0.8410
SGDClassifier,baseline_clean_train_adv_test,adv_test,0.6234,0.6234,0.6234,0.6234,0.6521,0.6234
...
```

### 对抗样本
```
X_train_adv.npy: shape=(30000, 4), dtype=float64
X_test_adv.npy: shape=(7500, 4), dtype=float64
y_train.npy: shape=(30000,), dtype=int32
y_test.npy: shape=(7500,), dtype=int32
```

---

## 🧭 按需求快速导航

### "我要快速理解这个框架"
1. 阅读本文 (5min) ← 你在这里
2. 阅读 [INNOVATION2_QUICKREF.md](INNOVATION2_QUICKREF.md) (15min)
3. 看代码 pipeline.py main() (10min)
4. **总用时**: 30min ✅

### "我要运行一个快速测试"
```bash
python pipeline.py \
    --feature-selection forward \
    --max-features 10 \
    --adversarial-method fgsm \
    --epochs 3 \
    --output-dir results/quick_test/
```
**文档**: [INNOVATION2_QUICKREF.md](INNOVATION2_QUICKREF.md) §5  
**时间**: 5min ✅

### "我要完整理解每个算法"
1. 特征选择: [INNOVATION2_CODE_GUIDE.md](INNOVATION2_CODE_GUIDE.md) §1
2. Thompson Sampling: [INNOVATION2_CODE_GUIDE.md](INNOVATION2_CODE_GUIDE.md) §4
3. Pipeline集成: [INNOVATION2_CODE_GUIDE.md](INNOVATION2_CODE_GUIDE.md) §2
4. **总用时**: 1-2小时

### "我要修改代码或扩展功能"
1. 查看源代码文件导航 (本文 §2)
2. 找到对应文件和行号
3. 参考 [INNOVATION2_CODE_GUIDE.md](INNOVATION2_CODE_GUIDE.md) 的代码片段
4. 查看相关论文/博客
5. **总用时**: 2-4小时

### "我的实验出现问题"
1. 查看 [INNOVATION2_CODE_GUIDE.md](INNOVATION2_CODE_GUIDE.md) §7 常见问题
2. 检查参数配置 [INNOVATION2_CODE_GUIDE.md](INNOVATION2_CODE_GUIDE.md) §6
3. 查看日志输出
4. 运行调试脚本

### "我要跨数据集对比"
1. 准备多个normalized_data_*.csv
2. 运行 [INNOVATION2_QUICKREF.md](INNOVATION2_QUICKREF.md) §5 的for循环命令
3. 对比 results/*/metrics.csv
4. **参考**: 性能基准表 [INNOVATION2_QUICKREF.md](INNOVATION2_QUICKREF.md) §3

---

## 🎯 关键参数一览表

| 参数 | 默认值 | 范围 | 影响 | 参考位置 |
|-----|-------|------|------|--------|
| `--max-features` | 20 | 1-100 | 特征数↓ → 轻量化↑ | pipeline.py L322 |
| `--feature-selection` | None | {forward,backward,...} | 特征筛选方法 | pipeline.py L315 |
| `--epsilon` | 0.3 | 0.1-1.0 | FGSM扰动强度↑ | pipeline.py L254 |
| `--owc-epochs` | 100 | 10-200 | 对抗生成质量 | pipeline.py L341 |
| `--adversarial-method` | fgsm | {fgsm,owc-sawn,both} | 对抗方法 | pipeline.py L334 |
| `--epochs` | 15 | 5-50 | DNN训练轮数 | pipeline.py L279 |
| `--batch-size` | 256 | 32-1024 | 内存占用 | pipeline.py L285 |
| `--max-iter` | 2000 | 500-5000 | SGD收敛速度 | pipeline.py L303 |

**调优建议**:
- 快速实验: `--epochs 3 --owc-epochs 0 --max-features 10`
- 完整实验: `--epochs 15 --owc-epochs 50 --max-features 15`
- 高精度: `--epochs 30 --owc-epochs 100 --max-features 20`

---

## 📋 检查清单

完成新开发者培训:

- [ ] 读完本文 (INNOVATION2_INDEX.md)
- [ ] 读完快速参考 (INNOVATION2_QUICKREF.md)
- [ ] 运行快速测试命令
- [ ] 读完详细指南 (INNOVATION2_CODE_GUIDE.md)
- [ ] 理解4种特征选择算法
- [ ] 理解Thompson Sampling原理
- [ ] 理解pipeline数据流
- [ ] 修改1个参数并重新运行
- [ ] 查看并理解输出文件格式
- [ ] 可以独立解释某个性能指标下降的原因

---

## 🔗 相关资源链接

### 代码文件
- [feature_selection.py](feature_selection.py) - 特征选择实现
- [pipeline.py](pipeline.py) - Pipeline主逻辑
- [owc_sawn/](owc_sawn/) - 对抗样本生成
- [preprocessing.py](../lab-ids-anta-main/Dataset/preprocessing.py) - 数据预处理

### 文档文件
- [INNOVATION2_CODE_GUIDE.md](INNOVATION2_CODE_GUIDE.md) - 详细指南
- [INNOVATION2_QUICKREF.md](INNOVATION2_QUICKREF.md) - 快速参考
- [FEATURE_SELECTION.md](FEATURE_SELECTION.md) - 特征选择使用说明
- [README.md](README.md) - 项目整体说明

### 外部参考
- scikit-learn特性选择: https://scikit-learn.org/stable/modules/feature_selection.html
- Thompson Sampling论文: https://arxiv.org/abs/1111.1797
- FGSM对抗例子: https://arxiv.org/abs/1412.6572
- GAN基础: https://arxiv.org/abs/1406.2661

---

**版本**: 1.0  
**最后更新**: 2024  
**建议**: 👉 **先读本文** → 再读QUICKREF → 动手实验 → 读CODE_GUIDE深入学习

---

### 常用快速链接

| 需求 | 直达 | 时间 |
|-----|-----|------|
| 5分钟了解框架 | 本文(前3节) | 5min ⏱️ |
| 15分钟快速入门 | [QUICKREF](INNOVATION2_QUICKREF.md) | 15min ⏱️ |
| 30分钟能运行测试 | 本文 + [QUICKREF](INNOVATION2_QUICKREF.md) 命令 | 30min ⏱️ |
| 2小时完全理解 | [CODE_GUIDE](INNOVATION2_CODE_GUIDE.md) | 2h ⏱️ |
| 调试问题 | [CODE_GUIDE](INNOVATION2_CODE_GUIDE.md) §7 + 本文 §3 | varies |

