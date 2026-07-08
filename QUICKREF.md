# 特征选择快速参考

## 快速命令

### 基础使用
```bash
# 激活虚拟环境
.\lab-ids-anta\Scripts\Activate.ps1
cd hybrid_scheme1

# 不使用特征选择
python pipeline.py

# 使用Forward Selection
python pipeline.py --feature-selection forward --max-features 20

# 使用Backward Elimination
python pipeline.py --feature-selection backward --max-features 15

# 使用Correlation-based
python pipeline.py --feature-selection correlation --max-features 20

# 使用Importance-based
python pipeline.py --feature-selection importance --max-features 20

# 自动比较并选择最佳方法
python pipeline.py --feature-selection compare --max-features 20
```

### 评估和测试
```bash
# 快速测试
python test_feature_selection.py

# 详细评估
python evaluate_feature_selection.py \
    --dataset ..\lab-ids-anta-main\Dataset\encoded_features_2017.csv \
    --max-features 20
```

## 四种方法对比

| 方法 | 速度 | 精度 | 适用场景 |
|------|------|------|----------|
| **Forward** | ⭐⭐ | ⭐⭐⭐⭐ | 特征多(>100),需要高精度 |
| **Backward** | ⭐⭐ | ⭐⭐⭐⭐ | 特征适中(50-100),考虑交互 |
| **Correlation** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 大数据集,快速筛选 |
| **Importance** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 树模型,快速有效 |

## 参数说明

```bash
--feature-selection METHOD    # forward/backward/correlation/importance/compare
--max-features N              # 最大选择特征数 (默认: 20)
--fs-cv K                     # 交叉验证折数 (默认: 3)
```

## 输出文件

```
results/
├── feature_selection.json          # 特征选择信息
├── metrics.csv                     # 模型评估指标
├── X_train_adv.npy                # 训练集对抗样本
└── X_test_adv.npy                 # 测试集对抗样本

results/feature_selection_eval/     # 评估脚本输出
├── evaluation_results.csv
├── feature_selection_details.json
├── feature_selection_comparison.png
└── features_vs_performance.png
```

## Python API

```python
from feature_selection import ForwardSelection
from sklearn.linear_model import SGDClassifier

# 创建分类器和选择器
clf = SGDClassifier(loss='log_loss', max_iter=1000, random_state=42)
selector = ForwardSelection(clf, max_features=20, cv=3, verbose=True)

# 拟合和转换
X_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# 获取选择的特征
print(selector.selected_features_)  # [0, 5, 10, 15, ...]
print(len(selector.selected_features_))  # 20
```

## 常见问题

**Q: 哪个方法最好?**  
A: 使用 `--feature-selection compare` 自动比较

**Q: 选择多少特征?**  
A: 原始特征数的25-50% (如78→20-40)

**Q: 运行时间太长?**  
A: 使用Correlation或Importance,或减少 `--fs-cv`

**Q: 性能反而下降?**  
A: 增加max_features或尝试其他方法

## 完整示例

```bash
# 1. 测试模块
python test_feature_selection.py

# 2. 评估不同方法
python evaluate_feature_selection.py \
    --dataset ..\lab-ids-anta-main\Dataset\encoded_features_2017.csv

# 3. 运行完整pipeline
python pipeline.py \
    --dataset ..\lab-ids-anta-main\Dataset\encoded_features_2017.csv \
    --feature-selection compare \
    --max-features 20 \
    --epsilon 0.3 \
    --epochs 15

# 4. 生成对比图
python plot_metrics.py --metric f1
```

## 文档链接

- 详细文档: `FEATURE_SELECTION.md`
- 完成报告: `SUMMARY.md`
- 主文档: `README.md`
