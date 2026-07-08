# 指标表格生成完整指南

## 已生成的指标文件

### 📊 原始数据
- `metrics.csv` - CSV格式指标数据
- `metrics.json` - JSON格式指标数据
- `feature_selection.json` - 特征选择信息

### 📝 格式化表格
- `metrics_table.txt` - 控制台友好格式,包含详细统计
- `metrics_table.md` - Markdown格式,适合文档
- `metrics_table.tex` - LaTeX格式,适合论文

### 📈 可视化图表
- `scenario_comparison.png` - 三场景F1对比柱状图
- `metrics_heatmap.png` - 所有指标热图表格
- `metrics_comparison.png` - 分组柱状对比图
- `metrics_radar.png` - 雷达图,综合性能展示

### 📄 完整报告
- `METRICS_REPORT.md` - 完整的性能分析报告

---

## 快速查看指标

### 1. 控制台查看

```powershell
# 查看格式化表格
Get-Content .\results\metrics_table.txt

# 或直接运行脚本
python generate_metrics_table.py
```

### 2. 查看Markdown表格

```powershell
# 在VS Code中打开
code .\results\metrics_table.md

# 或在浏览器中查看转换后的HTML
```

### 3. 查看可视化图表

```powershell
# 在默认图片查看器中打开
.\results\metrics_heatmap.png
.\results\metrics_comparison.png
.\results\metrics_radar.png
```

### 4. 查看完整报告

```powershell
# 在VS Code中打开Markdown报告
code .\results\METRICS_REPORT.md
```

---

## 生成命令总结

```powershell
# 1. 运行pipeline生成指标
python pipeline.py --feature-selection forward --max-features 20

# 2. 生成格式化表格
python generate_metrics_table.py

# 3. 生成可视化图表
python generate_visual_tables.py

# 4. 生成场景对比图
python plot_metrics.py --metric f1

# 一键生成所有表格和图表
python generate_metrics_table.py; python generate_visual_tables.py; python plot_metrics.py --metric f1
```

---

## 表格内容说明

### 控制台表格 (metrics_table.txt)

包含三部分:

1. **详细指标表格**: 按场景分组展示所有指标
2. **场景对比表**: F1分数三场景对比
3. **统计摘要**: 均值、标准差、最大最小值、最佳性能

### Markdown表格 (metrics_table.md)

结构化的Markdown表格,按场景分组,便于嵌入文档。

### LaTeX表格 (metrics_table.tex)

可直接插入LaTeX论文的表格代码:

```latex
\begin{table}[htbp]
\centering
\caption{IDS Performance Metrics with Feature Selection}
\label{tab:metrics}
...
\end{table}
```

---

## 可视化说明

### 热图表格 (metrics_heatmap.png)

- **布局**: 2个子图,分别展示SGD和Ridge的性能
- **颜色**: 绿色=高性能,红色=低性能
- **包含**: 所有4个场景的5个指标
- **适用**: 快速对比不同场景和模型

### 对比柱状图 (metrics_comparison.png)

- **布局**: 3个子图,分别展示Accuracy、F1、AUC
- **对比**: 3个主要场景(Clean→Clean, Clean→Adv, Adv→Adv)
- **颜色**: 蓝色=SGD,紫色=Ridge
- **适用**: 强调对抗训练的效果

### 雷达图 (metrics_radar.png)

- **布局**: 2个子图,分别展示SGD和Ridge
- **维度**: 5个指标(Accuracy, Precision, Recall, F1, AUC)
- **场景**: 3条线,代表3个主要场景
- **适用**: 综合性能展示,多维度对比

### 场景对比图 (scenario_comparison.png)

- **原有图表**: 展示F1分数的三场景对比
- **布局**: 简洁的分组柱状图
- **适用**: 论文图表,直观展示防御效果

---

## 关键指标解读

### 准确率 (Accuracy)
- **定义**: (TP+TN)/(TP+TN+FP+FN)
- **最佳**: 77.75% (SGD, Adv→Adv)
- **平均**: 69.92%

### 精确率 (Precision)
- **定义**: TP/(TP+FP)
- **最佳**: 76.78% (SGD, Adv→Adv)
- **平均**: 69.88%

### 召回率 (Recall)
- **定义**: TP/(TP+FN)
- **最佳**: 99.59% (Ridge, Clean→Clean)
- **平均**: 97.56% (非常高!)

### F1分数 (F1-Score)
- **定义**: 2×(Precision×Recall)/(Precision+Recall)
- **最佳**: 85.30% (SGD, Adv→Adv)
- **平均**: 81.39%

### AUC
- **定义**: ROC曲线下面积
- **最佳**: 76.17% (SGD, Adv→Adv)
- **平均**: 68.55%

---

## 论文使用建议

### 插入表格

**Markdown文档**:
```markdown
直接复制metrics_table.md的内容
```

**LaTeX论文**:
```latex
\input{results/metrics_table.tex}
```

**Word文档**:
1. 在Excel中打开metrics.csv
2. 格式化后复制粘贴

### 插入图表

**论文图表**:
- 使用 `metrics_comparison.png` 展示对抗训练效果
- 使用 `metrics_heatmap.png` 展示全面性能
- 使用 `scenario_comparison.png` 作为主要结果图

**PPT展示**:
- 使用 `metrics_radar.png` 展示综合性能
- 使用 `metrics_comparison.png` 强调关键指标

---

## 自定义生成

### 修改输出格式

编辑 `generate_metrics_table.py`:

```python
# 修改小数位数
def format_percentage(value, decimals=2):
    return f"{value * 100:.{decimals}f}%"

# 修改表格列
metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
```

### 修改可视化样式

编辑 `generate_visual_tables.py`:

```python
# 修改颜色方案
colors = ['#2E86AB', '#A23B72', '#F18F01']

# 修改图表大小
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 修改DPI
plt.savefig(output_path, dpi=300, bbox_inches='tight')
```

---

## 常见问题

**Q: 表格显示不完整?**  
A: 使用等宽字体查看txt文件,或使用Markdown/LaTeX版本

**Q: 图表不清晰?**  
A: 所有图表都是300 DPI高分辨率,适合打印和出版

**Q: 如何只生成特定格式?**  
A: 使用 `--formats` 参数
```bash
python generate_metrics_table.py --formats markdown
```

**Q: 如何批量处理多个实验?**  
A: 修改脚本循环处理不同的metrics.csv文件

**Q: 图表文字太小?**  
A: 修改 `generate_visual_tables.py` 中的 `fontsize` 参数

---

## 完整工作流示例

```powershell
# Step 1: 激活虚拟环境
.\lab-ids-anta\Scripts\Activate.ps1
cd hybrid_scheme1

# Step 2: 运行pipeline
python pipeline.py \
    --feature-selection forward \
    --max-features 20 \
    --epochs 15

# Step 3: 生成所有表格和图表
python generate_metrics_table.py
python generate_visual_tables.py
python plot_metrics.py --metric f1

# Step 4: 查看结果
code .\results\METRICS_REPORT.md
.\results\metrics_heatmap.png

# Step 5: 用于论文/报告
# - 复制metrics_table.md到文档
# - 插入metrics_comparison.png到论文
# - 使用metrics_table.tex到LaTeX文档
```

---

## 输出文件清单

```
results/
├── 📊 原始数据
│   ├── metrics.csv
│   ├── metrics.json
│   └── feature_selection.json
│
├── 📝 格式化表格
│   ├── metrics_table.txt    (控制台格式+统计)
│   ├── metrics_table.md     (Markdown格式)
│   └── metrics_table.tex    (LaTeX格式)
│
├── 📈 可视化图表
│   ├── scenario_comparison.png    (场景对比柱状图)
│   ├── metrics_heatmap.png       (热图表格)
│   ├── metrics_comparison.png    (分组柱状图)
│   └── metrics_radar.png         (雷达图)
│
├── 📄 完整报告
│   └── METRICS_REPORT.md         (详细分析报告)
│
└── 💾 数据文件
    ├── X_train_adv.npy
    ├── X_test_adv.npy
    ├── y_train.npy
    └── y_test.npy
```

---

**生成时间**: 2025年12月11日  
**状态**: ✅ 所有表格和图表已生成  
**下一步**: 查看METRICS_REPORT.md获取详细分析
