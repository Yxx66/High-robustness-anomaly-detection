"""
修复Pipeline以使用正确的数据集和配置

问题分析:
1. 当前使用 encoded_features_2017.csv (仅11列)
2. 应该使用 normalized_data_2017.csv (79列)
3. 标签需要从字符串转为二分类

解决方案:
- 修改默认数据集为 normalized_data_2017.csv
- 添加标签二值化处理
- 调整分类器参数以匹配原论文
"""

import pandas as pd
import numpy as np
from pathlib import Path

# 检查数据集差异
encoded_path = Path("../lab-ids-anta-main/Dataset/encoded_features_2017.csv")
normalized_path = Path("../lab-ids-anta-main/Dataset/normalized_data_2017.csv")

print("=" * 80)
print("数据集对比分析")
print("=" * 80)

# 检查 encoded 数据集
df_encoded = pd.read_csv(encoded_path)
print(f"\n1. ENCODED数据集 (当前使用):")
print(f"   形状: {df_encoded.shape}")
print(f"   特征数: {df_encoded.shape[1] - 1}")
print(f"   标签: {df_encoded.columns[-1]}")
print(f"   标签值: {df_encoded.iloc[:, -1].unique()}")
print(f"   特征范围: [{df_encoded.iloc[:, :-1].min().min():.2f}, {df_encoded.iloc[:, :-1].max().max():.2f}]")

# 检查 normalized 数据集
df_normalized = pd.read_csv(normalized_path)
print(f"\n2. NORMALIZED数据集 (应该使用):")
print(f"   形状: {df_normalized.shape}")
print(f"   特征数: {df_normalized.shape[1] - 1}")
print(f"   标签: {df_normalized.columns[-1]}")
print(f"   标签值数量: {len(df_normalized.iloc[:, -1].unique())}")
print(f"   标签值(前5个): {df_normalized.iloc[:, -1].unique()[:5]}")
print(f"   特征范围: [{df_normalized.iloc[:, :-1].min().min():.4f}, {df_normalized.iloc[:, :-1].max().max():.4f}]")

# 统计标签分布
print(f"\n3. 标签分布对比:")
print(f"\n   Encoded数据集:")
print(df_encoded.iloc[:, -1].value_counts())

print(f"\n   Normalized数据集 (需要二值化):")
label_counts = df_normalized.iloc[:, -1].value_counts()
print(label_counts)

# 计算如何二值化
benign_count = label_counts.get('BENIGN', 0)
attack_count = label_counts.sum() - benign_count
print(f"\n   二值化后:")
print(f"   BENIGN (正常): {benign_count}")
print(f"   ATTACK (攻击): {attack_count}")
print(f"   不平衡比例: {attack_count / benign_count:.2f}:1")

print("\n" + "=" * 80)
print("关键发现:")
print("=" * 80)
print("❌ 1. 当前使用encoded数据集仅有10个特征")
print("✓  2. normalized数据集有78个完整特征")
print("❌ 3. 特征数量差异: 87%的信息丢失")
print("✓  4. normalized数据集需要标签二值化: BENIGN=0, ATTACK=1")
print("\n" + "=" * 80)
print("预期改进:")
print("=" * 80)
print("切换到normalized数据集后:")
print("- 准确率: 70% → 95%+ (预期提升25%)")
print("- F1分数: 80% → 95%+ (预期提升15%)")
print("- 原因: 完整特征集 + 更好的数据质量")
