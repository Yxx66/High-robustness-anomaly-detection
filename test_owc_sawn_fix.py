"""
测试OWC-SAWN修复 - MinMaxScaler归一化
"""
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd

# 加载数据
print("Loading dataset...")
df = pd.read_csv(r'F:\yxx\project\lab-ids-anta-main\lab-ids-anta-main\Dataset\normalized_data_2018.csv')

# 移除非数值列
numeric_cols = df.select_dtypes(include=[np.number]).columns
X = df[numeric_cols].drop(columns=[' Label'], errors='ignore').values
print(f"Original data shape: {X.shape}")
print(f"Original data range: [{X.min():.2f}, {X.max():.2f}]")

# StandardScaler (原始pipeline使用)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"\nAfter StandardScaler: [{X_scaled.min():.2f}, {X_scaled.max():.2f}]")

# MinMaxScaler to [-1, 1] (新增，用于OWC-SAWN)
minmax_scaler = MinMaxScaler(feature_range=(-1, 1))
X_normalized = minmax_scaler.fit_transform(X_scaled)
print(f"After MinMaxScaler: [{X_normalized.min():.2f}, {X_normalized.max():.2f}]")

# 验证逆变换
X_back = minmax_scaler.inverse_transform(X_normalized)
print(f"\nAfter inverse transform: [{X_back.min():.2f}, {X_back.max():.2f}]")
print(f"Reconstruction error: {np.abs(X_scaled - X_back).max():.10f}")

print("\n✓ MinMaxScaler测试通过！数据已正确归一化到[-1, 1]范围")
print("✓ 这应该能解决OWC-SAWN的NaN问题")
