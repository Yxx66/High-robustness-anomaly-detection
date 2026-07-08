"""
最小化OWC-SAWN测试 - 直接诊断NaN问题
"""
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import sys
sys.path.append('.')

from owc_sawn import ConditionalGenerator, WeightedDiscriminator, OWCSAWNTrainer

print("="*80)
print("最小化OWC-SAWN测试")
print("="*80)

# 1. 加载和预处理数据
print("\n1. 加载数据...")
df = pd.read_csv(r'F:\yxx\project\lab-ids-anta-main\lab-ids-anta-main\Dataset\normalized_data_2018.csv')
df = df.select_dtypes(include=[np.number])
X = df.drop(columns=[' Label'], errors='ignore').values[:5000]  # 只用5000样本快速测试
y = (df[' Label'].values[:5000] != 0).astype(np.int32) if ' Label' in df.columns else np.zeros(5000, dtype=np.int32)

print(f"数据形状: {X.shape}")
print(f"原始范围: [{X.min():.2f}, {X.max():.2f}]")

# 2. 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"StandardScaler后: [{X_scaled.min():.2f}, {X_scaled.max():.2f}]")

# 3. MinMaxScaler到[-1, 1]  🔴 关键步骤
minmax = MinMaxScaler(feature_range=(-1, 1))
X_norm = minmax.fit_transform(X_scaled)
print(f"MinMaxScaler后: [{X_norm.min():.2f}, {X_norm.max():.2f}]")

# 4. 分割数据
X_train, X_val, y_train, y_val = train_test_split(X_norm, y, test_size=0.2, random_state=42)

print(f"\n训练集: {X_train.shape}")
print(f"验证集: {X_val.shape}")

# 5. 创建极简模型
print("\n2. 创建模型...")
input_dim = X_train.shape[1]
num_classes = 2
latent_dim = 32  # 减小到32

generator = ConditionalGenerator(
    latent_dim=latent_dim,
    output_dim=input_dim,
    num_classes=num_classes,
    hidden_layers=[64, 128, 64],  # 非常简单
    dropout_rate=0.1,
    embedding_dim=16
)

discriminator = WeightedDiscriminator(
    input_dim=input_dim,
    num_classes=num_classes,
    hidden_layers=[64, 32],  # 非常简单
    dropout_rate=0.1,
    embedding_dim=16,
    use_class_weights=False
)

# 6. 创建trainer
print("\n3. 创建Trainer...")
trainer = OWCSAWNTrainer(
    generator=generator,
    discriminator=discriminator,
    latent_dim=latent_dim,
    generator_lr=0.0001,  # 非常小的学习率
    discriminator_lr=0.0001,
    beta_1=0.5,
    beta_2=0.999,
    use_gradient_penalty=False,  # 关闭梯度惩罚
    gp_lambda=0.0,
    n_discriminator_steps=1,  # 1:1训练
    checkpoint_dir=None,
    log_dir=None
)

# 7. 训练
print("\n4. 开始训练...")
print("="*80)

try:
    trainer.fit(
        X_train, y_train,
        epochs=10,  # 只训练10轮看看
        batch_size=128,
        validation_data=(X_val, y_val),
        verbose=1
    )
    print("\n✅ 训练成功完成！没有NaN！")
    
    # 生成样本测试
    print("\n5. 生成样本测试...")
    samples = trainer.generate_samples(num_samples=10, labels=np.array([0,1,0,1,0,1,0,1,0,1]))
    print(f"生成样本范围: [{samples.min():.2f}, {samples.max():.2f}]")
    print("✅ 样本生成成功！")
    
except Exception as e:
    print(f"\n❌ 训练失败: {e}")
    import traceback
    traceback.print_exc()
