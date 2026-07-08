# OWC-SAWN: Optimized Weighted Conditional Stepwise Adversarial Network

## 概述

OWC-SAWN是一个条件对抗网络模块,用于为入侵检测系统生成对抗样本。它结合了以下特性:

- **条件生成**: 根据类别标签生成特定类型的攻击样本
- **加权判别**: 使用类别权重来平衡不同攻击类型的学习
- **渐进式训练**: 交替更新生成器和判别器,提升训练稳定性
- **梯度惩罚**: 使用WGAN-GP风格的梯度惩罚来约束训练

## 架构

### 1. 生成器 (ConditionalGenerator)

条件生成器将随机噪声和类别标签结合,生成对抗样本:

```
输入: [噪声向量(latent_dim), 类别标签]
     ↓
标签嵌入(Embedding) + 噪声拼接
     ↓
全连接层 × N (BatchNorm + LeakyReLU)
     ↓
输出层(Tanh激活)
     ↓
输出: 生成样本(output_dim)
```

**关键参数**:
- `latent_dim`: 噪声向量维度 (默认: 100)
- `output_dim`: 输出特征维度 (应与IDS数据集特征数匹配)
- `num_classes`: 类别数量
- `embedding_dim`: 标签嵌入维度 (默认: 50)
- `hidden_layers`: 隐藏层大小列表 (默认: [256, 512, 256])

### 2. 判别器 (WeightedDiscriminator)

加权判别器使用类别权重来区分真实样本和生成样本:

```
输入: [样本(input_dim), 类别标签]
     ↓
标签嵌入(Embedding) + 样本拼接
     ↓
全连接层 × N (Dropout + LeakyReLU)
     ↓
输出层(Sigmoid激活)
     ↓
输出: 真实性概率 + 类别权重
```

**关键参数**:
- `input_dim`: 输入特征维度
- `num_classes`: 类别数量
- `hidden_layers`: 隐藏层大小列表 (默认: [256, 256, 128, 64])
- `dropout_rate`: Dropout率 (默认: 0.3)
- `use_class_weights`: 是否使用类别权重

### 3. 训练器 (OWCSAWNTrainer)

训练器实现了渐进式对抗训练流程:

- 每个epoch中多次更新判别器,一次更新生成器
- 使用梯度惩罚来稳定训练
- 支持检查点保存和TensorBoard日志
- 自动跟踪训练指标

## 使用方法

### 基本训练流程

```python
from owc_sawn import ConditionalGenerator, WeightedDiscriminator, OWCSAWNTrainer

# 1. 创建模型
generator = ConditionalGenerator(
    latent_dim=100,
    output_dim=78,  # IDS特征数
    num_classes=2,  # 正常/攻击
    hidden_layers=[256, 512, 256]
)

discriminator = WeightedDiscriminator(
    input_dim=78,
    num_classes=2,
    hidden_layers=[256, 256, 128, 64],
    use_class_weights=True
)

# 2. 创建训练器
trainer = OWCSAWNTrainer(
    generator=generator,
    discriminator=discriminator,
    latent_dim=100,
    generator_lr=0.0002,
    discriminator_lr=0.0002,
    n_discriminator_steps=5,
    use_gradient_penalty=True,
    checkpoint_dir="checkpoints",
    log_dir="logs"
)

# 3. 训练
trainer.fit(
    X_train,  # 训练数据
    y_train,  # 类别标签
    epochs=100,
    batch_size=128,
    validation_data=(X_val, y_val),
    verbose=1
)

# 4. 生成对抗样本
adversarial_samples = trainer.generate_samples(
    num_samples=1000,
    labels=np.array([0]*500 + [1]*500)  # 生成500个正常,500个攻击
)
```

### 使用工具函数

```python
from owc_sawn.utils import (
    generate_adversarial_samples,
    evaluate_sample_quality,
    augment_data_with_gan,
    build_generator,
    build_discriminator
)

# 快速构建模型
generator = build_generator(
    latent_dim=100,
    output_dim=78,
    conditional=True,
    num_classes=2
)

discriminator = build_discriminator(
    input_dim=78,
    conditional=True,
    num_classes=2,
    discriminator_type="weighted"
)

# 生成对抗样本
samples, labels = generate_adversarial_samples(
    generator=generator,
    num_samples=1000,
    num_classes=2,
    latent_dim=100
)

# 评估样本质量
real_samples = X_train  # 真实数据
quality_metrics = evaluate_sample_quality(real_samples, samples)
print(f"Quality Score: {quality_metrics['overall_quality_score']:.4f}")
print(f"Diversity: {quality_metrics['diversity']:.4f}")
print(f"Coverage: {quality_metrics['coverage']:.4f}")

# 数据增强
X_augmented, y_augmented = augment_data_with_gan(
    X_train, y_train,
    generator=generator,
    augmentation_ratio=0.5  # 增加50%的生成样本
)
```

### 检查点和恢复

```python
# 保存检查点
trainer.save_checkpoint(epoch=50)

# 加载检查点
trainer.restore_checkpoint()

# 或指定特定检查点
trainer.checkpoint.restore("checkpoints/ckpt-10")

# 导出训练历史
history = trainer.get_training_history()
print(f"Generator Loss: {history['gen_loss']}")
print(f"Discriminator Loss: {history['disc_loss']}")
```

### TensorBoard监控

```python
# 在训练时启用TensorBoard
trainer = OWCSAWNTrainer(
    generator=generator,
    discriminator=discriminator,
    log_dir="logs/experiment_1"
)

# 训练后启动TensorBoard
# tensorboard --logdir=logs
```

## 高级用法

### 自定义损失函数

```python
def custom_loss(real_output, fake_output, real_labels, fake_labels):
    # 实现自定义损失逻辑
    real_loss = tf.reduce_mean(real_output)
    fake_loss = tf.reduce_mean(fake_output)
    return real_loss + fake_loss

# 使用自定义损失
total_loss, loss_dict = discriminator.compute_loss(
    real_output, fake_output,
    real_labels, fake_labels,
    loss_fn=custom_loss
)
```

### 辅助判别器

```python
from owc_sawn import AuxiliaryDiscriminator

# 创建辅助判别器(同时预测真假和类别)
aux_discriminator = AuxiliaryDiscriminator(
    input_dim=78,
    num_classes=2,
    hidden_layers=[256, 256, 128]
)

# 前向传播
real_fake_pred, class_pred = aux_discriminator(samples)
```

## 参数调优建议

### 生成器参数

- **latent_dim**: 
  - 较小(50-100): 训练快,生成样本多样性低
  - 较大(100-200): 训练慢,生成样本多样性高
  
- **hidden_layers**:
  - 浅层(2-3层): 适合简单数据分布
  - 深层(4-6层): 适合复杂数据分布

### 判别器参数

- **n_discriminator_steps**: 
  - 建议值: 3-5
  - 越大训练越稳定,但生成器更新慢
  
- **dropout_rate**:
  - 0.2-0.3: 适合小数据集
  - 0.4-0.5: 适合大数据集

### 训练参数

- **learning_rate**:
  - Generator: 0.0001-0.0002
  - Discriminator: 0.0001-0.0002 (可以略高于生成器)
  
- **gradient_penalty**:
  - lambda: 5-10 (WGAN-GP风格)
  - 建议启用以稳定训练

## 与FGSM对比

| 特性 | OWC-SAWN | FGSM |
|------|----------|------|
| **生成方式** | 学习数据分布 | 基于梯度扰动 |
| **条件生成** | ✓ 支持类别条件 | ✗ 无条件 |
| **训练时间** | 较长(需要训练) | 短(直接计算) |
| **样本多样性** | 高 | 中 |
| **可控性** | 高 | 低 |
| **适用场景** | 数据增强,生成新样本 | 鲁棒性测试,快速攻击 |

## 集成到pipeline

```python
# 在pipeline.py中使用OWC-SAWN
from owc_sawn import ConditionalGenerator, OWCSAWNTrainer

# 训练OWC-SAWN
owc_trainer = train_owc_sawn(
    X_train, y_train,
    epochs=100,
    checkpoint_dir="owc_checkpoints"
)

# 生成对抗样本用于增强训练集
X_adversarial = owc_trainer.generate_samples(
    num_samples=len(X_train),
    labels=y_train
)

# 结合原始数据和对抗样本
X_combined = np.vstack([X_train, X_adversarial])
y_combined = np.hstack([y_train, y_train])

# 训练检测器
detector.fit(X_combined, y_combined)
```

## 性能指标

训练完成后,可以查看以下指标:

- **生成器损失** (gen_loss): 越低越好,表示生成样本越逼真
- **判别器损失** (disc_loss): 应保持在合理范围,过高或过低都不好
- **判别器准确率** (disc_real_acc, disc_fake_acc): 50%左右表示平衡
- **样本质量分数** (quality_score): 0-1之间,越高越好
- **多样性** (diversity): 0-1之间,越高表示生成样本越多样
- **覆盖率** (coverage): 0-1之间,越高表示覆盖真实分布越好

## 故障排除

### 训练不稳定

- 降低学习率
- 增加判别器更新步数
- 启用梯度惩罚
- 减少dropout率

### 生成样本质量差

- 增加训练epoch
- 调整网络深度
- 增大embedding维度
- 检查数据归一化

### 模式崩溃 (Mode Collapse)

- 增大latent_dim
- 使用多个判别器
- 增加类别权重
- 定期重启训练

## 引用

如果使用本模块,请引用:

```
@article{owc-sawn,
  title={Optimized Weighted Conditional Stepwise Adversarial Network for Intrusion Detection},
  author={Your Name},
  journal={Journal Name},
  year={2024}
}
```

## 许可证

MIT License
