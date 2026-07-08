# OWC-SAWN集成示例

## 测试集成是否成功

```powershell
# 1. 查看pipeline帮助信息
python pipeline.py --help

# 2. 查看OWC-SAWN独立脚本帮助信息
python run_owc_sawn.py --help

# 3. 测试OWC-SAWN模块
python test_owc_sawn.py
```

## 使用示例

### 1. 在主pipeline中使用FGSM (默认)

```powershell
python pipeline.py \
    --dataset ..\Dataset\normalized_data_2017.csv \
    --epochs 15 \
    --output-dir results_fgsm
```

### 2. 在主pipeline中使用OWC-SAWN

```powershell
python pipeline.py \
    --dataset ..\Dataset\normalized_data_2017.csv \
    --adversarial-method owc-sawn \
    --owc-epochs 50 \
    --epochs 15 \
    --output-dir results_owc_sawn
```

### 3. 组合FGSM和OWC-SAWN

```powershell
python pipeline.py \
    --dataset ..\Dataset\normalized_data_2017.csv \
    --adversarial-method both \
    --owc-epochs 50 \
    --epochs 15 \
    --output-dir results_both
```

### 4. 独立训练OWC-SAWN

```powershell
python run_owc_sawn.py \
    --dataset ..\Dataset\normalized_data_2017.csv \
    --epochs 100 \
    --batch-size 128 \
    --latent-dim 100 \
    --generate-samples 5000 \
    --save-generated \
    --output-dir owc_output_2017
```

### 5. 结合特征选择和OWC-SAWN

```powershell
python pipeline.py \
    --dataset ..\Dataset\normalized_data_2017.csv \
    --feature-selection importance \
    --max-features 20 \
    --adversarial-method owc-sawn \
    --owc-epochs 50 \
    --output-dir results_fs_owc
```

## 输出说明

### Pipeline输出 (使用OWC-SAWN)

```
results_owc_sawn/
├── metrics.csv                      # 性能指标
├── metrics.json                     # JSON格式指标
├── owc_sawn_info.json              # OWC-SAWN训练信息
├── X_train_adv.npy                 # 对抗样本(训练集)
├── X_test_adv.npy                  # 对抗样本(测试集)
├── owc_sawn_checkpoints/           # 模型检查点
│   ├── ckpt-1.data-00000-of-00001
│   ├── ckpt-1.index
│   └── checkpoint
└── owc_sawn_logs/                  # TensorBoard日志
    └── events.out.tfevents...
```

### 独立OWC-SAWN输出

```
owc_output_2017/
├── training_summary.json           # 完整训练总结
├── training_history.png            # 训练曲线图
├── sample_quality_metrics.json     # 样本质量评估
├── generated_samples.csv           # 生成的对抗样本
├── checkpoints/                    # 模型检查点
└── logs/                           # TensorBoard日志
```

## 监控训练

启动TensorBoard查看训练过程:

```powershell
# Pipeline中的OWC-SAWN
tensorboard --logdir=results_owc_sawn/owc_sawn_logs

# 独立OWC-SAWN
tensorboard --logdir=owc_output_2017/logs
```

访问 http://localhost:6006 查看:
- 生成器和判别器损失
- 真假样本准确率
- 训练稳定性

## 性能对比

| 方法 | 训练时间 | 样本质量 | 多样性 | 适用场景 |
|------|---------|---------|--------|---------|
| **FGSM** | 快速 (~10分钟) | 中等 | 低 | 快速测试、梯度攻击 |
| **OWC-SAWN** | 较慢 (~1小时) | 高 | 高 | 数据增强、真实攻击 |
| **BOTH** | 中等 (~1.2小时) | 高 | 最高 | 综合防御训练 |

## 故障排除

### 1. 内存不足

```powershell
# 减少batch size
python pipeline.py --adversarial-method owc-sawn --batch-size 64

# 减少OWC-SAWN epochs
python pipeline.py --adversarial-method owc-sawn --owc-epochs 30
```

### 2. 训练不稳定

```powershell
# 查看训练日志
tensorboard --logdir=results_owc_sawn/owc_sawn_logs

# 检查判别器和生成器准确率是否在50%附近
# 如果判别器准确率过高(>90%)，生成器学习困难
# 如果判别器准确率过低(<20%)，可能发生模式崩溃
```

### 3. 生成样本质量差

查看 `owc_sawn_info.json` 中的质量指标:
- `overall_quality_score` < 0.4: 增加训练epochs
- `diversity` < 0.3: 增加latent_dim
- `coverage` < 0.4: 检查数据归一化

## 下一步

1. **比较不同方法**: 分别运行FGSM、OWC-SAWN和BOTH,比较`metrics.csv`
2. **调优参数**: 调整`--owc-epochs`、`--latent-dim`等参数
3. **特征选择**: 结合`--feature-selection`降低维度
4. **可视化**: 使用`generate_visual_tables.py`生成对比图表
