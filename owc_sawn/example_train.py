"""
Example: Training OWC-SAWN on CICIDS Dataset

This script demonstrates how to train the OWC-SAWN adversarial network
on the CICIDS2017/2018/2019 datasets for adversarial sample generation.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from owc_sawn import (
    ConditionalGenerator,
    WeightedDiscriminator,
    OWCSAWNTrainer,
    generate_adversarial_samples,
    evaluate_sample_quality,
    augment_data_with_gan
)


def load_ids_data(dataset_path: str, dataset_year: str = "2017"):
    """
    Load and prepare IDS dataset.
    
    Args:
        dataset_path: Path to dataset folder
        dataset_year: Year of dataset ("2017", "2018", or "2019")
    
    Returns:
        X_train, y_train, X_test, y_test
    """
    print(f"Loading CICIDS{dataset_year} dataset...")
    
    # Load normalized data
    data_path = Path(dataset_path) / f"normalized_data_{dataset_year}.csv"
    df = pd.read_csv(data_path)
    
    # Separate features and labels
    if 'Label' in df.columns:
        X = df.drop('Label', axis=1).values
        y = df['Label'].values
    else:
        # Assume last column is label
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
    
    # Convert labels to binary (0: benign, 1: attack)
    y_binary = (y != 0).astype(np.int32)
    
    # Split train/test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
    )
    
    print(f"Train samples: {len(X_train)} (Benign: {np.sum(y_train==0)}, Attack: {np.sum(y_train==1)})")
    print(f"Test samples: {len(X_test)} (Benign: {np.sum(y_test==0)}, Attack: {np.sum(y_test==1)})")
    print(f"Feature dimension: {X_train.shape[1]}")
    
    return X_train, y_train, X_test, y_test


def train_owc_sawn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 100,
    batch_size: int = 128,
    checkpoint_dir: str = "owc_checkpoints",
    log_dir: str = "owc_logs"
):
    """
    Train OWC-SAWN on IDS data.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        epochs: Number of training epochs
        batch_size: Batch size
        checkpoint_dir: Directory to save checkpoints
        log_dir: Directory for TensorBoard logs
    
    Returns:
        Trained OWCSAWNTrainer
    """
    print("\n" + "="*70)
    print("Training OWC-SAWN Adversarial Network")
    print("="*70)
    
    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train))
    
    # Create models
    print(f"\nCreating models...")
    print(f"  Input dimension: {input_dim}")
    print(f"  Number of classes: {num_classes}")
    
    generator = ConditionalGenerator(
        latent_dim=100,
        output_dim=input_dim,
        num_classes=num_classes,
        hidden_layers=[256, 512, 512, 256],
        embedding_dim=50
    )
    
    discriminator = WeightedDiscriminator(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_layers=[256, 256, 128, 64],
        dropout_rate=0.3,
        embedding_dim=50,
        use_class_weights=True
    )
    
    # Create trainer
    trainer = OWCSAWNTrainer(
        generator=generator,
        discriminator=discriminator,
        latent_dim=100,
        generator_lr=0.0002,
        discriminator_lr=0.0002,
        beta_1=0.5,
        beta_2=0.999,
        use_gradient_penalty=True,
        gp_lambda=10.0,
        n_discriminator_steps=5,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir
    )
    
    # Train
    print(f"\nStarting training for {epochs} epochs...")
    print(f"  Batch size: {batch_size}")
    print(f"  Discriminator steps: 5")
    print(f"  Gradient penalty: Enabled (lambda=10.0)")
    
    trainer.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        verbose=1
    )
    
    print("\n✓ Training completed!")
    
    return trainer


def evaluate_generated_samples(
    trainer: OWCSAWNTrainer,
    X_real: np.ndarray,
    num_samples: int = 1000
):
    """
    Evaluate quality of generated samples.
    
    Args:
        trainer: Trained OWCSAWNTrainer
        X_real: Real samples for comparison
        num_samples: Number of samples to generate
    """
    print("\n" + "="*70)
    print("Evaluating Generated Samples")
    print("="*70)
    
    # Generate samples for each class
    num_classes = trainer.generator.num_classes
    samples_per_class = num_samples // num_classes
    
    labels = []
    for c in range(num_classes):
        labels.extend([c] * samples_per_class)
    labels = np.array(labels)
    
    print(f"\nGenerating {num_samples} samples...")
    X_generated = trainer.generate_samples(num_samples, labels)
    
    # Evaluate quality
    print("\nComputing quality metrics...")
    metrics = evaluate_sample_quality(X_real, X_generated)
    
    print("\nQuality Metrics:")
    print(f"  Overall Quality Score: {metrics['overall_quality_score']:.4f}")
    print(f"  Diversity: {metrics['diversity']:.4f}")
    print(f"  Coverage: {metrics['coverage']:.4f}")
    print(f"  Mean Distance: {metrics['mean_distance']:.4f}")
    print(f"  Std Distance: {metrics['std_distance']:.4f}")
    
    return X_generated, labels, metrics


def visualize_training_history(trainer: OWCSAWNTrainer, save_path: str = "owc_training.png"):
    """
    Visualize training history.
    
    Args:
        trainer: Trained OWCSAWNTrainer
        save_path: Path to save plot
    """
    history = trainer.get_training_history()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Generator loss
    axes[0, 0].plot(history['gen_loss'], label='Generator Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Generator Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Discriminator loss
    axes[0, 1].plot(history['disc_loss'], label='Discriminator Loss', color='orange')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Discriminator Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Discriminator real accuracy
    axes[1, 0].plot(history['disc_real_acc'], label='Real Acc', color='green')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_title('Discriminator Real Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    axes[1, 0].set_ylim([0, 1])
    
    # Discriminator fake accuracy
    axes[1, 1].plot(history['disc_fake_acc'], label='Fake Acc', color='red')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_title('Discriminator Fake Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    axes[1, 1].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\n✓ Training history plot saved to {save_path}")
    plt.close()


def augment_dataset_with_owc_sawn(
    trainer: OWCSAWNTrainer,
    X_train: np.ndarray,
    y_train: np.ndarray,
    augmentation_ratio: float = 0.5
):
    """
    Augment training dataset with generated samples.
    
    Args:
        trainer: Trained OWCSAWNTrainer
        X_train: Original training features
        y_train: Original training labels
        augmentation_ratio: Ratio of generated samples to add
    
    Returns:
        X_augmented, y_augmented
    """
    print("\n" + "="*70)
    print("Data Augmentation with OWC-SAWN")
    print("="*70)
    
    print(f"\nOriginal dataset size: {len(X_train)}")
    print(f"Augmentation ratio: {augmentation_ratio}")
    
    # Use utility function
    X_augmented, y_augmented = augment_data_with_gan(
        X_train, y_train,
        generator=trainer.generator,
        augmentation_ratio=augmentation_ratio,
        latent_dim=100
    )
    
    print(f"Augmented dataset size: {len(X_augmented)}")
    print(f"Added {len(X_augmented) - len(X_train)} generated samples")
    
    return X_augmented, y_augmented


def main():
    """Main execution function."""
    
    # Configuration
    DATASET_PATH = "../Dataset"
    DATASET_YEAR = "2017"
    EPOCHS = 100
    BATCH_SIZE = 128
    CHECKPOINT_DIR = "owc_checkpoints_2017"
    LOG_DIR = "owc_logs_2017"
    
    # Load data
    X_train, y_train, X_test, y_test = load_ids_data(DATASET_PATH, DATASET_YEAR)
    
    # Split validation set
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
    )
    
    # Train OWC-SAWN
    trainer = train_owc_sawn(
        X_train, y_train,
        X_val, y_val,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        checkpoint_dir=CHECKPOINT_DIR,
        log_dir=LOG_DIR
    )
    
    # Visualize training
    visualize_training_history(trainer, f"owc_training_{DATASET_YEAR}.png")
    
    # Evaluate generated samples
    X_generated, labels_generated, quality_metrics = evaluate_generated_samples(
        trainer, X_train, num_samples=1000
    )
    
    # Augment dataset
    X_augmented, y_augmented = augment_dataset_with_owc_sawn(
        trainer, X_train, y_train, augmentation_ratio=0.5
    )
    
    # Save augmented data
    print("\nSaving augmented dataset...")
    augmented_df = pd.DataFrame(X_augmented)
    augmented_df['Label'] = y_augmented
    augmented_df.to_csv(f"augmented_data_{DATASET_YEAR}.csv", index=False)
    print(f"✓ Augmented data saved to augmented_data_{DATASET_YEAR}.csv")
    
    # Print summary
    print("\n" + "="*70)
    print("Training Summary")
    print("="*70)
    history = trainer.get_training_history()
    print(f"\nFinal Metrics:")
    print(f"  Generator Loss: {history['gen_loss'][-1]:.4f}")
    print(f"  Discriminator Loss: {history['disc_loss'][-1]:.4f}")
    print(f"  Disc Real Accuracy: {history['disc_real_acc'][-1]:.4f}")
    print(f"  Disc Fake Accuracy: {history['disc_fake_acc'][-1]:.4f}")
    print(f"\nSample Quality:")
    print(f"  Quality Score: {quality_metrics['overall_quality_score']:.4f}")
    print(f"  Diversity: {quality_metrics['diversity']:.4f}")
    print(f"  Coverage: {quality_metrics['coverage']:.4f}")
    print(f"\nAugmented Dataset:")
    print(f"  Original size: {len(X_train)}")
    print(f"  Augmented size: {len(X_augmented)}")
    print(f"  Increase: {(len(X_augmented) - len(X_train)) / len(X_train) * 100:.1f}%")
    
    print("\n✓ All tasks completed successfully!")
    print(f"\nCheckpoints saved to: {CHECKPOINT_DIR}")
    print(f"TensorBoard logs saved to: {LOG_DIR}")
    print(f"\nTo view training progress:")
    print(f"  tensorboard --logdir={LOG_DIR}")


if __name__ == "__main__":
    main()
