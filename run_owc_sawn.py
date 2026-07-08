"""
Quick launcher for OWC-SAWN adversarial network training.

This script provides a simplified interface for training OWC-SAWN on IDS datasets
without running the full pipeline.

Usage:
    python run_owc_sawn.py --dataset ../Dataset/normalized_data_2017.csv --epochs 100
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from owc_sawn import (
    ConditionalGenerator,
    WeightedDiscriminator,
    OWCSAWNTrainer,
    evaluate_sample_quality,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train OWC-SAWN on IDS dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to normalized/preprocessed IDS dataset CSV",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default="Label",
        help="Name of the label column in dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("owc_sawn_output"),
        help="Directory to save outputs",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=100,
        help="Latent dimension for generator",
    )
    parser.add_argument(
        "--generator-lr",
        type=float,
        default=0.0002,
        help="Generator learning rate",
    )
    parser.add_argument(
        "--discriminator-lr",
        type=float,
        default=0.0002,
        help="Discriminator learning rate",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test split ratio",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.1,
        help="Validation split ratio (from training set)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--generate-samples",
        type=int,
        default=1000,
        help="Number of samples to generate after training",
    )
    parser.add_argument(
        "--save-generated",
        action="store_true",
        help="Save generated samples to CSV",
    )
    
    return parser.parse_args()


def load_and_preprocess(dataset_path: Path, label_column: str, test_size: float, 
                        val_size: float, random_state: int):
    """Load and preprocess dataset."""
    print(f"\n{'='*80}")
    print("Loading Dataset")
    print(f"{'='*80}")
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    df = pd.read_csv(dataset_path)
    print(f"Loaded dataset: {dataset_path.name}")
    print(f"Shape: {df.shape}")
    
    if label_column not in df.columns:
        raise ValueError(f"Column '{label_column}' not found in dataset")
    
    # Separate features and labels
    X = df.drop(columns=[label_column]).values.astype(np.float32)
    y = df[label_column].values.astype(np.int32)
    
    # Convert to binary if multi-class
    if len(np.unique(y)) > 2:
        print(f"Converting {len(np.unique(y))} classes to binary (0=benign, 1=attack)")
        y = (y != 0).astype(np.int32)
    
    print(f"\nClass distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for cls, count in zip(unique, counts):
        print(f"  Class {cls}: {count} samples ({count/len(y)*100:.1f}%)")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, stratify=y_train, random_state=random_state
    )
    
    print(f"\nData splits:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Test: {len(X_test)} samples")
    
    # Normalize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)
    
    print(f"\nFeature dimension: {X_train.shape[1]}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_owc_sawn(X_train, y_train, X_val, y_val, args, output_dir):
    """Train OWC-SAWN network."""
    print(f"\n{'='*80}")
    print("Training OWC-SAWN")
    print(f"{'='*80}")
    
    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train))
    
    checkpoint_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs"
    
    print(f"\nConfiguration:")
    print(f"  Input dimension: {input_dim}")
    print(f"  Number of classes: {num_classes}")
    print(f"  Latent dimension: {args.latent_dim}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Generator LR: {args.generator_lr}")
    print(f"  Discriminator LR: {args.discriminator_lr}")
    
    # Create models
    generator = ConditionalGenerator(
        latent_dim=args.latent_dim,
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
        latent_dim=args.latent_dim,
        generator_lr=args.generator_lr,
        discriminator_lr=args.discriminator_lr,
        beta_1=0.5,
        beta_2=0.999,
        use_gradient_penalty=True,
        gp_lambda=10.0,
        n_discriminator_steps=5,
        checkpoint_dir=str(checkpoint_dir),
        log_dir=str(log_dir)
    )
    
    # Train
    print(f"\nStarting training...")
    print(f"Checkpoints will be saved to: {checkpoint_dir}")
    print(f"TensorBoard logs will be saved to: {log_dir}")
    print(f"\nTo monitor training, run:")
    print(f"  tensorboard --logdir={log_dir}")
    
    trainer.fit(
        X_train, y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(X_val, y_val),
        verbose=1
    )
    
    print(f"\n✓ Training completed!")
    
    return trainer


def visualize_training(trainer, output_dir):
    """Create training visualization."""
    print(f"\n{'='*80}")
    print("Visualizing Training History")
    print(f"{'='*80}")
    
    history = trainer.get_training_history()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Generator loss
    axes[0, 0].plot(history['gen_loss'], label='Generator Loss', color='blue')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Generator Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Discriminator loss
    axes[0, 1].plot(history['disc_loss'], label='Discriminator Loss', color='orange')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Discriminator Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Discriminator real accuracy
    axes[1, 0].plot(history['disc_real_acc'], label='Real Accuracy', color='green')
    axes[1, 0].axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Target (0.5)')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_title('Discriminator Real Accuracy')
    axes[1, 0].set_ylim([0, 1])
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Discriminator fake accuracy
    axes[1, 1].plot(history['disc_fake_acc'], label='Fake Accuracy', color='red')
    axes[1, 1].axhline(y=0.5, color='g', linestyle='--', alpha=0.5, label='Target (0.5)')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_title('Discriminator Fake Accuracy')
    axes[1, 1].set_ylim([0, 1])
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = output_dir / "training_history.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Training history plot saved to {plot_path}")
    plt.close()


def evaluate_and_generate(trainer, X_real, y_real, args, output_dir):
    """Evaluate sample quality and generate samples."""
    print(f"\n{'='*80}")
    print("Generating and Evaluating Samples")
    print(f"{'='*80}")
    
    num_classes = len(np.unique(y_real))
    samples_per_class = args.generate_samples // num_classes
    
    # Generate balanced samples
    labels = []
    for c in range(num_classes):
        labels.extend([c] * samples_per_class)
    labels = np.array(labels)
    
    print(f"\nGenerating {len(labels)} samples...")
    print(f"  {samples_per_class} samples per class")
    
    generated_samples = trainer.generate_samples(len(labels), labels)
    
    # Evaluate quality
    print(f"\nEvaluating sample quality...")
    metrics = evaluate_sample_quality(X_real, generated_samples)
    
    print(f"\n{'='*80}")
    print("Sample Quality Metrics")
    print(f"{'='*80}")
    print(f"  Overall Quality Score: {metrics['overall_quality_score']:.4f}")
    print(f"  Diversity: {metrics['diversity']:.4f}")
    print(f"  Coverage: {metrics['coverage']:.4f}")
    print(f"  Mean Distance: {metrics['mean_distance']:.4f}")
    print(f"  Std Distance: {metrics['std_distance']:.4f}")
    
    # Save metrics
    metrics_path = output_dir / "sample_quality_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"\n✓ Metrics saved to {metrics_path}")
    
    # Save generated samples
    if args.save_generated:
        samples_path = output_dir / "generated_samples.csv"
        samples_df = pd.DataFrame(generated_samples)
        samples_df['Label'] = labels
        samples_df.to_csv(samples_path, index=False)
        print(f"✓ Generated samples saved to {samples_path}")
    
    return generated_samples, labels, metrics


def save_summary(trainer, metrics, args, output_dir):
    """Save training summary."""
    history = trainer.get_training_history()
    
    summary = {
        "configuration": {
            "dataset": str(args.dataset),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "latent_dim": args.latent_dim,
            "generator_lr": args.generator_lr,
            "discriminator_lr": args.discriminator_lr,
        },
        "final_training_metrics": {
            "gen_loss": float(history['gen_loss'][-1]),
            "disc_loss": float(history['disc_loss'][-1]),
            "disc_real_acc": float(history['disc_real_acc'][-1]),
            "disc_fake_acc": float(history['disc_fake_acc'][-1]),
        },
        "sample_quality": {
            "overall_quality_score": float(metrics['overall_quality_score']),
            "diversity": float(metrics['diversity']),
            "coverage": float(metrics['coverage']),
        },
        "training_history": {
            "gen_loss": [float(x) for x in history['gen_loss']],
            "disc_loss": [float(x) for x in history['disc_loss']],
            "disc_real_acc": [float(x) for x in history['disc_real_acc']],
            "disc_fake_acc": [float(x) for x in history['disc_fake_acc']],
        }
    }
    
    summary_path = output_dir / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n✓ Training summary saved to {summary_path}")


def main():
    """Main execution function."""
    args = parse_args()
    
    # Create output directory
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print("OWC-SAWN Training Launcher")
    print(f"{'='*80}")
    print(f"Output directory: {output_dir}")
    
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess(
        args.dataset,
        args.label_column,
        args.test_size,
        args.val_size,
        args.random_state
    )
    
    # Train OWC-SAWN
    trainer = train_owc_sawn(X_train, y_train, X_val, y_val, args, output_dir)
    
    # Visualize training
    visualize_training(trainer, output_dir)
    
    # Generate and evaluate samples
    generated_samples, labels, metrics = evaluate_and_generate(
        trainer, X_train, y_train, args, output_dir
    )
    
    # Save summary
    save_summary(trainer, metrics, args, output_dir)
    
    # Print final summary
    print(f"\n{'='*80}")
    print("Training Complete!")
    print(f"{'='*80}")
    print(f"\nAll outputs saved to: {output_dir}")
    print(f"\nKey files:")
    print(f"  - training_summary.json: Complete training summary")
    print(f"  - training_history.png: Loss and accuracy plots")
    print(f"  - sample_quality_metrics.json: Generated sample quality")
    print(f"  - checkpoints/: Model checkpoints")
    print(f"  - logs/: TensorBoard logs")
    
    if args.save_generated:
        print(f"  - generated_samples.csv: Generated adversarial samples")
    
    print(f"\nTo view training progress in TensorBoard:")
    print(f"  tensorboard --logdir={output_dir / 'logs'}")
    
    print(f"\n✓ All tasks completed successfully!")


if __name__ == "__main__":
    main()
