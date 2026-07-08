"""
Utility functions for OWC-SAWN.

Includes functions for sample generation, quality evaluation, and visualization.
"""

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
from typing import Tuple, Dict, Optional, List
import matplotlib.pyplot as plt


def generate_adversarial_samples(
    generator,
    num_samples: int,
    labels: Optional[np.ndarray] = None,
    num_classes: int = 2,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate adversarial samples using trained generator.
    
    Args:
        generator: Trained generator model
        num_samples: Number of samples to generate
        labels: Class labels. If None, generate balanced samples
        num_classes: Number of classes
        seed: Random seed
        
    Returns:
        Tuple of (generated_samples, labels)
    """
    if labels is None:
        # Generate balanced samples
        samples_per_class = num_samples // num_classes
        labels = np.repeat(np.arange(num_classes), samples_per_class)
        
        # Add remaining samples to last class
        remaining = num_samples - len(labels)
        if remaining > 0:
            labels = np.concatenate([labels, np.full(remaining, num_classes - 1)])
    
    samples = generator.generate_samples(num_samples, labels, seed)
    
    return samples, labels


def evaluate_sample_quality(
    real_samples: np.ndarray,
    fake_samples: np.ndarray,
    classifier=None,
    real_labels: Optional[np.ndarray] = None,
    fake_labels: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Evaluate quality of generated samples.
    
    Metrics:
    - Statistical distance (mean, std comparison)
    - Nearest neighbor distance
    - Diversity (intra-class distance)
    - Classifier performance (if classifier provided)
    
    Args:
        real_samples: Real data samples
        fake_samples: Generated samples
        classifier: Optional classifier to test samples
        real_labels: Labels for real samples
        fake_labels: Labels for fake samples
        
    Returns:
        Dictionary of quality metrics
    """
    metrics = {}
    
    # 1. Statistical distance
    real_mean = np.mean(real_samples, axis=0)
    fake_mean = np.mean(fake_samples, axis=0)
    mean_distance = np.linalg.norm(real_mean - fake_mean)
    
    real_std = np.std(real_samples, axis=0)
    fake_std = np.std(fake_samples, axis=0)
    std_distance = np.linalg.norm(real_std - fake_std)
    
    metrics['mean_distance'] = float(mean_distance)
    metrics['std_distance'] = float(std_distance)
    metrics['statistical_similarity'] = float(1.0 / (1.0 + mean_distance + std_distance))
    
    # 2. Nearest neighbor distance (how close fake samples are to real ones)
    nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
    nn.fit(real_samples)
    distances, _ = nn.kneighbors(fake_samples)
    avg_nn_distance = np.mean(distances)
    
    metrics['avg_nearest_neighbor_distance'] = float(avg_nn_distance)
    
    # 3. Diversity (average pairwise distance within fake samples)
    if len(fake_samples) > 1:
        pairwise_dist = cdist(fake_samples, fake_samples, metric='euclidean')
        # Exclude diagonal (distance to self)
        np.fill_diagonal(pairwise_dist, np.nan)
        diversity = np.nanmean(pairwise_dist)
        metrics['diversity'] = float(diversity)
    else:
        metrics['diversity'] = 0.0
    
    # 4. Coverage (what percentage of real sample space is covered)
    # Use k-nearest neighbors from fake to real
    nn_coverage = NearestNeighbors(n_neighbors=5, metric='euclidean')
    nn_coverage.fit(fake_samples)
    distances, _ = nn_coverage.kneighbors(real_samples)
    coverage = np.mean(distances[:, 0] < avg_nn_distance * 2)  # Within 2x avg distance
    metrics['coverage'] = float(coverage)
    
    # 5. Classifier performance (if provided)
    if classifier is not None and real_labels is not None and fake_labels is not None:
        # Train on real, test on fake
        classifier.fit(real_samples, real_labels)
        fake_pred = classifier.predict(fake_samples)
        
        accuracy = accuracy_score(fake_labels, fake_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            fake_labels, fake_pred, average='weighted', zero_division=0
        )
        
        metrics['classifier_accuracy'] = float(accuracy)
        metrics['classifier_precision'] = float(precision)
        metrics['classifier_recall'] = float(recall)
        metrics['classifier_f1'] = float(f1)
    
    # 6. Overall quality score (weighted combination)
    quality_score = (
        0.3 * metrics['statistical_similarity'] +
        0.2 * (1.0 / (1.0 + metrics['avg_nearest_neighbor_distance'])) +
        0.2 * min(1.0, metrics['diversity'] / np.std(real_samples)) +
        0.3 * metrics['coverage']
    )
    metrics['overall_quality_score'] = float(quality_score)
    
    return metrics


def plot_sample_distribution(
    real_samples: np.ndarray,
    fake_samples: np.ndarray,
    feature_indices: List[int] = [0, 1],
    save_path: Optional[str] = None
):
    """
    Plot distribution comparison of real vs generated samples.
    
    Args:
        real_samples: Real data samples
        fake_samples: Generated samples
        feature_indices: Which features to plot (max 2 for 2D plot)
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 2D scatter plot
    ax = axes[0]
    ax.scatter(
        real_samples[:, feature_indices[0]],
        real_samples[:, feature_indices[1]],
        alpha=0.5,
        s=20,
        label='Real',
        color='blue'
    )
    ax.scatter(
        fake_samples[:, feature_indices[0]],
        fake_samples[:, feature_indices[1]],
        alpha=0.5,
        s=20,
        label='Generated',
        color='red'
    )
    ax.set_xlabel(f'Feature {feature_indices[0]}', fontsize=12)
    ax.set_ylabel(f'Feature {feature_indices[1]}', fontsize=12)
    ax.set_title('2D Feature Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Distribution comparison (histograms)
    ax = axes[1]
    feature_idx = feature_indices[0]
    ax.hist(
        real_samples[:, feature_idx],
        bins=50,
        alpha=0.5,
        label='Real',
        color='blue',
        density=True
    )
    ax.hist(
        fake_samples[:, feature_idx],
        bins=50,
        alpha=0.5,
        label='Generated',
        color='red',
        density=True
    )
    ax.set_xlabel(f'Feature {feature_idx} Value', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'Feature {feature_idx} Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None
):
    """
    Plot training history.
    
    Args:
        history: Training history dictionary
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Generator loss
    ax = axes[0, 0]
    ax.plot(history['gen_loss'], label='Generator Loss', color='blue', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title('Generator Loss', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend()
    
    # Discriminator loss
    ax = axes[0, 1]
    ax.plot(history['disc_loss'], label='Discriminator Loss', color='red', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title('Discriminator Loss', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend()
    
    # Discriminator accuracies
    ax = axes[1, 0]
    ax.plot(history['disc_real_acc'], label='Real Accuracy', color='green', linewidth=2)
    ax.plot(history['disc_fake_acc'], label='Fake Accuracy', color='orange', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.set_title('Discriminator Accuracies', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend()
    
    # Epoch time
    ax = axes[1, 1]
    ax.plot(history['epoch_time'], label='Epoch Time', color='purple', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Time (seconds)', fontsize=11)
    ax.set_title('Training Time per Epoch', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def compute_inception_score(
    samples: np.ndarray,
    classifier,
    num_splits: int = 10
) -> Tuple[float, float]:
    """
    Compute Inception Score for generated samples.
    
    The Inception Score measures both quality and diversity of generated samples.
    Higher is better.
    
    Args:
        samples: Generated samples
        classifier: Classifier to compute class probabilities
        num_splits: Number of splits for computing mean and std
        
    Returns:
        Tuple of (mean_score, std_score)
    """
    # Get predicted class probabilities
    if hasattr(classifier, 'predict_proba'):
        preds = classifier.predict_proba(samples)
    else:
        # If classifier doesn't have predict_proba, use one-hot encoding of predictions
        pred_labels = classifier.predict(samples)
        num_classes = len(np.unique(pred_labels))
        preds = np.eye(num_classes)[pred_labels]
    
    # Split into groups
    split_size = samples.shape[0] // num_splits
    scores = []
    
    for i in range(num_splits):
        part = preds[i * split_size:(i + 1) * split_size]
        
        # Compute KL divergence
        py = np.mean(part, axis=0)
        scores_part = []
        
        for j in range(part.shape[0]):
            pyx = part[j, :]
            scores_part.append(np.sum(pyx * (np.log(pyx + 1e-10) - np.log(py + 1e-10))))
        
        scores.append(np.exp(np.mean(scores_part)))
    
    return float(np.mean(scores)), float(np.std(scores))


def augment_data_with_gan(
    X_train: np.ndarray,
    y_train: np.ndarray,
    generator,
    augmentation_ratio: float = 0.5,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Augment training data with GAN-generated samples.
    
    Args:
        X_train: Original training samples
        y_train: Original training labels
        generator: Trained generator
        augmentation_ratio: Ratio of synthetic to real samples
        seed: Random seed
        
    Returns:
        Tuple of (augmented_X, augmented_y)
    """
    num_synthetic = int(len(X_train) * augmentation_ratio)
    
    # Generate synthetic samples with same label distribution
    unique_labels, label_counts = np.unique(y_train, return_counts=True)
    label_probs = label_counts / len(y_train)
    
    synthetic_labels = np.random.choice(
        unique_labels,
        size=num_synthetic,
        p=label_probs
    )
    
    synthetic_samples, _ = generate_adversarial_samples(
        generator,
        num_synthetic,
        labels=synthetic_labels,
        seed=seed
    )
    
    # Combine real and synthetic
    X_augmented = np.vstack([X_train, synthetic_samples])
    y_augmented = np.concatenate([y_train, synthetic_labels])
    
    # Shuffle
    if seed is not None:
        np.random.seed(seed)
    indices = np.random.permutation(len(X_augmented))
    X_augmented = X_augmented[indices]
    y_augmented = y_augmented[indices]
    
    return X_augmented, y_augmented


if __name__ == "__main__":
    print("Testing OWC-SAWN utilities...")
    
    # Create synthetic data
    real_samples = np.random.randn(1000, 78)
    fake_samples = np.random.randn(500, 78) * 0.8 + 0.1  # Slightly different distribution
    
    # Evaluate quality
    print("\nEvaluating sample quality...")
    metrics = evaluate_sample_quality(real_samples, fake_samples)
    
    print("\nQuality Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    print("\n✓ Utility tests passed!")
