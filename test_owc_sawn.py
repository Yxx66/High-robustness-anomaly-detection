"""
Test script for OWC-SAWN module.

Tests all components of the OWC-SAWN adversarial network.
"""

import sys
from pathlib import Path
import numpy as np
import tensorflow as tf

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

print("Testing OWC-SAWN Module...")
print("=" * 80)

# Test 1: Import modules
print("\n[1/6] Testing imports...")
try:
    from owc_sawn.generator import Generator, ConditionalGenerator, build_generator
    from owc_sawn.discriminator import (
        Discriminator, WeightedDiscriminator,
        AuxiliaryDiscriminator, build_discriminator
    )
    from owc_sawn.trainer import OWCSAWNTrainer
    from owc_sawn.utils import (
        generate_adversarial_samples,
        evaluate_sample_quality,
        plot_training_history,
        augment_data_with_gan
    )
    print("✓ All modules imported successfully")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Generator
print("\n[2/6] Testing Generator...")
try:
    # Basic generator
    gen = Generator(latent_dim=100, output_dim=78)
    noise = tf.random.normal([32, 100])
    samples = gen(noise, training=False)
    assert samples.shape == (32, 78), f"Expected (32, 78), got {samples.shape}"
    
    # Conditional generator
    cond_gen = ConditionalGenerator(latent_dim=100, output_dim=78, num_classes=2)
    noise = tf.random.normal([32, 100])
    labels = tf.constant([0] * 16 + [1] * 16, dtype=tf.int32)
    samples = cond_gen([noise, labels], training=False)
    assert samples.shape == (32, 78), f"Expected (32, 78), got {samples.shape}"
    
    # Generate samples
    samples = cond_gen.generate_class_samples(100, target_class=1)
    assert samples.shape == (100, 78), f"Expected (100, 78), got {samples.shape}"
    
    print("✓ Generator tests passed")
except Exception as e:
    print(f"✗ Generator test failed: {e}")
    sys.exit(1)

# Test 3: Discriminator
print("\n[3/6] Testing Discriminator...")
try:
    # Basic discriminator
    disc = Discriminator(input_dim=78)
    samples = tf.random.normal([32, 78])
    output = disc(samples, training=False)
    assert output.shape == (32, 1), f"Expected (32, 1), got {output.shape}"
    
    # Weighted discriminator
    weighted_disc = WeightedDiscriminator(input_dim=78, num_classes=2)
    samples = tf.random.normal([32, 78])
    labels = tf.constant([0] * 16 + [1] * 16, dtype=tf.int32)
    output = weighted_disc([samples, labels], training=False)
    assert output.shape == (32, 1), f"Expected (32, 1), got {output.shape}"
    
    # Check class weights
    weights = weighted_disc.get_class_weights()
    assert len(weights) == 2, f"Expected 2 weights, got {len(weights)}"
    assert np.abs(np.sum(weights) - 1.0) < 1e-5, "Weights should sum to 1"
    
    # Auxiliary discriminator
    aux_disc = AuxiliaryDiscriminator(input_dim=78, num_classes=2)
    samples = tf.random.normal([32, 78])
    real_output, class_output = aux_disc(samples, training=False)
    assert real_output.shape == (32, 1), f"Expected (32, 1), got {real_output.shape}"
    assert class_output.shape == (32, 2), f"Expected (32, 2), got {class_output.shape}"
    
    print("✓ Discriminator tests passed")
except Exception as e:
    print(f"✗ Discriminator test failed: {e}")
    sys.exit(1)

# Test 4: Trainer
print("\n[4/6] Testing Trainer...")
try:
    # Create models
    generator = ConditionalGenerator(latent_dim=100, output_dim=78, num_classes=2)
    discriminator = WeightedDiscriminator(input_dim=78, num_classes=2)
    
    # Create trainer
    trainer = OWCSAWNTrainer(
        generator=generator,
        discriminator=discriminator,
        latent_dim=100,
        n_discriminator_steps=2,
        checkpoint_dir="test_checkpoints",
        log_dir="test_logs"
    )
    
    # Create synthetic data
    X_train = np.random.randn(500, 78).astype(np.float32)
    y_train = np.random.randint(0, 2, size=500).astype(np.int32)
    
    # Train for 2 epochs
    trainer.fit(X_train, y_train, epochs=2, batch_size=32, verbose=0)
    
    # Check history
    assert len(trainer.training_history['gen_loss']) == 2
    assert len(trainer.training_history['disc_loss']) == 2
    
    # Generate samples
    samples = trainer.generate_samples(10, labels=np.array([0]*5 + [1]*5))
    assert samples.shape == (10, 78), f"Expected (10, 78), got {samples.shape}"
    
    print("✓ Trainer tests passed")
except Exception as e:
    print(f"✗ Trainer test failed: {e}")
    sys.exit(1)

# Test 5: Utils
print("\n[5/6] Testing Utilities...")
try:
    # Generate adversarial samples
    samples, labels = generate_adversarial_samples(
        generator=trainer.generator,
        num_samples=100,
        num_classes=2
    )
    assert samples.shape == (100, 78), f"Expected (100, 78), got {samples.shape}"
    assert len(labels) == 100, f"Expected 100 labels, got {len(labels)}"
    
    # Evaluate sample quality
    real_samples = np.random.randn(200, 78)
    fake_samples = np.random.randn(100, 78) * 0.8
    
    metrics = evaluate_sample_quality(real_samples, fake_samples)
    assert 'overall_quality_score' in metrics
    assert 'diversity' in metrics
    assert 'coverage' in metrics
    
    # Data augmentation
    X_train = np.random.randn(100, 78).astype(np.float32)
    y_train = np.random.randint(0, 2, size=100).astype(np.int32)
    
    X_aug, y_aug = augment_data_with_gan(
        X_train, y_train,
        generator=trainer.generator,
        augmentation_ratio=0.5
    )
    expected_size = int(100 * 1.5)
    assert X_aug.shape[0] == expected_size, f"Expected {expected_size}, got {X_aug.shape[0]}"
    
    print("✓ Utility tests passed")
except Exception as e:
    print(f"✗ Utility test failed: {e}")
    sys.exit(1)

# Test 6: Integration test
print("\n[6/6] Testing full pipeline integration...")
try:
    # Create fresh models
    generator = build_generator(
        latent_dim=50,
        output_dim=20,
        conditional=True,
        num_classes=2,
        hidden_layers=[128, 64]
    )
    
    discriminator = build_discriminator(
        input_dim=20,
        conditional=True,
        num_classes=2,
        discriminator_type="weighted",
        hidden_layers=[64, 32]
    )
    
    # Create trainer
    trainer = OWCSAWNTrainer(
        generator=generator,
        discriminator=discriminator,
        latent_dim=50,
        generator_lr=0.001,
        discriminator_lr=0.001,
        n_discriminator_steps=3,
        checkpoint_dir="test_checkpoints",
        log_dir="test_logs"
    )
    
    # Create small dataset
    X_train = np.random.randn(200, 20).astype(np.float32)
    y_train = np.random.randint(0, 2, size=200).astype(np.int32)
    
    # Train
    trainer.fit(X_train, y_train, epochs=3, batch_size=32, verbose=0)
    
    # Generate and evaluate
    gen_samples, gen_labels = generate_adversarial_samples(
        generator=trainer.generator,
        num_samples=50,
        labels=np.array([0]*25 + [1]*25)
    )
    
    quality_metrics = evaluate_sample_quality(X_train, gen_samples)
    
    assert quality_metrics['overall_quality_score'] > 0, "Quality score should be positive"
    
    print("✓ Integration test passed")
except Exception as e:
    print(f"✗ Integration test failed: {e}")
    sys.exit(1)

print("\n" + "=" * 80)
print("✓ All OWC-SAWN tests passed successfully!")
print("=" * 80)

print("\nModule Summary:")
print(f"  Generator: {generator.__class__.__name__}")
print(f"  Discriminator: {discriminator.__class__.__name__}")
print(f"  Latent dim: {trainer.latent_dim}")
print(f"  Output dim: {X_train.shape[1]}")
print(f"  Training epochs: 3")
print(f"  Final generator loss: {trainer.training_history['gen_loss'][-1]:.4f}")
print(f"  Final discriminator loss: {trainer.training_history['disc_loss'][-1]:.4f}")
print(f"  Sample quality score: {quality_metrics['overall_quality_score']:.4f}")

print("\nNext steps:")
print("1. Integrate OWC-SAWN into main pipeline")
print("2. Train on real IDS datasets")
print("3. Evaluate adversarial sample quality")
print("4. Compare with FGSM-based generation")
