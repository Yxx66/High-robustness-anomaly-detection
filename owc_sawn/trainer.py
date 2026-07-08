"""
Trainer module for OWC-SAWN.

Implements stepwise adversarial training with optimized weighted loss.
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
import time
import json

from .generator import ConditionalGenerator
from .discriminator import WeightedDiscriminator


class OWCSAWNTrainer:
    """
    Trainer for Optimized Weighted Conditional Stepwise Adversarial Network.
    
    Features:
    - Stepwise training (alternating Generator and Discriminator updates)
    - Weighted loss for different attack classes
    - Gradient penalty for training stability
    - Comprehensive logging and checkpointing
    """
    
    def __init__(
        self,
        generator: ConditionalGenerator,
        discriminator: WeightedDiscriminator,
        latent_dim: int = 100,
        generator_lr: float = 0.0002,
        discriminator_lr: float = 0.0002,
        beta_1: float = 0.5,
        beta_2: float = 0.999,
        use_gradient_penalty: bool = True,
        gp_lambda: float = 10.0,
        n_discriminator_steps: int = 5,
        checkpoint_dir: Optional[str] = None,
        log_dir: Optional[str] = None
    ):
        """
        Initialize OWC-SAWN Trainer.
        
        Args:
            generator: Generator model
            discriminator: Discriminator model
            latent_dim: Dimension of latent noise
            generator_lr: Learning rate for generator
            discriminator_lr: Learning rate for discriminator
            beta_1: Adam beta_1 parameter
            beta_2: Adam beta_2 parameter
            use_gradient_penalty: Whether to use gradient penalty
            gp_lambda: Gradient penalty coefficient
            n_discriminator_steps: Number of discriminator steps per generator step
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory for tensorboard logs
        """
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim
        
        self.use_gradient_penalty = use_gradient_penalty
        self.gp_lambda = gp_lambda
        self.n_discriminator_steps = n_discriminator_steps
        
        # Optimizers
        self.generator_optimizer = keras.optimizers.Adam(
            learning_rate=generator_lr,
            beta_1=beta_1,
            beta_2=beta_2
        )
        
        self.discriminator_optimizer = keras.optimizers.Adam(
            learning_rate=discriminator_lr,
            beta_1=beta_1,
            beta_2=beta_2
        )
        
        # Loss function
        self.bce_loss = keras.losses.BinaryCrossentropy(from_logits=False)
        
        # Metrics
        self.gen_loss_metric = keras.metrics.Mean(name='gen_loss')
        self.disc_loss_metric = keras.metrics.Mean(name='disc_loss')
        self.disc_real_acc_metric = keras.metrics.BinaryAccuracy(name='disc_real_acc')
        self.disc_fake_acc_metric = keras.metrics.BinaryAccuracy(name='disc_fake_acc')
        
        # Checkpointing
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            self.checkpoint = tf.train.Checkpoint(
                generator=generator,
                discriminator=discriminator,
                generator_optimizer=self.generator_optimizer,
                discriminator_optimizer=self.discriminator_optimizer
            )
            
            self.checkpoint_manager = tf.train.CheckpointManager(
                self.checkpoint,
                str(self.checkpoint_dir),
                max_to_keep=5
            )
        
        # Logging
        self.log_dir = Path(log_dir) if log_dir else None
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.summary_writer = tf.summary.create_file_writer(str(self.log_dir))
        
        self.training_history = {
            'gen_loss': [],
            'disc_loss': [],
            'disc_real_acc': [],
            'disc_fake_acc': [],
            'epoch_time': []
        }
    
    def gradient_penalty(
        self,
        real_samples: tf.Tensor,
        fake_samples: tf.Tensor,
        labels: tf.Tensor
    ) -> tf.Tensor:
        """
        Compute gradient penalty for WGAN-GP style training.
        
        Args:
            real_samples: Real data samples
            fake_samples: Generated fake samples
            labels: Class labels
            
        Returns:
            Gradient penalty loss
        """
        batch_size = tf.shape(real_samples)[0]
        alpha = tf.random.uniform([batch_size, 1], 0.0, 1.0)
        
        # Interpolate between real and fake samples
        interpolated = alpha * real_samples + (1 - alpha) * fake_samples
        
        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            pred = self.discriminator([interpolated, labels], training=True)
        
        # Compute gradients
        grads = tape.gradient(pred, interpolated)
        
        # Compute gradient norm
        grad_norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=1))
        
        # Gradient penalty: (||grad|| - 1)^2
        gp = tf.reduce_mean((grad_norm - 1.0) ** 2)
        
        return gp
    
    @tf.function
    def train_discriminator_step(
        self,
        real_samples: tf.Tensor,
        labels: tf.Tensor
    ) -> Dict[str, tf.Tensor]:
        """
        Single discriminator training step.
        
        Args:
            real_samples: Batch of real samples
            labels: Batch of labels
            
        Returns:
            Dictionary of losses
        """
        batch_size = tf.shape(real_samples)[0]
        
        # Generate fake samples
        noise = tf.random.normal([batch_size, self.latent_dim])
        fake_samples = self.generator([noise, labels], training=False)
        
        with tf.GradientTape() as tape:
            # Discriminator predictions
            real_output = self.discriminator([real_samples, labels], training=True)
            fake_output = self.discriminator([fake_samples, labels], training=True)
            
            # Compute weighted loss
            disc_loss, loss_dict = self.discriminator.compute_weighted_loss(
                real_output, fake_output, labels, labels, self.bce_loss
            )
            
            # Add gradient penalty if enabled
            if self.use_gradient_penalty:
                gp = self.gradient_penalty(real_samples, fake_samples, labels)
                disc_loss += self.gp_lambda * gp
                loss_dict['gradient_penalty'] = gp
        
        # Update discriminator
        gradients = tape.gradient(disc_loss, self.discriminator.trainable_variables)
        # Clip gradients to prevent explosion
        gradients = [tf.clip_by_norm(g, 1.0) if g is not None else g for g in gradients]
        self.discriminator_optimizer.apply_gradients(
            zip(gradients, self.discriminator.trainable_variables)
        )
        
        # Update metrics
        self.disc_loss_metric.update_state(disc_loss)
        self.disc_real_acc_metric.update_state(tf.ones_like(real_output), real_output)
        self.disc_fake_acc_metric.update_state(tf.zeros_like(fake_output), fake_output)
        
        return loss_dict
    
    @tf.function
    def train_generator_step(
        self,
        batch_size: int,
        labels: tf.Tensor
    ) -> tf.Tensor:
        """
        Single generator training step.
        
        Args:
            batch_size: Size of batch
            labels: Batch of labels
            
        Returns:
            Generator loss
        """
        noise = tf.random.normal([batch_size, self.latent_dim])
        
        with tf.GradientTape() as tape:
            # Generate fake samples
            fake_samples = self.generator([noise, labels], training=True)
            
            # Discriminator prediction on fake samples
            fake_output = self.discriminator([fake_samples, labels], training=False)
            
            # Generator wants discriminator to think samples are real
            gen_loss = self.bce_loss(tf.ones_like(fake_output), fake_output)
        
        # Update generator
        gradients = tape.gradient(gen_loss, self.generator.trainable_variables)
        # Clip gradients to prevent explosion
        gradients = [tf.clip_by_norm(g, 1.0) if g is not None else g for g in gradients]
        self.generator_optimizer.apply_gradients(
            zip(gradients, self.generator.trainable_variables)
        )
        
        # Update metrics
        self.gen_loss_metric.update_state(gen_loss)
        
        return gen_loss
    
    def train_epoch(
        self,
        dataset: tf.data.Dataset,
        epoch: int
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            dataset: Training dataset
            epoch: Current epoch number
            
        Returns:
            Dictionary of epoch metrics
        """
        start_time = time.time()
        
        # Reset metrics
        self.gen_loss_metric.reset_states()
        self.disc_loss_metric.reset_states()
        self.disc_real_acc_metric.reset_states()
        self.disc_fake_acc_metric.reset_states()
        
        for step, (real_samples, labels) in enumerate(dataset):
            # Train discriminator for n_discriminator_steps
            for _ in range(self.n_discriminator_steps):
                disc_loss_dict = self.train_discriminator_step(real_samples, labels)
            
            # Train generator
            batch_size = tf.shape(real_samples)[0]
            gen_loss = self.train_generator_step(batch_size, labels)
        
        epoch_time = time.time() - start_time
        
        # Collect metrics
        metrics = {
            'gen_loss': float(self.gen_loss_metric.result()),
            'disc_loss': float(self.disc_loss_metric.result()),
            'disc_real_acc': float(self.disc_real_acc_metric.result()),
            'disc_fake_acc': float(self.disc_fake_acc_metric.result()),
            'epoch_time': epoch_time
        }
        
        # Update history
        for key, value in metrics.items():
            self.training_history[key].append(value)
        
        # Log to tensorboard
        if self.summary_writer:
            with self.summary_writer.as_default():
                for key, value in metrics.items():
                    tf.summary.scalar(key, value, step=epoch)
                
                # Log class weights
                class_weights = self.discriminator.get_class_weights()
                for i, weight in enumerate(class_weights):
                    tf.summary.scalar(f'class_weight_{i}', weight, step=epoch)
        
        return metrics
    
    def fit(
        self,
        train_data: np.ndarray,
        train_labels: np.ndarray,
        epochs: int = 100,
        batch_size: int = 128,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        verbose: int = 1,
        save_interval: int = 10,
        sample_callback: Optional[Callable] = None
    ):
        """
        Train the OWC-SAWN model.
        
        Args:
            train_data: Training samples
            train_labels: Training labels
            epochs: Number of training epochs
            batch_size: Batch size
            validation_data: Optional validation data (X_val, y_val)
            verbose: Verbosity level
            save_interval: Save checkpoint every N epochs
            sample_callback: Optional callback to visualize samples
        """
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices(
            (train_data.astype(np.float32), train_labels.astype(np.int32))
        )
        dataset = dataset.shuffle(buffer_size=10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        if verbose:
            print(f"\nTraining OWC-SAWN for {epochs} epochs")
            print(f"Dataset: {train_data.shape[0]} samples, {train_data.shape[1]} features")
            print(f"Batch size: {batch_size}")
            print(f"Discriminator steps per generator step: {self.n_discriminator_steps}")
            print("=" * 80)
        
        # Training loop
        for epoch in range(epochs):
            metrics = self.train_epoch(dataset, epoch)
            
            if verbose:
                print(
                    f"Epoch {epoch+1}/{epochs} | "
                    f"G Loss: {metrics['gen_loss']:.4f} | "
                    f"D Loss: {metrics['disc_loss']:.4f} | "
                    f"D Real Acc: {metrics['disc_real_acc']:.4f} | "
                    f"D Fake Acc: {metrics['disc_fake_acc']:.4f} | "
                    f"Time: {metrics['epoch_time']:.2f}s"
                )
            
            # Save checkpoint
            if self.checkpoint_manager and (epoch + 1) % save_interval == 0:
                save_path = self.checkpoint_manager.save()
                if verbose:
                    print(f"Checkpoint saved at: {save_path}")
            
            # Sample callback
            if sample_callback and (epoch + 1) % save_interval == 0:
                sample_callback(self.generator, epoch)
        
        if verbose:
            print("=" * 80)
            print("Training completed!")
        
        # Save final model
        if self.checkpoint_dir:
            self.save_models(self.checkpoint_dir / "final")
            
            # Save training history
            history_path = self.checkpoint_dir / "training_history.json"
            with open(history_path, 'w') as f:
                json.dump(self.training_history, f, indent=2)
            
            if verbose:
                print(f"Models and history saved to {self.checkpoint_dir}")
    
    def save_models(self, save_dir: Path):
        """Save generator and discriminator models."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        self.generator.save_weights(str(save_dir / "generator_weights.h5"))
        self.discriminator.save_weights(str(save_dir / "discriminator_weights.h5"))
    
    def load_models(self, load_dir: Path):
        """Load generator and discriminator models."""
        load_dir = Path(load_dir)
        
        self.generator.load_weights(str(load_dir / "generator_weights.h5"))
        self.discriminator.load_weights(str(load_dir / "discriminator_weights.h5"))
    
    def generate_samples(
        self,
        num_samples: int,
        labels: Optional[np.ndarray] = None,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate synthetic samples using trained generator.
        
        Args:
            num_samples: Number of samples to generate
            labels: Class labels for samples
            seed: Random seed
            
        Returns:
            Generated samples
        """
        return self.generator.generate_samples(num_samples, labels, seed)


if __name__ == "__main__":
    print("Testing OWC-SAWN Trainer...")
    
    from .generator import ConditionalGenerator
    from .discriminator import WeightedDiscriminator
    
    # Create models
    generator = ConditionalGenerator(
        latent_dim=100,
        output_dim=78,
        num_classes=2
    )
    
    discriminator = WeightedDiscriminator(
        input_dim=78,
        num_classes=2
    )
    
    # Create trainer
    trainer = OWCSAWNTrainer(
        generator=generator,
        discriminator=discriminator,
        latent_dim=100
    )
    
    # Test with synthetic data
    X_train = np.random.randn(1000, 78).astype(np.float32)
    y_train = np.random.randint(0, 2, size=1000).astype(np.int32)
    
    print("\nTraining for 2 epochs (test)...")
    trainer.fit(
        X_train,
        y_train,
        epochs=2,
        batch_size=32,
        verbose=1
    )
    
    # Generate samples
    samples = trainer.generate_samples(10, labels=np.array([0]*5 + [1]*5))
    print(f"\nGenerated samples shape: {samples.shape}")
    
    print("\n✓ Trainer tests passed!")
