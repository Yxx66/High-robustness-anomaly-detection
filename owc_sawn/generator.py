"""
Generator module for OWC-SAWN.

Implements conditional generator that creates adversarial samples
conditioned on class labels.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Tuple, Optional, List


class Generator(keras.Model):
    """
    Base Generator network for adversarial sample generation.
    
    Architecture:
    - Input: noise vector (latent_dim)
    - Hidden layers: fully connected with batch normalization
    - Output: synthetic data samples matching input feature dimension
    """
    
    def __init__(
        self,
        latent_dim: int = 100,
        output_dim: int = 78,
        hidden_layers: List[int] = [256, 512, 256],
        dropout_rate: float = 0.3,
        name: str = "generator",
        **kwargs
    ):
        """
        Initialize Generator.
        
        Args:
            latent_dim: Dimension of random noise input
            output_dim: Dimension of output (number of features)
            hidden_layers: List of hidden layer sizes
            dropout_rate: Dropout rate for regularization
            name: Model name
        """
        super(Generator, self).__init__(name=name, **kwargs)
        
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.hidden_layers_sizes = hidden_layers
        self.dropout_rate = dropout_rate
        
        # Build layers
        self.dense_layers = []
        self.batch_norm_layers = []
        self.dropout_layers = []
        
        for i, units in enumerate(hidden_layers):
            self.dense_layers.append(
                layers.Dense(
                    units,
                    kernel_initializer='he_normal',
                    name=f'dense_{i}'
                )
            )
            self.batch_norm_layers.append(
                layers.BatchNormalization(name=f'bn_{i}')
            )
            self.dropout_layers.append(
                layers.Dropout(dropout_rate, name=f'dropout_{i}')
            )
        
        # Output layer
        self.output_layer = layers.Dense(
            output_dim,
            activation='tanh',  # Output in [-1, 1] range
            kernel_initializer='glorot_normal',
            name='output'
        )
    
    def call(self, inputs, training=False):
        """
        Forward pass.
        
        Args:
            inputs: Noise tensor of shape (batch_size, latent_dim)
            training: Whether in training mode
            
        Returns:
            Generated samples of shape (batch_size, output_dim)
        """
        x = inputs
        
        # Pass through hidden layers
        for dense, bn, dropout in zip(
            self.dense_layers,
            self.batch_norm_layers,
            self.dropout_layers
        ):
            x = dense(x)
            x = bn(x, training=training)
            x = tf.nn.leaky_relu(x, alpha=0.2)
            x = dropout(x, training=training)
        
        # Output layer
        output = self.output_layer(x)
        
        return output
    
    def generate_samples(
        self,
        num_samples: int,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate synthetic samples.
        
        Args:
            num_samples: Number of samples to generate
            seed: Random seed for reproducibility
            
        Returns:
            Generated samples as numpy array
        """
        if seed is not None:
            tf.random.set_seed(seed)
        
        noise = tf.random.normal([num_samples, self.latent_dim])
        samples = self(noise, training=False)
        
        return samples.numpy()


class ConditionalGenerator(keras.Model):
    """
    Conditional Generator that generates samples based on class labels.
    
    This is the core component of OWC-SAWN, allowing controlled generation
    of adversarial samples for specific attack classes.
    """
    
    def __init__(
        self,
        latent_dim: int = 100,
        output_dim: int = 78,
        num_classes: int = 2,
        hidden_layers: List[int] = [256, 512, 512, 256],
        dropout_rate: float = 0.3,
        embedding_dim: int = 50,
        name: str = "conditional_generator",
        **kwargs
    ):
        """
        Initialize Conditional Generator.
        
        Args:
            latent_dim: Dimension of random noise input
            output_dim: Dimension of output features
            num_classes: Number of class labels
            hidden_layers: List of hidden layer sizes
            dropout_rate: Dropout rate
            embedding_dim: Dimension of label embedding
            name: Model name
        """
        super(ConditionalGenerator, self).__init__(name=name, **kwargs)
        
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.num_classes = num_classes
        self.hidden_layers_sizes = hidden_layers
        self.dropout_rate = dropout_rate
        self.embedding_dim = embedding_dim
        
        # Label embedding layer
        self.label_embedding = layers.Embedding(
            num_classes,
            embedding_dim,
            name='label_embedding'
        )
        
        # Concatenate noise and label embedding
        self.concat = layers.Concatenate(name='concat_noise_label')
        
        # Build hidden layers
        self.dense_layers = []
        self.batch_norm_layers = []
        self.dropout_layers = []
        
        input_dim = latent_dim + embedding_dim
        
        for i, units in enumerate(hidden_layers):
            self.dense_layers.append(
                layers.Dense(
                    units,
                    kernel_initializer=keras.initializers.GlorotUniform(),
                    bias_initializer='zeros',
                    name=f'dense_{i}'
                )
            )
            self.batch_norm_layers.append(
                layers.BatchNormalization(momentum=0.8, epsilon=1e-5, name=f'bn_{i}')
            )
            self.dropout_layers.append(
                layers.Dropout(dropout_rate, name=f'dropout_{i}')
            )
        
        # Output layer
        self.output_layer = layers.Dense(
            output_dim,
            activation='tanh',
            kernel_initializer='glorot_normal',
            name='output'
        )
    
    def call(self, inputs, training=False):
        """
        Forward pass with conditional input.
        
        Args:
            inputs: Tuple of (noise, labels)
                - noise: tensor of shape (batch_size, latent_dim)
                - labels: tensor of shape (batch_size,) with class indices
            training: Whether in training mode
            
        Returns:
            Generated samples of shape (batch_size, output_dim)
        """
        noise, labels = inputs
        
        # Embed labels
        label_embedding = self.label_embedding(labels)
        # Flatten if necessary
        if len(label_embedding.shape) > 2:
            label_embedding = tf.squeeze(label_embedding, axis=1)
        elif len(label_embedding.shape) == 2 and label_embedding.shape[1] != self.embedding_dim:
            # If shape is (batch, 1, embedding_dim), squeeze middle dimension
            label_embedding = tf.reshape(label_embedding, [-1, self.embedding_dim])
        
        # Concatenate noise and label embedding
        x = self.concat([noise, label_embedding])
        
        # Pass through hidden layers (WITHOUT BatchNorm for stability)
        for dense, bn, dropout in zip(
            self.dense_layers,
            self.batch_norm_layers,
            self.dropout_layers
        ):
            x = dense(x)
            # SKIP BatchNorm - causes NaN in epoch 2
            # x = bn(x, training=training)
            x = tf.nn.leaky_relu(x, alpha=0.01)
            x = dropout(x, training=training)
        
        # Output layer
        output = self.output_layer(x)
        
        return output
    
    def generate_samples(
        self,
        num_samples: int,
        labels: Optional[np.ndarray] = None,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate conditional synthetic samples.
        
        Args:
            num_samples: Number of samples to generate
            labels: Class labels for each sample. If None, randomly sample.
            seed: Random seed
            
        Returns:
            Generated samples as numpy array
        """
        if seed is not None:
            tf.random.set_seed(seed)
        
        noise = tf.random.normal([num_samples, self.latent_dim])
        
        if labels is None:
            # Random labels
            labels = np.random.randint(0, self.num_classes, size=num_samples)
        
        labels = tf.constant(labels, dtype=tf.int32)
        
        samples = self([noise, labels], training=False)
        
        return samples.numpy()
    
    def generate_class_samples(
        self,
        num_samples: int,
        target_class: int,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate samples for a specific class.
        
        Args:
            num_samples: Number of samples to generate
            target_class: Target class index
            seed: Random seed
            
        Returns:
            Generated samples for the target class
        """
        labels = np.full(num_samples, target_class, dtype=np.int32)
        return self.generate_samples(num_samples, labels, seed)


def build_generator(
    latent_dim: int = 100,
    output_dim: int = 78,
    conditional: bool = True,
    num_classes: int = 2,
    **kwargs
) -> keras.Model:
    """
    Factory function to build generator.
    
    Args:
        latent_dim: Dimension of latent noise
        output_dim: Output feature dimension
        conditional: Whether to use conditional generator
        num_classes: Number of classes (for conditional generator)
        **kwargs: Additional arguments for generator
        
    Returns:
        Generator model
    """
    if conditional:
        return ConditionalGenerator(
            latent_dim=latent_dim,
            output_dim=output_dim,
            num_classes=num_classes,
            **kwargs
        )
    else:
        return Generator(
            latent_dim=latent_dim,
            output_dim=output_dim,
            **kwargs
        )


if __name__ == "__main__":
    # Test generator
    print("Testing Generator...")
    
    gen = Generator(latent_dim=100, output_dim=78)
    noise = tf.random.normal([32, 100])
    samples = gen(noise, training=False)
    print(f"Generator output shape: {samples.shape}")
    
    print("\nTesting Conditional Generator...")
    cond_gen = ConditionalGenerator(
        latent_dim=100,
        output_dim=78,
        num_classes=2
    )
    noise = tf.random.normal([32, 100])
    labels = tf.constant([0] * 16 + [1] * 16, dtype=tf.int32)
    samples = cond_gen([noise, labels], training=False)
    print(f"Conditional Generator output shape: {samples.shape}")
    
    # Test sample generation
    samples = cond_gen.generate_class_samples(100, target_class=1)
    print(f"Generated {samples.shape[0]} samples for class 1")
    
    print("\n✓ Generator tests passed!")
