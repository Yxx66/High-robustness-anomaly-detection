"""
Discriminator module for OWC-SAWN.

Implements weighted discriminator that distinguishes real from generated samples
with optimized loss weighting for different attack classes.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Tuple, Optional, List, Dict


class Discriminator(keras.Model):
    """
    Base Discriminator network for adversarial detection.
    
    Architecture:
    - Input: data samples (feature_dim)
    - Hidden layers: fully connected with dropout
    - Output: probability of being real (sigmoid)
    """
    
    def __init__(
        self,
        input_dim: int = 78,
        hidden_layers: List[int] = [256, 128, 64],
        dropout_rate: float = 0.3,
        name: str = "discriminator",
        **kwargs
    ):
        """
        Initialize Discriminator.
        
        Args:
            input_dim: Dimension of input features
            hidden_layers: List of hidden layer sizes
            dropout_rate: Dropout rate for regularization
            name: Model name
        """
        super(Discriminator, self).__init__(name=name, **kwargs)
        
        self.input_dim = input_dim
        self.hidden_layers_sizes = hidden_layers
        self.dropout_rate = dropout_rate
        
        # Build layers
        self.dense_layers = []
        self.dropout_layers = []
        
        for i, units in enumerate(hidden_layers):
            self.dense_layers.append(
                layers.Dense(
                    units,
                    kernel_initializer='he_normal',
                    name=f'dense_{i}'
                )
            )
            self.dropout_layers.append(
                layers.Dropout(dropout_rate, name=f'dropout_{i}')
            )
        
        # Output layer (binary classification)
        self.output_layer = layers.Dense(
            1,
            activation='sigmoid',
            kernel_initializer='glorot_normal',
            name='output'
        )
    
    def call(self, inputs, training=False):
        """
        Forward pass.
        
        Args:
            inputs: Data tensor of shape (batch_size, input_dim)
            training: Whether in training mode
            
        Returns:
            Probability of being real, shape (batch_size, 1)
        """
        x = inputs
        
        # Pass through hidden layers
        for dense, dropout in zip(self.dense_layers, self.dropout_layers):
            x = dense(x)
            x = tf.nn.leaky_relu(x, alpha=0.2)
            x = dropout(x, training=training)
        
        # Output layer
        output = self.output_layer(x)
        
        return output
    
    def predict_real_prob(self, samples: np.ndarray) -> np.ndarray:
        """
        Predict probability of samples being real.
        
        Args:
            samples: Input samples
            
        Returns:
            Probabilities of being real
        """
        probs = self(samples, training=False)
        return probs.numpy()


class WeightedDiscriminator(keras.Model):
    """
    Weighted Discriminator with class-conditional discrimination.
    
    This discriminator not only distinguishes real from fake samples,
    but also uses class labels to apply weighted loss, emphasizing
    harder-to-detect attack classes.
    """
    
    def __init__(
        self,
        input_dim: int = 78,
        num_classes: int = 2,
        hidden_layers: List[int] = [256, 256, 128, 64],
        dropout_rate: float = 0.3,
        embedding_dim: int = 50,
        use_class_weights: bool = True,
        name: str = "weighted_discriminator",
        **kwargs
    ):
        """
        Initialize Weighted Discriminator.
        
        Args:
            input_dim: Dimension of input features
            num_classes: Number of class labels
            hidden_layers: List of hidden layer sizes
            dropout_rate: Dropout rate
            embedding_dim: Dimension of label embedding
            use_class_weights: Whether to use class-based weighting
            name: Model name
        """
        super(WeightedDiscriminator, self).__init__(name=name, **kwargs)
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_layers_sizes = hidden_layers
        self.dropout_rate = dropout_rate
        self.embedding_dim = embedding_dim
        self.use_class_weights = use_class_weights
        
        # Label embedding layer
        self.label_embedding = layers.Embedding(
            num_classes,
            embedding_dim,
            name='label_embedding'
        )
        
        # Concatenate data and label embedding
        self.concat = layers.Concatenate(name='concat_data_label')
        
        # Build hidden layers
        self.dense_layers = []
        self.dropout_layers = []
        
        for i, units in enumerate(hidden_layers):
            self.dense_layers.append(
                layers.Dense(
                    units,
                    kernel_initializer='he_normal',
                    name=f'dense_{i}'
                )
            )
            self.dropout_layers.append(
                layers.Dropout(dropout_rate, name=f'dropout_{i}')
            )
        
        # Output layer (real/fake classification)
        self.output_layer = layers.Dense(
            1,
            activation='sigmoid',
            kernel_initializer='glorot_normal',
            name='output'
        )
        
        # Class weights (learnable parameters)
        if use_class_weights:
            self.class_weights = tf.Variable(
                tf.ones([num_classes], dtype=tf.float32),
                trainable=True,
                name='class_weights'
            )
    
    def call(self, inputs, training=False):
        """
        Forward pass with conditional input.
        
        Args:
            inputs: Tuple of (data, labels)
                - data: tensor of shape (batch_size, input_dim)
                - labels: tensor of shape (batch_size,) with class indices
            training: Whether in training mode
            
        Returns:
            Probability of being real, shape (batch_size, 1)
        """
        data, labels = inputs
        
        # Embed labels
        label_embedding = self.label_embedding(labels)
        # Flatten if necessary
        if len(label_embedding.shape) > 2:
            label_embedding = tf.squeeze(label_embedding, axis=1)
        elif len(label_embedding.shape) == 2 and label_embedding.shape[1] != self.embedding_dim:
            # If shape is (batch, 1, embedding_dim), squeeze middle dimension
            label_embedding = tf.reshape(label_embedding, [-1, self.embedding_dim])
        
        # Concatenate data and label embedding
        x = self.concat([data, label_embedding])
        
        # Pass through hidden layers
        for dense, dropout in zip(self.dense_layers, self.dropout_layers):
            x = dense(x)
            x = tf.nn.leaky_relu(x, alpha=0.2)
            x = dropout(x, training=training)
        
        # Output layer
        output = self.output_layer(x)
        
        return output
    
    def compute_weighted_loss(
        self,
        real_output: tf.Tensor,
        fake_output: tf.Tensor,
        real_labels: tf.Tensor,
        fake_labels: tf.Tensor,
        loss_fn: Optional[keras.losses.Loss] = None
    ) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """
        Compute weighted discriminator loss.
        
        Args:
            real_output: Discriminator output for real samples
            fake_output: Discriminator output for fake samples
            real_labels: Labels for real samples
            fake_labels: Labels for fake samples
            loss_fn: Loss function (default: BinaryCrossentropy)
            
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        if loss_fn is None:
            loss_fn = keras.losses.BinaryCrossentropy(from_logits=False)
        
        # Clip outputs to prevent log(0) and improve stability
        epsilon = 1e-7
        real_output_clipped = tf.clip_by_value(real_output, epsilon, 1.0 - epsilon)
        fake_output_clipped = tf.clip_by_value(fake_output, epsilon, 1.0 - epsilon)
        
        # Real loss (should predict 1)
        real_loss = loss_fn(tf.ones_like(real_output_clipped), real_output_clipped)
        
        # Fake loss (should predict 0)
        fake_loss = loss_fn(tf.zeros_like(fake_output_clipped), fake_output_clipped)
        
        if self.use_class_weights:
            # Apply class weights
            # Normalize weights to sum to 1
            weights = tf.nn.softmax(self.class_weights)
            
            # Get weights for each sample
            real_weights = tf.gather(weights, real_labels)
            fake_weights = tf.gather(weights, fake_labels)
            
            # Apply weights
            real_loss = tf.reduce_mean(real_loss * real_weights)
            fake_loss = tf.reduce_mean(fake_loss * fake_weights)
        
        total_loss = real_loss + fake_loss
        
        loss_dict = {
            'total_loss': total_loss,
            'real_loss': real_loss,
            'fake_loss': fake_loss
        }
        
        if self.use_class_weights:
            # Add class weights to loss dict for monitoring
            for i in range(self.num_classes):
                loss_dict[f'class_weight_{i}'] = weights[i]
        
        return total_loss, loss_dict
    
    def get_class_weights(self) -> np.ndarray:
        """Get current class weights."""
        if self.use_class_weights:
            weights = tf.nn.softmax(self.class_weights)
            return weights.numpy()
        else:
            return np.ones(self.num_classes) / self.num_classes


class AuxiliaryDiscriminator(keras.Model):
    """
    Auxiliary Discriminator with additional auxiliary classifier.
    
    This discriminator not only distinguishes real from fake,
    but also classifies the attack type (auxiliary task).
    """
    
    def __init__(
        self,
        input_dim: int = 78,
        num_classes: int = 2,
        hidden_layers: List[int] = [256, 256, 128],
        dropout_rate: float = 0.3,
        name: str = "auxiliary_discriminator",
        **kwargs
    ):
        """
        Initialize Auxiliary Discriminator.
        
        Args:
            input_dim: Dimension of input features
            num_classes: Number of attack classes
            hidden_layers: List of hidden layer sizes
            dropout_rate: Dropout rate
            name: Model name
        """
        super(AuxiliaryDiscriminator, self).__init__(name=name, **kwargs)
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_layers_sizes = hidden_layers
        self.dropout_rate = dropout_rate
        
        # Shared layers
        self.dense_layers = []
        self.dropout_layers = []
        
        for i, units in enumerate(hidden_layers):
            self.dense_layers.append(
                layers.Dense(
                    units,
                    kernel_initializer='he_normal',
                    name=f'shared_dense_{i}'
                )
            )
            self.dropout_layers.append(
                layers.Dropout(dropout_rate, name=f'dropout_{i}')
            )
        
        # Real/Fake output
        self.real_fake_output = layers.Dense(
            1,
            activation='sigmoid',
            kernel_initializer='glorot_normal',
            name='real_fake_output'
        )
        
        # Class output (auxiliary classifier)
        self.class_output = layers.Dense(
            num_classes,
            activation='softmax',
            kernel_initializer='glorot_normal',
            name='class_output'
        )
    
    def call(self, inputs, training=False):
        """
        Forward pass.
        
        Args:
            inputs: Data tensor of shape (batch_size, input_dim)
            training: Whether in training mode
            
        Returns:
            Tuple of (real_prob, class_probs)
        """
        x = inputs
        
        # Pass through shared layers
        for dense, dropout in zip(self.dense_layers, self.dropout_layers):
            x = dense(x)
            x = tf.nn.leaky_relu(x, alpha=0.2)
            x = dropout(x, training=training)
        
        # Real/Fake output
        real_output = self.real_fake_output(x)
        
        # Class output
        class_output = self.class_output(x)
        
        return real_output, class_output


def build_discriminator(
    input_dim: int = 78,
    conditional: bool = True,
    num_classes: int = 2,
    discriminator_type: str = "weighted",
    **kwargs
) -> keras.Model:
    """
    Factory function to build discriminator.
    
    Args:
        input_dim: Input feature dimension
        conditional: Whether to use conditional discriminator
        num_classes: Number of classes
        discriminator_type: Type of discriminator ("basic", "weighted", "auxiliary")
        **kwargs: Additional arguments
        
    Returns:
        Discriminator model
    """
    if not conditional:
        return Discriminator(input_dim=input_dim, **kwargs)
    
    if discriminator_type == "weighted":
        return WeightedDiscriminator(
            input_dim=input_dim,
            num_classes=num_classes,
            **kwargs
        )
    elif discriminator_type == "auxiliary":
        return AuxiliaryDiscriminator(
            input_dim=input_dim,
            num_classes=num_classes,
            **kwargs
        )
    else:
        return Discriminator(input_dim=input_dim, **kwargs)


if __name__ == "__main__":
    # Test discriminator
    print("Testing Discriminator...")
    
    disc = Discriminator(input_dim=78)
    samples = tf.random.normal([32, 78])
    output = disc(samples, training=False)
    print(f"Discriminator output shape: {output.shape}")
    
    print("\nTesting Weighted Discriminator...")
    weighted_disc = WeightedDiscriminator(input_dim=78, num_classes=2)
    samples = tf.random.normal([32, 78])
    labels = tf.constant([0] * 16 + [1] * 16, dtype=tf.int32)
    output = weighted_disc([samples, labels], training=False)
    print(f"Weighted Discriminator output shape: {output.shape}")
    print(f"Class weights: {weighted_disc.get_class_weights()}")
    
    print("\nTesting Auxiliary Discriminator...")
    aux_disc = AuxiliaryDiscriminator(input_dim=78, num_classes=2)
    samples = tf.random.normal([32, 78])
    real_output, class_output = aux_disc(samples, training=False)
    print(f"Real/Fake output shape: {real_output.shape}")
    print(f"Class output shape: {class_output.shape}")
    
    print("\n✓ Discriminator tests passed!")
