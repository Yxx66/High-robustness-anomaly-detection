"""
OWC-SAWN (Optimized Weighted Conditional Stepwise Adversarial Network) Module

This module implements the conditional adversarial network for IDS attack detection,
featuring:
- Conditional Generator with class labels
- Weighted Discriminator with optimized loss
- Stepwise training strategy
- Quality assessment metrics
"""

__version__ = "1.0.0"
__author__ = "Hybrid IDS Research Team"

from .generator import Generator, ConditionalGenerator
from .discriminator import Discriminator, WeightedDiscriminator, AuxiliaryDiscriminator
from .trainer import OWCSAWNTrainer
from .utils import (
    generate_adversarial_samples,
    evaluate_sample_quality,
    augment_data_with_gan,
)

__all__ = [
    'Generator',
    'ConditionalGenerator',
    'Discriminator',
    'WeightedDiscriminator',
    'AuxiliaryDiscriminator',
    'OWCSAWNTrainer',
    'generate_adversarial_samples',
    'evaluate_sample_quality',
    'augment_data_with_gan',
]
