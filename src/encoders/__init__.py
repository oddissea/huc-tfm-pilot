# src/encoders/__init__.py
"""
Paper-compliant encoders for Supervised Contrastive Learning.

Implementations following Yengec-Tasdemir et al. (2024):
- BiT-M R50x1: 86.2% accuracy (paper)
- ResNet-50: 75.7% accuracy (paper baseline)
- ResNet-18: lightweight baseline for binary classification

All encoders use official pretrained weights from timm/torchvision.
"""

from .base_encoder import BaseEncoder
from .bit_encoder import BiTEncoder, load_bit_encoder
from .resnet_encoder import (
    ResNet18Encoder,
    ResNet50Encoder,
    ResNetEncoderFactory,
    load_resnet_encoder
)
from .dual_stream_encoder import DualStreamEncoder

__all__ = [
    'BaseEncoder',
    'BiTEncoder',
    'load_bit_encoder',
    'ResNet18Encoder',
    'ResNet50Encoder',
    'ResNetEncoderFactory',
    'load_resnet_encoder',
    'DualStreamEncoder',
]

__version__ = '1.0.0'