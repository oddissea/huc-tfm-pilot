# src/encoders/bit_encoder.py
"""
BiT (Big Transfer) encoder implementation using official pretrained models.

This module implements the BiT-M R50x1 encoder by loading the official
pretrained model from timm library, following Yengec-Tasdemir et al. (2024)
specifications where BiT-M achieves 86.2% accuracy.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, Any

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    timm = None
    TIMM_AVAILABLE = False

from .base_encoder import BaseEncoder


class BiTEncoder(BaseEncoder):
    """
    BiT-M R50x1 encoder using official pretrained weights.

    Loads the official BiT-M (Big Transfer Medium) model pretrained on ImageNet-21k
    as specified in Yengec-Tasdemir et al. (2024). This model achieves 86.2% accuracy
    in the paper, significantly outperforming ResNet-50 (75.7% accuracy).

    The BiT-M model features:
    - ResNet-50 architecture with BiT improvements
    - Group Normalization instead of Batch Normalization
    - Weight Standardization for better domain adaptation
    - Pretrained on ImageNet-21k (larger dataset)
    - 2048-dimensional output embeddings
    """

    def __init__(
        self,
        embedding_dim: int = 2048,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        model_name: str = 'resnetv2_50x1_bit.goog_in21k'
    ):
        """
        Initialize BiT-M encoder.

        Args:
            embedding_dim: Output embedding dimension (should be 2048 for BiT-M)
            pretrained: Whether to load ImageNet-21k pretrained weights
            freeze_backbone: Whether to freeze backbone for stage 2
            model_name: timm model name for BiT-M
        """
        if not TIMM_AVAILABLE:
            raise ImportError(
                "timm library is required for BiT models. Install with: pip install timm"
            )

        self.model_name = model_name

        if embedding_dim != 2048:
            print(f"Warning: BiT-M typically outputs 2048-dim features, got {embedding_dim}")

        super().__init__(embedding_dim, pretrained, freeze_backbone)

    def _load_pretrained_model(self) -> nn.Module:
        """Load official BiT-M pretrained model from timm."""
        try:
            model = timm.create_model(
                self.model_name,
                pretrained=self.pretrained,
                num_classes=0,
                global_pool=''
            )

            if hasattr(model, 'num_features'):
                actual_dim = model.num_features
                if actual_dim != self.embedding_dim:
                    self.embedding_dim = actual_dim

            return model

        except Exception as e:
            raise RuntimeError(f"Failed to load BiT model '{self.model_name}': {e}")

    def _build_feature_extractor(self) -> nn.Module:
        """Build feature extraction head for BiT encoder."""
        return nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.1)
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through BiT encoder.

        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)

        Returns:
            Feature embeddings of shape (batch_size, 2048)
        """
        features = self.backbone(x)
        embeddings = self.feature_extractor(features)
        return embeddings

    def get_bit_info(self) -> Dict[str, Any]:
        """Get BiT-specific model information."""
        return {
            'model_name': self.model_name,
            'pretraining_dataset': 'ImageNet-21k',
            'architecture_base': 'ResNet-50',
            'bit_improvements': [
                'Group Normalization',
                'Weight Standardization',
                'Large-scale pretraining'
            ],
            'paper_accuracy': 0.862,
            'domain_adaptation': 'Excellent',
            'batch_size_robustness': 'High'
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        param_info = self.get_num_parameters()
        bit_info = self.get_bit_info()

        return {
            'encoder_type': 'BiT-M R50x1',
            'embedding_dim': self.embedding_dim,
            'parameters': param_info,
            'pretrained': self.pretrained,
            'input_resolution': (224, 224),
            'normalization': 'Group Normalization',
            'paper_reference': 'Kolesnikov et al. (2020) + Yengec-Tasdemir et al. (2024)',
            **bit_info
        }

    def get_layer_features(self, x: Tensor, layer_name: str) -> Tensor:
        """
        Extract features from specific BiT layer.

        Args:
            x: Input tensor
            layer_name: Layer name (e.g., 'stages.0', 'stages.1', etc.)

        Returns:
            Features from specified layer
        """
        features = {}

        def hook_function(_module_ref, _input_tensor, output_tensor):
            features['output'] = output_tensor

        target_module = None
        for name, module in self.backbone.named_modules():
            if name == layer_name:
                target_module = module
                break

        if target_module is None:
            available_layers = [name for name, _ in self.backbone.named_modules()][:10]
            raise ValueError(f"Layer '{layer_name}' not found. Available layers: {available_layers}...")

        handle = target_module.register_forward_hook(hook_function)

        with torch.no_grad():
            self.forward(x)

        handle.remove()

        return features['output']

    @staticmethod
    def compare_with_resnet() -> Dict[str, Any]:
        """Compare BiT-M performance with ResNet-50 baseline."""
        return {
            'bit_m_accuracy': 0.862,
            'resnet50_accuracy': 0.757,
            'improvement': 0.105,
            'relative_improvement': '13.9%',
            'bit_advantages': [
                'Better domain adaptation',
                'Robust to small batch sizes',
                'Group Norm vs Batch Norm',
                'ImageNet-21k pretraining'
            ]
        }


def load_bit_encoder(
    pretrained: bool = True,
    model_variant: str = 'bitm_r50x1'
) -> BiTEncoder:
    """
    Factory function to load BiT encoder variants.

    Args:
        pretrained: Whether to use pretrained weights
        model_variant: BiT model variant

    Returns:
        Configured BiT encoder
    """
    model_mapping = {
        'bitm_r50x1': 'resnetv2_50x1_bit.goog_in21k',
        'bitm_r101x1': 'resnetv2_101x1_bit.goog_in21k',
        'bitm_r50x3': 'resnetv2_50x3_bit.goog_in21k'
    }

    if model_variant not in model_mapping:
        raise ValueError(f"Unknown BiT variant: {model_variant}. Available: {list(model_mapping.keys())}")

    timm_name = model_mapping[model_variant]
    embedding_dim = 2048

    return BiTEncoder(
        embedding_dim=embedding_dim,
        pretrained=pretrained,
        model_name=timm_name
    )