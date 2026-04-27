# src/encoders/resnet_encoder.py
"""
ResNet encoder implementations using official PyTorch pretrained models.

This module implements ResNet-18 and ResNet-50 encoders as baseline models
using official torchvision pretrained weights. ResNet-50 achieves 75.7% accuracy
in Yengec-Tasdemir et al. (2024) compared to BiT-M's 86.2%.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, Any

try:
    import torchvision.models as models
    TORCHVISION_AVAILABLE = True
except ImportError:
    models = None
    TORCHVISION_AVAILABLE = False

from .base_encoder import BaseEncoder


class ResNet18Encoder(BaseEncoder):
    """
    ResNet-18 encoder using official PyTorch pretrained weights.

    Lightweight baseline model for comparison with BiT and ResNet-50.
    Uses standard ImageNet pretrained weights from torchvision.
    Expected performance: 75-83% accuracy range.
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        pretrained: bool = True,
        freeze_backbone: bool = False
    ):
        """
        Initialize ResNet-18 encoder.

        Args:
            embedding_dim: Output embedding dimension (512 for ResNet-18)
            pretrained: Whether to load ImageNet pretrained weights
            freeze_backbone: Whether to freeze backbone for stage 2
        """
        if not TORCHVISION_AVAILABLE:
            raise ImportError(
                "torchvision is required for ResNet models. Install with: pip install torchvision"
            )

        if embedding_dim != 512:
            print(f"Warning: ResNet-18 outputs 512-dim features, got {embedding_dim}")

        super().__init__(embedding_dim, pretrained, freeze_backbone)

    def _load_pretrained_model(self) -> nn.Module:
        """Load official ResNet-18 pretrained model from torchvision."""
        try:
            if hasattr(models, 'ResNet18_Weights') and self.pretrained:
                weights = models.ResNet18_Weights.IMAGENET1K_V1
                model = models.resnet18(weights=weights)
            else:
                model = models.resnet18(pretrained=self.pretrained)

            model.fc = nn.Identity()

            self.embedding_dim = 512

            return model

        except Exception as e:
            raise RuntimeError(f"Failed to load ResNet-18 model: {e}")

    def _build_feature_extractor(self) -> nn.Module:
        """Build feature extraction head for ResNet-18."""
        return nn.Sequential(
            nn.Dropout(0.2)
        )

    def get_model_info(self) -> Dict[str, Any]:
        """Get ResNet-18 model information."""
        param_info = self.get_num_parameters()

        return {
            'encoder_type': 'ResNet-18',
            'embedding_dim': self.embedding_dim,
            'parameters': param_info,
            'pretrained_dataset': 'ImageNet-1k',
            'input_resolution': (224, 224),
            'normalization': 'Batch Normalization',
            'expected_accuracy_range': (0.75, 0.83),
            'paper_reference': 'He et al. (2016) + Yengec-Tasdemir et al. (2024)',
            'architecture_details': {
                'blocks': 'BasicBlock',
                'layers': [2, 2, 2, 2],
                'total_layers': 18
            }
        }


class ResNet50Encoder(BaseEncoder):
    """
    ResNet-50 encoder using official PyTorch pretrained weights.

    Traditional baseline model as specified in Yengec-Tasdemir et al. (2024).
    Achieves exactly 75.7% accuracy in the paper compared to BiT-M's 86.2%.
    Uses standard ImageNet pretrained weights from torchvision.
    """

    def __init__(
        self,
        embedding_dim: int = 2048,
        pretrained: bool = True,
        freeze_backbone: bool = False
    ):
        """
        Initialize ResNet-50 encoder.

        Args:
            embedding_dim: Output embedding dimension (2048 for ResNet-50)
            pretrained: Whether to load ImageNet pretrained weights
            freeze_backbone: Whether to freeze backbone for stage 2
        """
        if not TORCHVISION_AVAILABLE:
            raise ImportError(
                "torchvision is required for ResNet models. Install with: pip install torchvision"
            )

        if embedding_dim != 2048:
            print(f"Warning: ResNet-50 outputs 2048-dim features, got {embedding_dim}")

        super().__init__(embedding_dim, pretrained, freeze_backbone)

    def _load_pretrained_model(self) -> nn.Module:
        """Load official ResNet-50 pretrained model from torchvision."""
        try:
            if hasattr(models, 'ResNet50_Weights') and self.pretrained:
                weights = models.ResNet50_Weights.IMAGENET1K_V2
                model = models.resnet50(weights=weights)
            else:
                model = models.resnet50(pretrained=self.pretrained)

            model.fc = nn.Identity()

            self.embedding_dim = 2048

            return model

        except Exception as e:
            raise RuntimeError(f"Failed to load ResNet-50 model: {e}")

    def _build_feature_extractor(self) -> nn.Module:
        """Build feature extraction head for ResNet-50."""
        return nn.Sequential(
            nn.Dropout(0.1)
        )

    def get_layer_features(self, x: Tensor, layer_name: str) -> Tensor:
        """
        Extract features from specific ResNet layer.

        Args:
            x: Input tensor
            layer_name: Layer name ('layer1', 'layer2', 'layer3', 'layer4', 'avg_pool')

        Returns:
            Features from specified layer
        """
        features = {}

        def hook_function(_module_ref, _input_tensor, output_tensor):
            features['output'] = output_tensor

        layer_mapping = {
            'layer1': self.backbone.layer1,
            'layer2': self.backbone.layer2,
            'layer3': self.backbone.layer3,
            'layer4': self.backbone.layer4,
            'avg_pool': self.backbone.avg_pool
        }

        if layer_name not in layer_mapping:
            raise ValueError(f"Layer '{layer_name}' not found. Available: {list(layer_mapping.keys())}")

        target_layer = layer_mapping[layer_name]
        handle = target_layer.register_forward_hook(hook_function)

        with torch.no_grad():
            self.forward(x)

        handle.remove()

        return features['output']

    def get_model_info(self) -> Dict[str, Any]:
        """Get ResNet-50 model information."""
        param_info = self.get_num_parameters()

        return {
            'encoder_type': 'ResNet-50',
            'embedding_dim': self.embedding_dim,
            'parameters': param_info,
            'pretrained_dataset': 'ImageNet-1k',
            'input_resolution': (224, 224),
            'normalization': 'Batch Normalization',
            'paper_accuracy': 0.757,
            'expected_accuracy_range': (0.75, 0.80),
            'paper_reference': 'He et al. (2016) + Yengec-Tasdemir et al. (2024)',
            'architecture_details': {
                'blocks': 'Bottleneck',
                'layers': [3, 4, 6, 3],
                'total_layers': 50
            }
        }

    @staticmethod
    def compare_with_bit() -> Dict[str, Any]:
        """Compare ResNet-50 performance with BiT-M."""
        return {
            'resnet50_accuracy': 0.757,
            'bit_m_accuracy': 0.862,
            'performance_gap': -0.105,
            'bit_advantages': [
                'Group Normalization vs Batch Normalization',
                'ImageNet-21k vs ImageNet-1k pretraining',
                'Weight Standardization',
                'Better domain adaptation capabilities'
            ],
            'resnet50_advantages': [
                'Simpler architecture',
                'Widely available',
                'Faster inference',
                'Less memory usage'
            ]
        }


class ResNetEncoderFactory:
    """Factory for creating ResNet encoders using official pretrained models."""

    @staticmethod
    def create_resnet_encoder(
        architecture: str,
        pretrained: bool = True,
        **kwargs
    ) -> BaseEncoder:
        """
        Create ResNet encoder by architecture name.

        Args:
            architecture: 'resnet18' or 'resnet50'
            pretrained: Whether to use ImageNet pretrained weights
            **kwargs: Additional configuration parameters

        Returns:
            Configured ResNet encoder instance
        """
        architecture = architecture.lower().replace('_', '').replace('-', '')

        if architecture == 'resnet18':
            return ResNet18Encoder(pretrained=pretrained, **kwargs)
        elif architecture in ['resnet50', 'resnet50v2']:
            return ResNet50Encoder(pretrained=pretrained, **kwargs)
        else:
            raise ValueError(f"Unknown ResNet architecture: {architecture}. Supported: resnet18, resnet50")

    @staticmethod
    def get_recommended_config(architecture: str) -> Dict[str, Any]:
        """Get recommended configuration for ResNet architecture."""
        configs = {
            'resnet18': {
                'embedding_dim': 512,
                'dropout_rate': 0.2,
                'expected_performance': (0.75, 0.83),
                'batch_size': 32,
                'learning_rate': 0.01
            },
            'resnet50': {
                'embedding_dim': 2048,
                'dropout_rate': 0.1,
                'expected_performance': (0.75, 0.80),
                'paper_accuracy': 0.757,
                'batch_size': 16,
                'learning_rate': 0.001
            }
        }

        architecture = architecture.lower().replace('_', '').replace('-', '')
        return configs.get(architecture, configs['resnet50'])


def load_resnet_encoder(
    architecture: str,
    pretrained: bool = True
) -> BaseEncoder:
    """
    Factory function to load ResNet encoders.

    Args:
        architecture: 'resnet18' or 'resnet50'
        pretrained: Whether to use ImageNet pretrained weights

    Returns:
        Configured ResNet encoder
    """
    return ResNetEncoderFactory.create_resnet_encoder(
        architecture=architecture,
        pretrained=pretrained
    )