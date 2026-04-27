# src/encoders/base_encoder.py
"""
Base encoder abstract class for supervised contrastive learning using pretrained models.

This module provides the abstract base class for all encoder architectures
that wrap official pretrained models from PyTorch/timm libraries, following
Yengec-Tasdemir et al. (2024) methodology.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import torch
import torch.nn as nn
from torch import Tensor


class BaseEncoder(ABC, nn.Module):
    """
    Abstract base class for encoder architectures using pretrained models.

    This class wraps official pretrained models (ResNet, BiT-M) from PyTorch/timm
    libraries and adapts them for the two-stage supervised contrastive learning
    framework. All concrete implementations load official pretrained weights.
    """

    def __init__(
            self,
            embedding_dim: int,
            pretrained: bool = True,
            freeze_backbone: bool = False
    ) -> None:
        """
        Initialize the base encoder.

        Args:
            embedding_dim: Dimensionality of output embeddings
            pretrained: Whether to use pretrained weights
            freeze_backbone: Whether to freeze backbone parameters for stage 2
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.pretrained = pretrained
        self.freeze_backbone = freeze_backbone
        self._contrastive_mode = True

        self.backbone = self._load_pretrained_model()
        self.feature_extractor = self._build_feature_extractor()

        if freeze_backbone:
            self._freeze_backbone_parameters()

    @abstractmethod
    def _load_pretrained_model(self) -> nn.Module:
        """
        Load pretrained model from official source.

        Returns:
            Pretrained backbone model (ResNet, BiT-M, etc.)
        """
        pass

    @abstractmethod
    def _build_feature_extractor(self) -> nn.Module:
        """
        Build feature extraction head on top of backbone.

        Returns:
            Feature extractor module that outputs embeddings
        """
        pass

    def _freeze_backbone_parameters(self) -> None:
        """Freeze backbone parameters for stage 2 training."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def _unfreeze_backbone_parameters(self) -> None:
        """Unfreeze backbone parameters for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def set_contrastive_mode(self, mode: bool = True) -> None:
        """
        Set encoder mode for contrastive learning vs classification.

        Args:
            mode: True for contrastive mode (stage 1), False for classification mode (stage 2)
        """
        self._contrastive_mode = mode

        if not mode and not self.freeze_backbone:
            self._freeze_backbone_parameters()
        elif mode and self.freeze_backbone:
            self._unfreeze_backbone_parameters()

    def get_embedding_dim(self) -> int:
        """Return the dimensionality of output embeddings."""
        return self.embedding_dim

    def get_num_parameters(self) -> Dict[str, int]:
        """
        Get parameter count statistics.

        Returns:
            Dictionary with total, trainable, and frozen parameter counts
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params

        return {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': frozen_params
        }

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through encoder.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Feature embeddings of shape (batch_size, embedding_dim)
        """
        features = self.backbone(x)
        embeddings = self.feature_extractor(features)
        return embeddings

    def extract_features(self, x: Tensor) -> Tensor:
        """
        Extract features without gradient computation (for inference).

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Feature embeddings of shape (batch_size, embedding_dim)
        """
        with torch.no_grad():
            return self.forward(x)

    def get_backbone_features(self, x: Tensor) -> Tensor:
        """
        Get raw backbone features before feature extractor.

        Args:
            x: Input tensor

        Returns:
            Raw backbone features
        """
        with torch.no_grad():
            return self.backbone(x)

    def compute_embedding_statistics(self, x: Tensor) -> Dict[str, float]:
        """
        Compute statistics of output embeddings.

        Args:
            x: Input tensor

        Returns:
            Dictionary with embedding statistics
        """
        embeddings = self.extract_features(x)

        return {
            'mean_norm': torch.norm(embeddings, dim=1).mean().item(),
            'std_norm': torch.norm(embeddings, dim=1).std().item(),
            'mean_activation': embeddings.mean().item(),
            'std_activation': embeddings.std().item(),
            'min_activation': embeddings.min().item(),
            'max_activation': embeddings.max().item()
        }

    @property
    def is_contrastive_mode(self) -> bool:
        """Check if encoder is in contrastive mode."""
        return self._contrastive_mode

    @property
    def is_frozen(self) -> bool:
        """Check if backbone parameters are frozen."""
        return not any(p.requires_grad for p in self.backbone.parameters())

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model-specific information.

        Returns:
            Dictionary with model information
        """
        pass

    def __repr__(self) -> str:
        """String representation of the encoder."""
        param_info = self.get_num_parameters()
        mode = "contrastive" if self._contrastive_mode else "classification"
        frozen_status = "frozen" if self.is_frozen else "trainable"

        return (
            f"{self.__class__.__name__}(\n"
            f"  embedding_dim={self.embedding_dim},\n"
            f"  mode={mode},\n"
            f"  backbone={frozen_status},\n"
            f"  parameters={param_info['total']:,} "
            f"({param_info['trainable']:,} trainable, {param_info['frozen']:,} frozen)\n"
            f")"
        )