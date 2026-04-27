# src/encoders/dual_stream_encoder.py

import torch
import torch.nn as nn
# import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Literal

from .base_encoder import BaseEncoder


class DualStreamEncoder(nn.Module):
    """
    Wrapper para procesamiento dual-stream con pesos compartidos.

    Procesa simultáneamente imagen original y rebinned usando el mismo
    encoder, luego fusiona las características resultantes.
    """

    def __init__(
            self,
            base_encoder: BaseEncoder,
            fusion_mode: Literal['concat', 'add', 'attention'] = 'concat',
            freeze_backbone: bool = False
    ):
        """
        Args:
            base_encoder: Encoder preentrenado (ResNet/BiT)
            fusion_mode: Estrategia de fusión de features
            freeze_backbone: Congelar pesos del encoder base
        """
        super().__init__()

        self.encoder = base_encoder
        self.fusion_mode = fusion_mode
        self.base_embedding_dim = base_encoder.embedding_dim

        if fusion_mode == 'concat':
            self.output_dim = 2 * self.base_embedding_dim
        elif fusion_mode == 'add':
            self.output_dim = self.base_embedding_dim
        elif fusion_mode == 'attention':
            self.output_dim = self.base_embedding_dim
            self._build_attention_module()
        else:
            raise ValueError(f"fusion_mode inválido: {fusion_mode}")

        if freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def _build_attention_module(self):
        """Construir módulo de atención simple para fusión."""
        self.attention = nn.Sequential(
            nn.Linear(self.base_embedding_dim * 2, self.base_embedding_dim),
            nn.Tanh(),
            nn.Linear(self.base_embedding_dim, 2),
            nn.Softmax(dim=1)
        )

    def forward(
            self,
            x: Tuple[Tensor, Tensor]
    ) -> Tensor:
        """
        Forward pass dual-stream.

        Args:
            x: Fila (image_orig, image_rebin)
               Cada una de forma (batch_size, 3, 224, 224)

        Returns:
            Features fusionados de forma (batch_size, output_dim)
        """
        img_orig, img_rebin = x

        # Procesamiento paralelo con pesos compartidos
        feat_orig = self.encoder(img_orig)  # (B, embedding_dim)
        feat_rebin = self.encoder(img_rebin)  # (B, embedding_dim)

        # Fusión según estrategia configurada
        # IMPORTANTE: NO normalizar aquí - ProjectionHead ya lo hace
        if self.fusion_mode == 'concat':
            fused = torch.cat([feat_orig, feat_rebin], dim=1)
        elif self.fusion_mode == 'add':
            fused = feat_orig + feat_rebin
        elif self.fusion_mode == 'attention':
            # Atención ponderada
            stacked = torch.stack([feat_orig, feat_rebin], dim=1)  # (B, 2, emb_dim)
            combined = torch.cat([feat_orig, feat_rebin], dim=1)
            weights = self.attention(combined)  # (B, 2)
            weights = weights.unsqueeze(2)  # (B, 2, 1)
            fused = (stacked * weights).sum(dim=1)  # (B, emb_dim)
        else:
            raise RuntimeError(
                f"fusion_mode inválido: {self.fusion_mode}. "
                f"Este error no debería ocurrir si __init__ validó correctamente."
            )

        return fused

    def set_contrastive_mode(self, mode: bool = True):
        """Propagar modo contrastivo al encoder base."""
        self.encoder.set_contrastive_mode(mode)

    def get_num_parameters(self):
        """Obtener conteo de parámetros."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'total': total,
            'trainable': trainable,
            'frozen': total - trainable
        }