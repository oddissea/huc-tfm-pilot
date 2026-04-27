# src/models/slide_aggregator.py

"""
Agregador con atención para predicción a nivel de slide (Attention MIL).

Toma los embeddings de todos los patches de un slide y produce una
predicción única para el slide completo. Los pesos de atención indican
qué patches fueron más relevantes para la predicción.

Referencia: Ilse et al. "Attention-based Deep Multiple Instance Learning" (ICML 2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class AttentionMIL(nn.Module):
    """
    Attention-based Multiple Instance Learning para agregación de slide.

    Dado un bag de N embeddings (patches de un slide), calcula pesos de
    atención por patch y produce un embedding agregado del slide mediante
    suma ponderada. Un clasificador final predice la clase del slide.

    Arquitectura:
        patches (N, D) → Attention → weighted sum → (1, D) → Classifier → (1, C)
    """

    def __init__(
        self,
        embedding_dim: int = 4096,
        hidden_dim: int = 256,
        num_classes: int = 2,
        dropout: float = 0.25,
        attention_heads: int = 1,
    ):
        """
        Args:
            embedding_dim: Dimensión de los embeddings de entrada (4096 para F4 concat)
            hidden_dim: Dimensión de la capa oculta de atención
            num_classes: Número de clases de salida
            dropout: Dropout en el clasificador
            attention_heads: Número de cabezas de atención (1 = estándar Ilse)
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.attention_heads = attention_heads

        # Gated Attention (Ilse et al. 2018)
        # V branch: tanh
        self.attention_V = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.Tanh(),
        )
        # U branch: sigmoid gate
        self.attention_U = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.Sigmoid(),
        )
        # Attention weights
        self.attention_W = nn.Linear(hidden_dim, attention_heads)

        # Clasificador sobre el embedding agregado
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim * attention_heads, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def attention(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calcula pesos de atención para cada patch.

        Args:
            x: (N, D) embeddings de los N patches del slide

        Returns:
            A: (1, N) pesos de atención normalizados
            x_agg: (1, D * heads) embedding agregado del slide
        """
        # Gated attention
        V = self.attention_V(x)  # (N, hidden_dim)
        U = self.attention_U(x)  # (N, hidden_dim)
        A = self.attention_W(V * U)  # (N, heads)

        # Softmax sobre patches (dim=0 porque N es la primera dimensión)
        A = A.transpose(0, 1)  # (heads, N)
        A = F.softmax(A, dim=1)  # (heads, N)

        # Suma ponderada
        x_agg = torch.mm(A, x)  # (heads, D)
        x_agg = x_agg.view(1, -1)  # (1, D * heads)

        return A, x_agg

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: embedding de patches → predicción de slide.

        Args:
            x: (N, D) embeddings de los N patches del slide

        Returns:
            logits: (1, C) logits de clasificación del slide
            A: (heads, N) pesos de atención por patch
        """
        A, x_agg = self.attention(x)  # A: (heads, N), x_agg: (1, D*heads)
        logits = self.classifier(x_agg)  # (1, C)
        return logits, A

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Obtener solo los pesos de atención (para visualización/heatmaps).

        Args:
            x: (N, D) embeddings de los N patches del slide

        Returns:
            weights: (N,) pesos de atención normalizados
        """
        with torch.no_grad():
            A, _ = self.attention(x)
        return A.squeeze(0)  # (N,)
