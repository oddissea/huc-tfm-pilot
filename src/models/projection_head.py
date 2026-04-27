# src/models/projection_head.py
"""
Projection Head para Supervised Contrastive Learning (Stage 1).

Implementa la cabeza de proyección que mapea embeddings del encoder
a un espacio de menor dimensión para contrastive learning, siguiendo
Yengec-Tasdemir et al. (2024).

Arquitectura:
    Input: embeddings del encoder (512-D o 2048-D)
    Output: proyección normalizada (128-D por defecto)
    
Referencias:
    Yengec-Tasdemir et al. (2024): Two-stage framework con projection head de 128-D
    Khosla et al. (2020): Supervised Contrastive Learning paper original
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """
    Projection head para mapear embeddings a espacio contrastivo.
    
    Arquitectura simple: Linear layer seguida de normalización L2.
    No usa activaciones no lineales para mantener la estructura del espacio.
    
    Atributos:
        projection: Capa lineal de proyección
        embedding_dim: Dimensión de entrada (del encoder)
        projection_dim: Dimensión de salida (espacio contrastivo)
    """
    
    def __init__(
        self,
        embedding_dim: int,
        projection_dim: int = 128
    ):
        """
        Inicializa projection head.
        
        Args:
            embedding_dim: Dimensión de embeddings de entrada (512 o 2048)
            projection_dim: Dimensión de proyección de salida (128 por defecto)
            
        Examples:
            >>> # Para ResNet-18
            >>> proj_head = ProjectionHead(embedding_dim=512, projection_dim=128)
            
            >>> # Para ResNet-50 o BiT
            >>> proj_head = ProjectionHead(embedding_dim=2048, projection_dim=128)
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.projection_dim = projection_dim
        
        self.projection = nn.Linear(embedding_dim, projection_dim, bias=False)
        
        self._init_weights()
    
    def _init_weights(self):
        """Inicializa pesos usando Xavier uniform."""
        nn.init.xavier_uniform_(self.projection.weight)
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Proyecta embeddings a espacio contrastivo y normaliza.
        
        Args:
            embeddings: Tensor de embeddings [batch_size, embedding_dim]
        
        Returns:
            Tensor proyectado y normalizado L2 [batch_size, projection_dim]
            
        Raises:
            ValueError: Si las dimensiones de entrada no coinciden
        """
        if embeddings.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"Dimensión de entrada incorrecta: esperado {self.embedding_dim}, "
                f"recibido {embeddings.shape[-1]}"
            )
        
        projections = self.projection(embeddings)
        
        projections = F.normalize(projections, p=2, dim=1)
        
        return projections
    
    def __repr__(self) -> str:
        return (
            f"ProjectionHead(embedding_dim={self.embedding_dim}, "
            f"projection_dim={self.projection_dim})"
        )
