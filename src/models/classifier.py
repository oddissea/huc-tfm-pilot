# src/models/classifier.py
"""
MLP Classifier para Classification Fine-tuning (Stage 2).

Implementa el clasificador que se entrena sobre embeddings del encoder
congelado, siguiendo Yengec-Tasdemir et al. (2024).

Arquitectura:
    Input: embeddings del encoder congelado (512-D o 2048-D)
    Hidden: capa densa con dropout
    Output: logits de clasificación (NUM_CLASSES)
    
Referencias:
    Yengec-Tasdemir et al. (2024): Two-stage framework con clasificador MLP
"""

import torch
import torch.nn as nn


class MLPClassifier(nn.Module):
    """
    Clasificador MLP para Stage 2 del framework two-stage.
    
    Arquitectura simple con una capa oculta y dropout para regularización.
    Se entrena mientras el encoder permanece congelado.
    
    Atributos:
        hidden: Capa lineal oculta
        dropout: Dropout para regularización
        output: Capa de salida (logits)
        embedding_dim: Dimensión de entrada (del encoder)
        hidden_units: Unidades en capa oculta
        num_classes: Número de clases de salida
        dropout_rate: Tasa de dropout
    """
    
    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        hidden_units: int = 512,
        dropout_rate: float = 0.1
    ):
        """
        Inicializa clasificador MLP.
        
        Args:
            embedding_dim: Dimensión de embeddings de entrada (512 o 2048)
            num_classes: Número de clases de salida (2 para binario)
            hidden_units: Unidades en capa oculta (512 por defecto)
            dropout_rate: Tasa de dropout (0.1 por defecto)
            
        Examples:
            >>> # Clasificador binario con ResNet-18
            >>> classifier = MLPClassifier(
            ...     embedding_dim=512,
            ...     num_classes=2,
            ...     hidden_units=512,
            ...     dropout_rate=0.1
            ... )
            
            >>> # Clasificador multiclase con ResNet-50
            >>> classifier = MLPClassifier(
            ...     embedding_dim=2048,
            ...     num_classes=8,
            ...     hidden_units=512,
            ...     dropout_rate=0.1
            ... )
        """
        super().__init__()
        
        if dropout_rate < 0 or dropout_rate >= 1:
            raise ValueError(f"dropout_rate debe estar en [0, 1), recibido: {dropout_rate}")
        
        if num_classes < 2:
            raise ValueError(f"num_classes debe ser >= 2, recibido: {num_classes}")
        
        self.embedding_dim = embedding_dim
        self.hidden_units = hidden_units
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        self.hidden = nn.Linear(embedding_dim, hidden_units)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.output = nn.Linear(hidden_units, num_classes)
        
        self._init_weights()
    
    def _init_weights(self):
        """Inicializa pesos usando Xavier uniform."""
        nn.init.xavier_uniform_(self.hidden.weight)
        nn.init.zeros_(self.hidden.bias)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Clasificación sobre embeddings del encoder.
        
        Args:
            embeddings: Tensor de embeddings [batch_size, embedding_dim]
        
        Returns:
            Logits de clasificación [batch_size, num_classes]
            
        Raises:
            ValueError: Si las dimensiones de entrada no coinciden
        """
        if embeddings.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"Dimensión de entrada incorrecta: esperado {self.embedding_dim}, "
                f"recibido {embeddings.shape[-1]}"
            )
        
        x = self.hidden(embeddings)
        x = self.activation(x)
        x = self.dropout(x)
        logits = self.output(x)
        
        return logits
    
    def __repr__(self) -> str:
        return (
            f"MLPClassifier(embedding_dim={self.embedding_dim}, "
            f"num_classes={self.num_classes}, "
            f"hidden_units={self.hidden_units}, "
            f"dropout_rate={self.dropout_rate})"
        )
