# src/models/__init__.py
"""
Módulo models: componentes del modelo two-stage.

Contiene:
- ProjectionHead: cabeza de proyección para Stage 1 (contrastive learning)
- MLPClassifier: clasificador MLP para Stage 2 (fine-tuning)
"""

from .classifier import MLPClassifier
from .projection_head import ProjectionHead
from .slide_aggregator import AttentionMIL

__all__ = [
    'MLPClassifier',
    'ProjectionHead',
    'AttentionMIL',
]