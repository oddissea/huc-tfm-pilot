"""Instanciación y carga de los modelos F4 y AttnMIL en modo inferencia.

Los hiperparámetros están extraídos directamente de los checkpoints:

  F4 (config dict guardado en final_inference_model.pth):
      model_type=bitm, patch_mode=dual, fusion_mode=concat,
      num_classes=3, embedding_dim=4096, hidden_units=512,
      dropout=0.1, freeze_encoder=True

  AttnMIL ternary 512-d (forma de los tensores en seed_*/fold_*.pth):
      embedding_dim=512, hidden_dim=256, num_classes=3,
      dropout=0.25, attention_heads=1
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn

from src.encoders.bit_encoder import BiTEncoder
from src.encoders.dual_stream_encoder import DualStreamEncoder
from src.models.classifier import MLPClassifier
from src.models.slide_aggregator import AttentionMIL

logger = logging.getLogger(__name__)

CLASS_NAMES = ("ADE", "NOR", "CAR")  # orden alfabético del entrenamiento


@dataclass
class F4Bundle:
    """Modelo F4 cargado y listo para inferencia (encoder + classifier)."""
    encoder: nn.Module
    classifier: nn.Module
    device: torch.device


@dataclass
class AttnMILBundle:
    """Un modelo AttnMIL del ensemble ternario, identificado por (seed, fold)."""
    seed: int
    fold: int
    model: nn.Module


def _build_f4_architecture() -> tuple[nn.Module, nn.Module]:
    """Crea la arquitectura F4 vacía (sin pesos), lista para load_state_dict."""
    base = BiTEncoder(embedding_dim=2048, pretrained=False, freeze_backbone=True)
    encoder = DualStreamEncoder(
        base_encoder=base,
        fusion_mode="concat",
        freeze_backbone=True,
    )
    classifier = MLPClassifier(
        embedding_dim=4096,
        num_classes=3,
        hidden_units=512,
        dropout_rate=0.1,
    )
    return encoder, classifier


def load_f4(checkpoint_path: Path, device: torch.device) -> F4Bundle:
    """Carga el modelo F4 desde el checkpoint y lo deja en eval()."""
    logger.info("Loading F4 from %s", checkpoint_path)
    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)

    encoder, classifier = _build_f4_architecture()
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    classifier.load_state_dict(checkpoint["classifier_state_dict"])

    encoder = encoder.to(device).eval()
    classifier = classifier.to(device).eval()

    # Para liberar memoria innecesaria: el encoder ya no necesita gradientes.
    for p in encoder.parameters():
        p.requires_grad_(False)
    for p in classifier.parameters():
        p.requires_grad_(False)

    return F4Bundle(encoder=encoder, classifier=classifier, device=device)


def load_attnmil_ensemble(
    checkpoints: Iterable[tuple[int, int, Path]],
    device: torch.device,
) -> list[AttnMILBundle]:
    """Carga los 25 (o los que sean) modelos AttnMIL del ensemble ternario."""
    bundles: list[AttnMILBundle] = []
    for seed, fold, path in checkpoints:
        model = AttentionMIL(
            embedding_dim=512,
            hidden_dim=256,
            num_classes=3,
            dropout=0.25,
            attention_heads=1,
        )
        state_dict = torch.load(str(path), map_location=device, weights_only=False)
        model.load_state_dict(state_dict)
        model = model.to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        bundles.append(AttnMILBundle(seed=seed, fold=fold, model=model))
        logger.debug("loaded AttnMIL seed=%d fold=%d", seed, fold)
    logger.info("loaded %d AttnMIL models", len(bundles))
    return bundles
