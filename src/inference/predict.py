"""Funciones de inferencia.

Uso típico:

    bundle = load_f4(...)
    attnmil_ensemble = load_attnmil_ensemble(...)
    # patches: arrays uint8 (N, H, W, 3) — la conversión + resize se hace
    # por batch en GPU para no agotar RAM en slides grandes (ca_1534 = 3.177).
    out = predict_slide(bundle, attnmil_ensemble, patches_orig, patches_reb,
                        mode="ensemble")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F

from src.inference.model import AttnMILBundle, F4Bundle, CLASS_NAMES

TARGET_HW = 224

logger = logging.getLogger(__name__)


InferenceMode = Literal["single", "ensemble"]


@dataclass
class SlideResult:
    """Resultado de una predicción a nivel de portaobjetos."""
    probabilities_mean: torch.Tensor   # (3,) probabilidades softmax promediadas
    probabilities_std: torch.Tensor    # (3,) std entre miembros del ensemble (cero si single)
    predicted_class: str               # "ADE" | "NOR" | "CAR"
    predicted_index: int               # 0, 1, 2 según orden CLASS_NAMES
    n_patches: int                     # número de parches del portaobjetos
    n_models_used: int                 # 1, 5 o 25
    attention_weights_mean: torch.Tensor | None  # (N,) atención promedio si se pidió
    patch_probs: torch.Tensor | None = None       # (N, 3) softmax del classifier F4 por parche
    patch_predictions: torch.Tensor | None = None # (N,) argmax de patch_probs (idx 0..2)
    features: torch.Tensor | None = None          # (N, 512) embeddings post-ReLU del head F4
                                                  # (input del AttnMIL). Persistidos por el worker
                                                  # para abaratar futuros fine-tunes sin reforwardear F4.


def _to_model_tensor(
    chunk_uint8: np.ndarray,
    device: torch.device,
    target_hw: int = TARGET_HW,
) -> torch.Tensor:
    """(B, H, W, 3) uint8 → (B, 3, target_hw, target_hw) float32 [0,1] en device.

    La conversión + resize se hace por batch en lugar de pre-procesar el slide
    entero: un slide de 3.177 parches en float32 a 300×300 picaba ~12 GB de RAM
    (OOM en `g2-standard-4`). Con conversión por batch el pico se queda bajo.
    """
    if chunk_uint8.dtype != np.uint8:
        raise ValueError(f"Esperaba uint8, recibí {chunk_uint8.dtype}")
    if chunk_uint8.ndim != 4 or chunk_uint8.shape[-1] != 3:
        raise ValueError(f"Shape inesperado: {chunk_uint8.shape}, esperaba (B,H,W,3)")

    t = torch.from_numpy(chunk_uint8).permute(0, 3, 1, 2).float() / 255.0
    h, w = t.shape[-2:]
    if (h, w) != (target_hw, target_hw):
        t = F.interpolate(
            t,
            size=(target_hw, target_hw),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
    return t.contiguous().to(device, non_blocking=True)


def _f4_forward_to_features(
    f4: F4Bundle,
    patches_orig: np.ndarray,
    patches_reb: np.ndarray,
    batch_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pasa los parches por F4 y devuelve features 512-d + logits ternarios por parche.

    Args:
        f4: bundle del F4 cargado
        patches_orig: np.ndarray (N, H, W, 3) uint8 — parches originales sin convertir
        patches_reb:  np.ndarray (N, H, W, 3) uint8 — parches rebinneados sin convertir
        batch_size: tamaño de batch para inferencia (también limita el pico de RAM)

    Returns:
        - features 512-d (N, 512) post-ReLU del classifier (input del AttnMIL)
        - logits ternarios (N, 3) del classifier F4 (para predicción patch-level)
    """
    device = f4.device
    n = patches_orig.shape[0]
    feat_chunks: list[torch.Tensor] = []
    logit_chunks: list[torch.Tensor] = []

    classifier_hidden = f4.classifier.hidden     # nn.Linear(4096, 512)
    classifier_output = f4.classifier.output     # nn.Linear(512, 3)

    with torch.inference_mode():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            x_orig = _to_model_tensor(patches_orig[start:end], device)
            x_reb = _to_model_tensor(patches_reb[start:end], device)

            embeddings_4096 = f4.encoder((x_orig, x_reb))   # (B, 4096)
            hidden_pre_relu = classifier_hidden(embeddings_4096)  # (B, 512)
            hidden_post_relu = F.relu(hidden_pre_relu)            # input del AttnMIL
            patch_logits = classifier_output(hidden_post_relu)    # (B, 3)
            feat_chunks.append(hidden_post_relu.cpu())
            logit_chunks.append(patch_logits.cpu())

    return torch.cat(feat_chunks, dim=0), torch.cat(logit_chunks, dim=0)


def _select_attnmil_models(
    ensemble: list[AttnMILBundle],
    mode: InferenceMode,
    seed: int | None = None,
) -> list[AttnMILBundle]:
    """Filtra el ensemble según el modo de inferencia."""
    if mode == "single":
        if seed is None:
            raise ValueError("modo 'single' requiere seed")
        return [b for b in ensemble if b.seed == seed]
    if mode == "ensemble":
        return list(ensemble)
    raise ValueError(f"modo desconocido: {mode}")


def predict_slide(
    f4: F4Bundle,
    ensemble: list[AttnMILBundle],
    patches_orig: np.ndarray,
    patches_reb: np.ndarray,
    mode: InferenceMode = "ensemble",
    seed: int | None = None,
    return_attention: bool = False,
) -> SlideResult:
    """Inferencia completa: parches → F4 → features 512-d → AttnMIL → probabilidades.

    Args:
        f4: bundle del modelo F4 cargado
        ensemble: lista de AttnMILBundle (5 modelos del ensemble de producción)
        patches_orig: np.ndarray (N, H, W, 3) uint8 — sin convertir
        patches_reb:  np.ndarray (N, H, W, 3) uint8 — sin convertir
        mode: "single" | "ensemble"
        seed: requerido en modo "single"
        return_attention: si True, devuelve además los pesos medios de atención

    Returns:
        SlideResult con probabilidades + meta
    """
    selected = _select_attnmil_models(ensemble, mode, seed=seed)
    if not selected:
        raise ValueError(
            f"No hay modelos AttnMIL para modo={mode} (seed={seed})"
        )

    features_512, patch_logits = _f4_forward_to_features(f4, patches_orig, patches_reb)
    patch_probs = F.softmax(patch_logits, dim=-1)            # (N, 3)
    patch_preds = torch.argmax(patch_probs, dim=-1)          # (N,)
    features_512 = features_512.to(f4.device)

    probs_per_member: list[torch.Tensor] = []
    attentions_per_member: list[torch.Tensor] = []

    with torch.inference_mode():
        for member in selected:
            logits, attention = member.model(features_512)
            probs = F.softmax(logits, dim=-1).squeeze(0)  # (3,)
            probs_per_member.append(probs)
            if return_attention:
                attentions_per_member.append(attention.squeeze(0).cpu())

    probs_stack = torch.stack(probs_per_member, dim=0)  # (M, 3)
    probs_mean = probs_stack.mean(dim=0).cpu()
    probs_std = probs_stack.std(dim=0).cpu() if probs_stack.shape[0] > 1 else torch.zeros_like(probs_mean)

    pred_idx = int(torch.argmax(probs_mean).item())

    attention_mean: torch.Tensor | None = None
    if return_attention and attentions_per_member:
        attention_mean = torch.stack(attentions_per_member, dim=0).mean(dim=0)

    return SlideResult(
        probabilities_mean=probs_mean,
        probabilities_std=probs_std,
        predicted_class=CLASS_NAMES[pred_idx],
        predicted_index=pred_idx,
        n_patches=int(features_512.shape[0]),
        n_models_used=len(selected),
        attention_weights_mean=attention_mean,
        patch_probs=patch_probs,
        patch_predictions=patch_preds,
        features=features_512.detach().cpu(),
    )


def predict_synthetic(
    f4: F4Bundle,
    ensemble: list[AttnMILBundle],
    n_patches: int = 50,
    mode: InferenceMode = "ensemble",
) -> SlideResult:
    """Smoke test: genera parches uint8 aleatorios y predice.

    No tiene sentido clínico, sirve solo para validar que el pipeline (F4 →
    features 512-d → AttnMIL → softmax) está cableado correctamente.
    """
    rng = np.random.default_rng(0)
    patches_orig = rng.integers(0, 256, size=(n_patches, 224, 224, 3), dtype=np.uint8)
    patches_reb = rng.integers(0, 256, size=(n_patches, 224, 224, 3), dtype=np.uint8)
    return predict_slide(f4, ensemble, patches_orig, patches_reb, mode=mode)
