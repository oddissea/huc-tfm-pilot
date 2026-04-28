"""Lee un H5 generado por el pipeline TIFF→H5 y devuelve los tensores
de parches listos para inferencia con F4 (BiT-M).

Formato esperado del H5 (port directo de portaanalysis/util_hdf5.py):
    /patches            (N, 2, H, W, 3) uint8
                        [:,0] = original, [:,1] = rebinned
    /patch_positions    (N, 2) int       (y, x) esquina superior-izquierda
    /patch_categories   (N,)   |S3       "XXX" para inferencia
    /patch_numbers      (N,)   int32     id original de grid

Preprocesamiento aplicado (idéntico a `create_stage2_transforms('bitm')`):
    1. (N,2,H,W,3) uint8 → (N,3,H,W) float32 en [0,1]
    2. Resize bilineal antialias a 224×224 si H,W != 224
    3. NO se normaliza con stats ImageNet (BiT-M no lo requiere).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

TARGET_HW = 224


@dataclass
class H5Patches:
    patches_orig: torch.Tensor   # (N, 3, 224, 224) float32 [0,1]
    patches_reb: torch.Tensor    # (N, 3, 224, 224) float32 [0,1]
    positions: np.ndarray        # (N, 2) int — (y, x)
    source_image_name: str | None
    raw_size: int                # H=W del parche en el H5 (antes del resize)


def _patches_to_tensor(patches_np: np.ndarray) -> torch.Tensor:
    """(N, H, W, 3) uint8 → (N, 3, 224, 224) float32 en [0,1]."""
    if patches_np.dtype != np.uint8:
        raise ValueError(f"Esperaba uint8, recibí {patches_np.dtype}")
    if patches_np.ndim != 4 or patches_np.shape[-1] != 3:
        raise ValueError(f"Shape inesperado: {patches_np.shape}, esperaba (N,H,W,3)")

    # (N, H, W, 3) → (N, 3, H, W) float32 [0,1]
    t = torch.from_numpy(patches_np).permute(0, 3, 1, 2).float() / 255.0

    h, w = t.shape[-2:]
    if (h, w) != (TARGET_HW, TARGET_HW):
        t = F.interpolate(
            t,
            size=(TARGET_HW, TARGET_HW),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
    return t.contiguous()


def load_patches_from_h5(h5_path: Path) -> H5Patches:
    h5_path = Path(h5_path)
    if not h5_path.exists():
        raise FileNotFoundError(h5_path)

    with h5py.File(str(h5_path), "r") as f:
        if "patches" not in f:
            raise ValueError("H5 no contiene dataset 'patches'")

        patches = f["patches"][:]   # type: ignore[index]
        if patches.ndim != 5 or patches.shape[1] != 2:
            raise ValueError(
                f"Shape de 'patches' inesperado: {patches.shape}, "
                "esperaba (N, 2, H, W, 3) — ¿está en formato dual_stream?"
            )
        n, _, h, w, c = patches.shape
        if c != 3:
            raise ValueError(f"'patches' debe tener 3 canales, tiene {c}")
        if n == 0:
            raise ValueError("H5 sin parches útiles (N=0).")

        positions = (
            f["patch_positions"][:]   # type: ignore[index]
            if "patch_positions" in f else np.zeros((n, 2), dtype=np.int32)
        )

        source_name = None
        attrs = dict(f.attrs)   # type: ignore[arg-type]
        if "source_image_name" in attrs:
            v = attrs["source_image_name"]
            source_name = v.decode() if isinstance(v, bytes) else str(v)

    patches_orig = _patches_to_tensor(patches[:, 0])
    patches_reb = _patches_to_tensor(patches[:, 1])

    logger.info(
        "H5 cargado: N=%d, patch_size=%d → resize a %d, source=%s",
        n, h, TARGET_HW, source_name or "?"
    )

    return H5Patches(
        patches_orig=patches_orig,
        patches_reb=patches_reb,
        positions=np.asarray(positions),
        source_image_name=source_name,
        raw_size=h,
    )
