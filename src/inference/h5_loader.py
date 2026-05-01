"""Lee un H5 generado por el pipeline TIFF→H5 y devuelve los parches
en uint8 listos para que `predict.py` los convierta a float32 + 224×224
**en el loop de batches** (evita pico de RAM en slides con muchos parches).

Formato esperado del H5 (port directo de portaanalysis/util_hdf5.py):
    /patches            (N, 2, H, W, 3) uint8
                        [:,0] = original, [:,1] = rebinned
    /patch_positions    (N, 2) int       (y, x) esquina superior-izquierda
    /patch_categories   (N,)   |S3       "XXX" para inferencia
    /patch_numbers      (N,)   int32     id original de grid

Por qué uint8 en lugar de float32+resize de golpe: para `ca_1534.h5`
(3.177 parches), preconvertir a float32 picaba ~12 GB de RAM y mataba
la VM `g2-standard-4` (16 GB) por OOM. Manteniendo uint8 hasta el batch
loop, el pico baja a ~1,7 GB.

Preprocesamiento real (lo hace `predict._to_model_tensor` por batch):
    1. (B, H, W, 3) uint8 → (B, 3, H, W) float32 en [0,1]
    2. Resize bilineal antialias a 224×224 si H,W != 224
    3. NO se normaliza con stats ImageNet (BiT-M no lo requiere).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

logger = logging.getLogger(__name__)

TARGET_HW = 224


@dataclass
class H5Patches:
    patches_orig: np.ndarray     # (N, H, W, 3) uint8 — sin convertir
    patches_reb: np.ndarray      # (N, H, W, 3) uint8 — sin convertir
    positions: np.ndarray        # (N, 2) int — (y, x)
    source_image_name: str | None
    raw_size: int                # H=W del parche en el H5 (antes del resize)
    patch_categories: np.ndarray # (N,) str — etiqueta por parche ('XXX' = sin GT)

    @property
    def has_patch_gt(self) -> bool:
        """True si el H5 trae etiquetas patch-level útiles (no todo XXX/?)."""
        return bool(((self.patch_categories != "XXX") & (self.patch_categories != "?")).any())


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

        if "patch_categories" in f:
            cats_raw = f["patch_categories"][:]   # type: ignore[index]
            categories = np.array([
                c.decode() if isinstance(c, bytes) else str(c) for c in cats_raw
            ])
        else:
            categories = np.array(["?"] * n)

        source_name = None
        attrs = dict(f.attrs)   # type: ignore[arg-type]
        if "source_image_name" in attrs:
            v = attrs["source_image_name"]
            source_name = v.decode() if isinstance(v, bytes) else str(v)

    patches_orig = np.ascontiguousarray(patches[:, 0])
    patches_reb = np.ascontiguousarray(patches[:, 1])

    logger.info(
        "H5 cargado: N=%d, patch_size=%d (uint8, resize a %d en GPU), source=%s",
        n, h, TARGET_HW, source_name or "?"
    )

    return H5Patches(
        patches_orig=patches_orig,
        patches_reb=patches_reb,
        positions=np.asarray(positions),
        source_image_name=source_name,
        raw_size=h,
        patch_categories=categories,
    )
