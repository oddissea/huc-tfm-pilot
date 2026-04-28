"""Pipeline TIFF → H5 (port directo de portaanalysis del servidor doctor).

Función pública: `convert_tiff_to_h5(tiff_path, h5_path)`. Usa los defaults
del piloto (patch 300×300, white-percent 80, image-level 5).
"""

from __future__ import annotations

import logging
from pathlib import Path

from .tiff_to_h5 import (
    PortaHelper,
    generate_patches,
    rebin_patches,
    save_patches_to_hdf5,
)

logger = logging.getLogger(__name__)

PATCH_SIZE = 300
WHITE_PERCENT = 80.0
IMAGE_LEVEL = 5


def convert_tiff_to_h5(tiff_path: Path, h5_path: Path) -> int:
    """Convierte un TIFF piramidal en un H5 con parches dual-stream.

    Parámetros idénticos al pipeline del servidor doctor (300×300, 80%
    de blanco, level=5). El H5 resultante tiene `patches` (N,2,H,W,3)
    uint8 con orig en [:,0] y rebinned en [:,1].

    Devuelve N (parches útiles tras filtro de blanco + rebin con 8 vecinos).
    Lanza ValueError si N=0.
    """
    porta = PortaHelper(str(tiff_path), image_level=IMAGE_LEVEL)
    tissue, image_name, pixel_max = porta.extract_tissue()
    logger.info(
        "TIFF: tissue %s, pixel_max=%d", tissue.shape, pixel_max,
    )

    patch_list = generate_patches(
        tissue, [PATCH_SIZE, PATCH_SIZE], WHITE_PERCENT, pixel_max,
    )
    positions, indices, pairs = rebin_patches(patch_list, PATCH_SIZE)
    n_pre = len(patch_list)
    n_useful = len(pairs)
    logger.info("Parches: %d pre-rebin → %d con 8 vecinos", n_pre, n_useful)

    if n_useful == 0:
        raise ValueError(
            f"Sin parches útiles tras rebin: {n_pre} pre-rebin pero ninguno "
            "tiene los 8 vecinos requeridos. ¿TIFF demasiado pequeño o tejido fragmentado?"
        )

    categories = ["XXX"] * n_useful
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    save_patches_to_hdf5(
        pairs, str(h5_path), f"{image_name}.tif",
        positions, categories, indices,
        patch_size_x=PATCH_SIZE, patch_size_y=PATCH_SIZE, num_channels=3,
    )
    return n_useful


__all__ = ["convert_tiff_to_h5"]
