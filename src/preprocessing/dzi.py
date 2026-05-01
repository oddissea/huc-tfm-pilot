"""Genera tiles DZI (Deep Zoom Image) para el visor OpenSeadragon.

Dos paths de entrada:

1. **TIFF directo**: `generate_dzi(tiff_path, ...)` — pirámide del WSI
   completo, incluyendo zonas de fondo blanco que el filtro de tejido
   descartó. Lento para WSIs muy grandes (~3 min para 2 GPx).

2. **H5 stitched**: `generate_dzi_from_h5(h5_path, ...)` — reconstruye una
   imagen "solo tejido" stitcheando los parches del H5 en sus posiciones
   originales del WSI. Funciona también para uploads H5 sin TIFF original
   (cohort §5.9) y produce coordenadas IDÉNTICAS a las que usa el AttnMIL,
   así los overlays de predicción se alinean pixel-perfect.

OpenSeadragon consume `<basename>.dzi` y carga los tiles bajo demanda.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import h5py
import numpy as np
import pyvips

logger = logging.getLogger(__name__)

DEFAULT_TILE_SIZE = 256
DEFAULT_OVERLAP = 1
DEFAULT_QUALITY = 85       # JPEG quality (1-100)


def generate_dzi(
    source_image: Path,
    output_dir: Path,
    basename: str = "slide",
    tile_size: int = DEFAULT_TILE_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    quality: int = DEFAULT_QUALITY,
) -> Path:
    """Genera la pirámide DZI desde una imagen (TIFF/PNG/JPG/...).

    Args:
        source_image: imagen fuente (TIFF preferido para WSI)
        output_dir: directorio donde se escribirán `.dzi` + `_files/`
        basename: prefijo de los ficheros generados (sin extensión)
        tile_size: tamaño de los tiles cuadrados en píxeles
        overlap: solape entre tiles (mejora pan/zoom suave en bordes)
        quality: calidad JPEG (mayor = más calidad y más espacio)

    Returns:
        Path al fichero `.dzi`
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / basename       # libvips añade .dzi y _files/

    t0 = time.time()
    img = pyvips.Image.new_from_file(str(source_image), access="sequential")
    logger.info(
        "DZI source: %s (%dx%d, %d bandas)",
        source_image.name, img.width, img.height, img.bands,
    )
    img.dzsave(
        str(target),
        tile_size=tile_size,
        overlap=overlap,
        suffix=f".jpg[Q={quality}]",
        layout="dz",
    )
    elapsed = time.time() - t0
    dzi_path = target.with_suffix(".dzi")
    logger.info(
        "DZI generado en %.1fs: %s (tiles=%d px, Q=%d)",
        elapsed, dzi_path, tile_size, quality,
    )
    return dzi_path


def stitch_h5_to_image(h5_path: Path) -> tuple[np.ndarray, tuple[int, int]]:
    """Reconstruye una imagen "solo tejido" stitcheando los parches del H5
    en sus posiciones originales del WSI.

    Returns:
        canvas: np.ndarray (H_total, W_total, 3) uint8. Posiciones sin parche
                quedan en blanco (255).
        offset: (y_min, x_min) — origen del canvas en coordenadas del WSI.
                Para mapear una posición de parche del H5 a la imagen
                stitched: (py - y_min, px - x_min).
    """
    with h5py.File(str(h5_path), "r") as f:
        patches = np.asarray(f["patches"][:, 0])           # (N, H, W, 3) uint8 — stream original
        positions = np.asarray(f["patch_positions"][:])    # (N, 2) — (y, x)

    n, h, w, c = patches.shape
    if c != 3:
        raise ValueError(f"Esperaba 3 canales, recibí {c}")
    y_min = int(positions[:, 0].min())
    x_min = int(positions[:, 1].min())
    y_max = int(positions[:, 0].max()) + h
    x_max = int(positions[:, 1].max()) + w

    canvas = np.full((y_max - y_min, x_max - x_min, 3), 255, dtype=np.uint8)
    for i in range(n):
        py = int(positions[i, 0]) - y_min
        px = int(positions[i, 1]) - x_min
        canvas[py:py + h, px:px + w] = patches[i]

    logger.info(
        "Stitched %d parches → canvas %dx%d uint8 (~%.1f MB)",
        n, canvas.shape[1], canvas.shape[0], canvas.nbytes / 1e6,
    )
    return canvas, (y_min, x_min)


def generate_dzi_from_h5(
    h5_path: Path,
    output_dir: Path,
    basename: str = "slide",
    tile_size: int = DEFAULT_TILE_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    quality: int = DEFAULT_QUALITY,
) -> tuple[Path, tuple[int, int]]:
    """Genera DZI a partir de un H5 parcheado, stitcheando parches en una
    imagen 'solo tejido'. Devuelve (path al .dzi, offset (y_min, x_min)).

    El offset es necesario para que el visor pueda dibujar overlays sobre
    los parches usando sus posiciones originales del WSI.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / basename

    canvas, offset = stitch_h5_to_image(h5_path)
    h, w, _ = canvas.shape

    t0 = time.time()
    # pyvips desde memoria: evita pasar por TIFF intermedio
    img = pyvips.Image.new_from_memory(canvas.tobytes(), w, h, 3, "uchar")
    img.dzsave(
        str(target),
        tile_size=tile_size,
        overlap=overlap,
        suffix=f".jpg[Q={quality}]",
        layout="dz",
    )
    elapsed = time.time() - t0
    dzi_path = target.with_suffix(".dzi")
    logger.info(
        "DZI desde H5 generado en %.1fs: %s (canvas %dx%d, offset=%s)",
        elapsed, dzi_path, w, h, offset,
    )
    return dzi_path, offset


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=Path)
    ap.add_argument("--output-dir", type=Path, default=Path("./dzi_output"))
    ap.add_argument("--basename", default="slide")
    ap.add_argument("--tile-size", type=int, default=DEFAULT_TILE_SIZE)
    ap.add_argument("--quality", type=int, default=DEFAULT_QUALITY)
    args = ap.parse_args()
    generate_dzi(
        args.source, args.output_dir, args.basename,
        tile_size=args.tile_size, quality=args.quality,
    )
