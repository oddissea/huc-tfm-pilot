"""Genera tiles DZI (Deep Zoom Image) desde un TIFF para servirlos al
visor OpenSeadragon. La pirámide multi-resolución la calcula libvips, que
es órdenes de magnitud más rápido que cualquier alternativa Python pura.

Output:
    <output_dir>/<basename>.dzi              -- XML con metadata
    <output_dir>/<basename>_files/0/0_0.jpg  -- tiles por nivel
    <output_dir>/<basename>_files/1/0_0.jpg
    ...

OpenSeadragon consume `<basename>.dzi` y carga los tiles bajo demanda.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

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
