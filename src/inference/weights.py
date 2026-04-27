"""Descarga y cacheo de pesos del modelo desde GCS.

Estructura esperada en `gs://huc-tfm-pilot-models/`:

    F4/final_inference_model.pth                     -- pesos del modelo F4 patch-level
    attnmil/seed_{42,123,456,789,2026}/fold_{0..4}.pth   -- 25 modelos del ensemble ternario

Se descargan al directorio local `/app/weights/` (montado como volumen Docker
para que persista entre reinicios). En la siguiente arrancada, si los ficheros
ya existen, no se re-descargan.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable

from google.cloud import storage

logger = logging.getLogger(__name__)

BUCKET_NAME = os.environ.get("HUC_PILOT_BUCKET", "huc-tfm-pilot-models")
WEIGHTS_DIR = Path(os.environ.get("HUC_PILOT_WEIGHTS_DIR", "/app/weights"))

F4_BLOB = "F4/final_inference_model.pth"
ATTNMIL_SEEDS = (42, 123, 456, 789, 2026)
ATTNMIL_FOLDS = (0, 1, 2, 3, 4)


def f4_local_path() -> Path:
    """Ruta local donde se cachea el peso del F4."""
    return WEIGHTS_DIR / F4_BLOB


def attnmil_local_path(seed: int, fold: int) -> Path:
    """Ruta local donde se cachea un modelo concreto del ensemble AttnMIL."""
    return WEIGHTS_DIR / "attnmil" / f"seed_{seed}" / f"fold_{fold}.pth"


def list_attnmil_models() -> list[tuple[int, int]]:
    """Devuelve la lista canónica de (seed, fold) que conforma el ensemble."""
    return [(s, f) for s in ATTNMIL_SEEDS for f in ATTNMIL_FOLDS]


def _download_blob(client: storage.Client, blob_name: str, target: Path) -> None:
    """Descarga un objeto de GCS al disco local. No re-descarga si ya existe."""
    if target.exists():
        logger.debug("cache hit: %s", target)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(blob_name)
    logger.info("downloading gs://%s/%s -> %s", BUCKET_NAME, blob_name, target)
    blob.download_to_filename(str(target))


def ensure_weights(progress_cb=None) -> dict:
    """Garantiza que F4 y los 25 AttnMIL están descargados localmente.

    Args:
        progress_cb: opcional, función `progress_cb(done: int, total: int, msg: str)`
                     llamada tras cada descarga (para feedback en UI).

    Returns:
        Diccionario con las rutas locales:
            {
                "f4": Path,
                "attnmil": [(seed, fold, Path), ...],
            }
    """
    client = storage.Client()

    pairs = list_attnmil_models()
    total = 1 + len(pairs)
    done = 0

    if progress_cb is not None:
        progress_cb(done, total, "Descargando F4…")
    _download_blob(client, F4_BLOB, f4_local_path())
    done += 1

    attnmil_paths: list[tuple[int, int, Path]] = []
    for seed, fold in pairs:
        if progress_cb is not None:
            progress_cb(done, total, f"Descargando AttnMIL seed={seed} fold={fold}…")
        path = attnmil_local_path(seed, fold)
        _download_blob(client, f"attnmil/seed_{seed}/fold_{fold}.pth", path)
        attnmil_paths.append((seed, fold, path))
        done += 1

    if progress_cb is not None:
        progress_cb(done, total, "Descargas completadas.")

    return {"f4": f4_local_path(), "attnmil": attnmil_paths}
