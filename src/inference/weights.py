"""Descarga y cacheo de pesos del modelo desde GCS.

Estructura esperada en `gs://huc-tfm-pilot-models/`:

    F4/final_inference_model.pth                          -- pesos del modelo F4 patch-level
    attnmil_production/seed_{42,123,456,789,2026}/model.pth   -- 5 modelos del ensemble ternario

Cada `seed_N/model.pth` es un AttnMIL ternario 512-d entrenado sobre los 91
slides clínicos del cohort §5.9 **sin** validación cruzada (artefacto de
producción). §5.9 reporta el rendimiento esperado sobre slides nuevos
(92,8 ± 1,1 % accuracy en 5-fold CV multi-seed); este ensemble de 5 ofrece
robustez por reducción de varianza al consumir el modelo en producción.

Se descargan al directorio local `/app/weights/` (montado como volumen Docker
para que persista entre reinicios). En la siguiente arrancada, si los ficheros
ya existen, no se re-descargan.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from google.cloud import storage

logger = logging.getLogger(__name__)

BUCKET_NAME = os.environ.get("HUC_PILOT_BUCKET", "huc-tfm-pilot-models")
WEIGHTS_DIR = Path(os.environ.get("HUC_PILOT_WEIGHTS_DIR", "/app/weights"))

F4_BLOB = "F4/final_inference_model.pth"
ATTNMIL_SEEDS = (42, 123, 456, 789, 2026)
ATTNMIL_PREFIX = "attnmil_production"


def f4_local_path() -> Path:
    """Ruta local donde se cachea el peso del F4."""
    return WEIGHTS_DIR / F4_BLOB


def attnmil_local_path(seed: int) -> Path:
    """Ruta local donde se cachea un modelo concreto del ensemble AttnMIL."""
    return WEIGHTS_DIR / ATTNMIL_PREFIX / f"seed_{seed}" / "model.pth"


def list_attnmil_seeds() -> list[int]:
    """Devuelve la lista canónica de seeds que conforma el ensemble de producción."""
    return list(ATTNMIL_SEEDS)


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
    """Garantiza que F4 y los 5 AttnMIL están descargados localmente.

    Args:
        progress_cb: opcional, función `progress_cb(done: int, total: int, msg: str)`
                     llamada tras cada descarga (para feedback en UI).

    Returns:
        Diccionario con las rutas locales:
            {
                "f4": Path,
                "attnmil": [(seed, Path), ...],
            }
    """
    client = storage.Client()

    seeds = list_attnmil_seeds()
    total = 1 + len(seeds)
    done = 0

    if progress_cb is not None:
        progress_cb(done, total, "Descargando F4…")
    _download_blob(client, F4_BLOB, f4_local_path())
    done += 1

    attnmil_paths: list[tuple[int, Path]] = []
    for seed in seeds:
        if progress_cb is not None:
            progress_cb(done, total, f"Descargando AttnMIL seed={seed}…")
        path = attnmil_local_path(seed)
        _download_blob(client, f"{ATTNMIL_PREFIX}/seed_{seed}/model.pth", path)
        attnmil_paths.append((seed, path))
        done += 1

    if progress_cb is not None:
        progress_cb(done, total, "Descargas completadas.")

    return {"f4": f4_local_path(), "attnmil": attnmil_paths}
