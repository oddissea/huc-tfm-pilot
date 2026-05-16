"""Export de correcciones del patólogo a GCS antes de que el TTL las purgue.

Hito 0 del módulo de aprendizaje: ningún ``corrections.jsonl`` debería
desaparecer por el TTL de 24h del worker sin haber sido replicado a
``gs://huc-tfm-pilot-corrections/<job_id>/``. Este módulo expone las
primitivas que tanto el script CLI (``scripts/export_corrections.py``)
como el hook del prune del manager (``src/jobs/manager.py``) invocan.

Diseño:

- **Idempotente** vía sha256 local guardado en custom metadata del
  blob. Re-subir lo mismo es no-op.
- **Defensivo**: si GCS está caído o el bucket no es accesible, el
  caller decide qué hacer. ``export_job`` devuelve ``error`` no None
  pero NUNCA lanza excepción no manejada — la pasa al caller.
- **Trazabilidad**: además de ``corrections.jsonl``, subimos
  ``meta.json`` del job (filename original, n_patches, etc.) para que
  el reentrenamiento tenga el contexto completo del slide.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from google.cloud import storage

logger = logging.getLogger(__name__)

DEFAULT_BUCKET = "huc-tfm-pilot-corrections"


def _file_sha256(p: Path) -> str:
    """SHA256 hex digest del contenido del archivo."""
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def export_job(
    job_dir: Path,
    bucket: "storage.Bucket",
    *,
    dry_run: bool = False,
) -> dict:
    """Sube ``corrections.jsonl`` y ``meta.json`` de un ``job_dir``.

    Solo opera si el ``corrections.jsonl`` existe y tiene al menos una
    línea no vacía. Si no, marca el job como ``skipped`` y retorna.

    Idempotencia: comparamos sha256 local con el guardado en custom
    metadata del blob remoto. Si coincide, no resubimos el JSONL.
    ``meta.json`` se sobreescribe siempre (es pequeño y a veces se
    actualiza).

    Args:
        job_dir: ruta al ``<queue>/<job_id>/``.
        bucket: instancia ``google.cloud.storage.Bucket`` ya
            autenticada.
        dry_run: si ``True``, no sube nada, solo simula.

    Returns:
        Dict con keys: ``job_id``, ``n_corrections`` (líneas no
        vacías), ``uploaded_corrections`` (bool), ``uploaded_meta``
        (bool), ``skipped`` (bool — sin corrections para subir),
        ``error`` (str | None).
    """
    from google.cloud.exceptions import NotFound

    job_id = job_dir.name
    result: dict = {
        "job_id": job_id,
        "n_corrections": 0,
        "uploaded_corrections": False,
        "uploaded_meta": False,
        "skipped": False,
        "error": None,
    }

    corrections_path = job_dir / "corrections.jsonl"
    if not corrections_path.exists():
        result["skipped"] = True
        return result

    # Contar líneas no vacías (cheap sanity check + métrica).
    with corrections_path.open() as f:
        n_lines = sum(1 for line in f if line.strip())
    if n_lines == 0:
        result["skipped"] = True
        return result
    result["n_corrections"] = n_lines

    # Idempotencia vía sha256.
    local_hash = _file_sha256(corrections_path)
    blob_corr = bucket.blob(f"{job_id}/corrections.jsonl")
    remote_hash: str | None = None
    try:
        blob_corr.reload()
        if blob_corr.metadata:
            remote_hash = blob_corr.metadata.get("local_sha256")
    except NotFound:
        remote_hash = None
    except Exception as e:  # noqa: BLE001 — defensa anti-flake de GCS
        result["error"] = f"reload corrections failed: {e}"
        return result

    if remote_hash != local_hash:
        if not dry_run:
            try:
                blob_corr.metadata = {"local_sha256": local_hash}
                blob_corr.upload_from_filename(
                    str(corrections_path),
                    content_type="application/x-ndjson",
                )
            except Exception as e:  # noqa: BLE001
                result["error"] = f"upload corrections failed: {e}"
                return result
        result["uploaded_corrections"] = True

    # meta.json: overwrite siempre. Es pequeño.
    meta_path = job_dir / "meta.json"
    if meta_path.exists():
        blob_meta = bucket.blob(f"{job_id}/meta.json")
        if not dry_run:
            try:
                blob_meta.upload_from_filename(
                    str(meta_path),
                    content_type="application/json",
                )
            except Exception as e:  # noqa: BLE001
                result["error"] = f"upload meta failed: {e}"
                return result
        result["uploaded_meta"] = True

    return result


def export_all(
    queue_dir: Path,
    bucket_name: str = DEFAULT_BUCKET,
    *,
    dry_run: bool = False,
) -> list[dict]:
    """Itera ``queue_dir`` y exporta cada job que tenga correcciones.

    Args:
        queue_dir: directorio raíz de la cola (contiene ``<job_id>/``).
        bucket_name: nombre del bucket GCS (sin prefijo ``gs://``).
        dry_run: si ``True``, simula sin subir.

    Returns:
        Lista de dicts (uno por job procesado), tal como devuelve
        ``export_job()``. Jobs sin ``corrections.jsonl`` se filtran
        antes (no aparecen en la lista).
    """
    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    results: list[dict] = []
    for job_dir in sorted(queue_dir.iterdir()):
        if not job_dir.is_dir() or job_dir.name.startswith("."):
            continue
        result = export_job(job_dir, bucket, dry_run=dry_run)
        if not result["skipped"] or result["error"]:
            results.append(result)
    return results


def export_job_safe(
    job_dir: Path,
    *,
    bucket_name: str = DEFAULT_BUCKET,
) -> dict:
    """Versión "safe" de ``export_job`` para uso desde el prune del manager.

    Crea el cliente y bucket por su cuenta. Captura cualquier excepción
    no esperada (incluyendo errores de auth, network, etc.) y la
    devuelve como ``error`` en el dict, sin propagar — el caller
    (``manager.prune``) usa el ``error`` para decidir si borrar o no
    el job_dir.
    """
    try:
        from google.cloud import storage

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        return export_job(job_dir, bucket, dry_run=False)
    except Exception as e:  # noqa: BLE001 — esto es la safety net
        return {
            "job_id": job_dir.name,
            "n_corrections": 0,
            "uploaded_corrections": False,
            "uploaded_meta": False,
            "skipped": False,
            "error": f"export_job_safe failed: {e}",
        }
