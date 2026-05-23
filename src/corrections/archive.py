"""Archive local de correcciones del patólogo + features 512-d.

Hito 1 del módulo de aprendizaje: cuando el TTL del worker está a punto
de purgar un ``job_dir``, COPIAMOS los artefactos críticos a un archive
bind-montado en el host (NO subimos a la nube). Esto permite que el
reentrenamiento futuro recoja las correcciones aunque hayan pasado meses
desde la inferencia.

Reemplaza el ``export.py`` original basado en GCS. Decisión de la sesión
#64: en el HUC no habrá GCS — los datos del paciente no salen del
hospital. Una sola implementación del sink, idéntica en QA y producción,
elimina el riesgo de "el path de producción nunca se probó".

Diseño:

- **Idempotente** vía comparación directa de sha256 entre origen y
  destino. Para features.npy (1-3 MB típico) el coste de hashing es
  inferior al de la copia, así que no compensa optimizar con sidecars.
- **Copia atómica** vía fichero temporal + ``os.replace``. Si el proceso
  crashea a media copia, el destino final no queda corrupto.
- **Defensiva**: si la copia falla, devuelve ``error`` no None pero
  NUNCA lanza excepción no manejada — el caller decide qué hacer.
- **Trazabilidad**: archiva ``corrections.jsonl`` + ``features.npy`` +
  ``meta.json``. Suficiente para reentrenar el head F4 y el AttnMIL sin
  reforwardear el encoder.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_ARCHIVE_DIR = Path(
    os.environ.get("PILOT_ARCHIVE_DIR", "/var/archive")
)

CORRECTIONS_FILENAME = "corrections.jsonl"
FEATURES_FILENAME = "features.npy"
META_FILENAME = "meta.json"


def _file_sha256(p: Path) -> str:
    """SHA256 hex digest del contenido del archivo."""
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _files_identical(src: Path, dst: Path) -> bool:
    """True si dst existe y su sha256 coincide con src."""
    if not dst.exists():
        return False
    if src.stat().st_size != dst.stat().st_size:
        return False
    return _file_sha256(src) == _file_sha256(dst)


def _copy_atomic(src: Path, dst: Path) -> None:
    """Copia src → dst atómicamente vía fichero temporal.

    Si el proceso muere a media copia, ``dst`` final queda intacto (o
    inexistente). El ``.tmp`` puede quedar huérfano pero no corrompe.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


def archive_job(
    job_dir: Path,
    archive_dir: Path = DEFAULT_ARCHIVE_DIR,
    *,
    dry_run: bool = False,
) -> dict:
    """Copia ``corrections.jsonl`` + ``features.npy`` + ``meta.json`` al archive.

    Solo opera si el ``corrections.jsonl`` existe y tiene al menos una
    línea no vacía. Si no, marca el job como ``skipped`` y retorna sin
    tocar el archive — no tiene sentido archivar features sin la señal
    supervisada que justifica el reentrenamiento.

    Idempotencia: para cada fichero comparamos sha256 local con el del
    archive. Si coinciden, no se copia. ``meta.json`` se sobreescribe
    siempre (es pequeño y puede actualizarse: dzi_status, predicted_class,
    etc.).

    Args:
        job_dir: ruta al ``<queue>/<job_id>/``.
        archive_dir: raíz del archive (default: ``$PILOT_ARCHIVE_DIR``).
        dry_run: si ``True``, no copia nada, solo simula.

    Returns:
        Dict con keys: ``job_id``, ``n_corrections`` (líneas no vacías),
        ``archived_corrections`` (bool: archive tiene el fichero al día),
        ``archived_features`` (bool: idem), ``archived_meta`` (bool),
        ``skipped`` (bool — sin corrections para archivar),
        ``error`` (str | None).
    """
    job_id = job_dir.name
    result: dict = {
        "job_id": job_id,
        "n_corrections": 0,
        "archived_corrections": False,
        "archived_features": False,
        "archived_meta": False,
        "skipped": False,
        "error": None,
    }

    corrections_src = job_dir / CORRECTIONS_FILENAME
    if not corrections_src.exists():
        result["skipped"] = True
        return result

    # Sanity check: corrections.jsonl puede existir pero estar vacío si
    # el patólogo abrió el slide y no corrigió nada. No archivamos.
    with corrections_src.open() as f:
        n_lines = sum(1 for line in f if line.strip())
    if n_lines == 0:
        result["skipped"] = True
        return result
    result["n_corrections"] = n_lines

    if dry_run:
        result["archived_corrections"] = True
        result["archived_features"] = (job_dir / FEATURES_FILENAME).exists()
        result["archived_meta"] = (job_dir / META_FILENAME).exists()
        return result

    job_archive_dir = archive_dir / job_id

    # 1. corrections.jsonl — BLOQUEANTE: si falla, el caller NO debe borrar
    # el job_dir; reintenta en el siguiente prune. Las correcciones del
    # patólogo son irrecuperables si se pierden.
    corrections_dst = job_archive_dir / CORRECTIONS_FILENAME
    try:
        if not _files_identical(corrections_src, corrections_dst):
            _copy_atomic(corrections_src, corrections_dst)
        result["archived_corrections"] = True
    except Exception as e:  # noqa: BLE001 — defensa
        result["error"] = f"archive corrections failed: {e}"
        return result

    # 2. features.npy — también bloqueante (si está). Si falla, conservar
    # job_dir para reintentar: los features se generaron junto con el
    # forward del encoder y son la mitad útil del registro
    # (corrections sin features → re-forwardear F4 al reentrenar).
    features_src = job_dir / FEATURES_FILENAME
    if features_src.exists():
        features_dst = job_archive_dir / FEATURES_FILENAME
        try:
            if not _files_identical(features_src, features_dst):
                _copy_atomic(features_src, features_dst)
            result["archived_features"] = True
        except Exception as e:  # noqa: BLE001
            result["error"] = f"archive features failed: {e}"
            return result

    # 3. meta.json — overwrite siempre. Es pequeño y a veces se actualiza.
    meta_src = job_dir / META_FILENAME
    if meta_src.exists():
        meta_dst = job_archive_dir / META_FILENAME
        try:
            _copy_atomic(meta_src, meta_dst)
            result["archived_meta"] = True
        except Exception as e:  # noqa: BLE001
            result["error"] = f"archive meta failed: {e}"
            return result

    return result


def archive_all(
    queue_dir: Path,
    archive_dir: Path = DEFAULT_ARCHIVE_DIR,
    *,
    dry_run: bool = False,
) -> list[dict]:
    """Itera ``queue_dir`` y archiva cada job que tenga correcciones.

    Args:
        queue_dir: directorio raíz de la cola (contiene ``<job_id>/``).
        archive_dir: raíz del archive local.
        dry_run: si ``True``, simula sin copiar.

    Returns:
        Lista de dicts (uno por job procesado), tal como devuelve
        ``archive_job()``. Jobs sin ``corrections.jsonl`` se filtran
        antes (no aparecen en la lista).
    """
    results: list[dict] = []
    for job_dir in sorted(queue_dir.iterdir()):
        if not job_dir.is_dir() or job_dir.name.startswith("."):
            continue
        result = archive_job(job_dir, archive_dir, dry_run=dry_run)
        if not result["skipped"] or result["error"]:
            results.append(result)
    return results


def archive_job_safe(
    job_dir: Path,
    archive_dir: Path = DEFAULT_ARCHIVE_DIR,
) -> dict:
    """Versión "safe" de ``archive_job`` para uso desde el prune del manager.

    Captura cualquier excepción no esperada y la devuelve como ``error``
    en el dict, sin propagar — el caller (``manager.prune``) usa el
    ``error`` para decidir si borrar o no el ``job_dir``.
    """
    try:
        return archive_job(job_dir, archive_dir, dry_run=False)
    except Exception as e:  # noqa: BLE001 — safety net
        return {
            "job_id": job_dir.name,
            "n_corrections": 0,
            "archived_corrections": False,
            "archived_features": False,
            "archived_meta": False,
            "skipped": False,
            "error": f"archive_job_safe failed: {e}",
        }


def archive_stats(archive_dir: Path = DEFAULT_ARCHIVE_DIR) -> dict:
    """Lee el archive y devuelve estadísticas agregadas para la UI.

    Recorre cada subdirectorio del archive y suma tamaños + cuenta
    correcciones. No abre los .npy (sólo `stat()`), así que es barato
    incluso con cientos de jobs.

    Returns:
        Dict con:
        - ``n_jobs``: número de subdirectorios con al menos un fichero.
        - ``n_jobs_with_features``: subset con features.npy presente.
        - ``total_bytes``: suma de tamaños de los ficheros archivados.
        - ``n_corrections_total``: suma de líneas no vacías de los
          ``corrections.jsonl``.
        - ``last_archived_at``: timestamp epoch del fichero más reciente
          en el archive (`None` si vacío).
        - ``oldest_archived_at``: timestamp epoch del fichero más antiguo
          (`None` si vacío).
        - ``archive_dir``: ruta absoluta del archive como string.
        - ``exists``: ``False`` si el directorio aún no existe (primer
          uso del piloto).
    """
    result: dict = {
        "archive_dir": str(archive_dir),
        "exists": archive_dir.exists(),
        "n_jobs": 0,
        "n_jobs_with_features": 0,
        "total_bytes": 0,
        "n_corrections_total": 0,
        "last_archived_at": None,
        "oldest_archived_at": None,
    }
    if not archive_dir.exists():
        return result

    mtimes: list[float] = []
    for job_dir in archive_dir.iterdir():
        if not job_dir.is_dir() or job_dir.name.startswith("."):
            continue
        files = [p for p in job_dir.iterdir() if p.is_file()]
        if not files:
            continue
        result["n_jobs"] += 1
        for p in files:
            st = p.stat()
            result["total_bytes"] += st.st_size
            mtimes.append(st.st_mtime)
        if (job_dir / FEATURES_FILENAME).exists():
            result["n_jobs_with_features"] += 1
        corr = job_dir / CORRECTIONS_FILENAME
        if corr.exists():
            with corr.open() as f:
                result["n_corrections_total"] += sum(1 for line in f if line.strip())

    if mtimes:
        result["last_archived_at"] = max(mtimes)
        result["oldest_archived_at"] = min(mtimes)
    return result
