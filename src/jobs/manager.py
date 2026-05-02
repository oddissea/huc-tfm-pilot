"""Gestor de cola en disco efímero.

Cada job vive en `/tmp/queue/<uuid>/` con:
- `raw.tif` o `raw.h5` (el upload original)
- `meta.json` con filename, tipo, estado, timestamps, error opcional

Estados (`JobStatus`):
    queued   →  processing  →  converted (solo TIFF)  →  ready_for_inference
                                                      ↘
                                                       failed (error en meta.json)

`ready_for_inference` es el estado terminal de M4.3. M4.4 añadirá los estados
`predicting` → `done`.
"""

from __future__ import annotations

import json
import logging
import shutil
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import IO

logger = logging.getLogger(__name__)

QUEUE_ROOT = Path("/tmp/queue")
META_FILENAME = "meta.json"

# Extensiones aceptadas y nombre canónico del raw file
TIFF_EXTS = {".tif", ".tiff"}
H5_EXTS = {".h5", ".hdf5"}


class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    CONVERTED = "converted"           # TIFF → H5 hecho (intermedio)
    READY_FOR_INFERENCE = "ready_for_inference"
    PREDICTING = "predicting"         # M4.4
    DONE = "done"                     # M4.4
    FAILED = "failed"


@dataclass
class Job:
    job_id: str
    original_filename: str
    input_type: str                   # "tiff" | "h5"
    status: JobStatus
    created_at: float
    updated_at: float
    error: str | None = None
    extra: dict = field(default_factory=dict)

    @property
    def short_id(self) -> str:
        return self.job_id[:8]

    @property
    def job_dir(self) -> Path:
        return QUEUE_ROOT / self.job_id

    @property
    def raw_path(self) -> Path:
        ext = ".tif" if self.input_type == "tiff" else ".h5"
        return self.job_dir / f"raw{ext}"

    @property
    def h5_path(self) -> Path:
        """Path del H5 listo para inferencia (igual a raw_path si input_type=h5)."""
        return self.job_dir / "input.h5"

    @property
    def result_path(self) -> Path:
        """Path del JSON con probabilidades + meta tras la inferencia."""
        return self.job_dir / "result.json"

    @property
    def attention_path(self) -> Path:
        """Path del .npy con pesos medios de atención (uno por parche)."""
        return self.job_dir / "attention.npy"

    @property
    def patch_eval_path(self) -> Path:
        """Path del .npz con GT + predicciones por parche (solo si hay GT)."""
        return self.job_dir / "patch_eval.npz"

    @property
    def features_path(self) -> Path:
        """Path del .npy con features 512-d por parche (post-ReLU del head F4).
        Persistido para abaratar futuros fine-tunes del head sin reforwardear F4."""
        return self.job_dir / "features.npy"

    @property
    def dzi_path(self) -> Path:
        """Path del fichero DZI principal (XML) generado por pyvips a partir
        del TIFF para el visor OpenSeadragon. Solo existe si input_type=tiff
        — los H5 ya parcheados no tienen TIFF original que tilear."""
        return self.job_dir / "slide.dzi"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Job":
        d = dict(d)
        d["status"] = JobStatus(d["status"])
        return cls(**d)


class JobManager:
    """Persistencia + lectura de la cola.

    Thread-safe: un único lock para serializar enqueue/update/list. Las
    transiciones de estado las hace el worker; Streamlit solo encola y lee.
    """

    def __init__(self, root: Path = QUEUE_ROOT):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Escritura
    # ------------------------------------------------------------------

    def enqueue(
        self,
        file_obj: IO[bytes],
        original_filename: str,
        slide_gt: str | None = None,
    ) -> Job:
        ext = Path(original_filename).suffix.lower()
        if ext in TIFF_EXTS:
            input_type = "tiff"
        elif ext in H5_EXTS:
            input_type = "h5"
        else:
            raise ValueError(
                f"Extensión no soportada: '{ext}'. Acepto {sorted(TIFF_EXTS | H5_EXTS)}"
            )

        now = time.time()
        extra: dict = {}
        if slide_gt is not None:
            extra["slide_gt"] = slide_gt
        job = Job(
            job_id=str(uuid.uuid4()),
            original_filename=original_filename,
            input_type=input_type,
            status=JobStatus.QUEUED,
            created_at=now,
            updated_at=now,
            extra=extra,
        )

        with self._lock:
            job.job_dir.mkdir(parents=True, exist_ok=False)
            with open(job.raw_path, "wb") as f:
                shutil.copyfileobj(file_obj, f)
            self._write_meta(job)

        logger.info(
            "Encolado job %s (%s, %.1f MB)",
            job.short_id, original_filename, job.raw_path.stat().st_size / 1e6,
        )
        return job

    def update_status(
        self,
        job_id: str,
        status: JobStatus,
        error: str | None = None,
        extra: dict | None = None,
    ) -> Job:
        with self._lock:
            job = self._load_unlocked(job_id)
            job.status = status
            job.updated_at = time.time()
            if error is not None:
                job.error = error
            if extra:
                job.extra.update(extra)
            self._write_meta(job)
        logger.info("Job %s → %s", job.short_id, status.value)
        return job

    def update_extra(self, job_id: str, **kv) -> Job:
        """Actualiza job.extra sin tocar el status. Valor None → borra la clave."""
        with self._lock:
            job = self._load_unlocked(job_id)
            for k, v in kv.items():
                if v is None:
                    job.extra.pop(k, None)
                else:
                    job.extra[k] = v
            job.updated_at = time.time()
            self._write_meta(job)
        return job

    def delete(self, job_id: str) -> None:
        with self._lock:
            job_dir = self.root / job_id
            if job_dir.exists():
                shutil.rmtree(job_dir, ignore_errors=True)
                logger.info("Borrado job %s", job_id[:8])

    def prune(self, max_age_hours: float = 24.0) -> dict:
        """Limpieza periódica de la cola (M4.6 — TTL).

        Dos pasadas:
        1. Borra job_dirs en DONE/FAILED con `updated_at` más viejo que
           `max_age_hours`. Los jobs activos (QUEUED/PROCESSING/...) nunca
           se tocan, sin importar su edad — son responsabilidad del worker.
        2. Borra cualquier `raw.*` huérfano en jobs DONE/FAILED (no debería
           existir post-M4.6 porque _do_preprocess hace unlink, pero esto
           cubre estados heredados o casos donde el unlink falló).

        Devuelve dict con contadores para logging.
        """
        now = time.time()
        cutoff = now - max_age_hours * 3600
        pruned_dirs = 0
        pruned_raws = 0
        with self._lock:
            for job in self._list_unlocked():
                terminal = job.status in (JobStatus.DONE, JobStatus.FAILED)
                if not terminal:
                    continue
                if job.updated_at < cutoff:
                    shutil.rmtree(job.job_dir, ignore_errors=True)
                    pruned_dirs += 1
                    continue
                # job terminal todavía dentro del TTL → al menos asegurar
                # que no haya raw residual
                if job.raw_path.exists():
                    try:
                        job.raw_path.unlink()
                        pruned_raws += 1
                    except OSError:
                        logger.warning(
                            "prune: no pude borrar raw huérfano en %s",
                            job.short_id,
                        )
        return {"pruned_dirs": pruned_dirs, "pruned_raws": pruned_raws}

    # ------------------------------------------------------------------
    # Lectura
    # ------------------------------------------------------------------

    def list_jobs(self) -> list[Job]:
        """Lista todos los jobs (incluyendo terminados), ordenados por created_at desc."""
        jobs: list[Job] = []
        for d in self.root.iterdir():
            if not d.is_dir():
                continue
            meta_path = d / META_FILENAME
            if not meta_path.exists():
                continue
            try:
                jobs.append(self._load_unlocked(d.name))
            except Exception as e:
                logger.warning("No pude leer meta de %s: %s", d.name, e)
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return jobs

    def get(self, job_id: str) -> Job:
        with self._lock:
            return self._load_unlocked(job_id)

    def pop_next_pending(self) -> Job | None:
        """Alias retrocompatible de pop_next_queued()."""
        return self.pop_next_queued()

    def pop_next_queued(self) -> Job | None:
        """Devuelve el job más antiguo en QUEUED y lo marca PROCESSING (atómico)."""
        return self._pop_with_transition(JobStatus.QUEUED, JobStatus.PROCESSING)

    def pop_next_ready_for_inference(self) -> Job | None:
        """Devuelve el job más antiguo en READY_FOR_INFERENCE y lo marca PREDICTING."""
        return self._pop_with_transition(
            JobStatus.READY_FOR_INFERENCE, JobStatus.PREDICTING,
        )

    def _pop_with_transition(self, from_state: JobStatus, to_state: JobStatus) -> Job | None:
        with self._lock:
            candidates = [j for j in self._list_unlocked() if j.status == from_state]
            if not candidates:
                return None
            candidates.sort(key=lambda j: j.created_at)
            job = candidates[0]
            job.status = to_state
            job.updated_at = time.time()
            self._write_meta(job)
        logger.info(
            "Worker tomó job %s (%s) %s → %s",
            job.short_id, job.original_filename, from_state.value, to_state.value,
        )
        return job

    # ------------------------------------------------------------------
    # Helpers internos (asumen lock ya tomado o irrelevante)
    # ------------------------------------------------------------------

    def _list_unlocked(self) -> list[Job]:
        jobs: list[Job] = []
        for d in self.root.iterdir():
            if not d.is_dir():
                continue
            if not (d / META_FILENAME).exists():
                continue
            try:
                jobs.append(self._load_unlocked(d.name))
            except Exception:
                continue
        return jobs

    def _load_unlocked(self, job_id: str) -> Job:
        meta_path = self.root / job_id / META_FILENAME
        with open(meta_path) as f:
            return Job.from_dict(json.load(f))

    def _write_meta(self, job: Job) -> None:
        meta_path = job.job_dir / META_FILENAME
        tmp = meta_path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(job.to_dict(), f, indent=2)
        tmp.replace(meta_path)


# Singleton del proceso (Streamlit reusa el módulo entre reruns)
_manager: JobManager | None = None
_manager_lock = threading.Lock()


def get_manager() -> JobManager:
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = JobManager()
        return _manager
