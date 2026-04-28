"""Worker daemon que consume la cola de jobs.

Loop sencillo: cada `POLL_INTERVAL` segundos toma el job más antiguo en
`QUEUED` y lo procesa. M4.3 solo gestiona el paso pre-inferencia:

- Si `input_type == "tiff"`: stub que simula la conversión (M5.1 lo
  reemplazará por el pipeline real TIFF→H5 con extracción de parches).
- Si `input_type == "h5"`: copia/renombra a `input.h5` directamente.

En ambos casos el estado terminal es `READY_FOR_INFERENCE`. M4.4 lanzará
otro worker (o reutilizará éste) para hacer la inferencia F4 + AttnMIL.
"""

from __future__ import annotations

import logging
import shutil
import threading
import time
import traceback

from .manager import Job, JobManager, JobStatus, get_manager

logger = logging.getLogger(__name__)

POLL_INTERVAL = 1.0  # segundos entre comprobaciones de la cola
TIFF_STUB_DELAY = 3.0  # M4.3 solo: simula trabajo de conversión TIFF→H5

_worker_thread: threading.Thread | None = None
_worker_lock = threading.Lock()
_stop_event = threading.Event()


def _process_job(manager: JobManager, job: Job) -> None:
    try:
        if job.input_type == "tiff":
            logger.info("Job %s: simulando TIFF→H5 (stub M4.3)…", job.short_id)
            time.sleep(TIFF_STUB_DELAY)
            # Placeholder: aún no hay h5 real. M5.1 generará uno aquí.
            job.h5_path.touch()
            manager.update_status(
                job.job_id,
                JobStatus.CONVERTED,
                extra={"note": "stub M4.3 — H5 vacío"},
            )
            manager.update_status(job.job_id, JobStatus.READY_FOR_INFERENCE)

        elif job.input_type == "h5":
            shutil.copyfile(job.raw_path, job.h5_path)
            manager.update_status(job.job_id, JobStatus.READY_FOR_INFERENCE)

        else:
            raise ValueError(f"Tipo de input desconocido: {job.input_type}")

    except Exception as e:
        tb = traceback.format_exc()
        logger.exception("Job %s falló", job.short_id)
        manager.update_status(
            job.job_id,
            JobStatus.FAILED,
            error=f"{type(e).__name__}: {e}\n\n{tb}",
        )


def _worker_loop() -> None:
    manager = get_manager()
    logger.info("Worker iniciado (poll=%.1fs)", POLL_INTERVAL)
    while not _stop_event.is_set():
        job = manager.pop_next_pending()
        if job is None:
            time.sleep(POLL_INTERVAL)
            continue
        _process_job(manager, job)
    logger.info("Worker detenido")


def start_worker() -> threading.Thread:
    """Arranca el worker (idempotente entre reruns de Streamlit)."""
    global _worker_thread
    with _worker_lock:
        if _worker_thread is not None and _worker_thread.is_alive():
            return _worker_thread
        _stop_event.clear()
        _worker_thread = threading.Thread(
            target=_worker_loop,
            name="job-worker",
            daemon=True,
        )
        _worker_thread.start()
        return _worker_thread


def stop_worker(timeout: float = 5.0) -> None:
    _stop_event.set()
    if _worker_thread is not None:
        _worker_thread.join(timeout=timeout)
