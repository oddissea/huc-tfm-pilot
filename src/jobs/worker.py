"""Worker daemon que consume la cola de jobs.

Dos transiciones que el worker realiza:

1. **Pre-inferencia** (estado terminal: READY_FOR_INFERENCE)
   - TIFF: convert_tiff_to_h5() → input.h5 con parches dual-stream
   - H5:   copia raw.h5 → input.h5

2. **Inferencia** (estado terminal: DONE)
   - Carga input.h5 (parches 300×300 → resize 224×224)
   - F4 → features 512-d → AttnMIL ensemble (25 modelos)
   - Guarda result.json + attention.npy

Si los modelos no están cargados todavía, los jobs se quedan en
READY_FOR_INFERENCE y el worker reintentará en el siguiente poll. Esto
permite encolar uploads antes de que el patólogo pulse "Cargar modelos".
"""

from __future__ import annotations

import json
import logging
import shutil
import threading
import time
import traceback

from src.inference.h5_loader import load_patches_from_h5
from src.inference.predict import predict_slide
from src.inference.runtime import try_get_models
from src.preprocessing import convert_tiff_to_h5

from .manager import Job, JobManager, JobStatus, get_manager

logger = logging.getLogger(__name__)

POLL_INTERVAL = 1.0


_worker_thread: threading.Thread | None = None
_worker_lock = threading.Lock()
_stop_event = threading.Event()


# ---------------------------------------------------------------------------
# Etapa 1: pre-inferencia
# ---------------------------------------------------------------------------

def _do_preprocess(manager: JobManager, job: Job) -> None:
    try:
        if job.input_type == "tiff":
            t0 = time.time()
            n_patches = convert_tiff_to_h5(job.raw_path, job.h5_path)
            conversion_seconds = time.time() - t0
            manager.update_status(
                job.job_id,
                JobStatus.CONVERTED,
                extra={
                    "n_patches": n_patches,
                    "conversion_seconds": round(conversion_seconds, 2),
                },
            )
            manager.update_status(job.job_id, JobStatus.READY_FOR_INFERENCE)

        elif job.input_type == "h5":
            shutil.copyfile(job.raw_path, job.h5_path)
            manager.update_status(job.job_id, JobStatus.READY_FOR_INFERENCE)

        else:
            raise ValueError(f"Tipo de input desconocido: {job.input_type}")

    except Exception as e:
        logger.exception("Job %s falló en preprocesado", job.short_id)
        manager.update_status(
            job.job_id,
            JobStatus.FAILED,
            error=f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}",
        )


# ---------------------------------------------------------------------------
# Etapa 2: inferencia
# ---------------------------------------------------------------------------

def _do_inference(manager: JobManager, job: Job) -> None:
    models = try_get_models()
    if models is None:
        # No debería pasar (pop solo se llama si models_loaded), pero por seguridad
        logger.warning("Job %s tomado pero modelos no cargados; revierto", job.short_id)
        manager.update_status(job.job_id, JobStatus.READY_FOR_INFERENCE)
        return
    f4, ensemble = models

    try:
        h5 = load_patches_from_h5(job.h5_path)
        t0 = time.time()
        result = predict_slide(
            f4, ensemble,
            patches_orig=h5.patches_orig,
            patches_reb=h5.patches_reb,
            mode="ensemble_25",
            return_attention=True,
        )
        elapsed = time.time() - t0

        result_dict = {
            "predicted_class": result.predicted_class,
            "predicted_index": result.predicted_index,
            "probabilities_mean": result.probabilities_mean.tolist(),
            "probabilities_std": result.probabilities_std.tolist(),
            "n_patches": result.n_patches,
            "n_models_used": result.n_models_used,
            "elapsed_seconds": elapsed,
            "source_image_name": h5.source_image_name,
            "patch_raw_size": h5.raw_size,
        }
        with open(job.result_path, "w") as f:
            json.dump(result_dict, f, indent=2)

        if result.attention_weights_mean is not None:
            import numpy as np
            np.save(job.attention_path, result.attention_weights_mean.numpy())

        manager.update_status(
            job.job_id,
            JobStatus.DONE,
            extra={
                "predicted_class": result.predicted_class,
                "n_patches": result.n_patches,
                "elapsed_seconds": round(elapsed, 2),
            },
        )
        logger.info(
            "Job %s DONE: %s (%d parches, %.1fs)",
            job.short_id, result.predicted_class, result.n_patches, elapsed,
        )

    except Exception as e:
        logger.exception("Job %s falló en inferencia", job.short_id)
        manager.update_status(
            job.job_id,
            JobStatus.FAILED,
            error=f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}",
        )


# ---------------------------------------------------------------------------
# Loop principal
# ---------------------------------------------------------------------------

def _worker_loop() -> None:
    manager = get_manager()
    logger.info("Worker iniciado (poll=%.1fs)", POLL_INTERVAL)

    while not _stop_event.is_set():
        # Prioridad 1: avanzar inferencias si los modelos están listos
        if try_get_models() is not None:
            job = manager.pop_next_ready_for_inference()
            if job is not None:
                _do_inference(manager, job)
                continue

        # Prioridad 2: convertir TIFF/H5 → input.h5
        job = manager.pop_next_queued()
        if job is not None:
            _do_preprocess(manager, job)
            continue

        time.sleep(POLL_INTERVAL)

    logger.info("Worker detenido")


def start_worker() -> threading.Thread:
    global _worker_thread
    with _worker_lock:
        if _worker_thread is not None and _worker_thread.is_alive():
            return _worker_thread
        _stop_event.clear()
        _worker_thread = threading.Thread(
            target=_worker_loop, name="job-worker", daemon=True,
        )
        _worker_thread.start()
        return _worker_thread


def stop_worker(timeout: float = 5.0) -> None:
    _stop_event.set()
    if _worker_thread is not None:
        _worker_thread.join(timeout=timeout)
