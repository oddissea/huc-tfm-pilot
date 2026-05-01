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

import numpy as np

from src.inference.h5_loader import load_patches_from_h5
from src.inference.model import CLASS_NAMES
from src.inference.predict import predict_slide
from src.inference.runtime import try_get_models
from src.preprocessing import convert_tiff_to_h5

from .manager import Job, JobManager, JobStatus, get_manager

logger = logging.getLogger(__name__)

POLL_INTERVAL = 1.0

# Mapeo de etiquetas patch-level del H5 a clases ternarias (ADE/NOR/CAR).
# TUM se renombró a CAR en la memoria del TFM (sesión #36); el código
# interno y los H5 conservan "TUM". HIP (hiperplasia) y ART (artefacto)
# no forman parte de la tarea ternaria → se excluyen del cómputo.
RAW_TO_TERNARY = {"NOR": "NOR", "ADE": "ADE", "TUM": "CAR"}
EXCLUDED_RAW = {"HIP", "ART", "XXX", "?"}


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
            "has_patch_gt": h5.has_patch_gt,
        }

        if result.attention_weights_mean is not None:
            np.save(job.attention_path, result.attention_weights_mean.numpy())

        # Predicciones patch-level del clasificador F4: SIEMPRE se guardan
        # (independientemente de la GT) para que la UI pueda mostrar la
        # distribución de clases predichas por parche aunque el H5 no traiga
        # etiquetas. Si además hay GT (has_patch_gt=True), añadimos los
        # campos cats_raw/cats_ternary/valid_mask/gt_index para activar la
        # sección de validación con matriz de confusión.
        if result.patch_predictions is not None:
            npz_payload: dict = {
                "pred_index": result.patch_predictions.numpy().astype(np.int64),
                "pred_probs": result.patch_probs.numpy().astype(np.float32),
            }
            if h5.has_patch_gt:
                cats_raw = h5.patch_categories                       # (N,) str
                cats_ternary = np.array([
                    RAW_TO_TERNARY.get(c, "EXCLUDED") for c in cats_raw
                ])
                valid_mask = cats_ternary != "EXCLUDED"
                class_to_idx = {c: i for i, c in enumerate(CLASS_NAMES)}
                gt_idx_full = np.array([
                    class_to_idx[c] if c in class_to_idx else -1 for c in cats_ternary
                ], dtype=np.int64)
                npz_payload.update(
                    cats_raw=cats_raw,
                    cats_ternary=cats_ternary,
                    valid_mask=valid_mask,
                    gt_index=gt_idx_full,
                )
                n_valid = int(valid_mask.sum())
                n_excluded = int((~valid_mask).sum())
                result_dict["patch_eval"] = {
                    "n_valid": n_valid,
                    "n_excluded": n_excluded,
                    "excluded_breakdown": {
                        c: int((cats_raw == c).sum())
                        for c in EXCLUDED_RAW
                        if (cats_raw == c).any()
                    },
                }
            np.savez(job.patch_eval_path, **npz_payload)

        with open(job.result_path, "w") as f:
            json.dump(result_dict, f, indent=2)

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
