"""Worker daemon que consume la cola de jobs.

Dos transiciones que el worker realiza:

1. **Pre-inferencia** (estado terminal: READY_FOR_INFERENCE)
   - TIFF: convert_tiff_to_h5() → input.h5 con parches dual-stream
   - H5:   copia raw.h5 → input.h5

2. **Inferencia** (estado terminal: DONE)
   - Carga input.h5 (parches 300×300 → resize 224×224)
   - F4 → features 512-d → AttnMIL ensemble (5 modelos)
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

from src.config.runtime import get_ttl_hours
from src.inference.h5_loader import load_patches_from_h5
from src.inference.model import CLASS_NAMES
from src.inference.predict import predict_slide
from src.inference.runtime import try_get_models
from src.preprocessing import convert_tiff_to_h5

from .manager import Job, JobManager, JobStatus, get_manager

logger = logging.getLogger(__name__)

POLL_INTERVAL = 1.0

# TTL de la cola (M4.6): cada cuánto invocar manager.prune() y umbral de
# edad para borrar job_dirs DONE/FAILED. Se lee en cada loop del prune
# (no al import) vía get_ttl_hours(), que aplica esta cascada:
#   1. JSON persistente editable por la UI (pages/1_configuracion.py).
#   2. Env var PILOT_TTL_HOURS.
#   3. Default 24.0 horas.
# Cambios desde la UI surten efecto en el siguiente prune (max 5 min)
# sin reiniciar el container. El archive (Hito 1) preserva correcciones
# + features pase lo que pase con el job_dir, así que TTL más cortos
# son seguros.
PRUNE_INTERVAL_SECONDS = 300.0  # 5 min

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

def _generate_dzi_async(manager: JobManager, job: Job) -> None:
    """Genera el DZI en un thread paralelo a la inferencia. La inferencia
    usa GPU + lectura de embeddings; pyvips usa CPU + RAM → no compiten
    por recursos. El meta.json se actualiza con `dzi_status` en cada
    transición (generating/done/failed) para que la UI lo refleje."""
    try:
        from src.preprocessing.dzi import generate_dzi_from_h5
        t = time.time()
        _, (y_min, x_min) = generate_dzi_from_h5(
            job.h5_path, job.job_dir, basename="slide",
        )
        elapsed = round(time.time() - t, 2)
        manager.update_extra(
            job.job_id,
            dzi_status="done",
            has_dzi=True,
            dzi_seconds=elapsed,
            dzi_y_min=int(y_min),
            dzi_x_min=int(x_min),
        )
        logger.info(
            "DZI desde H5 (async) listo para %s en %.1fs (offset y=%d x=%d)",
            job.short_id, elapsed, y_min, x_min,
        )
    except Exception as dzi_e:
        manager.update_extra(job.job_id, dzi_status="failed", dzi_error=str(dzi_e))
        logger.warning(
            "Job %s: DZI gen async falló (%s) — el resto del flujo no se ve afectado",
            job.short_id, dzi_e,
        )


def _do_preprocess(manager: JobManager, job: Job) -> None:
    try:
        extra: dict = {}
        if job.input_type == "tiff":
            t0 = time.time()
            n_patches = convert_tiff_to_h5(job.raw_path, job.h5_path)
            conversion_seconds = time.time() - t0
            extra["n_patches"] = n_patches
            extra["conversion_seconds"] = round(conversion_seconds, 2)
        elif job.input_type == "h5":
            shutil.copyfile(job.raw_path, job.h5_path)
        else:
            raise ValueError(f"Tipo de input desconocido: {job.input_type}")

        # Privacy (M4.6): borrar el fichero crudo en cuanto input.h5
        # está garantizado. Reduce la ventana de exposición a ~segundos.
        # El DZI async lee de input.h5, no de raw, así que esto no
        # interfiere con la generación del visor.
        if job.raw_path.exists():
            job.raw_path.unlink()
            extra["raw_deleted"] = True

        # Marcar que vamos a empezar a generar el DZI (la UI muestra
        # spinner) ANTES de pasar a READY_FOR_INFERENCE.
        extra["dzi_status"] = "generating"
        manager.update_status(job.job_id, JobStatus.CONVERTED, extra=extra)
        manager.update_status(job.job_id, JobStatus.READY_FOR_INFERENCE)

        # Genera DZI en paralelo a la inferencia. La inferencia arranca
        # tras el siguiente poll del worker; el DZI corre como daemon
        # thread sin bloquear ese flujo. Si el DZI tarda más que la
        # inferencia, el usuario verá DONE primero y el visor aparece
        # poco después (la fragment de la cola hace rerun cuando detecta
        # has_dzi en el signature).
        threading.Thread(
            target=_generate_dzi_async,
            args=(manager, job),
            name=f"dzi-{job.short_id}",
            daemon=True,
        ).start()

    except Exception as e:
        logger.exception("Job %s falló en preprocesado", job.short_id)
        # Privacy (M4.6): incluso si la conversión falla, el raw no debe
        # quedarse residual en disco. El usuario lo verá como FAILED y
        # podrá re-subirlo si quiere.
        if job.raw_path.exists():
            try:
                job.raw_path.unlink()
            except OSError:
                logger.warning("Job %s: no pude borrar raw residual", job.short_id)
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
            mode="ensemble",
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

        # Persistir features 512-d para abaratar futuros fine-tunes del head
        # (Fase 0 del flujo human-in-the-loop). ~2 KB por parche.
        if result.features is not None:
            np.save(job.features_path, result.features.numpy().astype(np.float32))

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
    logger.info("Worker iniciado (poll=%.1fs, prune=%.0fs)", POLL_INTERVAL, PRUNE_INTERVAL_SECONDS)
    last_prune = time.time()

    while not _stop_event.is_set():
        # TTL prune periódico — al principio del loop para que ocurra
        # incluso en ciclos con `continue` que se saltan el sleep.
        now = time.time()
        if now - last_prune > PRUNE_INTERVAL_SECONDS:
            try:
                summary = manager.prune(max_age_hours=get_ttl_hours())
                if (
                    summary["pruned_dirs"]
                    or summary["pruned_raws"]
                    or summary.get("archived_corr")
                    or summary.get("archive_errors")
                ):
                    logger.info(
                        "TTL prune: %d job_dirs + %d raws huérfanos eliminados "
                        "(correcciones archivadas: %d, features archivados: %d, "
                        "errores: %d, borrados pospuestos por fallo de archive: %d)",
                        summary["pruned_dirs"],
                        summary["pruned_raws"],
                        summary.get("archived_corr", 0),
                        summary.get("archived_features", 0),
                        summary.get("archive_errors", 0),
                        summary.get("skipped_due_to_archive_err", 0),
                    )
            except Exception:
                logger.exception("TTL prune falló")
            last_prune = now

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
