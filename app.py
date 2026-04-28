"""HUC TFM Pilot — Streamlit app.

Estado actual (M4.4): subida múltiple de TIFF/H5, cola persistente en disco
efímero, worker secuencial que convierte TIFF→H5 (real) y ejecuta la
inferencia F4 + ensemble AttnMIL ternario sobre cada portaobjetos. Las
visualizaciones detalladas (Safety Score, overlays de atención) llegan
en M4.5.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime

import pandas as pd
import streamlit as st
import torch

from src.inference.model import CLASS_NAMES
from src.inference.predict import predict_synthetic
from src.inference.runtime import get_models, load_models, models_loaded, try_get_models
from src.jobs import JobStatus, start_worker
from src.jobs.manager import get_manager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

st.set_page_config(page_title="HUC TFM Pilot", page_icon="🩺", layout="wide")
st.title("🩺 HUC TFM Pilot")
st.caption(
    "Demo interactiva del modelo F4 (BiT-M doble canal) + AttnMIL ternario "
    "para clasificación de portaobjetos histopatológicos colorrectales."
)

# Worker daemon (idempotente entre reruns de Streamlit)
start_worker()
manager = get_manager()


# ---------------------------------------------------------------------------
# Sidebar: estado del sistema + carga de modelos
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Estado del sistema")
    cuda_ok = torch.cuda.is_available()
    if cuda_ok:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        st.success(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        st.error("GPU no disponible")
    st.write(f"PyTorch: `{torch.__version__}`")
    if cuda_ok:
        st.write(f"CUDA: `{torch.version.cuda}`")

    st.divider()
    st.header("Modelo")
    if models_loaded():
        st.success("Modelos cargados ✓")
    else:
        st.caption("Pulsa para descargar pesos desde GCS y cargar el ensemble.")
        if st.button("Cargar modelos desde GCS", type="primary"):
            progress = st.progress(0.0, text="Iniciando…")

            def _on_progress(done: int, total: int, msg: str) -> None:
                progress.progress(done / total, text=msg)

            t0 = time.time()
            with st.spinner("Descargando pesos y cargando modelos en GPU…"):
                load_models(progress_cb=_on_progress)
            progress.empty()
            st.success(f"Cargados en {time.time() - t0:.1f} s")
            st.rerun()


# ---------------------------------------------------------------------------
# Subida de portaobjetos
# ---------------------------------------------------------------------------

st.header("Subir portaobjetos")
st.caption(
    "Acepta TIFF (`.tif/.tiff`) o H5 ya parcheado (`.h5/.hdf5`). Los TIFF se "
    "convierten internamente a H5 antes de inferir. Los uploads se procesan "
    "uno tras otro en la cola."
)

if not models_loaded():
    st.warning("Los uploads se encolarán pero no se inferirán hasta que cargues los modelos (barra lateral).")

uploads = st.file_uploader(
    "Arrastra uno o varios ficheros",
    type=["tif", "tiff", "h5", "hdf5"],
    accept_multiple_files=True,
)

processed_ids: set[str] = st.session_state.setdefault("processed_uploads", set())
new_uploads = [u for u in (uploads or []) if u.file_id not in processed_ids]
for up in new_uploads:
    try:
        manager.enqueue(up, up.name)
        processed_ids.add(up.file_id)
    except Exception as e:
        st.error(f"No se pudo encolar `{up.name}`: {e}")
if new_uploads:
    st.success(f"Encolados {len(new_uploads)} fichero(s).")


# ---------------------------------------------------------------------------
# Cola de procesamiento
# ---------------------------------------------------------------------------

st.header("Cola")

STATUS_LABELS = {
    JobStatus.QUEUED: "🕒 En cola",
    JobStatus.PROCESSING: "⚙️ Convirtiendo",
    JobStatus.CONVERTED: "🔄 Convertido",
    JobStatus.READY_FOR_INFERENCE: "📦 Listo para inferir",
    JobStatus.PREDICTING: "🧠 Inferiendo",
    JobStatus.DONE: "✅ Finalizado",
    JobStatus.FAILED: "❌ Falló",
}

ACTIVE_STATES = {
    JobStatus.QUEUED,
    JobStatus.PROCESSING,
    JobStatus.CONVERTED,
    JobStatus.READY_FOR_INFERENCE,
    JobStatus.PREDICTING,
}


def _human_age(ts: float) -> str:
    delta = time.time() - ts
    if delta < 60:
        return f"hace {int(delta)}s"
    if delta < 3600:
        return f"hace {int(delta // 60)}m"
    return f"hace {int(delta // 3600)}h"


def _read_result(job) -> dict | None:
    if not job.result_path.exists():
        return None
    try:
        with open(job.result_path) as f:
            return json.load(f)
    except Exception:
        return None


@st.fragment(run_every=2)
def _render_queue():
    """Tabla de cola que se auto-rerenderiza cada 2 s mientras haya jobs activos.

    Streamlit re-ejecuta solo este fragmento; el resto de la página queda
    quieto. No depende de componentes externos (el wrapper original
    `streamlit-autorefresh` no carga bien tras nginx + BasicAuth).
    """
    jobs_local = manager.list_jobs()
    if not jobs_local:
        st.info("La cola está vacía. Sube algún fichero para empezar.")
        return

    rows = []
    for j in jobs_local:
        result = _read_result(j) if j.status == JobStatus.DONE else None
        if result is not None:
            pred_class = result["predicted_class"]
            conf = max(result["probabilities_mean"])
            pred_str = f"{pred_class} · {conf:.1%}"
            n_patches = result.get("n_patches", "")
            elapsed = result.get("elapsed_seconds")
            elapsed_str = f"{elapsed:.1f} s" if elapsed is not None else ""
        else:
            pred_str = ""
            n_patches = j.extra.get("n_patches", "")
            elapsed_str = ""

        rows.append({
            "ID": j.short_id,
            "Fichero": j.original_filename,
            "Tipo": j.input_type.upper(),
            "Estado": STATUS_LABELS.get(j.status, j.status.value),
            "Parches": n_patches,
            "Predicción": pred_str,
            "Tiempo": elapsed_str,
            "Subido": datetime.fromtimestamp(j.created_at).strftime("%H:%M:%S"),
            "Actualizado": _human_age(j.updated_at),
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, hide_index=True, use_container_width=True)


_render_queue()
jobs = manager.list_jobs()

if jobs:
    failed_jobs = [j for j in jobs if j.status == JobStatus.FAILED]
    if failed_jobs:
        with st.expander(f"Errores ({len(failed_jobs)})"):
            for j in failed_jobs:
                st.markdown(f"**{j.short_id}** — `{j.original_filename}`")
                st.code(j.error or "(sin detalle)")

    col_a, _ = st.columns([1, 5])
    with col_a:
        if st.button("Limpiar cola"):
            for j in jobs:
                manager.delete(j.job_id)
            st.rerun()


# ---------------------------------------------------------------------------
# Diagnóstico (smoke test sintético + GPU benchmark) — colapsado
# ---------------------------------------------------------------------------

with st.expander("Diagnóstico (smoke test + GPU benchmark)", expanded=False):
    st.subheader("Smoke test sintético")
    st.caption(
        "Verificación end-to-end: 50 parches aleatorios → F4 → features 512-d → "
        "AttnMIL ensemble (25 modelos). Sin sentido clínico, solo cableado."
    )

    if not models_loaded():
        st.info("Carga primero los modelos desde la barra lateral.")
    else:
        if st.button("Ejecutar smoke test"):
            f4, ensemble = get_models()
            with st.spinner("Generando parches y prediciendo…"):
                t0 = time.time()
                result = predict_synthetic(f4, ensemble, n_patches=50, mode="ensemble_25")
                dt = time.time() - t0

            st.success(f"OK — {dt * 1000:.0f} ms (50 parches × 25 modelos)")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Probabilidades (media del ensemble)**")
                for i, name in enumerate(CLASS_NAMES):
                    st.metric(
                        label=name,
                        value=f"{result.probabilities_mean[i].item():.3f}",
                        delta=f"± {result.probabilities_std[i].item():.3f}",
                    )
            with col2:
                st.markdown("**Meta**")
                st.write(f"Predicted class: `{result.predicted_class}`")
                st.write(f"N parches: `{result.n_patches}`")
                st.write(f"N modelos: `{result.n_models_used}`")

    st.divider()
    st.subheader("GPU benchmark (matmul 4096×4096)")
    if cuda_ok and st.button("Ejecutar matmul"):
        x = torch.randn(4096, 4096, device="cuda")
        y = torch.randn(4096, 4096, device="cuda")
        torch.cuda.synchronize()
        t0 = time.time()
        z = x @ y
        torch.cuda.synchronize()
        dt = time.time() - t0
        st.success(f"OK — {dt * 1000:.1f} ms. Norma: `{z.norm().item():.3e}`")
