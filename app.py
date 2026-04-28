"""HUC TFM Pilot — Streamlit app.

Estado actual (M4.3): subida múltiple de TIFF/H5, cola persistente en disco
efímero (`/tmp/queue/`), worker secuencial que (por ahora) solo gestiona el
paso pre-inferencia (stub TIFF→H5 + copia H5). Cargas reales y predicción
llegan en M4.4.

El smoke test sintético y el benchmark GPU se mantienen como herramientas
de diagnóstico bajo un expander.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime

import pandas as pd
import streamlit as st
import torch
from streamlit_autorefresh import st_autorefresh

from src.inference.model import CLASS_NAMES, load_attnmil_ensemble, load_f4
from src.inference.predict import predict_synthetic
from src.inference.weights import ensure_weights
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
    if st.session_state.get("models_loaded"):
        st.success("Modelos cargados ✓")
    else:
        st.caption("Pulsa para descargar pesos desde GCS y cargar el ensemble.")
        if st.button("Cargar modelos desde GCS", type="primary"):
            progress = st.progress(0.0, text="Iniciando…")

            def _on_progress(done: int, total: int, msg: str) -> None:
                progress.progress(done / total, text=msg)

            t0 = time.time()
            with st.spinner("Descargando pesos y cargando modelos en GPU…"):
                device = torch.device("cuda" if cuda_ok else "cpu")
                paths = ensure_weights(progress_cb=_on_progress)
                load_f4(paths["f4"], device=device)
                load_attnmil_ensemble(paths["attnmil"], device=device)
            progress.empty()
            st.session_state["models_loaded"] = True
            st.success(f"Cargados en {time.time() - t0:.1f} s")
            st.rerun()


# ---------------------------------------------------------------------------
# Cache de modelos (singleton del proceso Streamlit, usado por smoke test)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = ensure_weights()
    f4 = load_f4(paths["f4"], device=device)
    attnmil = load_attnmil_ensemble(paths["attnmil"], device=device)
    return f4, attnmil


# ---------------------------------------------------------------------------
# Subida de portaobjetos
# ---------------------------------------------------------------------------

st.header("Subir portaobjetos")
st.caption(
    "Acepta TIFF (`.tif/.tiff`) o H5 ya parcheado (`.h5/.hdf5`). Los TIFF se "
    "convierten internamente a H5 antes de inferir. Los uploads se procesan "
    "uno tras otro en la cola."
)

uploads = st.file_uploader(
    "Arrastra uno o varios ficheros",
    type=["tif", "tiff", "h5", "hdf5"],
    accept_multiple_files=True,
)

# `uploads` persiste entre reruns: nos quedamos solo con los file_ids nuevos
# para no encolar dos veces el mismo fichero al hacer autorefresh.
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
    JobStatus.PROCESSING: "⚙️ Procesando",
    JobStatus.CONVERTED: "🔄 Convertido (TIFF→H5)",
    JobStatus.READY_FOR_INFERENCE: "📦 Listo para inferir",
    JobStatus.PREDICTING: "🧠 Inferiendo",
    JobStatus.DONE: "✅ Finalizado",
    JobStatus.FAILED: "❌ Falló",
}

ACTIVE_STATES = {JobStatus.QUEUED, JobStatus.PROCESSING, JobStatus.PREDICTING}


def _human_age(ts: float) -> str:
    delta = time.time() - ts
    if delta < 60:
        return f"hace {int(delta)}s"
    if delta < 3600:
        return f"hace {int(delta // 60)}m"
    return f"hace {int(delta // 3600)}h"


jobs = manager.list_jobs()

if not jobs:
    st.info("La cola está vacía. Sube algún fichero para empezar.")
else:
    has_active = any(j.status in ACTIVE_STATES for j in jobs)
    if has_active:
        st_autorefresh(interval=2000, key="queue_refresh")

    rows = []
    for j in jobs:
        rows.append({
            "ID": j.short_id,
            "Fichero": j.original_filename,
            "Tipo": j.input_type.upper(),
            "Estado": STATUS_LABELS.get(j.status, j.status.value),
            "Subido": datetime.fromtimestamp(j.created_at).strftime("%H:%M:%S"),
            "Actualizado": _human_age(j.updated_at),
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, hide_index=True, use_container_width=True)

    failed_jobs = [j for j in jobs if j.status == JobStatus.FAILED]
    if failed_jobs:
        with st.expander(f"Errores ({len(failed_jobs)})"):
            for j in failed_jobs:
                st.markdown(f"**{j.short_id}** — `{j.original_filename}`")
                st.code(j.error or "(sin detalle)")

    col_a, col_b = st.columns([1, 5])
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

    if not st.session_state.get("models_loaded"):
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
