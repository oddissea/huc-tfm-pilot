"""HUC TFM Pilot — Streamlit app.

Estado actual: smoke test del container (PyTorch + CUDA + GPU passthrough)
+ carga del modelo F4 y del ensemble AttnMIL ternario desde GCS, con un
test sintético de extremo a extremo.

Próximas iteraciones añadirán: subida de TIFF, conversión a H5, inferencia
sobre WSI real, visualizaciones (Safety Score, overlay de atención).
"""

from __future__ import annotations

import logging
import time

import streamlit as st
import torch

from src.inference.model import CLASS_NAMES, load_attnmil_ensemble, load_f4
from src.inference.predict import predict_synthetic
from src.inference.weights import ensure_weights

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

st.set_page_config(page_title="HUC TFM Pilot", page_icon="🩺", layout="wide")
st.title("🩺 HUC TFM Pilot")
st.caption(
    "Demo interactiva del modelo F4 (BiT-M doble canal) + AttnMIL ternario "
    "para clasificación de portaobjetos histopatológicos colorrectales."
)


# ---------------------------------------------------------------------------
# Sidebar
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
    st.caption(
        "Pulsa para descargar pesos desde GCS y cargar el ensemble en memoria."
    )


# ---------------------------------------------------------------------------
# Cache de modelos (singleton del proceso Streamlit)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_models():
    """Descarga los pesos de GCS y devuelve (F4Bundle, lista de AttnMILBundle)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    paths = ensure_weights()  # F4 + 25 AttnMIL
    f4 = load_f4(paths["f4"], device=device)
    attnmil = load_attnmil_ensemble(paths["attnmil"], device=device)
    return f4, attnmil


# ---------------------------------------------------------------------------
# Carga de modelos (con feedback de progreso)
# ---------------------------------------------------------------------------

if st.sidebar.button("Cargar modelos desde GCS", type="primary"):
    placeholder = st.empty()
    progress = st.progress(0.0, text="Iniciando…")

    # Si los pesos ya están cacheados, ensure_weights es instantáneo. Si no,
    # descarga 1 + 25 ficheros (~151 MB total) en segundos sobre intra-región.
    def _on_progress(done: int, total: int, msg: str) -> None:
        progress.progress(done / total, text=msg)

    t0 = time.time()
    with st.spinner("Descargando pesos y cargando modelos en GPU…"):
        from src.inference.weights import ensure_weights as _ensure
        from src.inference.model import load_attnmil_ensemble as _load_ens
        from src.inference.model import load_f4 as _load_f4

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        paths = _ensure(progress_cb=_on_progress)
        f4_bundle = _load_f4(paths["f4"], device=device)
        ensemble = _load_ens(paths["attnmil"], device=device)
    elapsed = time.time() - t0
    progress.empty()

    st.session_state["models_loaded"] = True
    st.success(f"Modelos cargados en {elapsed:.1f} s — F4 + {len(ensemble)} AttnMIL en `{device}`.")


# ---------------------------------------------------------------------------
# Smoke test sintético
# ---------------------------------------------------------------------------

st.header("Smoke test")
st.caption(
    "Verificación end-to-end: genera 50 parches aleatorios, los pasa por F4 → "
    "features 512-d → AttnMIL ensemble (25 modelos) → softmax. No tiene sentido "
    "clínico, sirve solo para confirmar que el cableado funciona."
)

if not st.session_state.get("models_loaded"):
    st.info("Primero pulsa **‘Cargar modelos desde GCS’** en la barra lateral.")
else:
    if st.button("Ejecutar smoke test sintético"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # get_models() está cacheado tras la primera carga, así que es instantáneo.
        f4, ensemble = get_models()

        with st.spinner("Generando parches sintéticos y prediciendo…"):
            t0 = time.time()
            result = predict_synthetic(f4, ensemble, n_patches=50, mode="ensemble_25")
            dt = time.time() - t0

        st.success(f"OK — {dt * 1000:.0f} ms para 50 parches sintéticos × 25 modelos AttnMIL")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Probabilidades (media del ensemble)")
            for i, name in enumerate(CLASS_NAMES):
                st.metric(
                    label=name,
                    value=f"{result.probabilities_mean[i].item():.3f}",
                    delta=f"± {result.probabilities_std[i].item():.3f}",
                )
        with col2:
            st.subheader("Meta")
            st.write(f"**Predicted class:** `{result.predicted_class}`")
            st.write(f"**N parches:** `{result.n_patches}`")
            st.write(f"**N modelos usados:** `{result.n_models_used}` (ensemble completo)")
            st.caption(
                "Recuerda: sobre datos aleatorios el modelo no tiene información útil; "
                "esperar predicciones cercanas a uniforme (~0.33 cada clase) o sesgos "
                "del entrenamiento (clase mayoritaria)."
            )


# ---------------------------------------------------------------------------
# GPU benchmark (heredado del primer smoke test, sigue siendo útil)
# ---------------------------------------------------------------------------

st.divider()
st.header("GPU benchmark (matmul 4096×4096)")
st.caption("Sanity check de la GPU sin tocar los modelos.")

if cuda_ok and st.button("Ejecutar matmul"):
    x = torch.randn(4096, 4096, device="cuda")
    y = torch.randn(4096, 4096, device="cuda")
    torch.cuda.synchronize()
    t0 = time.time()
    z = x @ y
    torch.cuda.synchronize()
    dt = time.time() - t0
    st.success(f"OK — {dt * 1000:.1f} ms. Norma del resultado: `{z.norm().item():.3e}`")
