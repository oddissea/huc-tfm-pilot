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
import re
import time
from datetime import datetime

import pandas as pd
import streamlit as st
import torch

from src.inference.runtime import load_models, models_loaded
from src.jobs import JobStatus, start_worker
from src.jobs.manager import get_manager
from src.viz import render_session_metrics, render_slide_detail


# Detección de GT por nombre de fichero. Regex tolerantes a las dos
# convenciones del HUC:
#   `<id>_<año>_<clase>_<timestamp>.h5`  → `_no_`, `_ad_`, `_ca_`
#   `<id>_<año><clase>_<timestamp>.h5`   → `22no_`, `22ad_`, `22ca_`
#   `<clase>_<id>.h5` (cleaned)          → `ca_`, `ad_`, `no_`
# Se exige que el token de clase NO esté rodeado de letras (sólo dígitos,
# inicio de cadena o subrayado/punto), para no confundir `cML_*` con CAR.
_GT_REGEX = {
    "NOR": re.compile(r"(?<![a-z])(no|nor)(?![a-z])", re.IGNORECASE),
    "ADE": re.compile(r"(?<![a-z])(ad|ade)(?![a-z])", re.IGNORECASE),
    "CAR": re.compile(r"(?<![a-z])(ca|car|tum)(?![a-z])", re.IGNORECASE),
}


def _detect_gt_from_filename(filename: str) -> str | None:
    """Devuelve "NOR" / "ADE" / "CAR" según patrones del nombre, o None."""
    for label, pattern in _GT_REGEX.items():
        if pattern.search(filename):
            return label
    return None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

st.set_page_config(
    page_title="HUC TFM Pilot",
    page_icon="assets/oddissea.png",
    layout="wide",
)
st.logo("assets/oddissea.png", size="large")

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
        # st.empty() permite reemplazar el botón por la versión disabled
        # "⏳ Cargando…" mientras dura la operación (~25 s). Tras la carga,
        # st.rerun() refresca el panel a "Modelos cargados ✓" y descarta
        # cualquier elemento residual de este branch.
        btn_slot = st.empty()
        clicked = btn_slot.button(
            "Cargar modelos desde GCS", type="primary", key="btn_load_models",
        )
        if clicked:
            btn_slot.button(
                "⏳ Cargando…",
                type="primary",
                disabled=True,
                key="btn_load_models_loading",
            )
            with st.spinner("Descargando pesos y cargando modelos en GPU…"):
                load_models()
            # Limpiar el slot antes del rerun para que el botón disabled
            # del frame previo no quede visible como elemento stale al
            # cambiar al if branch (st.success "Modelos cargados ✓").
            btn_slot.empty()
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

auto_detect_gt = st.checkbox(
    "Auto-detectar GT del nombre del fichero",
    value=True,
    help="Reconoce los patrones del HUC: `_no_`/`no_` → NOR, `_ad_`/`ad_` → ADE, "
         "`_ca_`/`ca_`/`tum` → CAR. Para cualquier slide cuyo nombre no encaje, "
         "queda sin etiqueta y la puedes asignar manualmente en el editor de "
         "abajo. Util para subir tandas grandes (p. ej. los 91 del cohort §5.9).",
)

uploads = st.file_uploader(
    "Arrastra uno o varios ficheros",
    type=["tif", "tiff", "h5", "hdf5"],
    accept_multiple_files=True,
)

processed_ids: set[str] = st.session_state.setdefault("processed_uploads", set())
new_uploads = [u for u in (uploads or []) if u.file_id not in processed_ids]
detected_summary: dict[str | None, int] = {}
for up in new_uploads:
    slide_gt = _detect_gt_from_filename(up.name) if auto_detect_gt else None
    detected_summary[slide_gt] = detected_summary.get(slide_gt, 0) + 1
    try:
        manager.enqueue(up, up.name, slide_gt=slide_gt)
        processed_ids.add(up.file_id)
    except Exception as e:
        st.error(f"No se pudo encolar `{up.name}`: {e}")
if new_uploads:
    summary = ", ".join(
        f"{c} × {label or 'sin etiqueta'}"
        for label, c in sorted(detected_summary.items(), key=lambda kv: (kv[0] is None, kv[0] or ""))
    )
    st.success(f"Encolados {len(new_uploads)} fichero(s) — {summary}.")


# ---------------------------------------------------------------------------
# Cola de procesamiento (envuelta en un expander para no ocupar pantalla
# cuando el patólogo está mirando un resultado)
# ---------------------------------------------------------------------------

st.subheader("Cola acumulada")

_jobs_now = manager.list_jobs()
_n_active = sum(1 for _j in _jobs_now if _j.status in {
    JobStatus.QUEUED, JobStatus.PROCESSING, JobStatus.CONVERTED,
    JobStatus.READY_FOR_INFERENCE, JobStatus.PREDICTING,
})
_label = f"📋 Cola ({len(_jobs_now)} portaobjetos"
if _n_active:
    _label += f", {_n_active} en curso"
_label += ")"

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

    Si detecta que el número de jobs DONE cambió desde la última vez,
    dispara un rerun completo para que el resto de la página (sección
    "Detalle por portaobjetos") también se re-renderice. Sin esto, los
    nuevos resultados aparecen en la tabla pero no debajo.
    """
    jobs_local = manager.list_jobs()

    # Detectar transiciones a DONE / DZI → rerun completo para refrescar
    # la sección de detalle (vive fuera del fragmento) y la columna 'Visor'
    # de la tabla. dzi_status pasa por 'generating' → 'done' o 'failed'.
    done_signature = (
        sum(1 for j in jobs_local if j.status == JobStatus.DONE),
        sum(1 for j in jobs_local if j.status == JobStatus.FAILED),
        tuple(j.extra.get("dzi_status", "x") for j in jobs_local),
    )
    last_sig = st.session_state.get("queue_done_sig")
    st.session_state["queue_done_sig"] = done_signature
    if last_sig is not None and last_sig != done_signature:
        st.rerun()

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
            conv = j.extra.get("conversion_seconds")
            if elapsed is None:
                elapsed_str = ""
            elif conv is not None:
                elapsed_str = f"{conv:.1f} s + {elapsed:.1f} s"
            else:
                elapsed_str = f"{elapsed:.1f} s"
        else:
            pred_str = ""
            n_patches = j.extra.get("n_patches", "")
            elapsed_str = ""

        # Estado del visor OpenSeadragon (DZI generado async tras el H5)
        dzi_status = j.extra.get("dzi_status", "unknown")
        dzi_emoji = {
            "generating": "⏳",
            "done": "✅",
            "failed": "❌",
        }.get(dzi_status, "—")

        rows.append({
            "ID": j.short_id,
            "Fichero": j.original_filename,
            "Tipo": j.input_type.upper(),
            "GT": j.extra.get("slide_gt", "—"),
            "Estado": STATUS_LABELS.get(j.status, j.status.value),
            "Visor": dzi_emoji,
            "Parches": str(n_patches) if n_patches != "" else "",
            "Predicción": pred_str,
            "Tiempo": elapsed_str,
            "Subido": datetime.fromtimestamp(j.created_at).strftime("%H:%M:%S"),
            "Actualizado": _human_age(j.updated_at),
        })
    df = pd.DataFrame(rows)
    event = st.dataframe(
        df,
        hide_index=True,
        use_container_width=True,
        on_select="rerun",
        selection_mode="multi-row",
        key="queue_table",
    )

    # Acción "eliminar seleccionados" cuando el usuario marca filas. Se
    # gestiona dentro del fragment para que rerunee localmente.
    selected_rows: list[int] = []
    try:
        selected_rows = list(event.selection.rows)   # type: ignore[attr-defined]
    except Exception:
        selected_rows = []

    # Botones de acción debajo de la tabla. La descarga CSV está siempre
    # disponible cuando hay filas; el borrado sólo aparece si hay
    # filas seleccionadas.
    col_dl, col_del = st.columns([1, 5])
    with col_dl:
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Descargar CSV",
            data=csv_bytes,
            file_name=f"cola_huc_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            key="btn_dl_csv",
        )

    if selected_rows:
        n_sel = len(selected_rows)
        with col_del:
            if st.button(
                f"🗑️ Eliminar {n_sel} seleccionado{'s' if n_sel > 1 else ''}",
                key="btn_delete_selected",
                type="primary",
            ):
                for idx in selected_rows:
                    if 0 <= idx < len(jobs_local):
                        manager.delete(jobs_local[idx].job_id)
                st.rerun()


# Expander que envuelve la tabla + sub-expanders (errores, editor GT) +
# botón limpiar. Default: expanded si hay actividad o si está vacía
# (información útil), colapsado si todo está finalizado para no ocupar
# espacio.
with st.expander(_label, expanded=bool(_n_active) or len(_jobs_now) == 0):
    jobs = manager.list_jobs()
    failed_jobs = [j for j in jobs if j.status == JobStatus.FAILED] if jobs else []

    # Streamlit no permite expanders anidados → usamos pestañas para
    # tabla / editor GT / errores. Solo mostramos la pestaña de errores
    # cuando hay alguno (si no, ocupa espacio sin valor).
    tab_labels = ["📊 Tabla", "✏️ Editar GT"]
    if failed_jobs:
        tab_labels.append(f"⚠️ Errores ({len(failed_jobs)})")
    tabs = st.tabs(tab_labels)

    with tabs[0]:
        _render_queue()
        if jobs:
            col_a, _ = st.columns([1, 5])
            with col_a:
                if st.button("Limpiar cola"):
                    for j in jobs:
                        manager.delete(j.job_id)
                    st.rerun()

    with tabs[1]:
        if not jobs:
            st.caption("Sube algún fichero para empezar a editar etiquetas.")
        else:
            st.caption(
                "Corrige (o asigna) la etiqueta GT manualmente. Los cambios se "
                "guardan al pulsar fuera de la celda y se reflejan al instante en "
                "'Métricas acumuladas'."
            )
            editor_df = pd.DataFrame([
                {
                    "Fichero": j.original_filename,
                    "GT": j.extra.get("slide_gt") or "—",
                    "Predicción": j.extra.get("predicted_class", "—"),
                }
                for j in jobs
            ])
            edited = st.data_editor(
                editor_df,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "GT": st.column_config.SelectboxColumn(
                        options=["—", "NOR", "ADE", "CAR"],
                        required=False,
                    ),
                },
                disabled=["Fichero", "Predicción"],
                key="gt_editor",
            )
            for j, new_gt in zip(jobs, edited["GT"]):
                new_val: str | None = None if new_gt == "—" else new_gt
                current = j.extra.get("slide_gt")
                if new_val != current:
                    manager.update_extra(j.job_id, slide_gt=new_val)

    if failed_jobs:
        with tabs[2]:
            for j in failed_jobs:
                st.markdown(f"**{j.short_id}** — `{j.original_filename}`")
                st.code(j.error or "(sin detalle)")


# ---------------------------------------------------------------------------
# Detalle de portaobjetos
# ---------------------------------------------------------------------------

done_jobs = [j for j in jobs if j.status == JobStatus.DONE] if jobs else []
if done_jobs:
    render_session_metrics(done_jobs)

    st.divider()
    st.header("Detalle por portaobjetos")
    options = {f"{j.short_id} · {j.original_filename}": j for j in done_jobs}
    sel = st.selectbox(
        "Selecciona un portaobjetos finalizado",
        options=list(options.keys()),
        index=0,
    )
    if sel:
        render_slide_detail(options[sel])


