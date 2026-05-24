"""Página de configuración del piloto.

Accesible desde el sidebar de Streamlit (autodescubre pages/ si están
junto a app.py). Está pensada para que Eduardo (operador único + data
steward del HUC) administre el piloto sin tocar shell ni
docker-compose.

Tres secciones (scope mínimo viable, sesión #65):

1. **Retención** — TTL del prune en días. Persistente vía
   src/config/runtime.py.
2. **Archive** — estadísticas read-only (jobs archivados, MB, fechas).
3. **Acciones** — botones destructivos con doble confirmación: borrar
   jobs DONE de la cola y vaciar el archive.

Cositas previstas para iteraciones siguientes (no aquí): export del
archive como .zip descargable, toggle modo debug, versión activa del
modelo (Hito 5).
"""

from __future__ import annotations

import io
import shutil
import time
import zipfile
from datetime import datetime
from pathlib import Path

import streamlit as st

from src.config.runtime import get_ttl_hours, set_ttl_hours
from src.corrections.archive import DEFAULT_ARCHIVE_DIR, archive_stats
from src.jobs import JobStatus
from src.jobs.manager import get_manager


st.set_page_config(
    page_title="DualPath CRC — Configuración",
    page_icon="⚙️",
    layout="wide",
)


@st.cache_data(ttl=60, show_spinner="Comprimiendo archive…")
def _make_archive_zip(archive_dir_str: str, _cache_key: tuple) -> bytes:
    """Comprime archive_dir en .zip in-memory.

    `_cache_key` agrupa stats del archive (n_jobs, total_bytes,
    last_archived_at). Streamlit la usa como parte de la firma de
    cache: cuando cambia (porque hay archive nuevo), regenera; cuando
    es la misma, sirve la versión cacheada (~ms en vez de recompresar).
    """
    archive_dir = Path(archive_dir_str)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(archive_dir.rglob("*")):
            if p.is_file():
                zf.write(p, arcname=str(p.relative_to(archive_dir)))
    return buf.getvalue()

st.title("⚙️ Configuración")
st.caption(
    "Administración del piloto sin tocar línea de comandos. Los cambios "
    "surten efecto en el siguiente ciclo del worker (máx. 5 minutos), "
    "sin necesidad de reiniciar."
)


# ---------------------------------------------------------------------------
# Sección 1 — Retención (TTL del prune)
# ---------------------------------------------------------------------------

st.divider()
st.header("Retención de jobs")

ttl_hours_current = get_ttl_hours()
ttl_days_current = ttl_hours_current / 24.0

st.markdown(
    "Tiempo que un portaobjetos procesado permanece en la cola antes de "
    "ser borrado del disco efímero. Las correcciones del patólogo y los "
    "embeddings (features 512-d) **siempre** se copian al archive antes "
    "de borrar, así que TTL cortos son seguros."
)

col_ttl, col_info = st.columns([1, 2])

with col_ttl:
    ttl_days_new = st.number_input(
        "TTL en días",
        min_value=1.0,
        max_value=30.0,
        value=float(ttl_days_current),
        step=1.0,
        format="%.1f",
        help="Mínimo 1 día, máximo 30. Recomendado: 7 días para HUC.",
    )

with col_info:
    if ttl_days_new < 1:
        st.warning("TTL < 1 día es agresivo: los jobs se borran antes de que termines la jornada.")
    elif ttl_days_new > 14:
        st.info("TTL > 14 días: la cola acumulará portaobjetos. El archive tiene los datos críticos de todas formas.")
    st.metric("TTL actual", f"{ttl_days_current:.1f} días", f"{ttl_hours_current:.0f} h")

if ttl_days_new != ttl_days_current:
    if st.button("Guardar nuevo TTL", type="primary"):
        set_ttl_hours(ttl_days_new * 24.0)
        st.success(f"TTL actualizado a {ttl_days_new:.1f} días. Surte efecto en el próximo ciclo del worker.")
        time.sleep(1.5)
        st.rerun()


# ---------------------------------------------------------------------------
# Sección 2 — Estado del archive (read-only)
# ---------------------------------------------------------------------------

st.divider()
st.header("Estado del archive")

stats = archive_stats(DEFAULT_ARCHIVE_DIR)

if not stats["exists"] or stats["n_jobs"] == 0:
    st.info(
        f"Archive vacío. Ruta: `{stats['archive_dir']}`. "
        "Aparecerán entradas cuando empieces a corregir portaobjetos y "
        "el worker dispare el prune."
    )
else:
    col_a, col_b, col_c, col_d, col_e = st.columns(5)
    col_a.metric("Jobs archivados", stats["n_jobs"])
    col_b.metric("Con features 512-d", stats["n_jobs_with_features"])
    col_c.metric(
        "Con patch_eval",
        stats.get("n_jobs_with_patch_eval", 0),
        help=(
            "Jobs con patch_eval.npz archivado (predicciones del modelo "
            "por parche). Necesario para reentrenamiento Hito 2; jobs "
            "legacy archivados antes pueden no tenerlo."
        ),
    )
    col_d.metric("Correcciones totales", stats["n_corrections_total"])
    col_e.metric("Tamaño en disco", f"{stats['total_bytes'] / 1024 ** 2:.1f} MB")

    col_e, col_f = st.columns(2)
    with col_e:
        st.metric(
            "Último archivado",
            datetime.fromtimestamp(stats["last_archived_at"]).strftime("%Y-%m-%d %H:%M"),
        )
    with col_f:
        st.metric(
            "Más antiguo",
            datetime.fromtimestamp(stats["oldest_archived_at"]).strftime("%Y-%m-%d %H:%M"),
        )

    st.caption(
        f"Ruta en disco del host: `{stats['archive_dir']}`. Para llevar "
        "las correcciones al entorno de reentrenamiento, copia esta "
        "carpeta vía USB cifrado, rsync, o descarga el .zip de abajo."
    )

    # Export .zip descargable. Cache invalida cuando el archive cambia.
    zip_bytes = _make_archive_zip(
        stats["archive_dir"],
        (stats["n_jobs"], stats["total_bytes"], stats["last_archived_at"]),
    )
    st.download_button(
        label=f"📦 Descargar archive completo ({len(zip_bytes) / 1024 ** 2:.1f} MB .zip)",
        data=zip_bytes,
        file_name=f"dualpath-crc-archive-{datetime.now().strftime('%Y%m%d-%H%M')}.zip",
        mime="application/zip",
        help=(
            "Comprime el contenido completo del archive (corrections.jsonl "
            "+ features.npy + meta.json de cada job) en un único .zip. "
            "Útil cuando la transferencia es por VPN/mail en vez de USB "
            "físico. La compresión se cachea 60 s; si subes nuevas "
            "correcciones, el zip se regenera automáticamente."
        ),
    )


# ---------------------------------------------------------------------------
# Sección 3 — Acciones (con doble confirmación)
# ---------------------------------------------------------------------------

st.divider()
st.header("Acciones")
st.caption(
    "Operaciones destructivas con doble confirmación. La caja de selección "
    "actúa como confirmación previa al botón."
)

manager = get_manager()


# --- Acción 1: borrar jobs DONE de la cola -----------------------------------

st.subheader("Limpiar cola: borrar jobs DONE/FAILED")

jobs = manager.list_jobs()
n_done = sum(1 for j in jobs if j.status in (JobStatus.DONE, JobStatus.FAILED))

st.markdown(
    f"Hay **{n_done}** job(s) en estado DONE/FAILED en la cola. Borrarlos "
    "ejecuta `prune(0)`: archiva correcciones + features primero y luego "
    "elimina los job_dirs del disco efímero."
)

confirm_prune = st.checkbox(
    "Sí, entiendo que se archivará y borrará lo no activo",
    key="confirm_prune_done",
)
if st.button("Ejecutar prune ahora", type="secondary", disabled=not confirm_prune or n_done == 0):
    summary = manager.prune(max_age_hours=0.0)
    st.success(
        f"prune(0) ejecutado: borrados {summary['pruned_dirs']} job_dirs, "
        f"archivadas {summary['archived_corr']} correcciones "
        f"({summary['archived_features']} con features). "
        f"{summary['archive_errors']} errores."
    )
    # Nota: Streamlit prohíbe modificar session_state[key] tras instanciar
    # el widget con ese key (StreamlitAPIException). El checkbox de
    # confirmación queda marcado tras el rerun; el usuario lo desmarca
    # manualmente la próxima vez. Si conviene auto-reset, refactorizar
    # con on_click callback.
    time.sleep(2.0)
    st.rerun()


# --- Acción 2: vaciar el archive --------------------------------------------

st.subheader("Vaciar archive completo")

st.markdown(
    "Borra **todo** el contenido del archive (correcciones + features de "
    "todos los jobs ya procesados). Hazlo SOLO después de haber recogido "
    "una tanda completa para reentrenamiento. **Irreversible**."
)

confirm_archive_1 = st.checkbox(
    "Sí, he recogido las correcciones que quiero conservar",
    key="confirm_archive_1",
)
confirm_archive_2 = st.checkbox(
    "Sí, entiendo que este borrado es irreversible",
    key="confirm_archive_2",
)
disabled = not (confirm_archive_1 and confirm_archive_2 and stats["n_jobs"] > 0)
if st.button("Vaciar archive AHORA", type="primary", disabled=disabled):
    # Borrado seguro: iteramos subdirs en lugar de rmtree del root, por si
    # alguien ha bind-montado algo más en /var/archive.
    archive_dir = Path(stats["archive_dir"])
    deleted = 0
    for sub in archive_dir.iterdir():
        if sub.is_dir() and not sub.name.startswith("."):
            shutil.rmtree(sub, ignore_errors=True)
            deleted += 1
    st.success(f"Archive vaciado: borrados {deleted} subdirectorios.")
    # Mismo caveat que en "Ejecutar prune ahora": Streamlit no permite
    # resetear los checkboxes tras instanciar; quedan marcados hasta que
    # el usuario los desmarque a mano.
    time.sleep(2.0)
    st.rerun()


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.divider()
st.caption(
    "Más opciones próximamente: export del archive como .zip descargable, "
    "modo debug, versión activa del modelo. Sugerencias bienvenidas."
)
