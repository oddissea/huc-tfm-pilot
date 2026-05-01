"""Vista detallada de un portaobjetos tras la inferencia.

Muestra:
- Barras de probabilidades por clase con error bars (std del ensemble)
- Gauge de confianza (max prob)
- Top-K parches con mayor atención del AttnMIL
- Scatter de las posiciones de los parches sobre el plano del slide,
  coloreadas por atención

Las funciones leen `result.json`, `attention.npy` y el `input.h5` que
deja el worker en `/tmp/queue/<uuid>/`.
"""

from __future__ import annotations

import base64
import io
import json
from typing import TYPE_CHECKING

import cv2
import h5py
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

if TYPE_CHECKING:
    from src.jobs.manager import Job

CLASS_NAMES = ("ADE", "NOR", "CAR")

# Colores consistentes con las figuras del TFM (sesión #45):
# CAR azul, ADE naranja. NOR verde para distinguirlo.
CLASS_COLORS = {
    "ADE": "#ff7f0e",   # naranja
    "NOR": "#2ca02c",   # verde
    "CAR": "#1f77b4",   # azul
}

# Mismos colores en formato RGB (0-1) para el overlay de atención
CLASS_COLORS_RGB = {
    "ADE": (1.00, 0.50, 0.00),
    "NOR": (0.18, 0.80, 0.20),
    "CAR": (0.00, 0.40, 1.00),
}


# ---------------------------------------------------------------------------
# Lectura de artefactos del job
# ---------------------------------------------------------------------------

def _load_result(job: "Job") -> dict | None:
    if not job.result_path.exists():
        return None
    with open(job.result_path) as f:
        return json.load(f)


def _load_attention(job: "Job") -> np.ndarray | None:
    if not job.attention_path.exists():
        return None
    return np.load(job.attention_path)


def _load_patch_eval(job: "Job") -> dict | None:
    """Carga el `.npz` con GT + preds patch-level (solo si existe)."""
    if not job.patch_eval_path.exists():
        return None
    npz = np.load(job.patch_eval_path)
    return {k: npz[k] for k in npz.files}


def _load_h5_meta(job: "Job") -> tuple[np.ndarray, np.ndarray] | None:
    """Devuelve (positions (N,2), categories (N,)) del input.h5, o None."""
    if not job.h5_path.exists():
        return None
    with h5py.File(str(job.h5_path), "r") as f:
        positions = f["patch_positions"][:] if "patch_positions" in f else None
        cats = f["patch_categories"][:] if "patch_categories" in f else None
    if positions is None:
        return None
    cats_decoded: np.ndarray
    if cats is not None:
        cats_decoded = np.array([
            c.decode() if isinstance(c, bytes) else c for c in cats
        ])
    else:
        cats_decoded = np.array(["?"] * len(positions))
    return positions, cats_decoded


def _load_top_patches(job: "Job", indices: list[int]) -> list[np.ndarray]:
    """Lee del H5 los parches originales en las posiciones indicadas."""
    if not job.h5_path.exists():
        return []
    with h5py.File(str(job.h5_path), "r") as f:
        ds = f["patches"]
        # patches[:, 0] es el original
        return [np.asarray(ds[i, 0]) for i in indices]


def _patch_to_data_uri(patch_np: np.ndarray) -> str:
    """Convierte un parche (H,W,3) uint8 en un data-URI PNG base64.

    Necesario para esquivar el media manager de Streamlit, que construye
    URLs relativas al servidor y falla con 'not connected to a server!'
    detrás de nginx + WebSocket en algunas combinaciones de cliente/proxy.
    Embebido directamente en <img>, no necesita servidor.
    """
    img = Image.fromarray(patch_np)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _load_all_originals(job: "Job") -> tuple[np.ndarray, int] | None:
    """Lee del H5 todos los parches originales `patches[:, 0]` y devuelve
    (array (N,H,W,3) uint8, patch_size H=W).
    """
    if not job.h5_path.exists():
        return None
    with h5py.File(str(job.h5_path), "r") as f:
        patches = np.asarray(f["patches"][:, 0])
    return patches, int(patches.shape[1])


# ---------------------------------------------------------------------------
# Componentes visuales
# ---------------------------------------------------------------------------

def _probability_bars(probs: list[float], stds: list[float], pred_class: str) -> go.Figure:
    """Barras horizontales de probabilidades con error bars del ensemble."""
    colors = [
        CLASS_COLORS[c] if c == pred_class else "#cccccc"
        for c in CLASS_NAMES
    ]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=probs,
        y=list(CLASS_NAMES),
        orientation="h",
        marker=dict(color=colors),
        error_x=dict(
            type="data", array=stds, visible=True,
            color="#444", thickness=1.5, width=4,
        ),
        text=[f"{p:.1%} ± {s:.1%}" for p, s in zip(probs, stds)],
        textposition="auto",
        hovertemplate="%{y}: %{x:.1%} ± %{customdata:.1%}<extra></extra>",
        customdata=stds,
    ))
    fig.update_layout(
        title="Probabilidades por clase (media ± std del ensemble de 25 modelos)",
        xaxis=dict(range=[0, 1], tickformat=".0%", title=""),
        yaxis=dict(title=""),
        height=260,
        margin=dict(l=10, r=10, t=50, b=20),
        showlegend=False,
    )
    return fig


def _confidence_gauge(max_prob: float, pred_class: str) -> go.Figure:
    """Gauge 0-100 % de la confianza (probabilidad de la clase predicha)."""
    color = CLASS_COLORS[pred_class]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=max_prob * 100,
        number=dict(suffix=" %", font=dict(size=36, color=color)),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1, tickcolor="#888"),
            bar=dict(color=color, thickness=0.55),
            bgcolor="white",
            steps=[
                dict(range=[0, 50], color="#ffe5e5"),     # rojo claro
                dict(range=[50, 75], color="#fff5d8"),    # amarillo claro
                dict(range=[75, 100], color="#e3f4e3"),   # verde claro
            ],
            threshold=dict(line=dict(color="#222", width=3), value=50),
        ),
        title=dict(text=f"Confianza ({pred_class})", font=dict(size=14)),
    ))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=10))
    return fig


def _attention_overlay(
    positions: np.ndarray,
    attention: np.ndarray,
    patches_orig: np.ndarray,
    patch_raw_size: int,
    pred_class: str,
    thumb_size: int = 48,
    opacity: float = 0.85,
) -> np.ndarray:
    """Construye un mosaico de los parches reales con la capa de atención
    superpuesta, coloreada según la clase predicha.

    El estilo replica los overlays del TFM (sesión #45): los parches se
    colocan en su posición de grid original y la atención se pinta como
    una capa transparente del color de la clase (CAR=azul, ADE=naranja,
    NOR=verde) con alpha proporcional a `attention / max(attention)`.

    Devuelve uint8 RGB (n_rows*thumb_size, n_cols*thumb_size, 3).
    """
    pos = np.asarray(positions, dtype=np.int64)
    n = len(pos)
    if n == 0:
        return np.ones((thumb_size, thumb_size, 3), dtype=np.uint8) * 255

    y_min, x_min = int(pos[:, 0].min()), int(pos[:, 1].min())
    rows = (pos[:, 0] - y_min) // patch_raw_size
    cols = (pos[:, 1] - x_min) // patch_raw_size
    n_rows = int(rows.max()) + 1
    n_cols = int(cols.max()) + 1

    s = thumb_size
    canvas = np.full((n_rows * s, n_cols * s, 3), 255, dtype=np.uint8)
    for i in range(n):
        r, c = int(rows[i]), int(cols[i])
        thumb = cv2.resize(patches_orig[i], (s, s), interpolation=cv2.INTER_AREA)
        canvas[r * s:(r + 1) * s, c * s:(c + 1) * s] = thumb

    color = CLASS_COLORS_RGB.get(pred_class, (0.5, 0.5, 0.5))
    a_max = float(attention.max()) or 1.0
    w_norm = attention / a_max

    overlay_rgba = np.zeros((n_rows * s, n_cols * s, 4), dtype=np.float32)
    color_arr = np.array(color, dtype=np.float32)
    for i in range(n):
        r, c = int(rows[i]), int(cols[i])
        a = float(w_norm[i]) * opacity
        overlay_rgba[r * s:(r + 1) * s, c * s:(c + 1) * s, :3] = color_arr
        overlay_rgba[r * s:(r + 1) * s, c * s:(c + 1) * s, 3] = a

    canvas_f = canvas.astype(np.float32) / 255.0
    alpha = overlay_rgba[..., 3:4]
    blended = canvas_f * (1.0 - alpha) + overlay_rgba[..., :3] * alpha
    return (np.clip(blended, 0, 1) * 255).astype(np.uint8)


def _attention_scatter(
    positions: np.ndarray,
    attention: np.ndarray,
    categories: np.ndarray,
) -> go.Figure:
    """Scatter de los parches sobre el plano del slide, coloreado por atención.

    `positions` se asume (N, 2) con (y, x) esquinas. Invertimos Y para que el
    norte del slide quede arriba (consistente con cómo se ven las miniaturas
    de microscopía). Hover incluye el índice `#i` del parche para poder
    cruzar con los thumbnails del top-K.
    """
    if positions.shape[1] >= 2:
        ys, xs = positions[:, 0], positions[:, 1]
    else:
        ys, xs = np.arange(len(attention)), np.zeros(len(attention))

    a_max = float(attention.max()) or 1.0
    rel = attention / a_max  # fracción del máximo del slide

    has_labels = bool((categories != "?").any() and (categories != "XXX").any())
    n = len(attention)
    idx_arr = np.arange(n)

    if has_labels:
        customdata = np.column_stack([idx_arr, attention, rel, categories])
        hover = (
            "#%{customdata[0]}<br>"
            "x=%{x}, y=%{y}<br>"
            "atención=%{customdata[1]:.4f} "
            "(%{customdata[2]:.0%} del máximo)<br>"
            "categoría=%{customdata[3]}<extra></extra>"
        )
    else:
        customdata = np.column_stack([idx_arr, attention, rel])
        hover = (
            "#%{customdata[0]}<br>"
            "x=%{x}, y=%{y}<br>"
            "atención=%{customdata[1]:.4f} "
            "(%{customdata[2]:.0%} del máximo)<extra></extra>"
        )

    fig = go.Figure(go.Scatter(
        x=xs,
        y=-ys,
        mode="markers",
        marker=dict(
            size=11,
            color=attention,
            colorscale="Blues",
            showscale=True,
            colorbar=dict(title="Atención", thickness=12, len=0.7),
            line=dict(width=0.5, color="#444"),
        ),
        customdata=customdata,
        hovertemplate=hover,
    ))
    fig.update_layout(
        title="Mapa de atención del AttnMIL (cada punto = un parche)",
        xaxis=dict(title="x (px en slide)", scaleanchor="y", scaleratio=1, showgrid=False),
        yaxis=dict(title="y (px, invertido)", showgrid=False),
        height=520,
        margin=dict(l=10, r=10, t=50, b=20),
    )
    return fig


def _attention_overlay_figure(
    positions: np.ndarray,
    attention: np.ndarray,
    patches_orig: np.ndarray,
    patch_raw_size: int,
    pred_class: str,
    thumb_size: int = 48,
    opacity: float = 0.85,
) -> go.Figure:
    """Versión Plotly del overlay TFM. Encima del mosaico se superpone una
    capa invisible de scatter centrada en cada parche para dar hover con
    `#índice` y `atención` — así se puede cruzar con los thumbnails del
    top-K. Sin botón de pantalla completa de Streamlit (Plotly trae su
    propia barra de zoom/pan).
    """
    overlay = _attention_overlay(
        positions, attention, patches_orig, patch_raw_size, pred_class,
        thumb_size=thumb_size, opacity=opacity,
    )
    h, w, _ = overlay.shape

    pos = np.asarray(positions, dtype=np.int64)
    n = len(pos)
    y_min, x_min = int(pos[:, 0].min()), int(pos[:, 1].min())
    rows = (pos[:, 0] - y_min) // patch_raw_size
    cols = (pos[:, 1] - x_min) // patch_raw_size

    s = thumb_size
    centers_x = cols * s + s / 2
    centers_y = rows * s + s / 2

    a_max = float(attention.max()) or 1.0
    rel = attention / a_max
    customdata = np.column_stack([np.arange(n), attention, rel])

    fig = go.Figure()
    fig.add_trace(go.Image(z=overlay, hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=centers_x,
        y=centers_y,
        mode="markers",
        marker=dict(size=max(8, s * 0.7), color="rgba(0,0,0,0)"),
        customdata=customdata,
        hovertemplate=(
            "#%{customdata[0]}<br>"
            "atención=%{customdata[1]:.4f} "
            "(%{customdata[2]:.0%} del máximo)<extra></extra>"
        ),
        showlegend=False,
    ))
    fig.update_layout(
        height=min(700, max(320, h)),
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(visible=False, range=[0, w]),
        yaxis=dict(visible=False, range=[h, 0]),
        dragmode="pan",
    )
    return fig


# ---------------------------------------------------------------------------
# Validación patch-level (matriz de confusión + métricas)
# ---------------------------------------------------------------------------

def _confusion_matrix(gt: np.ndarray, pred: np.ndarray, k: int = 3) -> np.ndarray:
    """Matriz de confusión kxk con filas=real, columnas=predicho."""
    cm = np.zeros((k, k), dtype=np.int64)
    for g, p in zip(gt, pred):
        cm[int(g), int(p)] += 1
    return cm


def _per_class_metrics(cm: np.ndarray) -> dict[str, dict[str, float | int]]:
    """Precision / recall / F1 por clase a partir de la matriz de confusión."""
    out: dict[str, dict[str, float | int]] = {}
    for i, name in enumerate(CLASS_NAMES):
        tp = int(cm[i, i])
        fn = int(cm[i, :].sum() - tp)
        fp = int(cm[:, i].sum() - tp)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        support = int(cm[i, :].sum())
        out[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "tp": tp,
        }
    return out


def _confusion_heatmap(cm: np.ndarray, level: str = "parche") -> go.Figure:
    """Heatmap 3x3 de la matriz de confusión.

    El degradado y el % entre paréntesis están normalizados **por columna**
    (perspectiva "predicción"): cada columna suma 100 % y el color se satura
    donde una predicción de clase X realmente corresponde a la clase X. Es
    complementario a la tabla per-class de la derecha, que ofrece la vista
    por fila (recall + precisión + F1).

    `level` se inserta en el título: "parche" (M4.7a) o "slide" (M4.7b).
    """
    col_sums = cm.sum(axis=0, keepdims=True)
    cm_norm = np.where(col_sums > 0, cm / np.maximum(col_sums, 1), 0.0)

    text = [
        [
            f"<b>{cm[i, j]}</b><br>({cm_norm[i, j]:.1%})"
            for j in range(cm.shape[1])
        ]
        for i in range(cm.shape[0])
    ]

    fig = go.Figure(go.Heatmap(
        z=cm_norm,
        x=list(CLASS_NAMES),
        y=list(CLASS_NAMES),
        colorscale="Blues",
        zmin=0, zmax=1,
        text=text,
        texttemplate="%{text}",
        hovertemplate="real=%{y} · predicho=%{x}<br>%{text}<extra></extra>",
        showscale=False,
    ))
    fig.update_layout(
        title=dict(
            text=f"Matriz de confusión a nivel de {level} (filas: real, columnas: predicho)",
            x=0.5,
            xanchor="center",
        ),
        xaxis=dict(title="Predicho", side="bottom", constrain="domain"),
        yaxis=dict(
            title="Real",
            autorange="reversed",
            scaleanchor="x",
            scaleratio=1,
        ),
        height=380,
        margin=dict(l=10, r=10, t=60, b=20),
    )
    return fig


def _patch_predictions_bars(pred_index: np.ndarray) -> go.Figure:
    """Bar chart con la distribución de clases predichas por parche."""
    n = len(pred_index)
    counts = np.bincount(pred_index, minlength=len(CLASS_NAMES))
    pcts = counts / max(n, 1)
    text = [f"<b>{int(c)}</b> ({p:.1%})" for c, p in zip(counts, pcts)]
    colors = [CLASS_COLORS[c] for c in CLASS_NAMES]
    y_max = int(counts.max()) if counts.size else 1
    fig = go.Figure(go.Bar(
        x=list(CLASS_NAMES),
        y=counts,
        marker=dict(color=colors),
        text=text,
        textposition="outside",
        hovertemplate="%{x}: %{y} parches (%{customdata:.1%})<extra></extra>",
        customdata=pcts,
    ))
    fig.update_layout(
        title=f"Distribución de predicciones del clasificador F4 sobre los {n} parches",
        xaxis=dict(title=""),
        yaxis=dict(
            title="parches",
            range=[0, y_max * 1.18],   # holgura para que el texto "outside" no se corte
        ),
        bargap=0.55,                   # barras más estrechas (≈45 % del slot)
        height=320,
        margin=dict(l=10, r=10, t=50, b=20),
        showlegend=False,
    )
    return fig


def _render_patch_predictions(patch_eval: dict) -> None:
    """Sección 'Predicciones por parche' (sin GT). Siempre disponible si el
    worker dejó patch_eval.npz."""
    pred_index = np.asarray(patch_eval.get("pred_index"), dtype=np.int64)
    if pred_index.size == 0:
        return
    st.divider()
    st.subheader("Predicciones a nivel de parche")
    st.caption(
        "Distribución de la salida del clasificador F4 (no del AttnMIL) sobre "
        "cada parche del portaobjetos. Refleja la heterogeneidad interna del "
        "tejido independientemente del veredicto agregado a nivel de slide. "
        "El AttnMIL puede asignar más peso a una minoría de parches y por "
        "eso la predicción slide-level no tiene por qué coincidir con la "
        "clase mayoritaria de las barras."
    )
    st.plotly_chart(_patch_predictions_bars(pred_index), use_container_width=True)


def _render_patch_validation(patch_eval: dict, result: dict) -> None:
    """Sección 'Validación a nivel de parche' bajo el detalle del slide.

    Solo se llama si el H5 trae etiquetas patch-level útiles. Replica las
    cifras del TFM (acc, F1 macro, CAR→NOR, CAR→ADE) sobre los parches de
    este único portaobjetos.
    """
    valid_mask = patch_eval["valid_mask"]
    if not valid_mask.any():
        st.info("El H5 trae etiquetas, pero ninguna corresponde a la tarea ternaria (todas HIP/ART/XXX).")
        return

    gt = patch_eval["gt_index"][valid_mask]
    pred = patch_eval["pred_index"][valid_mask]
    n_valid = int(valid_mask.sum())
    n_excluded = int((~valid_mask).sum())

    cm = _confusion_matrix(gt, pred)
    accuracy = (gt == pred).mean() if n_valid else 0.0
    metrics = _per_class_metrics(cm)
    f1_macro = float(np.mean([m["f1"] for m in metrics.values()]))

    # Tasas críticas (estilo Safety Score del TFM): CAR→NOR y CAR→ADE
    car_idx = CLASS_NAMES.index("CAR")
    nor_idx = CLASS_NAMES.index("NOR")
    ade_idx = CLASS_NAMES.index("ADE")
    car_total = int(cm[car_idx, :].sum())
    car_to_nor = int(cm[car_idx, nor_idx])
    car_to_ade = int(cm[car_idx, ade_idx])
    car_to_nor_rate = car_to_nor / car_total if car_total else 0.0
    car_to_ade_rate = car_to_ade / car_total if car_total else 0.0

    st.divider()
    st.subheader("Validación a nivel de parche")
    st.caption(
        f"Comparación de la predicción del clasificador F4 con la etiqueta "
        f"del H5 sobre **{n_valid} parches** ternarios. "
        + (f"Excluidos {n_excluded} parches HIP/ART/XXX. " if n_excluded else "")
        + "Las cifras replican el estilo de las tablas §5 del TFM."
    )

    cols = st.columns(4)
    cols[0].metric("Accuracy patch-level", f"{accuracy:.1%}")
    cols[1].metric("F1 macro", f"{f1_macro:.3f}")
    cols[2].metric("CAR→NOR", f"{car_to_nor_rate:.1%} ({car_to_nor}/{car_total})" if car_total else "—")
    cols[3].metric("CAR→ADE", f"{car_to_ade_rate:.1%} ({car_to_ade}/{car_total})" if car_total else "—")

    col_cm, col_table = st.columns([3, 2])
    with col_cm:
        st.plotly_chart(_confusion_heatmap(cm, level="parche"), use_container_width=True)
    with col_table:
        st.markdown(
            "<div style='text-align:center;'><strong>Métricas por clase</strong></div>",
            unsafe_allow_html=True,
        )
        rows = []
        for name, m in metrics.items():
            rows.append({
                "Clase": name,
                "Precisión": f"{m['precision']:.1%}",
                "Sensibilidad": f"{m['recall']:.1%} ({m['tp']}/{m['support']})",
                "F1": f"{m['f1']:.3f}",
                "Soporte": m["support"],
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        breakdown = result.get("patch_eval", {}).get("excluded_breakdown", {})
        if breakdown:
            counts = ", ".join(f"{k}={v}" for k, v in breakdown.items())
            st.caption(f"Parches excluidos por categoría: {counts}")


# ---------------------------------------------------------------------------
# Métricas acumuladas slide-level (M4.7b)
# ---------------------------------------------------------------------------

def render_session_metrics(jobs: list) -> None:
    """Sección 'Métricas acumuladas': agrega todos los DONE con GT slide-level
    y construye matriz de confusión 3x3 + métricas per-class.

    La GT se introduce al subir (radio en `app.py`) y se persiste en
    `job.extra['slide_gt']`. Solo se cuentan los jobs con GT y predicción
    válida en {ADE, NOR, CAR}.
    """
    pairs: list[tuple[int, int, str]] = []  # (gt_idx, pred_idx, filename)
    for j in jobs:
        gt = j.extra.get("slide_gt")
        if gt not in CLASS_NAMES:
            continue
        result = _load_result(j)
        if result is None:
            continue
        pred = result.get("predicted_class")
        if pred not in CLASS_NAMES:
            continue
        pairs.append((CLASS_NAMES.index(gt), CLASS_NAMES.index(pred), j.original_filename))

    if not pairs:
        return  # Sin GT, no se muestra nada

    gt_arr = np.array([p[0] for p in pairs])
    pred_arr = np.array([p[1] for p in pairs])
    cm = _confusion_matrix(gt_arr, pred_arr)
    accuracy = (gt_arr == pred_arr).mean()
    metrics = _per_class_metrics(cm)
    f1_macro = float(np.mean([m["f1"] for m in metrics.values()]))

    car_idx = CLASS_NAMES.index("CAR")
    nor_idx = CLASS_NAMES.index("NOR")
    ade_idx = CLASS_NAMES.index("ADE")
    car_total = int(cm[car_idx, :].sum())
    car_to_nor = int(cm[car_idx, nor_idx])
    car_to_ade = int(cm[car_idx, ade_idx])
    car_to_nor_rate = car_to_nor / car_total if car_total else 0.0
    car_to_ade_rate = car_to_ade / car_total if car_total else 0.0

    st.divider()
    st.subheader("Métricas acumuladas (slide-level)")
    st.caption(
        f"Agregado de **{len(pairs)} portaobjetos** con etiqueta GT introducida al subir. "
        "Cada portaobjetos cuenta una vez. Útil para validar el modelo sobre un "
        "lote etiquetado por ti (p. ej. los 91 del cohort §5.9)."
    )

    cols = st.columns(4)
    cols[0].metric("Slides evaluados", f"{len(pairs)}")
    cols[1].metric("Accuracy", f"{accuracy:.1%}")
    cols[2].metric("F1 macro", f"{f1_macro:.3f}")
    cols[3].metric(
        "CAR→NOR",
        f"{car_to_nor_rate:.1%} ({car_to_nor}/{car_total})" if car_total else "—",
    )

    col_cm, col_table = st.columns([3, 2])
    with col_cm:
        st.plotly_chart(_confusion_heatmap(cm, level="slide"), use_container_width=True)
    with col_table:
        st.markdown(
            "<div style='text-align:center;'><strong>Métricas por clase</strong></div>",
            unsafe_allow_html=True,
        )
        rows = []
        for name, m in metrics.items():
            rows.append({
                "Clase": name,
                "Precisión": f"{m['precision']:.1%}",
                "Sensibilidad": f"{m['recall']:.1%} ({m['tp']}/{m['support']})",
                "F1": f"{m['f1']:.3f}",
                "Soporte": m["support"],
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        if car_total:
            st.caption(
                f"CAR→ADE: {car_to_ade_rate:.1%} ({car_to_ade}/{car_total}) · "
                f"CAR→NOR: {car_to_nor_rate:.1%} ({car_to_nor}/{car_total})"
            )

    with st.expander("Detalle por portaobjetos"):
        detail_rows = [
            {
                "Fichero": fname,
                "GT": CLASS_NAMES[g],
                "Predicción": CLASS_NAMES[p],
                "Acierto": "✓" if g == p else "✗",
            }
            for g, p, fname in pairs
        ]
        st.dataframe(pd.DataFrame(detail_rows), hide_index=True, use_container_width=True)


# ---------------------------------------------------------------------------
# Función pública
# ---------------------------------------------------------------------------

def render_slide_detail(job: "Job", top_k: int = 5) -> None:
    """Renderiza la vista detallada de un job en estado DONE."""
    result = _load_result(job)
    if result is None:
        st.warning("No hay resultado para este job (¿aún en proceso?).")
        return

    probs = list(map(float, result["probabilities_mean"]))
    stds = list(map(float, result["probabilities_std"]))
    pred_class = result["predicted_class"]
    max_prob = max(probs)

    # ─── Encabezado: meta del slide ─────────────────────────────────────────
    st.subheader(f"Resultado · {job.original_filename}")
    cols = st.columns(4)
    cols[0].metric("Predicción", pred_class)
    cols[1].metric("Confianza", f"{max_prob:.1%}")
    cols[2].metric("Parches", str(result["n_patches"]))
    cols[3].metric("Tiempo", f"{result['elapsed_seconds']:.2f} s")

    # ─── Aviso clínico sobre la interpretación de la confianza ──────────────
    st.info(
        "**La confianza no es una probabilidad de acierto.** Es la media del "
        "*softmax* del ensemble (25 modelos AttnMIL) en la clase predicha. "
        "Un valor alto indica que los modelos del ensemble coinciden con "
        "*softmax* saturado, **no** que la predicción sea correcta esa "
        "proporción de veces. El *softmax* no está calibrado: interprétalo "
        "como **seguridad relativa del modelo**, no como certeza diagnóstica. "
        "Las barras de error de la sección siguiente miden la dispersión "
        "entre los 25 modelos del ensemble: una *std* alta indica desacuerdo "
        "entre miembros."
    )

    # ─── Probabilidades + gauge ─────────────────────────────────────────────
    col_bars, col_gauge = st.columns([3, 2])
    with col_bars:
        st.plotly_chart(
            _probability_bars(probs, stds, pred_class),
            use_container_width=True,
        )
    with col_gauge:
        st.plotly_chart(
            _confidence_gauge(max_prob, pred_class),
            use_container_width=True,
        )

    # ─── Atención: requiere attention.npy + positions del H5 ────────────────
    attention = _load_attention(job)
    if attention is None:
        st.info("Sin pesos de atención disponibles para este job.")
        return

    h5_meta = _load_h5_meta(job)
    if h5_meta is None:
        st.info("No se pudo leer las posiciones del H5 para el mapa de atención.")
        return
    positions, categories = h5_meta

    # Overlay tipo TFM: mosaico de los parches reales + capa coloreada
    # según clase predicha, alpha proporcional a la atención normalizada.
    # Se renderiza con Plotly para que el hover muestre #índice + atención
    # (cruzable con los thumbnails del top-K) y para evitar el modal de
    # pantalla completa de st.image.
    n_patches = len(attention)
    st.markdown(
        f"**Mapa de atención sobre el slide** "
        f"— color de la clase predicha ({pred_class}), "
        f"intensidad ∝ atención del AttnMIL. Pasa el ratón para "
        "ver el índice del parche."
    )
    with st.spinner(
        f"Generando overlay de atención ({n_patches} parches)…"
    ):
        originals = _load_all_originals(job)
        if originals is not None:
            patches_arr, patch_size = originals
            if len(patches_arr) == len(attention):
                fig_overlay = _attention_overlay_figure(
                    positions, attention, patches_arr, patch_size, pred_class,
                )
            else:
                fig_overlay = None
        else:
            fig_overlay = None
        if fig_overlay is not None:
            st.plotly_chart(
                fig_overlay,
                use_container_width=True,
                config={"displayModeBar": True, "scrollZoom": True},
            )

    # Top-K parches por atención (debajo del overlay para contexto detallado)
    st.markdown(f"**Top {top_k} parches por atención del AttnMIL**")
    k = min(top_k, len(attention))
    top_idx = np.argsort(attention)[-k:][::-1].tolist()
    with st.spinner(f"Cargando top-{k} parches…"):
        top_patches = _load_top_patches(job, top_idx)

    if top_patches:
        cols = st.columns(k)
        for i, (idx, patch) in enumerate(zip(top_idx, top_patches)):
            with cols[i]:
                cat = categories[idx] if idx < len(categories) else "?"
                cat_label = f" · {cat}" if cat not in ("?", "XXX") else ""
                uri = _patch_to_data_uri(patch)
                st.markdown(
                    f'<img src="{uri}" style="width:100%;border-radius:4px;'
                    f'border:1px solid #e0e0e0;">'
                    f'<div style="text-align:center;font-size:0.85rem;'
                    f'color:#555;margin-top:4px;">'
                    f'#{idx} · α={attention[idx]:.4f}{cat_label}</div>',
                    unsafe_allow_html=True,
                )

    # Scatter interactivo (Plotly) en expander para inspección hover
    with st.expander("Mapa interactivo de atención (hover para detalle)"):
        st.plotly_chart(
            _attention_scatter(positions, attention, categories),
            use_container_width=True,
        )

    # Predicciones por parche del clasificador F4: distribución siempre
    # disponible, validación con matriz de confusión solo si el H5 trae GT.
    # Slot fijo con st.empty() para que Streamlit reconcilie limpio al
    # cambiar de slide.
    patch_section_slot = st.empty()
    patch_eval = _load_patch_eval(job)
    if patch_eval is not None:
        with patch_section_slot.container():
            with st.spinner("Cargando predicciones a nivel de parche…"):
                _render_patch_predictions(patch_eval)
                if result.get("has_patch_gt"):
                    _render_patch_validation(patch_eval, result)
