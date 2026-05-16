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
import os
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import h5py
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from src.corrections import (
    CORRECTION_LABELS,
    latest_slide_label_entry,
    list_corrections,
    list_slide_label_history,
    record_correction,
    record_slide_label,
    summarize_corrections,
)
from src.viz.osd_component import osd_viewer

if TYPE_CHECKING:
    from src.jobs.manager import Job

CLASS_NAMES = ("ADE", "NOR", "CAR")

# Colores consistentes con las figuras del TFM (sesión #45):
# CAR azul, ADE naranja. NOR verde para distinguirlo. Los hex son los
# defaults de matplotlib/plotly que usan las figuras del documento.
CLASS_COLORS = {
    "ADE": "#ff7f0e",   # naranja
    "NOR": "#2ca02c",   # verde
    "CAR": "#1f77b4",   # azul
}


def _hex_to_rgb01(hx: str) -> tuple[float, float, float]:
    h = hx.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))  # type: ignore[return-value]


# Mismos colores en formato RGB (0-1) — derivados de CLASS_COLORS para
# garantizar que cualquier consumidor RGB y cualquier consumidor hex usen
# EXACTAMENTE el mismo color en pantalla. Antes existía una divergencia
# manual (NOR matplotlib #2ca02c=44,160,44 pero CLASS_COLORS_RGB=46,204,51) que
# producía un verde más saturado en el visor que en el resto de la UI.
CLASS_COLORS_RGB = {k: _hex_to_rgb01(v) for k, v in CLASS_COLORS.items()}


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


@st.cache_data(show_spinner=False, max_entries=8)
def _load_all_originals_cached(job_id: str, h5_path_str: str) -> tuple[np.ndarray, int] | None:
    """Versión cacheada por job_id (clicks subsiguientes leen de RAM)."""
    h5_path = Path(h5_path_str)
    if not h5_path.exists():
        return None
    with h5py.File(str(h5_path), "r") as f:
        patches = np.asarray(f["patches"][:, 0])
    return patches, int(patches.shape[1])


def _load_all_originals(job: "Job") -> tuple[np.ndarray, int] | None:
    """Lee del H5 todos los parches originales `patches[:, 0]` y devuelve
    (array (N,H,W,3) uint8, patch_size H=W). Cacheado entre reruns.
    """
    return _load_all_originals_cached(job.job_id, str(job.h5_path))


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
        title="Probabilidades por clase (media ± std del ensemble de 5 modelos)",
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


@st.cache_data(show_spinner=False, max_entries=8, hash_funcs={"_thread.RLock": lambda _: None})
def _attention_overlay_figure_cached(
    job_id: str, thumb_size: int, opacity: float,
    pred_class: str, _patches_arr: np.ndarray, _positions: np.ndarray,
    _attention: np.ndarray, patch_raw_size: int,
) -> go.Figure:
    return _attention_overlay_figure(
        _positions, _attention, _patches_arr, patch_raw_size, pred_class,
        thumb_size=thumb_size, opacity=opacity,
    )


@st.cache_data(show_spinner=False, max_entries=8, hash_funcs={"_thread.RLock": lambda _: None})
def _patch_predictions_overlay_figure_cached(
    job_id: str, thumb_size: int, border_thickness: int,
    _pred_index: np.ndarray, _patches_arr: np.ndarray, _positions: np.ndarray,
    _attention: np.ndarray | None, patch_raw_size: int,
) -> go.Figure:
    return _patch_predictions_overlay_figure(
        _positions, _pred_index, _patches_arr, patch_raw_size,
        attention=_attention, thumb_size=thumb_size, border_thickness=border_thickness,
    )


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


def _render_openseadragon_viewer(
    job: "Job",
    positions: np.ndarray | None = None,
    pred_index: np.ndarray | None = None,
    patch_raw_size: int | None = None,
    attention: np.ndarray | None = None,
    slide_pred_class: str = "CAR",
    show_predictions: bool = True,
    show_attention: bool = False,
    dzi_offset: tuple[int, int] = (0, 0),
    height: int = 620,
    selected_idx: int | None = None,
    view_corrected: bool = False,
    show_selected_borders: bool = True,
    pan_to_selected: bool = False,
    show_out_of_task: bool = True,
    enable_click_capture: bool = False,
) -> dict | None:
    """Si el job tiene `slide.dzi`, embebe un visor OpenSeadragon
    apuntando a `/dzi/<job_id>/slide.dzi`. Si se pasan posiciones +
    predicciones, dibuja un overlay SVG con un rectángulo del color de la
    clase predicha sobre cada parche en sus coordenadas del WSI stitched.

    Devuelve el último click sobre un parche en formato `{"idx", "ts"}` o
    None si nunca se hizo click (o si no hay DZI).

    nginx sirve `queue/<job_id>/` como static bajo `/dzi/<job_id>/`
    (ver `nginx.conf` location /dzi/). El navegador hereda BasicAuth
    same-origin para los tiles.
    """
    if not job.dzi_path.exists():
        return None
    dzi_url = f"/dzi/{job.job_id}/slide.dzi"

    # Cargar correcciones existentes para mostrar marcador distintivo
    # en el visor sobre los parches ya corregidos. Deduplicado last-wins
    # por patch_idx — la última corrección registrada es la que cuenta.
    corrections_by_idx: dict[int, str] = {}
    for c in list_corrections(job.job_dir):
        corrections_by_idx[int(c.patch_idx)] = c.label_corr

    # Construye el JSON con posiciones + clase + atención de cada parche,
    # restando el offset del DZI stitched (las posiciones del H5 están en
    # coords del WSI completo; el DZI stitched empieza en (y_min, x_min)).
    overlays_json = "[]"
    items: list[dict] = []
    y_off, x_off = dzi_offset
    sr, sg, sb = CLASS_COLORS_RGB.get(slide_pred_class, (0.5, 0.5, 0.5))
    # Carga las pred_probs (N, 3) del patch_eval.npz para incluirlas en el
    # hover de cada parche.
    pe = _load_patch_eval(job)
    pred_probs_arr = (
        np.asarray(pe["pred_probs"])
        if (pe is not None and "pred_probs" in pe and pred_index is not None
            and pe["pred_probs"].shape[0] == len(pred_index))
        else None
    )
    # GT por parche. La X de la esquina inferior izquierda aparece en
    # dos casos:
    # 1. GT ternaria válida (NOR/ADE/CAR) y pred != gt: X del color de
    #    la GT real. El modelo se equivocó en la tarea ternaria.
    # 2. GT real es HIP o ART (no-ternaria): X gris. El modelo predijo
    #    una de las 3 clases pero el parche realmente no es de la
    #    tarea ternaria — útil para que el patólogo identifique zonas
    #    de tejido no-tumoral o artefactos sobre los que el modelo está
    #    forzando una predicción.
    gt_index_arr = (
        np.asarray(pe["gt_index"]).astype(np.int64)
        if (pe is not None and "gt_index" in pe)
        else None
    )
    valid_mask_arr = (
        np.asarray(pe["valid_mask"]).astype(bool)
        if (pe is not None and "valid_mask" in pe)
        else None
    )
    cats_raw_arr = (
        np.asarray(pe["cats_raw"]).astype(str)
        if (pe is not None and "cats_raw" in pe)
        else None
    )

    if (positions is not None and pred_index is not None
            and patch_raw_size is not None and len(positions) == len(pred_index)):
        att_arr = np.asarray(attention) if attention is not None else None
        att_max = float(att_arr.max()) if (att_arr is not None and att_arr.size > 0) else 0.0
        for i, (pos, p) in enumerate(zip(positions, pred_index)):
            cls = CLASS_NAMES[int(p)]
            r, g, b = CLASS_COLORS_RGB[cls]
            color = f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"
            item = {
                "x": int(pos[1]) - int(x_off),
                "y": int(pos[0]) - int(y_off),
                "size": int(patch_raw_size),
                "color": color,
                "idx": i,
                "cls": cls,
                "pos": [int(pos[0]), int(pos[1])],
            }
            if att_arr is not None:
                a = float(att_arr[i])
                rel = a / att_max if att_max > 0 else 0.0
                item["att"] = round(a, 4)
                item["att_rel"] = round(rel, 3)
                item["att_fill"] = (
                    f"rgba({int(sr*255)},{int(sg*255)},{int(sb*255)},"
                    f"{round(min(rel * 0.85, 0.85), 3)})"
                )
            if pred_probs_arr is not None:
                pp = pred_probs_arr[i]
                # Probs en orden CLASS_NAMES = (ADE, NOR, CAR)
                item["probs"] = [round(float(v), 3) for v in pp]
            if i in corrections_by_idx:
                item["corrected"] = corrections_by_idx[i]
            # Marca de error en dos casos (descritos arriba).
            if (gt_index_arr is not None and valid_mask_arr is not None
                    and bool(valid_mask_arr[i])):
                # Caso 1: GT ternaria válida y pred != gt.
                gt_idx = int(gt_index_arr[i])
                if 0 <= gt_idx < len(CLASS_NAMES) and gt_idx != int(p):
                    item["error"] = True
                    item["gt_class"] = CLASS_NAMES[gt_idx]
                    item["error_kind"] = "ternary"
            elif cats_raw_arr is not None and i < len(cats_raw_arr):
                # Caso 2: GT no-ternaria (HIP / ART). Cualquier predicción
                # del modelo en clases ternarias es trivialmente "errónea"
                # (no hay opción correcta), pero conviene marcarlo.
                raw = str(cats_raw_arr[i])
                if raw in ("HIP", "ART"):
                    item["error"] = True
                    item["gt_class"] = raw  # para mostrar en hover
                    item["error_kind"] = "non_ternary"
            items.append(item)
        overlays_json = json.dumps(items)

    # Modo custom component: bidireccional (captura clicks → devuelve
    # {idx, ts}). Requiere que el path-based component esté registrado
    # y que nginx tenga X-Frame-Options SAMEORIGIN. Más sofisticado, pero
    # acepta clicks del visor y los puede llevar al panel de correcciones.
    if enable_click_capture:
        return osd_viewer(
            dzi_url=dzi_url,
            overlays=items,
            height=height,
            show_predictions=show_predictions,
            show_attention=show_attention,
            selected_idx=selected_idx,
            view_corrected=view_corrected,
            show_selected_borders=show_selected_borders,
            pan_to_selected=pan_to_selected,
            show_out_of_task=show_out_of_task,
            key=f"osd_{job.job_id}",
        )

    # Modo inline (st.components.v1.html): unidireccional (solo render),
    # sin captura de clicks. Más estable y autocontenido. Es el default
    # para no introducir el riesgo del custom component cuando no hace
    # falta capturar clicks.
    show_pred_js = "true" if show_predictions else "false"
    show_att_js = "true" if show_attention else "false"
    selected_idx_js = "null" if selected_idx is None else str(int(selected_idx))
    view_corrected_js = "true" if view_corrected else "false"

    html = f"""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.1.1/openseadragon.min.css">
    <style>
      .osd-patch {{ box-sizing: border-box; pointer-events: auto; }}
      .osd-patch svg {{ display: block; width: 100%; height: 100%; pointer-events: none; }}
    </style>
    <div id="osd-{job.job_id}" style="width:100%;height:{height}px;background:transparent;border-radius:6px;"></div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.1.1/openseadragon.min.js"></script>
    <script>
      const viewer = OpenSeadragon({{
        id: "osd-{job.job_id}",
        prefixUrl: "https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.1.1/images/",
        tileSources: "{dzi_url}",
        background: "transparent",
        showNavigator: true,
        navigatorPosition: "BOTTOM_RIGHT",
        navigatorHeight: 100,
        navigatorWidth: 130,
        gestureSettingsMouse: {{ scrollToZoom: true, clickToZoom: false }},
        showRotationControl: false,
        animationTime: 0.5,
        immediateRender: true,
        crossOriginPolicy: "Anonymous",
        loadTilesWithAjax: true,
        ajaxWithCredentials: true,
        // Permite hacer zoom hasta 5× más allá de la resolución nativa de
        // los tiles (por defecto 1.1x). El patólogo puede inspeccionar
        // morfología fina sin necesidad del panel del inspector aparte.
        // No tiene coste server-side: solo escala client-side (más allá
        // se ve pixelado, pero útil para verificar bordes y formas).
        maxZoomPixelRatio: 5,
      }});

      const overlays = {overlays_json};
      const SHOW_PRED = {show_pred_js};
      const SHOW_ATT = {show_att_js};
      const SELECTED_IDX = {selected_idx_js};
      const VIEW_CORRECTED = {view_corrected_js};
      const SVG_NS = "http://www.w3.org/2000/svg";
      const CLASSES = ["ADE", "NOR", "CAR"];
      const CORR_COLORS = {{
        "ADE": "rgb(255,127,14)", "NOR": "rgb(44,160,44)", "CAR": "rgb(31,119,180)",
        "HIP": "rgb(136,136,136)", "ART": "rgb(136,136,136)", "EXCLUDED": "rgb(136,136,136)",
      }};

      viewer.addHandler("open", function() {{
        let selectedOverlay = null;
        for (const o of overlays) {{
          const svg = document.createElementNS(SVG_NS, "svg");
          svg.setAttribute("viewBox", "0 0 1 1");
          svg.setAttribute("preserveAspectRatio", "none");

          // Fill (atención): rect completo del color de la clase del slide,
          // alpha proporcional a la atención (incluida en o.att_fill rgba).
          if (SHOW_ATT && o.att_fill) {{
            const fr = document.createElementNS(SVG_NS, "rect");
            fr.setAttribute("x", "0"); fr.setAttribute("y", "0");
            fr.setAttribute("width", "1"); fr.setAttribute("height", "1");
            fr.setAttribute("fill", o.att_fill);
            fr.setAttribute("stroke", "none");
            svg.appendChild(fr);
          }}

          // Stroke (predicción): borde del color de la clase predicha
          // del parche. Si el toggle 'ver correcciones aplicadas' está
          // ON y el parche tiene corrección, usamos el color de la
          // corrección en lugar del de la predicción del modelo.
          if (SHOW_PRED) {{
            let strokeColor = o.color;
            if (VIEW_CORRECTED && o.corrected && CORR_COLORS[o.corrected]) {{
              strokeColor = CORR_COLORS[o.corrected];
            }}
            const sr = document.createElementNS(SVG_NS, "rect");
            sr.setAttribute("x", "0.015"); sr.setAttribute("y", "0.015");
            sr.setAttribute("width", "0.97"); sr.setAttribute("height", "0.97");
            sr.setAttribute("fill", "none");
            sr.setAttribute("stroke", strokeColor);
            sr.setAttribute("stroke-width", "0.06");
            svg.appendChild(sr);
          }}

          // Selección: borde amarillo grueso encima de la predicción si
          // este parche es el seleccionado en el panel de correcciones.
          // Sirve de feedback visual entre el number_input/botón
          // 'siguiente más incierto' del panel y el visor.
          if (SELECTED_IDX !== null && o.idx === SELECTED_IDX) {{
            const sel = document.createElementNS(SVG_NS, "rect");
            sel.setAttribute("x", "0.04"); sel.setAttribute("y", "0.04");
            sel.setAttribute("width", "0.92"); sel.setAttribute("height", "0.92");
            sel.setAttribute("fill", "none");
            sel.setAttribute("stroke", "#fff200");
            sel.setAttribute("stroke-width", "0.10");
            svg.appendChild(sel);
            selectedOverlay = o;
          }}

          // Marcador de "ya corregido": círculo coloreado en la esquina
          // superior derecha del parche con el color de la clase
          // corregida. Permite al patólogo ver de un vistazo qué parches
          // ya ha tocado sin tener que revisar el JSONL.
          // ADE=naranja, NOR=verde, CAR=azul, HIP/ART/EXCLUDED=gris.
          if (o.corrected) {{
            const corrColors = {{
              "ADE": "#ff7f0e", "NOR": "#2ca02c", "CAR": "#1f77b4",
              "HIP": "#888888", "ART": "#888888", "EXCLUDED": "#888888",
            }};
            const dotColor = corrColors[o.corrected] || "#888888";
            const dot = document.createElementNS(SVG_NS, "circle");
            dot.setAttribute("cx", "0.85"); dot.setAttribute("cy", "0.15");
            dot.setAttribute("r", "0.10");
            dot.setAttribute("fill", dotColor);
            dot.setAttribute("stroke", "#fff");
            dot.setAttribute("stroke-width", "0.025");
            svg.appendChild(dot);
          }}

          const div = document.createElement("div");
          div.className = "osd-patch";
          // Hint completo en multilínea — replicaba lo que mostraba el
          // antiguo panel del inspector (ahora retirado): clase, 3 probs
          // del clasificador F4, atención AttnMIL absoluta + relativa,
          // y posición del parche en el slide. Los browsers renderizan
          // los \\n como saltos de línea en el title nativo.
          let lines = [
            `parche #${{o.idx}}`,
            `predicción F4: ${{o.cls}}`,
          ];
          if (o.probs) {{
            const parts = o.probs.map((p, i) => `${{CLASSES[i]}}=${{p.toFixed(3)}}`);
            lines.push(`probs F4: ${{parts.join(" · ")}}`);
          }}
          if (o.att !== undefined) {{
            const pct = (o.att_rel * 100).toFixed(0);
            lines.push(`atención AttnMIL: ${{o.att.toFixed(4)}} (${{pct}}% del máximo)`);
          }}
          if (o.corrected) {{
            lines.push(`✓ corregido como: ${{o.corrected}}`);
          }}
          lines.push(`posición: y=${{o.pos[0]}}, x=${{o.pos[1]}}`);
          div.title = lines.join("\\n");
          div.appendChild(svg);

          viewer.addOverlay({{
            element: div,
            location: viewer.viewport.imageToViewportRectangle(o.x, o.y, o.size, o.size),
          }});
        }}

        // Si hay un parche seleccionado, paneamos la vista hasta él para
        // que el patólogo lo vea inmediatamente sin tener que buscar.
        // Zoom moderado (3x) para ver el parche con detalle pero
        // manteniendo algo de contexto alrededor.
        if (selectedOverlay) {{
          const target = viewer.viewport.imageToViewportCoordinates(
            selectedOverlay.x + selectedOverlay.size / 2,
            selectedOverlay.y + selectedOverlay.size / 2
          );
          viewer.viewport.zoomTo(3.0, target, true);
          viewer.viewport.panTo(target, true);
        }}
      }});
    </script>
    """
    import streamlit.components.v1 as components
    components.html(html, height=height + 20, scrolling=False)
    # st.components.v1.html no devuelve eventos del cliente. Devolvemos un
    # dict vacío para indicar "viewer renderizado, sin click capturable".
    return {}


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

    # Colorscale Oddissea: degradado del crema del logo al marrón oscuro.
    # Conserva la lectura clásica de matrices de confusión (más oscuro =
    # más concentración) en la paleta del proyecto.
    _ODDISSEA_SCALE = [
        [0.0, "#FBF7F0"],
        [0.25, "#E5D6BE"],
        [0.5, "#A8845E"],
        [0.75, "#7A5A3F"],
        [1.0, "#261B17"],
    ]
    # Texto adaptativo: claro sobre celdas oscuras, oscuro sobre claras
    # (umbral 0.55 para que el cambio caiga dentro del marrón medio).
    text = [
        [
            (
                f'<span style="color:'
                f'{"#FBF7F0" if cm_norm[i, j] >= 0.55 else "#261B17"}">'
                f"<b>{cm[i, j]}</b><br>({cm_norm[i, j]:.1%})</span>"
            )
            for j in range(cm.shape[1])
        ]
        for i in range(cm.shape[0])
    ]

    fig = go.Figure(go.Heatmap(
        z=cm_norm,
        x=list(CLASS_NAMES),
        y=list(CLASS_NAMES),
        colorscale=_ODDISSEA_SCALE,
        zmin=0, zmax=1,
        text=text,
        texttemplate="%{text}",
        textfont=dict(size=15),
        hovertemplate="real=%{y} · predicho=%{x}<br>%{text}<extra></extra>",
        showscale=False,
    ))
    fig.update_layout(
        title=dict(
            text=f"Matriz de confusión a nivel de {level} (filas: real, columnas: predicho)",
            x=0.5,
            xanchor="center",
            font=dict(size=15, color="#261B17"),
        ),
        xaxis=dict(
            title=dict(text="Predicho", font=dict(size=14, color="#261B17")),
            side="bottom",
            constrain="domain",
            tickfont=dict(size=14, color="#261B17"),
        ),
        yaxis=dict(
            title=dict(text="Real", font=dict(size=14, color="#261B17")),
            autorange="reversed",
            scaleanchor="x",
            scaleratio=1,
            tickfont=dict(size=14, color="#261B17"),
        ),
        height=440,
        margin=dict(l=10, r=10, t=70, b=20),
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


def _patch_predictions_overlay(
    positions: np.ndarray,
    pred_index: np.ndarray,
    patches_orig: np.ndarray,
    patch_raw_size: int,
    thumb_size: int = 48,
    border_thickness: int = 3,
) -> np.ndarray:
    """Mosaico de los parches reales con un **borde coloreado** por la clase
    predicha de cada parche (verde=NOR, naranja=ADE, azul=CAR). El tejido
    queda visible al 100 %, sin opacity blending, para que el patólogo pueda
    hacer zoom y comprobar el contenido de cada parche.

    Entre parches dejamos un hueco blanco (`gap`) para que los bordes de
    parches adyacentes con clases distintas NO se toquen pixel a pixel y
    cada parche se lea como una "tarjeta" independiente.
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
    gap = max(2, thumb_size // 16)        # 2 px @ 48, 8 px @ 128
    inner = s - 2 * gap

    canvas = np.full((n_rows * s, n_cols * s, 3), 255, dtype=np.uint8)
    for i in range(n):
        r, c = int(rows[i]), int(cols[i])
        # Thumb más pequeño centrado dentro de la celda, dejando `gap` blanco a
        # los lados. El borde se pinta sobre el thumb shrunken.
        thumb = cv2.resize(
            patches_orig[i], (inner, inner), interpolation=cv2.INTER_AREA,
        )
        y0, x0 = r * s + gap, c * s + gap
        canvas[y0:y0 + inner, x0:x0 + inner] = thumb
        cls = CLASS_NAMES[int(pred_index[i])]
        color_rgb = tuple(int(v * 255) for v in CLASS_COLORS_RGB[cls])
        cv2.rectangle(
            canvas,
            (x0, y0),
            (x0 + inner - 1, y0 + inner - 1),
            color_rgb,
            thickness=border_thickness,
        )
    return canvas


def _patch_predictions_overlay_figure(
    positions: np.ndarray,
    pred_index: np.ndarray,
    patches_orig: np.ndarray,
    patch_raw_size: int,
    attention: np.ndarray | None,
    thumb_size: int = 48,
    border_thickness: int = 3,
) -> go.Figure:
    """Plotly versión del overlay de predicciones por parche. El mosaico
    está coloreado solo por borde (intensidad uniforme); `attention` se usa
    en el hover para mostrar el peso del AttnMIL pero no en el visual.
    """
    overlay = _patch_predictions_overlay(
        positions, pred_index, patches_orig, patch_raw_size,
        thumb_size=thumb_size, border_thickness=border_thickness,
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

    pred_class_names = np.array([CLASS_NAMES[int(p)] for p in pred_index])
    if attention is not None and attention.size:
        a_max = float(attention.max()) or 1.0
        rel = attention / a_max
        customdata = np.column_stack([np.arange(n), pred_class_names, attention, rel])
        hover = (
            "#%{customdata[0]}<br>"
            "predicción=%{customdata[1]}<br>"
            "atención=%{customdata[2]:.4f} "
            "(%{customdata[3]:.0%} del máximo)<extra></extra>"
        )
    else:
        customdata = np.column_stack([np.arange(n), pred_class_names])
        hover = (
            "#%{customdata[0]}<br>"
            "predicción=%{customdata[1]}<extra></extra>"
        )

    fig = go.Figure()
    fig.add_trace(go.Image(z=overlay, hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=centers_x,
        y=centers_y,
        mode="markers",
        marker=dict(size=max(8, s * 0.7), color="rgba(0,0,0,0)"),
        customdata=customdata,
        hovertemplate=hover,
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


def _render_patch_predictions(
    patch_eval: dict,
    positions: np.ndarray | None = None,
    patches_arr: np.ndarray | None = None,
    patch_size: int | None = None,
    attention: np.ndarray | None = None,
    job: "Job | None" = None,
    slide_pred_class: str = "CAR",
    show_out_of_task: bool = True,
) -> None:
    """Sección 'Predicciones por parche' (sin GT). Bar chart de distribución
    + visor OpenSeadragon con overlay de bordes coloreados (si hay DZI) +
    inspector individual con la imagen del parche a tamaño nativo.

    `show_out_of_task` controla si el bar chart cuenta los parches HIP/ART
    (cuando hay GT que los identifique). En default True usa todos los
    parches (output crudo del modelo). En False filtra HIP/ART/EXCLUDED
    para mostrar solo los parches del espacio ternario que vio el modelo
    en entrenamiento — coherente con el toggle del visor.
    """
    pred_index_raw = np.asarray(patch_eval.get("pred_index"), dtype=np.int64)
    if pred_index_raw.size == 0:
        return
    valid_mask = patch_eval.get("valid_mask")
    n_total = int(pred_index_raw.size)
    n_excluded = 0
    if not show_out_of_task and valid_mask is not None:
        valid_mask_arr = np.asarray(valid_mask)
        if valid_mask_arr.size == n_total and not valid_mask_arr.all():
            pred_index = pred_index_raw[valid_mask_arr]
            n_excluded = int((~valid_mask_arr).sum())
        else:
            pred_index = pred_index_raw
    else:
        pred_index = pred_index_raw
    job_id = job.job_id if job is not None else None
    st.divider()
    st.subheader("Predicciones a nivel de parche")
    base_caption = (
        "Distribución de la salida del clasificador F4 (no del AttnMIL) sobre "
        "cada parche del portaobjetos. Refleja la heterogeneidad interna del "
        "tejido independientemente del veredicto agregado a nivel de slide. "
        "El AttnMIL puede asignar más peso a una minoría de parches y por "
        "eso la predicción slide-level no tiene por qué coincidir con la "
        "clase mayoritaria de las barras."
    )
    if n_excluded:
        base_caption += (
            f" **Toggle 'fuera de tarea' OFF**: se excluyeron {n_excluded} "
            f"parches HIP/ART/EXCLUDED del conteo "
            f"(quedan {len(pred_index)}/{n_total})."
        )
    st.caption(base_caption)
    st.plotly_chart(_patch_predictions_bars(pred_index), use_container_width=True)

    # Antiguo 'Inspector parche en detalle' eliminado: con el visor
    # OpenSeadragon arriba (maxZoomPixelRatio=4) el patólogo puede hacer
    # zoom hasta 4× la resolución nativa del DZI sobre cualquier parche,
    # y el hover sobre cada parche muestra ya toda la info que daba el
    # inspector (clase, 3 probs F4, atención AttnMIL absoluta + relativa,
    # posición). El bar chart de distribución por clase (arriba) y la
    # matriz de confusión patch-level (abajo si hay GT) cubren el resto.


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
                "Sensibilidad (recall)": f"{m['recall']:.1%} ({m['tp']}/{m['support']})",
                "F1-score": f"{m['f1']:.3f}",
                "Soporte": m["support"],
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        breakdown = result.get("patch_eval", {}).get("excluded_breakdown", {})
        if breakdown:
            counts = ", ".join(f"{k}={v}" for k, v in breakdown.items())
            st.caption(f"Parches excluidos por categoría: {counts}")

        # Cajita explicativa de las métricas, siempre visible (estilo
        # del aviso de confianza del modo Atención). Define cada métrica
        # con su fórmula y el dual nombre español/inglés.
        st.info(
            "**Métricas por clase** (definidas sobre la matriz de confusión 3×3 a "
            "nivel de parche, normalizada por columna en el heatmap):\n\n"
            "- **Precisión** (*precision*) = TP / (TP + FP) — De los parches "
            "que el modelo predijo como esta clase, ¿qué fracción era realmente "
            "esta clase? Mide cuán fiables son las predicciones positivas.\n"
            "- **Sensibilidad** (*recall*, también *true positive rate*) = "
            "TP / (TP + FN) — De los parches realmente de esta clase, ¿qué "
            "fracción captó el modelo? Mide cuántos *positivos reales* no se "
            "escapan.\n"
            "- **F1** (*F1-score*) = 2·precisión·recall / (precisión + recall) — "
            "Media armónica de precisión y recall. Penaliza cuando una de las "
            "dos es muy baja (típico de clases minoritarias).\n"
            "- **Soporte** (*support*) — Número de parches reales de esta clase "
            "(el denominador de la sensibilidad: TP + FN).\n\n"
            "Donde *TP* = true positives (clasificados correctamente), "
            "*FP* = false positives (otra clase que el modelo predijo como "
            "esta), *FN* = false negatives (esta clase real que el modelo "
            "predijo como otra)."
        )


# ---------------------------------------------------------------------------
# Panel de correcciones del patólogo (Fase 0 — captura)
# ---------------------------------------------------------------------------

def _patologo_id() -> str:
    """ID del patólogo para auditoría de correcciones.

    Pilot GCP: viene del env var PILOT_USER (configurable en docker-compose)
    o, por defecto, 'anon'. En el HUC PC se establecerá a 'eduardo'.
    """
    return os.environ.get("PILOT_USER", "Patólogo")


def _model_version() -> str:
    """Versión del bundle de modelos activo. Placeholder hasta que se
    implemente production.json (Nivel 1 del flujo human-in-the-loop)."""
    return os.environ.get("MODEL_VERSION", "head_v1+attnmil_v1")


def _entropy_per_patch(probs: np.ndarray) -> np.ndarray:
    """Entropía Shannon por parche (mayor → más incierto el modelo).
    Útil como ordering del active learning para la corrección."""
    eps = 1e-12
    return -np.sum(probs * np.log(probs + eps), axis=1)


def _render_slide_label_panel(
    job: "Job",
    *,
    pred_class: str | None = None,
    pred_probs: list[float] | None = None,
) -> None:
    """Panel para asignar/cambiar la etiqueta clínica slide-level.

    Estados:
    - **Sin etiqueta** (`slide_gt` no en {ADE, NOR, CAR}): aviso + selector
      para asignarla. Sin etiqueta el slide no entra en las 'Métricas
      acumuladas (slide-level)'.
    - **Con etiqueta**: muestra la actual con su color + indicador de si
      coincide con la predicción del modelo o si la corrige (mostrando el
      valor previo si lo hay). Botón 'Cambiar' abre el selector.

    Persiste en `job.extra['slide_gt']` vía manager.update_extra. Mismo
    campo que el radio del upload, así que las métricas acumuladas la
    recogen automáticamente sin más cambios.

    Además registra cada asignación/cambio en `slide_label_audit.jsonl`
    vía record_slide_label — paralelo al `corrections.jsonl` patch-level
    para auditoría y futuro fine-tune del AttnMIL slide-level.
    """
    from src.jobs.manager import get_manager  # import local para evitar ciclo

    current_gt = job.extra.get("slide_gt")
    has_label = current_gt in CLASS_NAMES
    edit_key = f"slide_label_edit_{job.job_id}"
    is_editing = bool(st.session_state.get(edit_key, False)) or not has_label
    last_audit = latest_slide_label_entry(job.job_dir)

    st.divider()
    if not has_label:
        st.warning(
            "🏷️ **Etiqueta clínica del portaobjetos:** sin asignar — "
            "el slide no entrará en las *Métricas acumuladas (slide-level)* "
            "hasta que le pongas una etiqueta."
        )
    elif not is_editing:
        color = CLASS_COLORS.get(current_gt, "#888")
        col_lbl, col_change, col_del = st.columns([3, 1, 1])
        with col_lbl:
            st.markdown(
                f"🏷️ **Etiqueta clínica:** "
                f"<span style='color:{color};font-weight:600;font-size:1.1em;'>"
                f"{current_gt}</span>",
                unsafe_allow_html=True,
            )
        with col_change:
            if st.button(
                "✏️ Cambiar",
                key=f"slide_label_change_btn_{job.job_id}",
                use_container_width=True,
            ):
                st.session_state[edit_key] = True
                st.rerun()
        with col_del:
            # Detectar si hay correcciones posteriores al upload — para
            # ofrecer "deshacer correcciones (volver al upload)" además
            # del borrado total. Simétrico con el patch-level: 'borrar
            # correcciones' en parche elimina las anotaciones del
            # patólogo; aquí, restaura la etiqueta del upload (que es
            # GT clínica a priori, no respuesta a la predicción).
            history = list_slide_label_history(job.job_dir)
            upload_entry = next(
                (e for e in history if e.action == "upload"), None,
            )
            has_corrections_post_upload = (
                upload_entry is not None
                and any(e is not upload_entry for e in history)
            )

            with st.popover("🗑️ Gestionar", use_container_width=True):
                # Opción A: deshacer correcciones (solo si hay).
                if has_corrections_post_upload:
                    st.markdown(
                        f"**Deshacer correcciones** — restaura la etiqueta "
                        f"del upload (`{upload_entry.label_to}`) y elimina "
                        f"todas las correcciones posteriores del historial."
                    )
                    if st.button(
                        f"↩️ Volver a etiqueta del upload ({upload_entry.label_to})",
                        key=f"slide_label_undo_btn_{job.job_id}",
                        use_container_width=True,
                    ):
                        # Reescribir el JSONL dejando solo la entrada del
                        # upload. Restaurar slide_gt al valor del upload.
                        audit_path = job.job_dir / "slide_label_audit.jsonl"
                        with open(audit_path, "w", encoding="utf-8") as f:
                            f.write(upload_entry.to_jsonl() + "\n")
                        get_manager().update_extra(
                            job.job_id, slide_gt=upload_entry.label_to,
                        )
                        st.success(
                            f"Correcciones deshechas — etiqueta restaurada a "
                            f"**{upload_entry.label_to}** (upload)."
                        )
                        st.rerun()
                    st.divider()

                # Opción B: borrar todo (siempre disponible).
                st.warning(
                    "**Borrar etiqueta y audit log:** elimina `slide_gt` "
                    "y todo el histórico. **Irreversible.** El slide "
                    "volverá a *sin etiqueta* y dejará de contar para "
                    "las métricas acumuladas."
                )
                phrase = "borrar etiqueta clínica"
                confirm = st.text_input(
                    f"Para confirmar, escribe exactamente: `{phrase}`",
                    key=f"slide_label_delete_confirm_{job.job_id}",
                    placeholder=phrase,
                )
                match = (confirm or "").strip().lower() == phrase
                if st.button(
                    "🗑️ Borrar definitivamente",
                    key=f"slide_label_delete_btn_{job.job_id}",
                    disabled=not match,
                    type="primary",
                ):
                    get_manager().update_extra(job.job_id, slide_gt=None)
                    audit_path = job.job_dir / "slide_label_audit.jsonl"
                    if audit_path.exists():
                        audit_path.unlink()
                    st.success("Etiqueta clínica y audit log borrados.")
                    st.rerun()

        # Indicador de coincidencia con la predicción + histórico breve
        # de la última asignación / corrección.
        bits = []
        if pred_class:
            if pred_class == current_gt:
                bits.append(f"✓ Coincide con la predicción del modelo (**{pred_class}**)")
            else:
                bits.append(
                    f"✏️ Distinta de la predicción del modelo "
                    f"(modelo: **{pred_class}**, patólogo: **{current_gt}**)"
                )
        if last_audit is not None:
            # Registros antiguos guardaban "anon" como default; los
            # mostramos como "Patólogo" en captions sin tocar el JSONL.
            patologo = last_audit.patologo_id or "?"
            if patologo == "anon":
                patologo = "Patólogo"
            ts = last_audit.ts.replace("T", " ").replace("Z", " UTC")
            if last_audit.action == "cambiada" and last_audit.label_from:
                bits.append(
                    f"corregida desde **{last_audit.label_from}** "
                    f"por *{patologo}* el {ts}"
                )
            else:
                bits.append(f"asignada por *{patologo}* el {ts}")
        if bits:
            st.caption(" · ".join(bits))
        return

    # Modo edición — selector + guardar (+ cancelar si ya había etiqueta).
    new_label = st.segmented_control(
        "Etiqueta clínica del portaobjetos",
        options=list(CLASS_NAMES),
        default=current_gt if has_label else None,
        key=f"slide_label_input_{job.job_id}",
    )
    cols = st.columns([1, 1, 4])
    with cols[0]:
        if st.button(
            "💾 Guardar",
            key=f"slide_label_save_{job.job_id}",
            disabled=new_label is None,
            type="primary",
            use_container_width=True,
        ):
            # Persistir en meta.json (slide_gt) — alimenta las métricas
            # acumuladas. Después registramos en audit log para histórico
            # y trazabilidad.
            get_manager().update_extra(job.job_id, slide_gt=new_label)
            record_slide_label(
                job.job_dir,
                slide_uuid=job.job_id,
                label_to=new_label,
                label_from=current_gt if has_label else None,
                pred_orig=pred_class,
                pred_orig_probs=pred_probs,
                patologo_id=_patologo_id(),
            )
            st.session_state[edit_key] = False
            st.success(f"Etiqueta clínica guardada: **{new_label}**")
            st.rerun()
    if has_label:
        with cols[1]:
            if st.button(
                "❌ Cancelar",
                key=f"slide_label_cancel_{job.job_id}",
                use_container_width=True,
            ):
                st.session_state[edit_key] = False
                st.rerun()


def _parse_idx_range_text(raw: str, n_patches: int) -> tuple[tuple[int, ...], list[str]]:
    """Parser de la sintaxis de rango usada en el panel de correcciones v4.

    Acepta tokens coma-separados; cada token puede ser un entero (``12``)
    o un rango inclusivo (``20-30``). Tokens vacíos se ignoran (permite
    ``", 12, , 15,"``).

    Args:
        raw: cadena tecleada por el patólogo. Espacios extra ignorados.
        n_patches: tamaño total para filtrar fuera de rango.

    Returns:
        ``(idxs, errors)`` donde ``idxs`` es una tupla ordenada y sin
        duplicados de enteros válidos en ``[0, n_patches-1]``, y
        ``errors`` es una lista de mensajes humanos de tokens inválidos
        (formato no reconocido, fuera de rango, etc.).
    """
    if not raw or not raw.strip():
        return (), []
    seen: set[int] = set()
    errors: list[str] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok and not tok.startswith("-"):
            # Rango inclusivo "a-b".
            try:
                lo_str, hi_str = tok.split("-", 1)
                lo, hi = int(lo_str), int(hi_str)
            except ValueError:
                errors.append(f"Token «{tok}» no es un rango válido.")
                continue
            if lo > hi:
                lo, hi = hi, lo  # tolerante: "30-20" → 20..30
            for v in range(lo, hi + 1):
                if 0 <= v < n_patches:
                    seen.add(v)
                else:
                    errors.append(f"#{v} fuera de rango [0, {n_patches - 1}].")
        else:
            # Entero suelto.
            try:
                v = int(tok)
            except ValueError:
                errors.append(f"Token «{tok}» no es un entero válido.")
                continue
            if 0 <= v < n_patches:
                seen.add(v)
            else:
                errors.append(f"#{v} fuera de rango [0, {n_patches - 1}].")
    return tuple(sorted(seen)), errors


def _render_corrections_panel(
    job: "Job",
    *,
    pred_index: np.ndarray,
    patch_probs: np.ndarray | None,
    attention: np.ndarray,
) -> None:
    """Panel de captura de correcciones del patólogo.

    Layout en expander (default colapsado para no saturar el detalle):
      - selectbox del parche, ordenado por incertidumbre descendente
      - segmented_control con las 6 etiquetas válidas
      - text input para comentario opcional
      - botón guardar
      - resumen 'mis correcciones' con conteo por clase

    Las correcciones se persisten append-only en `<job_dir>/corrections.jsonl`
    vía `src.corrections.record_correction`. No tocan el modelo — sólo
    construyen dataset para futuros fine-tunes (ver §8 de
    `docs/deployment/MEJORA_CON_CORRECCIONES.md`).
    """
    n_patches = int(pred_index.shape[0])
    if n_patches == 0:
        return

    st.divider()
    with st.expander("✏️ Correcciones del patólogo (captura — no modifica el modelo)"):
        st.caption(
            "Las correcciones se guardan en `corrections.jsonl` por portaobjetos "
            "y servirán como dataset patch-level para futuros reentrenamientos. "
            "El modelo activo no cambia con cada corrección."
        )

        # Selector de métrica de incertidumbre. El patólogo elige entre
        # entropía Shannon (más fiel teóricamente, captura indecisión
        # entre las 3 clases) y 1-max(probs) (más intuitivo, "menor
        # max = más incierto"). Default: entropía.
        if patch_probs is not None:
            ranking_mode = st.segmented_control(
                "Métrica de incertidumbre para el ranking",
                options=["Entropía Shannon", "Probabilidad máxima"],
                default="Entropía Shannon",
                key=f"corr_ranking_mode_{job.job_id}",
                help=(
                    "**Entropía Shannon**: mide la indecisión del "
                    "modelo entre las 3 clases simultáneamente. Un parche con "
                    "probs [0.40, 0.35, 0.25] tiene entropía alta (las 3 compiten). "
                    "Uno con [0.50, 0.49, 0.01] tiene entropía baja aunque max sea "
                    "similar (en realidad solo 2 clases compiten).\n\n"
                    "**Probabilidad máxima**: ordena por `max(probs)` ascendente "
                    "(menor max = más incierto). Más intuitivo, ignora qué hacen "
                    "las clases secundarias. Útil cuando una clase ya está casi "
                    "descartada y solo importa la confianza de la dominante."
                ),
            )
            if ranking_mode == "Probabilidad máxima":
                # max ascendente: parches con menor max van primero.
                max_probs = patch_probs.max(axis=1)
                order = np.argsort(max_probs)
            else:
                # entropía descendente: parches con mayor entropía primero.
                entropy = _entropy_per_patch(patch_probs)
                order = np.argsort(-entropy)
            confidences = patch_probs.max(axis=1)
        else:
            order = np.arange(n_patches)
            confidences = np.full(n_patches, np.nan)

        # Estado canónico v4: el «set» de parches seleccionados (tuple
        # ordenada sin duplicados) y el «last» (último idx tocado, ancla
        # del pan/zoom y del caption en single-select). Ambos viven
        # alongside del widget_key del number_input para conservar la UX
        # actual de single-patch. La sincronización se hace en los
        # callbacks on_change del number_input y del text_input de rango.
        widget_key = f"corr_idx_{job.job_id}"
        range_text_key = f"corr_idx_range_{job.job_id}"
        idx_set_key = f"corr_idx_set_{job.job_id}"
        idx_last_key = f"corr_idx_last_{job.job_id}"
        pending_key = f"corr_pending_target_{job.job_id}"
        pending_set_key = f"corr_pending_set_{job.job_id}"
        pending_action_key = f"corr_pending_action_{job.job_id}"
        pending_pan_key = f"corr_pending_pan_{job.job_id}"
        range_errors_key = f"corr_range_errors_{job.job_id}"
        # Patrón "double-key" para la segmented_control de etiqueta: el
        # widget puede ser purgado por Streamlit cuando la función hace
        # early-return (sin parche seleccionado) y por tanto no lo
        # renderiza. Guardamos el valor en una key persistente paralela
        # para usarla como `default=` cuando el widget vuelva a
        # instanciarse, garantizando que la etiqueta seleccionada
        # sobreviva al ciclo "limpio number_input → patch_idx=None →
        # early-return → reescribo number_input".
        label_key_widget = f"corr_label_{job.job_id}"
        label_key_persist = f"corr_label_persist_{job.job_id}"
        comment_key = f"corr_comment_{job.job_id}"
        if (
            label_key_widget in st.session_state
            and st.session_state[label_key_widget] is not None
        ):
            st.session_state[label_key_persist] = st.session_state[label_key_widget]

        # Patrón "pending clear" para el reset del form tras guardar:
        # Streamlit prohíbe modificar la session_state de un widget tras
        # haberse instanciado en el mismo rerun (StreamlitAPIException).
        # En su lugar, el handler del save button solo activa el flag
        # `pending_clear` + escribe el mensaje de éxito, y aquí
        # —al principio del siguiente rerun, ANTES de instanciar los
        # widgets de input— consumimos el flag y limpiamos.
        pending_clear_key = f"_corr_pending_clear_{job.job_id}"
        pending_success_key = f"_corr_save_success_{job.job_id}"
        if st.session_state.get(pending_clear_key, False):
            # Mostrar mensaje de éxito del guardado anterior.
            success_msg = st.session_state.pop(pending_success_key, None)
            if success_msg:
                st.success(success_msg)
            # Limpiar state canónico.
            st.session_state[idx_set_key] = ()
            st.session_state[idx_last_key] = None
            st.session_state[range_errors_key] = []
            # Limpiar widgets (excepto el number_input: conserva ancla).
            if range_text_key in st.session_state:
                st.session_state[range_text_key] = ""
            if label_key_widget in st.session_state:
                st.session_state[label_key_widget] = None
            if label_key_persist in st.session_state:
                st.session_state[label_key_persist] = None
            if comment_key in st.session_state:
                st.session_state[comment_key] = ""
            # Consumir el flag.
            del st.session_state[pending_clear_key]
        # NOTA: la aplicación pending_key → widget_key se hace en
        # render_slide_detail (antes del visor) para evitar
        # desincronización visor↔combo. Aquí solo escribimos en
        # pending_key desde el botón 'siguiente más incierto'.
        # El tecleado en number_input usa on_change para señalizar que
        # es navegación explícita y debe panear/zoomear el visor.

        # Inicialización idempotente del estado canónico v4.
        st.session_state.setdefault(idx_set_key, ())
        st.session_state.setdefault(idx_last_key, None)

        def _on_idx_typed() -> None:
            # Cuando el patólogo teclea un #idx + Enter, el flujo es
            # navegación explícita (igual que el botón siguiente): el
            # visor debe centrar y zoomear ese parche, no solo
            # marcarlo en amarillo silenciosamente fuera del viewport.
            raw = st.session_state.get(widget_key)
            if raw is None:
                # number_input limpiado: vaciar el set canónico también.
                st.session_state[idx_set_key] = ()
                st.session_state[idx_last_key] = None
                return
            val = int(raw)
            st.session_state[idx_set_key] = (val,)
            st.session_state[idx_last_key] = val
            st.session_state[pending_pan_key] = True
            # Limpiar el text_input de rango para evitar inconsistencia
            # visual (si el patólogo había tecleado un rango antes y
            # ahora cambia a single-patch via number_input).
            if range_text_key in st.session_state:
                st.session_state[range_text_key] = ""

        def _on_range_typed() -> None:
            raw = (st.session_state.get(range_text_key) or "").strip()
            if not raw:
                # El patólogo limpió el text_input. No tocamos el
                # number_input ni el set canónico — si tenía un valor
                # previo, sigue válido.
                st.session_state[range_errors_key] = []
                return
            parsed, errors = _parse_idx_range_text(raw, n_patches)
            st.session_state[range_errors_key] = errors
            if not parsed:
                # Input totalmente inválido (cero tokens utilizables).
                # Limpiamos el estado canónico y el number_input para
                # que el patólogo no se confunda con un valor stale
                # del #idx anterior y guarde por accidente sobre él.
                # La purga del segmented_control de etiqueta (que
                # provoca el early-return cuando patch_idx=None) está
                # mitigada por el patrón double-key (label_key_persist)
                # definido al inicio de _render_corrections_panel.
                st.session_state[idx_set_key] = ()
                st.session_state[idx_last_key] = None
                if widget_key in st.session_state:
                    st.session_state[widget_key] = None
                return
            st.session_state[idx_set_key] = parsed
            st.session_state[idx_last_key] = parsed[-1]
            st.session_state[pending_pan_key] = True
            # Reflejar el ancla del lote en el number_input para que el
            # visor (que aún consume widget_key) siga panéando/zoomeando
            # al último elemento del set. El text_input conserva el
            # rango tecleado, lo que da al patólogo doble feedback:
            # número de ancla y composición del lote. Streamlit permite
            # escribir el widget key de otro widget desde un callback.
            if widget_key in st.session_state:
                st.session_state[widget_key] = parsed[-1]

        # Set de parches ya corregidos — para excluirlos del cálculo del
        # 'siguiente más incierto'.
        corrected_idxs = {
            int(c.patch_idx) for c in list_corrections(job.job_dir)
        }

        st.markdown(f"**Parche a corregir** (0–{n_patches - 1})")
        col_idx, col_range, col_next = st.columns([2, 2, 1])
        with col_idx:
            raw_idx = st.number_input(
                "Parche a corregir",
                min_value=0, max_value=n_patches - 1, step=1,
                value=None,  # arranca vacío (placeholder visible)
                key=widget_key,
                on_change=_on_idx_typed,
                help="Teclea el #índice que ves en el hover del visor o pulsa 'Siguiente más incierto'.",
                placeholder="—",
                label_visibility="collapsed",
            )
        with col_range:
            st.text_input(
                "Rango de parches",
                key=range_text_key,
                on_change=_on_range_typed,
                placeholder="o p. ej. 12, 15, 20-30",
                help=(
                    "Selección por lote: lista de índices separados por "
                    "comas, con rangos inclusivos. Ejemplos: `12, 15` "
                    "selecciona dos parches sueltos; `20-30` selecciona "
                    "del 20 al 30 (inclusive); `12, 20-30, 45` combina "
                    "ambas formas. Sustituye al `#idx` del number_input "
                    "al teclear."
                ),
                label_visibility="collapsed",
            )
        # Avisos del parser (tokens inválidos o fuera de rango). Se
        # muestran como st.warning bajo los inputs y se preservan entre
        # reruns hasta que el patólogo cambia el texto del rango.
        for err in st.session_state.get(range_errors_key, []) or []:
            st.warning(err)

        # `patch_idx` ancla la UI single-patch (caption, save, visor).
        # Prioridad: (1) el last canónico si hay set no vacío,
        # (2) el number_input clásico, (3) None.
        idx_set_current: tuple[int, ...] = tuple(st.session_state.get(idx_set_key, ()))
        if idx_set_current:
            patch_idx: int | None = int(
                st.session_state.get(idx_last_key) or idx_set_current[-1]
            )
        elif raw_idx is not None:
            patch_idx = int(raw_idx)
        else:
            patch_idx = None

        # Feedback inmediato cuando hay un lote >1 (sin caption aún
        # — eso va en próxima sesión). El usuario ve cuántos parches
        # tiene seleccionados y el last como ancla.
        if len(idx_set_current) > 1:
            st.caption(
                f"📌 Lote actual: **{len(idx_set_current)} parches** · "
                f"ancla `#{patch_idx}` (último seleccionado)."
            )

        with col_next:
            # next_uncertain: primer parche no corregido del ranking.
            # Si patch_idx existe y está en el ranking, partimos de él
            # para avanzar. Si no, partimos del principio (más incierto).
            order_list = [int(x) for x in order]
            if patch_idx is not None and patch_idx in order_list:
                cur_pos = order_list.index(patch_idx)
            else:
                cur_pos = -1  # arrancar antes del primero → primero es +1
            next_uncertain = None
            for offset in range(1, len(order_list) + 1):
                cand = order_list[(cur_pos + offset) % len(order_list)]
                if cand not in corrected_idxs:
                    next_uncertain = cand
                    break
            if next_uncertain is None:
                next_uncertain = order_list[(cur_pos + 1) % len(order_list)] if order_list else 0

            if st.button(
                f"💡 Siguiente más incierto (#{next_uncertain})",
                key=f"corr_next_{job.job_id}",
                use_container_width=True,
                help="Salta los parches que ya tienen corrección registrada.",
            ):
                # pending_pan=True para que el visor navegue al destino.
                st.session_state[pending_key] = next_uncertain
                st.session_state[f"corr_pending_pan_{job.job_id}"] = True
                # Al saltar a un parche nuevo, deseleccionamos la
                # etiqueta y vaciamos el comentario para forzar al
                # patólogo a tomar una decisión consciente sobre el
                # nuevo parche (mismo razonamiento que post-save).
                # Streamlit permite mutar estas keys aquí porque la
                # segmented_control y el text_input de comentario
                # aún no se han instanciado en este rerun (están
                # después del early-return de patch_idx None).
                if label_key_widget in st.session_state:
                    st.session_state[label_key_widget] = None
                if label_key_persist in st.session_state:
                    st.session_state[label_key_persist] = None
                if comment_key in st.session_state:
                    st.session_state[comment_key] = ""
                st.rerun()

        # Toggle: ver el visor con la predicción del modelo (default)
        # vs con las correcciones aplicadas. Disponible siempre.
        col_t1, col_t2, col_t3 = st.columns(3)
        with col_t1:
            st.toggle(
                "🎨 Mostrar correcciones aplicadas en el visor",
                key=f"view_corrected_{job.job_id}",
                help="OFF: bordes con la predicción del modelo (color por clase F4). "
                     "ON: bordes con la etiqueta corregida (donde la haya).",
            )
        with col_t2:
            st.toggle(
                "🔲 Bordes del parche seleccionado",
                value=True,
                key=f"show_sel_borders_{job.job_id}",
                help="OFF: oculta el borde de etiqueta y el highlight amarillo "
                     "del parche seleccionado para examinarlo sin ruido visual. "
                     "El círculo de 'corregido' (esquina) sigue siempre visible.",
            )
        with col_t3:
            st.toggle(
                "● Mostrar parches fuera de tarea (HIP/ART)",
                value=True,
                key=f"show_out_of_task_{job.job_id}",
                help="ON (default): muestra el disco rojo sobre los parches con "
                     "GT HIP/ART y los marcadores rojos de correcciones a "
                     "HIP/ART/EXCLUDED. OFF: oculta esa información, dejando el "
                     "visor coherente con el espacio ternario de la tarea (lo "
                     "que vio el modelo en entrenamiento).",
            )

        # Sin parche seleccionado: pista para empezar + saltamos a
        # renderizar el resumen al final (sin etiquetar / guardar).
        if patch_idx is None:
            st.info("Teclea un `#índice` o pulsa **Siguiente más incierto** para empezar a corregir.")
            summary = summarize_corrections(job.job_dir)
            if summary["n_total"] > 0:
                _render_corrections_summary(summary, job)
            return

        # Info del parche seleccionado: replica el contenido del hover
        # del visor para que el patólogo confirme que va a corregir el
        # parche correcto. Si ya está corregido, lo decimos en una
        # segunda línea con el icono ✓.
        pred_str_sel = CLASS_NAMES[int(pred_index[patch_idx])]
        att_sel = float(attention[patch_idx]) if patch_idx < len(attention) else 0.0
        if patch_probs is not None:
            probs_sel = patch_probs[patch_idx]
            probs_str = " · ".join(
                f"{c}={probs_sel[i]:.3f}" for i, c in enumerate(CLASS_NAMES)
            )
            conf_sel = float(probs_sel.max())
            conf_str_sel = f"{conf_sel:.0%}"
        else:
            probs_str = "?"
            conf_str_sel = "?"
        caption_text = (
            f"**#{patch_idx}** · predicción F4: **{pred_str_sel}** "
            f"({conf_str_sel}) · probs: {probs_str} · atención: {att_sel:.4f}"
        )
        # Última corrección registrada para este parche (last-wins).
        existing_corr = next(
            (c.label_corr for c in reversed(list_corrections(job.job_dir))
             if int(c.patch_idx) == patch_idx),
            None,
        )
        if existing_corr:
            caption_text += f"\n\n✓ **Corregido a {existing_corr}**"
        st.caption(caption_text)

        # Selector de clase. segmented_control para que sea un click
        # directo. `default=` rescata el valor de label_key_persist si
        # el widget fue purgado en un rerun anterior (ver double-key).
        new_label = st.segmented_control(
            "Etiqueta corregida",
            options=list(CORRECTION_LABELS),
            key=label_key_widget,
            default=st.session_state.get(label_key_persist),
        )

        # Coloreado de los botones por contenido de texto. Streamlit no
        # expone API nativa y los selectores CSS son frágiles entre
        # versiones, así que inyectamos un iframe invisible (height=0)
        # con un script que recorre window.parent.document y aplica
        # estilos a los botones cuyo texto coincide con cada etiqueta.
        # MutationObserver re-aplica los estilos si Streamlit re-renderiza
        # el segmented_control (lo hace en cada rerun).
        import streamlit.components.v1 as _components
        _components.html(
            """
            <script>
            const COLORS = {
              "ADE": "#ff7f0e", "NOR": "#2ca02c", "CAR": "#1f77b4",
              "HIP": "#888888", "ART": "#888888", "EXCLUDED": "#888888"
            };
            const FILLS = {
              "ADE": "rgba(255,127,14,0.25)", "NOR": "rgba(44,160,44,0.25)",
              "CAR": "rgba(31,119,180,0.25)",
              "HIP": "rgba(136,136,136,0.25)", "ART": "rgba(136,136,136,0.25)",
              "EXCLUDED": "rgba(136,136,136,0.25)"
            };
            function paint() {
              const doc = window.parent.document;
              // Buscamos botones/labels cuyo texto coincida exactamente con
              // una de las CORRECTION_LABELS. Evitamos colorear cosas
              // ajenas (ej. matrices con celdas que dicen "ADE").
              doc.querySelectorAll('button, label').forEach(el => {
                const txt = (el.textContent || '').trim();
                if (COLORS[txt] && txt.length <= 10) {
                  el.style.borderColor = COLORS[txt];
                  el.style.color = COLORS[txt];
                  el.style.borderWidth = '2px';
                  el.style.borderStyle = 'solid';
                  el.style.fontWeight = '600';
                  // Si está seleccionado (BaseWeb usa aria-checked o input:checked)
                  const checkedInput = el.querySelector('input:checked');
                  const ariaChecked = el.getAttribute('aria-checked') === 'true';
                  if (checkedInput || ariaChecked) {
                    el.style.backgroundColor = FILLS[txt];
                  } else {
                    el.style.backgroundColor = '';
                  }
                }
              });
            }
            paint();
            // Re-pintar cada 300ms — barato y atrapa cambios tras click.
            setInterval(paint, 300);
            </script>
            """,
            height=0,
        )

        comment = st.text_input(
            "Comentario (opcional)",
            key=f"corr_comment_{job.job_id}",
            placeholder="p. ej. 'morfología poco clara, posible artefacto de tinción'",
        )

        # Resolución del conjunto a guardar: el set canónico v4 si tiene
        # contenido, o el patch_idx single en su defecto (backward-compat).
        # El botón muestra el conteo cuando es lote para que el patólogo
        # tenga feedback inmediato antes de pulsar.
        save_idxs: tuple[int, ...]
        if idx_set_current:
            save_idxs = idx_set_current
        else:
            save_idxs = (patch_idx,)
        save_label = (
            "💾 Guardar corrección"
            if len(save_idxs) <= 1
            else f"💾 Aplicar a {len(save_idxs)} parches"
        )
        # Si el parser tiene errores activos (texto inválido sin tokens
        # útiles) deshabilitamos el guardado para que el patólogo no
        # registre por error una corrección sobre el #idx stale del
        # number_input mientras el text_input contiene basura. El
        # caption del col_info indica la causa.
        range_errors_active = bool(st.session_state.get(range_errors_key))

        col_save, col_info = st.columns([1, 4])
        with col_save:
            if st.button(
                save_label,
                key=f"corr_save_{job.job_id}",
                disabled=(new_label is None or range_errors_active),
                type="primary",
            ):
                # Lote: aplicamos la misma etiqueta a todos los idx del
                # set en orden. Cada uno persiste un registro independiente
                # en corrections.jsonl (append-only) con su pred_orig y
                # probs_orig propios — last-wins se resuelve al leer.
                for save_idx in save_idxs:
                    pred_orig_str = CLASS_NAMES[int(pred_index[save_idx])]
                    probs_orig = (
                        patch_probs[save_idx].tolist()
                        if patch_probs is not None
                        else None
                    )
                    record_correction(
                        job.job_dir,
                        slide_uuid=job.job_id,
                        patch_idx=save_idx,
                        label_corr=new_label,
                        pred_orig=pred_orig_str,
                        probs_orig=probs_orig,
                        patologo_id=_patologo_id(),
                        model_version=_model_version(),
                        comment=comment or "",
                    )
                # Activamos el flag de "pending clear" y guardamos el
                # mensaje de éxito. El limpiado de widgets ocurre al
                # inicio del próximo rerun, ANTES de instanciar los
                # widgets (única ventana permitida por Streamlit). No
                # podemos modificar aquí widget_key/range_text_key/
                # label_key_widget/comment_key directamente porque ya
                # están instanciados en este rerun.
                if len(save_idxs) <= 1:
                    msg = (
                        f"Corrección guardada: parche #{save_idxs[0]} → "
                        f"{new_label}"
                    )
                else:
                    msg = (
                        f"Correcciones guardadas: {len(save_idxs)} parches "
                        f"→ {new_label}"
                    )
                st.session_state[pending_success_key] = msg
                st.session_state[pending_clear_key] = True
                st.rerun()
        with col_info:
            if range_errors_active:
                st.caption(
                    "Hay errores en el rango de parches. Corrígelo o "
                    "vacíalo para activar el guardado."
                )
            elif new_label is None:
                st.caption("Selecciona una etiqueta para activar el guardado.")

        # Resumen de correcciones de este slide (con parche seleccionado).
        summary = summarize_corrections(job.job_dir)
        if summary["n_total"] > 0:
            _render_corrections_summary(summary, job)


_DELETE_CONFIRMATION_PHRASE = "borrar todas las correcciones"


def _render_corrections_summary(summary: dict, job: "Job") -> None:
    """Resumen del panel de correcciones: 3 métricas + breakdown por etiqueta
    + expander con botón de borrado masivo (con confirmación por frase).

    Dividido entre ternarias (ADE/NOR/CAR) que entran al fine-tune del head,
    y no-ternarias (HIP/ART/EXCLUDED) que se persisten como dataset latente
    para modelos cuaternarios o filtro de calidad.
    """
    ternary = sum(
        summary["by_label"].get(c, 0) for c in ("ADE", "NOR", "CAR")
    )
    non_ternary = sum(
        summary["by_label"].get(c, 0) for c in ("HIP", "ART", "EXCLUDED")
    )
    st.markdown("**Correcciones registradas para este portaobjetos:**")
    cols = st.columns(3)
    cols[0].metric("Parches únicos", summary["n_unique_patches"])
    cols[1].metric(
        "Ternarias (ADE/NOR/CAR)", ternary,
        help="Estas correcciones entrarán al fine-tune del head ternario.",
    )
    cols[2].metric(
        "No ternarias (HIP/ART/EXCLUDED)", non_ternary,
        help="Persistidas como dataset latente — útiles para modelos "
             "con más clases o como filtro de calidad. No entran al "
             "fine-tune ternario actual.",
    )
    if summary["by_label"]:
        breakdown = " · ".join(
            f"**{c}**: {n}" for c, n in sorted(summary["by_label"].items())
        )
        st.caption(f"Por etiqueta: {breakdown}")

    # Borrado masivo de correcciones del slide. Confirmación por frase
    # exacta (no por timeout) para minimizar el riesgo de borrar
    # accidentalmente trabajo del patólogo.
    # Usamos st.popover (no st.expander) porque el panel completo ya
    # vive dentro de un st.expander y Streamlit prohíbe anidar
    # expanders. popover es un dropdown botón que no cuenta como expander.
    with st.popover("🗑️ Borrar todas las correcciones de este portaobjetos"):
        st.warning(
            "Esta acción borra el `corrections.jsonl` del portaobjetos. "
            "**Es irreversible.** Las correcciones desaparecen de la "
            "vista, los círculos del visor, el resumen y el JSONL en disco."
        )
        confirm = st.text_input(
            f"Para confirmar, escribe exactamente: `{_DELETE_CONFIRMATION_PHRASE}`",
            key=f"corr_delete_confirm_{job.job_id}",
            placeholder=_DELETE_CONFIRMATION_PHRASE,
        )
        match = (confirm or "").strip().lower() == _DELETE_CONFIRMATION_PHRASE
        if st.button(
            "🗑️ Borrar definitivamente",
            key=f"corr_delete_btn_{job.job_id}",
            disabled=not match,
            type="primary",
        ):
            corrections_path = job.job_dir / "corrections.jsonl"
            if corrections_path.exists():
                corrections_path.unlink()
            st.success(
                f"Correcciones borradas: se eliminó `{corrections_path.name}`."
            )
            st.rerun()


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
                "Sensibilidad (recall)": f"{m['recall']:.1%} ({m['tp']}/{m['support']})",
                "F1-score": f"{m['f1']:.3f}",
                "Soporte": m["support"],
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        if car_total:
            st.caption(
                f"CAR→ADE: {car_to_ade_rate:.1%} ({car_to_ade}/{car_total}) · "
                f"CAR→NOR: {car_to_nor_rate:.1%} ({car_to_nor}/{car_total})"
            )

    # No repetimos la cajita explicativa de las métricas — ya aparece
    # bajo la tabla patch-level en _render_patch_validation. Las
    # definiciones (precisión, sensibilidad, F1, soporte, TP/FP/FN) son
    # las mismas independientemente del nivel.

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
    """Renderiza la vista detallada de un job en estado DONE.

    Layout:
        1. Encabezado con filename
        2. Visor OpenSeadragon (capa según el modo activo)
        3. Selector segmentado: 'Atención' | 'Predicciones a nivel de parche'
        4. Sección condicional según el modo:
           - Atención: métricas slide-level + barras + aviso confianza + top-K
           - Predicciones: bar chart distribución + inspector + matriz GT (si)
    """
    result = _load_result(job)
    if result is None:
        st.warning("No hay resultado para este job (¿aún en proceso?).")
        return

    probs = list(map(float, result["probabilities_mean"]))
    stds = list(map(float, result["probabilities_std"]))
    pred_class = result["predicted_class"]
    max_prob = max(probs)

    # ─── Encabezado ─────────────────────────────────────────────────────────
    st.subheader(f"Resultado · {job.original_filename}")

    # ─── Carga de artefactos compartidos por ambas vistas ───────────────────
    attention = _load_attention(job)
    if attention is None:
        st.info("Sin pesos de atención disponibles para este job.")
        return

    h5_meta = _load_h5_meta(job)
    if h5_meta is None:
        st.info("No se pudo leer las posiciones del H5.")
        return
    positions, categories = h5_meta

    originals = _load_all_originals(job)
    patches_arr, patch_size = originals if originals is not None else (None, None)

    patch_eval = _load_patch_eval(job)
    pred_index = (
        np.asarray(patch_eval["pred_index"], dtype=np.int64)
        if patch_eval is not None and "pred_index" in patch_eval
        else None
    )
    pred_probs = (
        np.asarray(patch_eval["pred_probs"], dtype=np.float32)
        if patch_eval is not None and "pred_probs" in patch_eval
        else None
    )

    # ─── Selector segmentado: Atención | Predicciones por parche ───────────
    OPT_ATT = "👁️ Atención"
    OPT_PRED = "🔮 Predicciones a nivel de parche"
    mode = st.segmented_control(
        "Modo de visualización",
        options=[OPT_ATT, OPT_PRED],
        default=OPT_ATT,
        key=f"view_mode_{job.job_id}",
        label_visibility="collapsed",
    )
    if mode is None:
        mode = OPT_ATT
    show_att = mode == OPT_ATT
    show_pred = mode == OPT_PRED

    # ─── Visor OpenSeadragon con la capa correspondiente ────────────────────
    dzi_status = job.extra.get("dzi_status", "unknown")
    if job.dzi_path.exists() and pred_index is not None:
        osd_offset = (
            int(job.extra.get("dzi_y_min", 0)),
            int(job.extra.get("dzi_x_min", 0)),
        )
        # Si estamos en modo predicciones, leemos el target del panel
        # de correcciones (el number_input) para que el visor dibuje el
        # borde amarillo y centre la vista en él. También leemos el
        # toggle 'ver correcciones aplicadas' para repintar bordes con
        # el color de la corrección.
        sel_idx = None
        view_corrected_flag = False
        show_sel_borders_flag = True
        pan_to_selected_flag = False
        show_out_of_task_flag = True
        if show_pred:
            widget_key = f"corr_idx_{job.job_id}"
            pending_key = f"corr_pending_target_{job.job_id}"
            pending_pan_key = f"corr_pending_pan_{job.job_id}"
            # CRÍTICO: aplicar pending_key ANTES de renderizar el visor.
            # El botón 'siguiente más incierto' encola en pending_key
            # (con pending_pan=True para que el visor panee al destino).
            # El click capturado del visor también encola en pending_key
            # pero con pending_pan=False (ya estás mirando ahí, no panees).
            if pending_key in st.session_state:
                st.session_state[widget_key] = st.session_state.pop(pending_key)
                pan_to_selected_flag = bool(st.session_state.pop(pending_pan_key, False))
            elif pending_pan_key in st.session_state:
                # Tecleado directo en number_input: on_change marcó
                # pending_pan_key=True. widget_key ya tiene el nuevo
                # valor (Streamlit lo actualizó antes del callback);
                # solo nos queda consumir el flag de pan/zoom.
                pan_to_selected_flag = bool(st.session_state.pop(pending_pan_key, False))
            if widget_key in st.session_state:
                v = st.session_state[widget_key]
                if v is not None:
                    sel_idx = int(v)
            view_corrected_flag = bool(
                st.session_state.get(f"view_corrected_{job.job_id}", False)
            )
            show_sel_borders_flag = bool(
                st.session_state.get(f"show_sel_borders_{job.job_id}", True)
            )
            show_out_of_task_flag = bool(
                st.session_state.get(f"show_out_of_task_{job.job_id}", True)
            )
        # En modo predicciones, usar el custom component (con click
        # capture). En modo atención, mantener el inline (más simple,
        # sin necesidad de captura — el patólogo solo está mirando los
        # hot-spots, no corrigiendo).
        clicked = _render_openseadragon_viewer(
            job,
            positions=positions,
            pred_index=pred_index,
            patch_raw_size=patch_size,
            attention=attention,
            slide_pred_class=pred_class,
            show_predictions=show_pred,
            show_attention=show_att,
            dzi_offset=osd_offset,
            selected_idx=sel_idx,
            view_corrected=view_corrected_flag,
            show_selected_borders=show_sel_borders_flag,
            pan_to_selected=pan_to_selected_flag,
            show_out_of_task=show_out_of_task_flag,
            enable_click_capture=show_pred,
        )

        # Si llegó un click nuevo, lo encolamos en pending_key — al
        # principio del próximo rerun se aplicará a widget_key (la del
        # number_input). pending_pan=False: el patólogo ya está mirando
        # el parche cliqueado, mover la vista despistaría.
        if isinstance(clicked, dict) and "ts" in clicked and show_pred:
            last_seen_key = f"corr_last_click_ts_{job.job_id}"
            if st.session_state.get(last_seen_key) != clicked["ts"]:
                st.session_state[last_seen_key] = clicked["ts"]
                st.session_state[f"corr_pending_target_{job.job_id}"] = int(clicked["idx"])
                st.session_state[f"corr_pending_pan_{job.job_id}"] = False
                st.rerun()
        st.caption(
            "Pan con arrastrar, zoom con rueda. Pasa el ratón sobre un "
            "parche para ver `#índice · clase · atención`. Las áreas blancas "
            "son zonas que el filtro de tejido descartó al parchear."
        )
        # Leyenda visual compacta (una sola línea). Estructura:
        # las tres clases NOR/ADE/CAR con su código de color y, entre
        # paréntesis, las marcas que usan ese mismo código (bordes,
        # ⭕ correcciones y ✕✕✕ errores ternarios — solo si el slide
        # trae GT por parche). El disco rojo pequeño de HIP/ART queda
        # fuera del paréntesis: es semánticamente distinto (fuera de
        # tarea, no error). HTML inline para que las X y el disco
        # coincidan en color con lo que dibuja el SVG del visor.
        if show_pred:
            has_gt = bool(result.get("has_patch_gt"))
            inside = "bordes y ⭕ correcciones"
            if has_gt:
                inside = (
                    "bordes, ⭕ correcciones y "
                    "<span style='color:#2ca02c;font-weight:700'>✕</span>"
                    "<span style='color:#ff7f0e;font-weight:700'>✕</span>"
                    "<span style='color:#1f77b4;font-weight:700'>✕</span>"
                    " errores"
                )
            legend_html = (
                "🟢 NOR · 🟠 ADE · 🔵 CAR (" + inside + ")"
            )
            if has_gt:
                legend_html += (
                    " · <span style='color:rgb(214,39,40);font-size:0.7em;"
                    "vertical-align:middle;'>●</span>"
                    " HIP/ART (fuera de tarea)"
                )
            st.markdown(
                "<div style='font-size:0.875rem;opacity:0.85;'>"
                + legend_html
                + "</div>",
                unsafe_allow_html=True,
            )
    elif dzi_status == "generating":
        # El thread async de DZI todavía está corriendo. La cola fragment
        # rerunna cada 2 s y dispara rerun global cuando has_dzi cambia,
        # así que el visor aparecerá automáticamente al terminar.
        st.info(
            "⏳ **Generando visor multi-resolución…** El visor "
            "(OpenSeadragon) tarda en construirse según el tamaño del "
            "portaobjetos: ~5 s para slides pequeños, hasta 1-3 minutos "
            "para WSIs grandes. Aparece automáticamente aquí en cuanto "
            "esté listo. Las predicciones y métricas ya están disponibles "
            "más abajo."
        )
        st.progress(0, text="Construyendo pirámide de tiles…")
    elif dzi_status == "failed":
        err = job.extra.get("dzi_error", "(sin detalle)")
        st.error(
            f"❌ **Visor multi-resolución no disponible.** La generación "
            f"de tiles falló: `{err}`. Las predicciones y métricas siguen "
            "funcionando con normalidad más abajo."
        )
    else:
        st.info(
            "El visor multi-resolución no está disponible para este job "
            "(legado: subido antes de la integración OpenSeadragon). Las "
            "métricas y mapas se muestran de todos modos."
        )

    # ─── Vista 'Atención': etiqueta clínica + top-K + métricas + barras ────
    if show_att:
        # Etiqueta clínica del slide: solo visible en modo Atención
        # (lectura global del slide, donde la decisión clínica
        # cuadra naturalmente con el flujo del patólogo). En modo
        # Predicciones se omite para no competir con las correcciones
        # parche a parche.
        _render_slide_label_panel(job, pred_class=pred_class, pred_probs=probs)

        # Top-K justo debajo del visor: extensión espacial inmediata de los
        # parches con mayor atención que se ven destacados arriba.
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

        st.divider()

        # Métricas slide-level del veredicto del AttnMIL
        cols = st.columns(4)
        cols[0].metric("Predicción", pred_class)
        cols[1].metric("Confianza", f"{max_prob:.1%}")
        cols[2].metric("Parches", str(result["n_patches"]))
        cols[3].metric("Tiempo", f"{result['elapsed_seconds']:.2f} s")

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

        st.info(
            "**La confianza no es una probabilidad de acierto.** Es la media "
            "del *softmax* del ensemble de 5 modelos *AttnMIL* en la clase "
            "predicha. Un valor alto indica que los 5 modelos coinciden con "
            "*softmax* saturado, **no** que la predicción sea correcta esa "
            "proporción de veces. El *softmax* no está calibrado: "
            "interprétalo como **seguridad relativa del modelo**, no como "
            "certeza diagnóstica.\n\n"
            "**TFM vs producción.** La memoria del TFM (§5.9) reporta "
            "**92,8 ± 1,1 %** *accuracy* mediante validación cruzada 5-fold "
            "*multi-seed* sobre los 91 portaobjetos clínicos del HUC: esa "
            "es la estimación honest del rendimiento esperado sobre "
            "portaobjetos **nuevos**. El *ensemble* desplegado en esta app "
            "es un reentrenamiento posterior de **5** modelos sobre los 91 "
            "completos **sin holdout** (práctica estándar al pasar de "
            "evaluación a producción) — sobre portaobjetos del propio "
            "cohort §5.9 las predicciones serán muy seguras (todos los "
            "modelos los vieron en *training*), pero esa cifra **no es "
            "comparable** con §5.9. **Para portaobjetos nuevos esperar "
            "~92,8 % accuracy.**\n\n"
            "Las barras de error miden la dispersión entre los 5 modelos "
            "del *ensemble*: una *std* alta indica desacuerdo entre miembros."
        )

    # ─── Vista 'Predicciones por parche': panel correcciones + bar chart + matriz ────
    elif show_pred and patch_eval is not None:
        # Panel de correcciones bajo el visor — solo en este modo. El
        # patólogo navega por incertidumbre, corrige los parches dudosos.
        if pred_index is not None:
            _render_corrections_panel(
                job,
                pred_index=pred_index,
                patch_probs=pred_probs,
                attention=attention,
            )
        _render_patch_predictions(
            patch_eval,
            positions=positions,
            patches_arr=patches_arr,
            patch_size=patch_size,
            attention=attention,
            job=job,
            slide_pred_class=pred_class,
            show_out_of_task=show_out_of_task_flag,
        )
        if result.get("has_patch_gt"):
            _render_patch_validation(patch_eval, result)
