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
from pathlib import Path
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
    dzi_offset: tuple[int, int] = (0, 0),
    height: int = 620,
) -> bool:
    """Si el job tiene `slide.dzi` (sólo TIFFs), embebe un visor OpenSeadragon
    apuntando a `/dzi/<job_id>/slide.dzi`. Si se pasan posiciones + predicciones,
    dibuja un overlay HTML con un rectángulo del color de la clase predicha
    sobre cada parche en sus coordenadas reales del WSI. Devuelve True si se
    renderizó.

    nginx sirve el directorio `queue/<job_id>/` como static bajo `/dzi/<job_id>/`
    (ver `nginx.conf` location /dzi/). El navegador hereda BasicAuth
    same-origin para los tiles.
    """
    if not job.dzi_path.exists():
        return False
    dzi_url = f"/dzi/{job.job_id}/slide.dzi"

    # Construye el JSON con posiciones + clase de cada parche, restando
    # el offset del DZI stitched (las posiciones del H5 están en coords
    # del WSI completo; el DZI stitched empieza en (y_min, x_min)).
    overlays_json = "[]"
    y_off, x_off = dzi_offset
    if (positions is not None and pred_index is not None
            and patch_raw_size is not None and len(positions) == len(pred_index)):
        items = []
        for i, (pos, p) in enumerate(zip(positions, pred_index)):
            cls = CLASS_NAMES[int(p)]
            r, g, b = CLASS_COLORS_RGB[cls]
            color = f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"
            items.append({
                "x": int(pos[1]) - int(x_off),
                "y": int(pos[0]) - int(y_off),
                "size": int(patch_raw_size),
                "color": color,
                "idx": i,
                "cls": cls,
            })
        overlays_json = json.dumps(items)

    html = f"""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.1.1/openseadragon.min.css">
    <style>
      .osd-patch {{
        box-sizing: border-box;
        pointer-events: none;
      }}
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
      }});

      const overlays = {overlays_json};
      const SVG_NS = "http://www.w3.org/2000/svg";

      viewer.addHandler("open", function() {{
        for (const o of overlays) {{
          // SVG con viewBox 0..1 + stroke-width relativo (8% del lado del
          // parche). Al hacer zoom el SVG escala con el overlay, así el
          // borde se mantiene visualmente proporcional al parche en
          // cualquier nivel de zoom — a vista lejana queda como un pequeño
          // marquito con interior transparente, no como un relleno.
          const svg = document.createElementNS(SVG_NS, "svg");
          svg.setAttribute("viewBox", "0 0 1 1");
          svg.setAttribute("preserveAspectRatio", "none");
          svg.style.width = "100%";
          svg.style.height = "100%";
          svg.style.pointerEvents = "none";

          const rect = document.createElementNS(SVG_NS, "rect");
          rect.setAttribute("x", "0.04");
          rect.setAttribute("y", "0.04");
          rect.setAttribute("width", "0.92");
          rect.setAttribute("height", "0.92");
          rect.setAttribute("fill", "none");
          rect.setAttribute("stroke", o.color);
          rect.setAttribute("stroke-width", "0.035");
          svg.appendChild(rect);

          const div = document.createElement("div");
          div.className = "osd-patch";
          div.title = `#${{o.idx}} · ${{o.cls}}`;
          div.appendChild(svg);

          viewer.addOverlay({{
            element: div,
            location: viewer.viewport.imageToViewportRectangle(
              o.x, o.y, o.size, o.size
            ),
          }});
        }}
      }});
    </script>
    """
    import streamlit.components.v1 as components
    components.html(html, height=height + 20, scrolling=False)
    return True


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
    job_id: str | None = None,
) -> None:
    """Sección 'Predicciones por parche' (sin GT). Bar chart de distribución
    + overlay del slide coloreado por predicción de cada parche."""
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

    if (positions is not None and patches_arr is not None
            and patch_size is not None and len(patches_arr) == len(pred_index)):
        st.markdown(
            "**Mapa de predicciones por parche** — verde NOR, naranja ADE, "
            "azul CAR. Borde coloreado por la clase predicha del clasificador "
            "F4 sobre cada parche; el tejido queda visible para zoom."
        )
        thumb_options = {"48 px": 48, "64 px": 64, "96 px": 96, "128 px": 128}
        thumb_label = st.radio(
            "Resolución del mapa de predicciones",
            options=list(thumb_options.keys()),
            index=0,
            horizontal=True,
            key=f"thumb_size_pred_{job_id}" if job_id else "thumb_size_pred",
            help=(
                "Tamaño de cada parche en el mosaico. Resolución alta = más "
                "detalle al hacer zoom, pero más tiempo de render en slides "
                "con muchos parches."
            ),
        )
        thumb_size = thumb_options[thumb_label]
        n = len(pred_index)
        sel_key = f"detail_idx_{job_id}" if job_id else "detail_idx"
        if sel_key not in st.session_state:
            st.session_state[sel_key] = -1

        with st.spinner(
            f"Generando mapa de predicciones ({len(pred_index)} parches a {thumb_size} px)…"
        ):
            fig_pred = _patch_predictions_overlay_figure_cached(
                job_id=job_id or "no_id",
                thumb_size=thumb_size,
                border_thickness=3,
                _pred_index=pred_index,
                _patches_arr=patches_arr,
                _positions=positions,
                _attention=attention,
                patch_raw_size=patch_size,
            )
            # `on_select="rerun"` + selection_mode=("points",) hace que un
            # click sobre un parche del mosaico devuelva un evento con
            # customdata[0] = índice del parche → se inyecta en el selector
            # de abajo automáticamente.
            event = st.plotly_chart(
                fig_pred,
                use_container_width=True,
                config={"displayModeBar": True, "scrollZoom": True},
                on_select="rerun",
                selection_mode=("points",),
                key=f"plot_pred_{job_id}",
            )

        # Capturar click sobre parche del mosaico → seleccionar en el panel
        try:
            sel_points = event["selection"]["points"]
        except (KeyError, TypeError, AttributeError):
            sel_points = []
        if sel_points:
            pt = sel_points[-1]   # último click
            cd = pt.get("customdata") if isinstance(pt, dict) else None
            if cd and len(cd) > 0:
                clicked_idx = int(cd[0])
                if clicked_idx != st.session_state.get(sel_key, -1):
                    st.session_state[sel_key] = clicked_idx

        @st.fragment
        def _render_inspector():
            """Inspector aislado: navegación con ←/→ y selectbox solo
            rerunna este fragmento, NO redibuja el mosaico de arriba.
            (El click sobre un parche del mosaico Plotly sí dispara un
            rerun completo de la página por limitación de Streamlit.)"""
            st.markdown(
                "**Inspeccionar parche en detalle** — *click en un parche del "
                "mosaico para inspeccionarlo, o usa los controles abajo*"
            )

            def _nav_idx(delta: int, key: str, n_max: int) -> None:
                cur = st.session_state.get(key, -1)
                st.session_state[key] = max(0, min(n_max - 1, cur + delta))

            ctrl_cols = st.columns([1, 6, 1])
            with ctrl_cols[0]:
                st.button(
                    "←", key=f"prev_{sel_key}",
                    disabled=(st.session_state[sel_key] <= 0),
                    on_click=_nav_idx, args=(-1, sel_key, n),
                )
            with ctrl_cols[1]:
                options = [-1] + list(range(n))
                st.selectbox(
                    "Índice del parche (—  = ocultar)",
                    options=options,
                    format_func=lambda i: "—" if i == -1 else f"#{i}",
                    key=sel_key,
                    label_visibility="collapsed",
                )
            with ctrl_cols[2]:
                st.button(
                    "→", key=f"next_{sel_key}",
                    disabled=(st.session_state[sel_key] >= n - 1),
                    on_click=_nav_idx, args=(1, sel_key, n),
                )

            idx = st.session_state[sel_key]
            if 0 <= idx < n:
                patch_full = np.asarray(patches_arr[idx])
                pred_probs = patch_eval.get("pred_probs")
                probs_str = ""
                if pred_probs is not None:
                    pp = np.asarray(pred_probs)[idx]
                    probs_str = " · ".join(f"{c}={p:.3f}" for c, p in zip(CLASS_NAMES, pp))
                pred_cls = CLASS_NAMES[int(pred_index[idx])]
                color_hex = CLASS_COLORS[pred_cls]

                gt_label = None
                cats_ternary = patch_eval.get("cats_ternary")
                if cats_ternary is not None:
                    cat = str(cats_ternary[idx])
                    if cat in CLASS_NAMES:
                        gt_label = cat
                    elif cat == "EXCLUDED":
                        cats_raw = patch_eval.get("cats_raw")
                        if cats_raw is not None:
                            gt_label = f"excluido ({str(cats_raw[idx])})"

                att_val = float(attention[idx]) if attention is not None else None
                att_rel = (
                    att_val / float(attention.max())
                    if (att_val is not None and attention.max() > 0)
                    else None
                )

                img_col, info_col = st.columns([1, 1])
                with img_col:
                    uri = _patch_to_data_uri(patch_full)
                    st.markdown(
                        f'<img src="{uri}" style="width:100%;border:4px solid {color_hex};'
                        f'border-radius:6px;">'
                        f'<div style="text-align:center;font-size:0.9rem;'
                        f'color:#555;margin-top:6px;">parche #{idx} · '
                        f'tamaño nativo {patch_full.shape[1]}×{patch_full.shape[0]} px</div>',
                        unsafe_allow_html=True,
                    )
                with info_col:
                    st.metric("Predicción F4", pred_cls)
                    if probs_str:
                        st.caption(f"Probabilidades F4: {probs_str}")
                    if att_val is not None:
                        delta_text = (
                            f"{att_rel:.0%} del máximo" if att_rel is not None else None
                        )
                        st.metric(
                            "Atención AttnMIL",
                            f"{att_val:.4f}",
                            delta=delta_text,
                            delta_color="off",
                        )
                    if gt_label is not None:
                        st.metric("GT (etiqueta del H5)", gt_label)
                    pos_y, pos_x = int(positions[idx][0]), int(positions[idx][1])
                    st.caption(f"Posición en slide: y={pos_y}, x={pos_x}")

        _render_inspector()


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
        "*softmax* del ensemble de 5 modelos *AttnMIL* en la clase predicha. "
        "Un valor alto indica que los 5 modelos coinciden con *softmax* "
        "saturado, **no** que la predicción sea correcta esa proporción de "
        "veces. El *softmax* no está calibrado: interprétalo como "
        "**seguridad relativa del modelo**, no como certeza diagnóstica.\n\n"
        "**TFM vs producción.** La memoria del TFM (§5.9) reporta "
        "**92,8 ± 1,1 %** *accuracy* mediante validación cruzada 5-fold "
        "*multi-seed* sobre los 91 portaobjetos clínicos del HUC: esa es la "
        "estimación honest del rendimiento esperado sobre portaobjetos "
        "**nuevos**. El *ensemble* desplegado en esta app es un "
        "reentrenamiento posterior de **5** modelos sobre los 91 completos "
        "**sin holdout** (práctica estándar al pasar de evaluación a "
        "producción) — sobre portaobjetos del propio cohort §5.9 las "
        "predicciones serán muy seguras (todos los modelos los vieron en "
        "*training*), pero esa cifra **no es comparable** con §5.9. "
        "**Para portaobjetos nuevos esperar ~92,8 % accuracy.**\n\n"
        "Las barras de error de la sección siguiente miden la dispersión "
        "entre los 5 modelos del *ensemble*: una *std* alta indica "
        "desacuerdo entre miembros."
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

    # ─── Visor WSI profesional (OpenSeadragon) si hay DZI ───────────────────
    # Solo se genera DZI cuando se sube un TIFF original (los H5 ya parcheados
    # no tienen WSI completo). Si existe, lo mostramos antes de los mosaicos
    # de atención/predicciones para que sea la primera vista del patólogo.
    # Cargamos posiciones y predicciones por parche para dibujar overlays
    # coloreados sobre el WSI directamente (verde NOR / naranja ADE / azul CAR).
    _patch_eval_for_osd = _load_patch_eval(job)
    _h5_meta_for_osd = _load_h5_meta(job)
    osd_positions = _h5_meta_for_osd[0] if _h5_meta_for_osd is not None else None
    osd_pred_index = (
        np.asarray(_patch_eval_for_osd["pred_index"], dtype=np.int64)
        if _patch_eval_for_osd is not None and "pred_index" in _patch_eval_for_osd
        else None
    )
    osd_patch_size = result.get("patch_raw_size")
    osd_offset = (
        int(job.extra.get("dzi_y_min", 0)),
        int(job.extra.get("dzi_x_min", 0)),
    )
    if _render_openseadragon_viewer(
        job,
        positions=osd_positions,
        pred_index=osd_pred_index,
        patch_raw_size=osd_patch_size,
        dzi_offset=osd_offset,
    ):
        st.caption(
            "Visor profesional con tiles multi-resolución (OpenSeadragon). "
            "Pan con arrastrar, zoom con rueda. Cada parche se dibuja con un "
            "borde del color de la clase predicha (verde NOR / naranja ADE / "
            "azul CAR). Los mosaicos coloreados de atención y predicciones "
            "siguen disponibles más abajo."
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
    # Selector de resolución del mosaico (radio horizontal, en su propia
    # sección para que el patólogo pueda subir solo este mapa sin afectar
    # el de predicciones por parche).
    thumb_options = {"48 px": 48, "64 px": 64, "96 px": 96, "128 px": 128}
    thumb_label = st.radio(
        "Resolución del mapa de atención",
        options=list(thumb_options.keys()),
        index=0,
        horizontal=True,
        key=f"thumb_size_att_{job.job_id}",
        help=(
            "Tamaño de cada parche en el mosaico. Resolución alta = más "
            "detalle al hacer zoom, pero más tiempo de render en slides "
            "con muchos parches."
        ),
    )
    thumb_size_att = thumb_options[thumb_label]

    # Cargamos originals una sola vez para reusar luego en el mapa de
    # predicciones por parche (sección de abajo).
    originals = _load_all_originals(job)
    patches_arr, patch_size = originals if originals is not None else (None, None)

    with st.spinner(
        f"Generando overlay de atención ({n_patches} parches a {thumb_size_att} px)…"
    ):
        if patches_arr is not None and len(patches_arr) == len(attention):
            fig_overlay = _attention_overlay_figure_cached(
                job_id=job.job_id,
                thumb_size=thumb_size_att,
                opacity=0.85,
                pred_class=pred_class,
                _patches_arr=patches_arr,
                _positions=positions,
                _attention=attention,
                patch_raw_size=patch_size,
            )
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

    # Predicciones por parche del clasificador F4: distribución +
    # mapa de slide coloreado por clase predicha (verde/naranja/azul).
    # Validación con matriz de confusión solo si el H5 trae GT. Slot fijo
    # con st.empty() para que Streamlit reconcilie limpio al cambiar de slide.
    patch_section_slot = st.empty()
    patch_eval = _load_patch_eval(job)
    if patch_eval is not None:
        with patch_section_slot.container():
            # Reusamos los originals + patch_size del overlay de atención
            # (ya cargados arriba) para no leer el H5 dos veces.
            originals_data = originals if originals is not None else (None, None)
            _render_patch_predictions(
                patch_eval,
                positions=positions,
                patches_arr=originals_data[0] if originals_data[0] is not None else None,
                patch_size=originals_data[1] if originals_data[1] is not None else None,
                attention=attention,
                job_id=job.job_id,
            )
            if result.get("has_patch_gt"):
                _render_patch_validation(patch_eval, result)
