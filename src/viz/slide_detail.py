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

import json
from typing import TYPE_CHECKING

import h5py
import numpy as np
import plotly.graph_objects as go
import streamlit as st

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


def _attention_scatter(
    positions: np.ndarray,
    attention: np.ndarray,
    categories: np.ndarray,
) -> go.Figure:
    """Scatter de los parches sobre el plano del slide, coloreado por atención.

    `positions` se asume (N, 2) con (y, x) esquinas. Invertimos Y para que el
    norte del slide quede arriba (consistente con cómo se ven las miniaturas
    de microscopía).
    """
    if positions.shape[1] >= 2:
        ys, xs = positions[:, 0], positions[:, 1]
    else:
        # fallback raro
        ys, xs = np.arange(len(attention)), np.zeros(len(attention))

    # Normalizar atención a [0,1] para el colorbar (relativa a este slide)
    if attention.max() > attention.min():
        attn_norm = (attention - attention.min()) / (attention.max() - attention.min())
    else:
        attn_norm = np.zeros_like(attention)

    has_labels = bool((categories != "?").any() and (categories != "XXX").any())
    customdata = np.stack(
        [attention, attn_norm, categories], axis=-1,
    ) if has_labels else np.stack([attention, attn_norm], axis=-1)

    hover = (
        "x=%{x}, y=%{y}<br>"
        "atención=%{customdata[0]:.4f} (norm %{customdata[1]:.2f})"
        + ("<br>cat=%{customdata[2]}" if has_labels else "")
        + "<extra></extra>"
    )

    fig = go.Figure(go.Scatter(
        x=xs,
        y=-ys,   # invertir Y
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

    # Top-K parches por atención
    st.markdown(f"**Top {top_k} parches por atención del AttnMIL**")
    k = min(top_k, len(attention))
    top_idx = np.argsort(attention)[-k:][::-1].tolist()
    top_patches = _load_top_patches(job, top_idx)

    if top_patches:
        cols = st.columns(k)
        for i, (idx, patch) in enumerate(zip(top_idx, top_patches)):
            with cols[i]:
                cat = categories[idx] if idx < len(categories) else "?"
                cat_label = f" · {cat}" if cat not in ("?", "XXX") else ""
                st.image(
                    patch,
                    caption=f"#{idx} · α={attention[idx]:.4f}{cat_label}",
                    use_container_width=True,
                )

    # Mapa scatter
    st.plotly_chart(
        _attention_scatter(positions, attention, categories),
        use_container_width=True,
    )
