"""Streamlit custom component que envuelve OpenSeadragon con click-events.

Devuelve `{"idx": int, "ts": int}` cuando el usuario hace click sobre un
parche del visor. El `ts` (timestamp del browser) sirve para que Streamlit
detecte cambios aunque el usuario clique repetidamente el mismo parche.

El path-based component se construye contra `osd_component/index.html`,
que implementa el bridge de Streamlit a mano con `postMessage` (no usa
`streamlit-component-lib` porque el SDK no se carga automáticamente en
componentes path-based — fue la causa del fallo en sesión #54).
"""

from __future__ import annotations

from pathlib import Path

import streamlit.components.v1 as components

_COMPONENT_DIR = Path(__file__).parent / "osd_component"
_osd_component = components.declare_component("osd_viewer", path=str(_COMPONENT_DIR))


def osd_viewer(
    dzi_url: str,
    overlays: list[dict],
    height: int = 620,
    show_predictions: bool = True,
    show_attention: bool = False,
    selected_idx: int | None = None,
    view_corrected: bool = False,
    show_selected_borders: bool = True,
    key: str | None = None,
) -> dict | None:
    """Renderiza el visor OpenSeadragon con click-events y toggles.

    Args:
        dzi_url: URL del fichero `.dzi` (relativa al host del piloto).
        overlays: lista de dicts con keys `x, y, size, color, idx, cls`
                  y opcionalmente `att, att_rel, att_fill, probs, pos`.
                  Coordenadas en píxeles del DZI (después de restar el
                  offset stitched).
                  - `color`: stroke (borde) cuando se muestra predicciones.
                  - `att_fill`: fill rgba(...) — color de la clase del slide
                    con alpha proporcional a la atención. Visible sólo en
                    modo atención.
                  - `probs`: [p_ADE, p_NOR, p_CAR] del classifier F4 para el
                    parche, mostrado en el hover.
                  - `pos`: [y, x] en píxeles del slide stitched, para hover.
        height: altura del visor en px.
        show_predictions: si True (default), dibuja borde por predicción.
        show_attention: si True, dibuja relleno coloreado por atención.
        selected_idx: si se pasa, marca ese parche con el highlight amarillo
                      de selección al renderizar (sincroniza estado externo
                      con el visor — p. ej. el selectbox del panel de
                      correcciones).
        key: identificador único entre instancias del componente.

    Returns:
        None si todavía no ha habido click. `{"idx": <int>, "ts": <int>}`
        del último click. El `ts` cambia entre clicks consecutivos sobre
        el mismo parche para forzar el rerun de Streamlit.
    """
    return _osd_component(
        dzi_url=dzi_url,
        overlays=overlays,
        height=height,
        show_predictions=show_predictions,
        show_attention=show_attention,
        selected_idx=selected_idx,
        view_corrected=view_corrected,
        show_selected_borders=show_selected_borders,
        key=key,
        default=None,
    )
