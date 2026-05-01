"""Streamlit custom component que envuelve OpenSeadragon con click-events.

Devuelve `{"idx": int, "ts": int}` cuando el usuario hace click sobre un
parche del visor. El `ts` (timestamp del browser) sirve para que Streamlit
detecte cambios aunque el usuario clique repetidamente el mismo parche.

Estructura: este archivo + carpeta hermana `osd_component/index.html`.
Streamlit sirve el HTML estático a través de su propio handler.
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
    key: str | None = None,
) -> dict | None:
    """Renderiza el visor OpenSeadragon con click-events y toggles.

    Args:
        dzi_url: URL del fichero `.dzi` (relativa al host del piloto).
        overlays: lista de dicts con keys `x, y, size, color, idx, cls`
                  y opcionalmente `att, att_rel, att_fill`. Coordenadas
                  en píxeles del DZI (después de restar el offset stitched).
                  - `color` = stroke (borde) cuando se muestra predicciones.
                  - `att_fill` = fill rgba(...) — color de la clase del
                    slide con alpha proporcional a la atención. Se usa
                    cuando se muestra atención.
        height: altura del visor en px.
        show_predictions: si True (default), dibuja borde por predicción.
        show_attention: si True, dibuja relleno coloreado por atención.
                        Default False (la predicción es la vista por defecto).
        key: identificador único entre instancias del componente.

    Returns:
        None si todavía no ha habido click. `{"idx": <int>, "ts": <int>}`
        del último click.
    """
    return _osd_component(
        dzi_url=dzi_url,
        overlays=overlays,
        height=height,
        show_predictions=show_predictions,
        show_attention=show_attention,
        key=key,
        default=None,
    )
