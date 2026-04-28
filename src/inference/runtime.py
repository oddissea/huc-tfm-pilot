"""Singleton thread-safe de los modelos F4 + ensemble AttnMIL.

Streamlit decora con `@st.cache_resource` para sus runs, pero el worker
corre en un thread daemon sin contexto Streamlit. Este módulo expone los
modelos a ambos sin duplicar GPU memory.

Uso:
    from src.inference.runtime import load_models, get_models, models_loaded

    if not models_loaded():
        load_models(progress_cb=...)

    f4, ensemble = get_models()
"""

from __future__ import annotations

import logging
import threading
from typing import Callable

import torch

from src.inference.model import AttnMILBundle, F4Bundle, load_attnmil_ensemble, load_f4
from src.inference.weights import ensure_weights

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_f4: F4Bundle | None = None
_ensemble: list[AttnMILBundle] | None = None


def models_loaded() -> bool:
    return _f4 is not None and _ensemble is not None


def load_models(progress_cb: Callable[[int, int, str], None] | None = None) -> None:
    """Carga F4 + ensemble en GPU. Idempotente."""
    global _f4, _ensemble
    with _lock:
        if _f4 is not None and _ensemble is not None:
            logger.info("Modelos ya cargados, no recargo.")
            return

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Cargando modelos en %s", device)
        paths = ensure_weights(progress_cb=progress_cb)
        _f4 = load_f4(paths["f4"], device=device)
        _ensemble = load_attnmil_ensemble(paths["attnmil"], device=device)
        logger.info("Modelos listos: F4 + %d AttnMIL", len(_ensemble))


def get_models() -> tuple[F4Bundle, list[AttnMILBundle]]:
    """Devuelve los modelos. Lanza RuntimeError si no están cargados."""
    if _f4 is None or _ensemble is None:
        raise RuntimeError("Modelos no cargados. Llama a load_models() primero.")
    return _f4, _ensemble


def try_get_models() -> tuple[F4Bundle, list[AttnMILBundle]] | None:
    """Como get_models() pero devuelve None en vez de lanzar."""
    if _f4 is None or _ensemble is None:
        return None
    return _f4, _ensemble
