"""Configuración runtime del piloto editable desde la UI.

Vive en un JSON persistente dentro del bind mount de la cola (sobrevive
a reinicios del container). La página `pages/1_configuracion.py` lo
edita; el worker y otros consumers leen los valores en cada uso (NO al
import) para que los cambios surtan efecto sin tener que reiniciar.

Claves actualmente soportadas:

- ``ttl_hours`` (float): umbral de edad para que ``prune()`` borre
  jobs DONE/FAILED. Si no está en el JSON, fallback al env var
  ``PILOT_TTL_HOURS`` (default 24.0).

Diseño:

- **Lectura tolerante**: si el fichero no existe o está corrupto,
  devolvemos `{}` y loguemos warning. La app sigue funcionando con
  defaults.
- **Escritura atómica**: ``.tmp`` + ``os.replace`` evita configs medio
  escritas si el proceso muere a mitad.
- **Merge incremental**: ``save_config(updates)`` hace merge con el
  contenido previo en lugar de overwrite total. Permite que distintas
  páginas/componentes escriban claves distintas sin pisarse.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(
    os.environ.get("PILOT_CONFIG_PATH", "/tmp/queue/.pilot_config.json")
)


def _config_path() -> Path:
    return DEFAULT_CONFIG_PATH


def load_config() -> dict:
    """Lee el JSON de configuración. Devuelve `{}` si no existe o está corrupto."""
    path = _config_path()
    if not path.exists():
        return {}
    try:
        with path.open() as f:
            data = json.load(f)
        if not isinstance(data, dict):
            logger.warning("config %s no es un dict, ignorando", path)
            return {}
        return data
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("config %s ilegible (%s), usando defaults", path, e)
        return {}


def save_config(updates: dict) -> None:
    """Persiste un merge incremental con el config previo.

    Args:
        updates: claves a actualizar. Las que no aparezcan se conservan.
                 Para borrar una clave, pasar `None` como valor.
    """
    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    current = load_config()
    for k, v in updates.items():
        if v is None:
            current.pop(k, None)
        else:
            current[k] = v
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(current, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)
    logger.info("config actualizada: %s", list(updates.keys()))


def get_ttl_hours() -> float:
    """Devuelve el TTL del prune en horas, con cascada de precedencia:

    1. JSON persistente (clave ``ttl_hours``).
    2. Env var ``PILOT_TTL_HOURS``.
    3. Default 24.0.
    """
    cfg = load_config()
    if "ttl_hours" in cfg:
        try:
            return float(cfg["ttl_hours"])
        except (TypeError, ValueError):
            logger.warning("ttl_hours del config no es numérico (%r), ignorando", cfg["ttl_hours"])
    return float(os.environ.get("PILOT_TTL_HOURS", "24.0"))


def set_ttl_hours(hours: float) -> None:
    """Persiste el TTL del prune. Acepta floats; clamp [0, 1000] para evitar
    accidentes de UI."""
    h = max(0.0, min(1000.0, float(hours)))
    save_config({"ttl_hours": h})
