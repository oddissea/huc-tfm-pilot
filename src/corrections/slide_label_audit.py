"""Audit log de asignaciones/correcciones de etiqueta slide-level.

Paralelo a `corrections.jsonl` (que registra correcciones patch-level),
este módulo persiste cada asignación o cambio de la etiqueta clínica
de un portaobjetos. Útil para:

- Trazabilidad: cuándo, quién, qué cambió.
- Reentrenamiento futuro del AttnMIL slide-level: la última entrada
  por slide es el `slide_gt` definitivo; las entradas previas y la
  predicción original del modelo dan contexto para análisis de
  discrepancias.

Schema por entrada (JSONL, una línea por evento):

    {
      "slide_uuid": "<job_id>",
      "action": "asignada" | "cambiada",
      "label_to": "ADE" | "NOR" | "CAR",
      "label_from": "<previa>" | null,    # null si action == "asignada"
      "pred_orig": "<predicción del modelo en el momento>",
      "pred_orig_probs": [p_ADE, p_NOR, p_CAR],
      "patologo_id": "<usuario>",
      "ts": "2026-05-03T15:42:00Z",
      "comment": "" | "<texto libre>"
    }
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

SLIDE_LABEL_AUDIT_FILENAME = "slide_label_audit.jsonl"

ACTION_UPLOAD = "upload"      # etiqueta del radio al subir el slide
ACTION_ASSIGNED = "asignada"  # primera asignación desde el panel del detalle
ACTION_CHANGED = "cambiada"   # cambio sobre una etiqueta previa

_write_lock = threading.Lock()


@dataclass
class SlideLabelEntry:
    slide_uuid: str
    action: str
    label_to: str
    label_from: str | None
    pred_orig: str | None
    pred_orig_probs: list[float] | None
    patologo_id: str
    ts: str
    comment: str

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def _audit_path(job_dir: Path) -> Path:
    return job_dir / SLIDE_LABEL_AUDIT_FILENAME


def record_slide_label(
    job_dir: Path,
    *,
    slide_uuid: str,
    label_to: str,
    label_from: str | None = None,
    pred_orig: str | None = None,
    pred_orig_probs: list[float] | None = None,
    patologo_id: str = "anon",
    comment: str = "",
    action: str | None = None,
) -> SlideLabelEntry:
    """Registra asignación (label_from=None) o cambio (label_from=<previa>)
    de la etiqueta slide-level. Por defecto action se infiere:
    - asignada si label_from is None
    - cambiada si label_from != None
    Pasa action='upload' explícitamente si la entrada viene del radio del
    upload (no del panel del detalle)."""
    if action is None:
        action = ACTION_ASSIGNED if label_from is None else ACTION_CHANGED
    entry = SlideLabelEntry(
        slide_uuid=slide_uuid,
        action=action,
        label_to=label_to,
        label_from=label_from,
        pred_orig=pred_orig,
        pred_orig_probs=pred_orig_probs,
        patologo_id=patologo_id,
        ts=datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        comment=comment,
    )
    path = _audit_path(job_dir)
    with _write_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(entry.to_jsonl() + "\n")
    logger.info(
        "Slide label %s %s: %s → %s (modelo predijo: %s)",
        slide_uuid[:8], action,
        label_from or "—", label_to, pred_orig or "?",
    )
    return entry


def list_slide_label_history(job_dir: Path) -> list[SlideLabelEntry]:
    """Lee el historial completo de asignaciones/cambios para el slide."""
    path = _audit_path(job_dir)
    if not path.exists():
        return []
    out: list[SlideLabelEntry] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                out.append(SlideLabelEntry(**d))
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning("slide_label_audit entrada malformada: %s", e)
    return out


def latest_slide_label_entry(job_dir: Path) -> SlideLabelEntry | None:
    """Última entrada del audit log (la más reciente). None si no existe."""
    history = list_slide_label_history(job_dir)
    return history[-1] if history else None
