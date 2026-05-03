"""Captura de correcciones del patólogo (Fase 0 del flujo human-in-the-loop).

El módulo no toca el modelo. Sólo persiste correcciones a nivel de parche
en `<job_dir>/corrections.jsonl` para que un proceso offline (`trainer`
container, ver `docs/deployment/MEJORA_CON_CORRECCIONES.md`) pueda
consumirlas más adelante para fine-tunes del head F4 + AttnMIL.
"""

from src.corrections.slide_label_audit import (
    SlideLabelEntry,
    latest_slide_label_entry,
    list_slide_label_history,
    record_slide_label,
)
from src.corrections.store import (
    CORRECTION_LABELS,
    Correction,
    list_corrections,
    record_correction,
    summarize_corrections,
)

__all__ = [
    "CORRECTION_LABELS",
    "Correction",
    "SlideLabelEntry",
    "latest_slide_label_entry",
    "list_corrections",
    "list_slide_label_history",
    "record_correction",
    "record_slide_label",
    "summarize_corrections",
]
