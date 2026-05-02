"""Append-only store de correcciones del patólogo a nivel de parche.

Estructura: un fichero JSONL por slide (`<job_dir>/corrections.jsonl`)
con una entrada por corrección. Cada `record_correction` añade una línea
sin sobreescribir las anteriores — la última corrección de un parche es
la que cuenta (semántica de overwrite-by-recency en consumidores).

Schema de cada entrada (campos obligatorios marcados con *):

    {
      "slide_uuid": "<job_id>"*,
      "patch_idx": <int>*,                  # índice dentro del H5
      "label_corr": "ADE|NOR|CAR|HIP|ART|EXCLUDED"*,
      "pred_orig": "ADE|NOR|CAR",            # la que dio el modelo
      "probs_orig": [p_ade, p_nor, p_car],   # softmax F4 por parche
      "patologo_id": "eduardo",              # usuario BasicAuth
      "ts": "2026-05-02T11:23:00Z",          # ISO 8601 UTC
      "model_version": "head_v1+attnmil_v1", # bundle activo cuando se hizo
      "comment": "" | "<texto libre>",
      "source": "patologo_corregido" |       # corrección manual
                "auto_inherited"             # heredado del slide-GT (alta confianza)
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

# Etiquetas válidas en una corrección. ADE/NOR/CAR son las clases del
# modelo ternario. HIP/ART las acepta el patólogo si quiere marcar
# explícitamente "hiperplasia" o "artefacto"; en el reentrenamiento
# se mapean a EXCLUIR (no entran en la matriz ternaria) — la opción
# "EXCLUDED" explícita sirve para descartar parches por borde, fondo
# blanco o cualquier otra razón.
CORRECTION_LABELS = ("ADE", "NOR", "CAR", "HIP", "ART", "EXCLUDED")

# Fuentes posibles de la corrección. "patologo_corregido" para clicks
# explícitos del patólogo; "auto_inherited" para parches de alta
# confianza cuya predicción coincide con el slide-GT y se aceptan
# automáticamente como label patch-level (mina de oro §5.9).
SOURCE_PATOLOGO = "patologo_corregido"
SOURCE_AUTO_INHERITED = "auto_inherited"

CORRECTIONS_FILENAME = "corrections.jsonl"

# Lock global para el writer. Las correcciones llegan desde el thread
# de Streamlit y la escritura es por slide pero queremos consistencia
# si en algún momento se paraleliza por slide_uuid distinto.
_write_lock = threading.Lock()


@dataclass
class Correction:
    slide_uuid: str
    patch_idx: int
    label_corr: str
    pred_orig: str | None
    probs_orig: list[float] | None
    patologo_id: str
    ts: str
    model_version: str
    comment: str
    source: str

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def _corrections_path(job_dir: Path) -> Path:
    return job_dir / CORRECTIONS_FILENAME


def record_correction(
    job_dir: Path,
    *,
    slide_uuid: str,
    patch_idx: int,
    label_corr: str,
    pred_orig: str | None = None,
    probs_orig: list[float] | None = None,
    patologo_id: str = "anon",
    model_version: str = "unknown",
    comment: str = "",
    source: str = SOURCE_PATOLOGO,
) -> Correction:
    """Registra una corrección al final del JSONL del slide.

    No valida que `patch_idx` esté en rango — el caller (UI del visor)
    ya conoce `n_patches`. Los duplicados sobre el mismo `patch_idx` son
    legítimos: el consumidor (script de fine-tune) deduplica por última.
    """
    if label_corr not in CORRECTION_LABELS:
        raise ValueError(
            f"label_corr={label_corr!r} no válido; debe ser uno de {CORRECTION_LABELS}"
        )
    correction = Correction(
        slide_uuid=slide_uuid,
        patch_idx=int(patch_idx),
        label_corr=label_corr,
        pred_orig=pred_orig,
        probs_orig=probs_orig,
        patologo_id=patologo_id,
        ts=datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        model_version=model_version,
        comment=comment,
        source=source,
    )
    path = _corrections_path(job_dir)
    with _write_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(correction.to_jsonl() + "\n")
    logger.info(
        "Corrección %s patch=%d %s→%s (%s)",
        slide_uuid[:8], patch_idx, pred_orig or "?", label_corr, source,
    )
    return correction


def list_corrections(job_dir: Path) -> list[Correction]:
    """Lee todas las correcciones del slide. Vacío si el JSONL no existe."""
    path = _corrections_path(job_dir)
    if not path.exists():
        return []
    out: list[Correction] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                out.append(Correction(**d))
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning("Corrección malformada en %s: %s", path, e)
    return out


def summarize_corrections(job_dir: Path) -> dict:
    """Devuelve un resumen por clase corregida + nº de correcciones únicas
    (deduplicado por `patch_idx`, quedando la última)."""
    corrections = list_corrections(job_dir)
    if not corrections:
        return {"n_total": 0, "n_unique_patches": 0, "by_label": {}, "by_source": {}}

    # Deduplicar por patch_idx, conservando la última (orden de aparición = orden temporal)
    last_by_patch: dict[int, Correction] = {}
    for c in corrections:
        last_by_patch[c.patch_idx] = c

    by_label: dict[str, int] = {}
    by_source: dict[str, int] = {}
    for c in last_by_patch.values():
        by_label[c.label_corr] = by_label.get(c.label_corr, 0) + 1
        by_source[c.source] = by_source.get(c.source, 0) + 1

    return {
        "n_total": len(corrections),
        "n_unique_patches": len(last_by_patch),
        "by_label": by_label,
        "by_source": by_source,
    }
