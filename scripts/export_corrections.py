#!/usr/bin/env python3
"""CLI wrapper de ``src.corrections.export``.

Permite invocar el export manualmente o desde cron. La lógica core
vive en ``src/corrections/export.py``; aquí solo argumentos y logging.

Uso::

    python -m scripts.export_corrections                       # default
    python -m scripts.export_corrections --dry-run             # sim
    python -m scripts.export_corrections --bucket otro-bucket  # override
    python -m scripts.export_corrections --verbose             # debug

Exit code 0 si todo OK (o sin nada que subir), 1 si hubo errores
en al menos un job.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Si se ejecuta como script suelto (no como módulo), añadir el repo
# root al sys.path para que el import de src.corrections funcione.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.corrections.export import DEFAULT_BUCKET, export_all  # noqa: E402

logger = logging.getLogger("export_corrections")

DEFAULT_QUEUE_DIR = Path.home() / "huc-tfm-pilot" / "queue"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--queue-dir", type=Path, default=DEFAULT_QUEUE_DIR,
        help=f"Directorio de la cola (default: {DEFAULT_QUEUE_DIR})",
    )
    parser.add_argument(
        "--bucket", default=DEFAULT_BUCKET,
        help=f"Nombre del bucket GCS (default: {DEFAULT_BUCKET})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Simula sin subir nada (útil para auditar qué se subiría).",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Log a nivel DEBUG.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if not args.queue_dir.exists():
        logger.error("queue_dir no existe: %s", args.queue_dir)
        return 1

    logger.info(
        "Exportando %s → gs://%s/ (dry_run=%s)",
        args.queue_dir, args.bucket, args.dry_run,
    )

    results = export_all(args.queue_dir, args.bucket, dry_run=args.dry_run)

    n_uploaded = sum(1 for r in results if r["uploaded_corrections"])
    n_errors = sum(1 for r in results if r["error"])
    total_corr = sum(r["n_corrections"] for r in results)

    for r in results:
        if r["error"]:
            logger.error("FAIL %s — %s", r["job_id"], r["error"])
        elif r["uploaded_corrections"]:
            logger.info(
                "UP   %s — %d correcciones, meta=%s",
                r["job_id"], r["n_corrections"], r["uploaded_meta"],
            )
        elif r["n_corrections"] > 0:
            logger.debug(
                "SKIP %s — %d correcciones ya al día",
                r["job_id"], r["n_corrections"],
            )

    logger.info(
        "Resumen: %d jobs con correcciones (total %d entries), %d subidos, %d errores.",
        len(results), total_corr, n_uploaded, n_errors,
    )
    return 1 if n_errors > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
