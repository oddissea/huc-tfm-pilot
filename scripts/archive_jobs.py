#!/usr/bin/env python3
"""CLI wrapper de ``src.corrections.archive``.

Permite invocar el archivado manualmente o desde cron. La lógica core
vive en ``src/corrections/archive.py``; aquí solo argumentos y logging.

Uso::

    python -m scripts.archive_jobs                            # default
    python -m scripts.archive_jobs --dry-run                  # sim
    python -m scripts.archive_jobs --archive-dir /otra/ruta   # override
    python -m scripts.archive_jobs --verbose                  # debug

Exit code 0 si todo OK (o sin nada que archivar), 1 si hubo errores en
al menos un job.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Si se ejecuta como script suelto (no como módulo), añadir el repo
# root al sys.path para que el import de src.corrections funcione.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.corrections.archive import DEFAULT_ARCHIVE_DIR, archive_all  # noqa: E402

logger = logging.getLogger("archive_jobs")

# Orden de precedencia para la queue dir:
#   1. --queue-dir explícito en CLI.
#   2. $PILOT_QUEUE_DIR (lo seteamos en docker-compose para que
#      dentro del container resuelva a /tmp/queue).
#   3. ~/huc-tfm-pilot/queue (default razonable en host).
DEFAULT_QUEUE_DIR = Path(
    os.environ.get("PILOT_QUEUE_DIR")
    or Path.home() / "huc-tfm-pilot" / "queue"
)


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
        "--archive-dir", type=Path, default=DEFAULT_ARCHIVE_DIR,
        help=f"Directorio del archive local (default: {DEFAULT_ARCHIVE_DIR})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Simula sin copiar nada (útil para auditar qué se archivaría).",
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
        "Archivando %s → %s (dry_run=%s)",
        args.queue_dir, args.archive_dir, args.dry_run,
    )

    results = archive_all(args.queue_dir, args.archive_dir, dry_run=args.dry_run)

    n_archived_corr = sum(1 for r in results if r["archived_corrections"])
    n_archived_feat = sum(1 for r in results if r["archived_features"])
    n_errors = sum(1 for r in results if r["error"])
    total_corr = sum(r["n_corrections"] for r in results)

    for r in results:
        if r["error"]:
            logger.error("FAIL %s — %s", r["job_id"], r["error"])
        elif r["archived_corrections"]:
            logger.info(
                "OK   %s — %d correcciones, features=%s, meta=%s",
                r["job_id"], r["n_corrections"],
                r["archived_features"], r["archived_meta"],
            )
        elif r["n_corrections"] > 0:
            logger.debug(
                "SKIP %s — %d correcciones ya al día",
                r["job_id"], r["n_corrections"],
            )

    logger.info(
        "Resumen: %d jobs con correcciones (total %d entries), %d archivados "
        "(%d con features), %d errores.",
        len(results), total_corr, n_archived_corr, n_archived_feat, n_errors,
    )
    return 1 if n_errors > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
