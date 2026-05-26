#!/usr/bin/env python3
"""HTTP server con CORS para servir tiles DZI a OpenSeadragon.

Sustituye al servidor nginx `/dzi/` del despliegue cloud cuando el piloto
corre en HUC PC sin compose ni nginx (un solo container, dos procesos
internos: streamlit en puerto 8501 + este servidor en 8888).

Sirve `/tmp/queue/` (donde JobManager crea `<job_id>/slide.dzi` y la
carpeta `slide_files/` con los tiles) bajo `http://0.0.0.0:8888/`.

OpenSeadragon en el navegador hace requests a:
- `http://localhost:8888/<job_id>/slide.dzi` (XML pyramid descriptor).
- `http://localhost:8888/<job_id>/slide_files/<level>/<col>_<row>.jpg`
  (tiles, paths relativos al .dzi resueltos automáticamente por OSD).

Headers CORS necesarios porque puerto 8888 (DZI) != puerto 8501
(Streamlit), aunque ambos vivan en `localhost`: para el browser son
orígenes distintos y aplica same-origin policy estricto.
"""

from __future__ import annotations

import os
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler


DZI_ROOT = os.environ.get("DZI_ROOT", "/tmp/queue")
DZI_PORT = int(os.environ.get("DZI_PORT", "8888"))


class CORSRequestHandler(SimpleHTTPRequestHandler):
    """SimpleHTTPRequestHandler con CORS habilitado para cualquier origen.

    Equivalente al `add_header Access-Control-Allow-Origin "*"` que tenía
    el `location /dzi/` de la conf de nginx en el despliegue cloud
    (ver `pilot/nginx/nginx.conf`). Suficiente para uso local (HUC PC).
    En producción multi-tenant habría que restringir Origin al dominio
    concreto.
    """

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "public, max-age=86400")
        super().end_headers()

    def do_OPTIONS(self):
        """Preflight CORS — OpenSeadragon puede hacerlo antes de GET."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()

    def log_message(self, fmt, *args):  # noqa: A003 — match parent signature
        """Silenciar logs (SimpleHTTPRequestHandler es muy verboso por defecto)."""
        return


def main() -> None:
    if not os.path.isdir(DZI_ROOT):
        # Si no existe, lo creamos. JobManager también lo crea al inicializarse
        # pero el race con Streamlit puede hacer que arranquemos antes.
        os.makedirs(DZI_ROOT, exist_ok=True)

    os.chdir(DZI_ROOT)
    print(f"[serve_dzi] Sirviendo {DZI_ROOT} en http://0.0.0.0:{DZI_PORT}/", flush=True)
    sys.stdout.flush()
    HTTPServer(("0.0.0.0", DZI_PORT), CORSRequestHandler).serve_forever()


if __name__ == "__main__":
    main()
