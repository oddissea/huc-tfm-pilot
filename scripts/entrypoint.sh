#!/bin/bash
# Entrypoint del container huc-pilot (v1.0.3+).
#
# Lanza nginx en background (puerto 80, multiplexa Streamlit + DZIs) +
# streamlit en foreground (127.0.0.1:8501, solo accesible para nginx
# interno, NO expuesto al host).
#
# Arquitectura interna:
#
#   browser (Eduardo en HUC PC)
#         │
#         ▼
#   host:80 ◄──── docker -p 80:80 (o -p 8080:80)
#         │
#         ▼
#   nginx (escucha en 80, dentro del container)
#     ├── location /     → proxy a 127.0.0.1:8501 (Streamlit)
#     └── location /dzi/ → alias /tmp/queue/ (static, DZIs y tiles)
#
# Beneficio: Streamlit y DZIs bajo el MISMO origen → cero CORS issues.
# Replica el setup de la VM cloud sin necesidad de docker compose ni
# certbot (no necesitamos TLS en local).
#
# El container expone solo puerto 80:
#   docker run -p 80:80 ... (o -p 8080:80 si 80 está ocupado en el host)
#
# Si streamlit muere, el container termina (entrypoint hace `exec`).
# `--restart unless-stopped` lo relanza limpiamente.

set -e

# Aseguramos que existe el directorio raíz de DZIs antes de arrancar
# nginx (JobManager también lo crea, pero hay race con Streamlit).
mkdir -p /tmp/queue /var/log/nginx

# nginx en background. -g "daemon off;" mantiene el proceso en
# foreground del shell, pero con & queda en background del entrypoint.
# Logs van a /var/log/nginx/{access,error}.log dentro del container
# (visibles con `docker exec huc-pilot tail -f /var/log/nginx/access.log`).
nginx -g "daemon off;" &

# Streamlit en foreground bindeando a 127.0.0.1 — NO 0.0.0.0. Eso
# significa que Streamlit solo es accesible desde dentro del container
# (donde nginx vive). nginx hace proxy en /. El usuario externo solo
# accede via nginx (puerto 80), nunca directo a Streamlit (puerto 8501).
exec streamlit run app.py --server.port=8501 --server.address=127.0.0.1
