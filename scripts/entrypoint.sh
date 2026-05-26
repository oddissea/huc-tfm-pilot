#!/bin/bash
# Entrypoint del container huc-pilot.
#
# Lanza serve_dzi.py en background (puerto 8888, sirve tiles DZI a
# OpenSeadragon) + streamlit en foreground (puerto 8501).
#
# Sustituye al CMD original "streamlit run app.py ..." en el flujo
# despliegue HUC PC sin nginx ni docker compose. En el despliegue cloud
# con docker compose + nginx, este entrypoint también funciona (el
# serve_dzi en background no estorba, nginx puede seguir sirviendo
# `/dzi/` desde fuera del container si así lo prefieres).
#
# El container debe exponer ambos puertos:
#   docker run -p 8501:8501 -p 8888:8888 ...
#
# Si streamlit muere, el container termina (entrypoint hace `exec`).
# Eso permite que `--restart unless-stopped` relance limpiamente.

set -e

# Aseguramos que existe el directorio raíz de DZIs antes de arrancar
# serve_dzi (JobManager también lo crea, pero hay race con Streamlit).
mkdir -p /tmp/queue

# Servidor DZI en background. Logs a stdout del container (visible con
# `docker logs huc-pilot`).
python /app/scripts/serve_dzi.py &

# Streamlit en foreground. exec sustituye el shell por streamlit → PID 1
# del container, los signals (SIGTERM al docker stop) se propagan bien.
exec streamlit run app.py --server.port=8501 --server.address=0.0.0.0
