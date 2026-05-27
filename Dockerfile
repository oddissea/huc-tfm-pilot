FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime
# CUDA 12.8 + PyTorch 2.7 son la versión mínima que soporta RTX 50xx
# (Blackwell, compute capability SM 12.0). El equipo HUC tiene una RTX
# 5070 12 GB; cualquier imagen anterior daba "no kernel image is
# available for execution on the device" al primer forward en GPU.

# System deps:
# - libglib2.0-0: required by opencv-python-headless at runtime
# - ca-certificates, curl: HTTPS calls to GCS / pip indexes
# - libvips42, libvips-tools: necesarios para pyvips (genera tiles DZI
#   para el visor OpenSeadragon de slides en alta resolución)
# - nginx-light: servidor delante de Streamlit (deploy HUC PC v1.0.3+).
#   Replica la arquitectura "nginx delante" del deploy cloud para que
#   tanto Streamlit como los DZIs estén en el mismo origen → cero
#   problemas CORS, idéntico al setup VM en producción.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libvips42 \
    libvips-tools \
    ca-certificates \
    curl \
    nginx-light \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Source code (model classes, tiff_to_h5).
COPY src/ ./src/

# Streamlit config (upload limits, telemetría).
COPY .streamlit/ ./.streamlit/

# Assets (logo, etc.).
COPY assets/ ./assets/

# App entry point.
COPY app.py .

# Páginas adicionales del sidebar (multipage Streamlit). Streamlit las
# autodescubre si están en ./pages/ junto al script principal. Hoy
# contiene ⚙️ Configuración para auto-servicio del operador.
COPY pages/ ./pages/

# CLI scripts (archive_jobs, entrypoint, ...). Permite invocar
# `docker compose exec app python -m scripts.archive_jobs` desde cron en
# el host de producción (red de seguridad del hook del worker).
# serve_dzi.py se conserva pero NO se usa en v1.0.3+ (nginx interno lo
# reemplaza). Lo dejamos como artefacto histórico por si en algún
# escenario futuro hace falta el modo "sin nginx".
COPY scripts/ ./scripts/
RUN chmod +x /app/scripts/entrypoint.sh /app/scripts/serve_dzi.py

# Configuración nginx interna que sirve tanto Streamlit (proxy a
# 127.0.0.1:8501) como los DZIs (alias /dzi/ → /tmp/queue/). Un solo
# origen para el browser, equivalente al setup de la VM cloud.
COPY nginx-internal/nginx.conf /etc/nginx/nginx.conf
RUN mkdir -p /var/log/nginx /run /tmp/queue

# URL relativa para los DZIs (resuelto por nginx interno bajo /dzi/).
# En cloud deploy con docker-compose + nginx externo, mismo valor.
ENV DZI_BASE_URL=/dzi

# Solo puerto 80 expuesto al host. nginx interno multiplexa Streamlit
# (que escucha solo en 127.0.0.1:8501, no expuesto) y DZIs.
EXPOSE 80

# Entrypoint lanza nginx en background + streamlit en foreground.
# Streamlit bindea solo a 127.0.0.1 para que NO sea accesible
# directamente desde fuera del container — el único punto de entrada
# es nginx, que aplica la lógica de routing.
CMD ["/app/scripts/entrypoint.sh"]
