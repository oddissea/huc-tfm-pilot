FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# System deps:
# - libglib2.0-0: required by opencv-python-headless at runtime
# - ca-certificates, curl: HTTPS calls to GCS / pip indexes
# - libvips42, libvips-tools: necesarios para pyvips (genera tiles DZI
#   para el visor OpenSeadragon de slides en alta resolución)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libvips42 \
    libvips-tools \
    ca-certificates \
    curl \
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

# CLI scripts (export_corrections, etc.). Permite invocar
# `docker compose exec app python -m scripts.export_corrections` desde
# cron en el host de producción (red de seguridad del hook del worker).
COPY scripts/ ./scripts/

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
