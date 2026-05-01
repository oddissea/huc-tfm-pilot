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

# App entry point.
COPY app.py .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
