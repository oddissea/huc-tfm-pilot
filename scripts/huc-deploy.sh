#!/usr/bin/env bash
# huc-deploy.sh — wrapper de un solo comando para el despliegue (o
# actualización) de DualPath CRC en el ordenador del HUC.
#
# Acepta dos formatos de origen del .tar.gz:
#   1. Fichero local ya descargado.
#   2. URL o ID de Google Drive (requiere gdown instalado).
#
# Uso:
#   bash huc-deploy.sh <ruta-fichero-local>
#   bash huc-deploy.sh "<URL-de-Google-Drive>"
#   bash huc-deploy.sh <FILE_ID-de-Drive>
#
# Ejemplos:
#   bash huc-deploy.sh ~/Descargas/huc-pilot-with-weights.tar.gz
#   bash huc-deploy.sh "https://drive.google.com/file/d/1aBcDeF.../view"
#   bash huc-deploy.sh 1aBcDeF...
#
# Asume:
# - El repo del piloto ya está clonado en ~/huc-tfm-pilot.
# - Docker CE + NVIDIA Container Toolkit ya configurados (verificable
#   con `docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi`).

set -euo pipefail

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

if [[ $# -ne 1 ]]; then
    echo "Uso:"
    echo "  bash huc-deploy.sh <ruta-fichero-local>"
    echo "  bash huc-deploy.sh \"<URL-de-Google-Drive>\""
    echo "  bash huc-deploy.sh <FILE_ID-de-Drive>"
    echo ""
    echo "Ejemplos:"
    echo "  bash huc-deploy.sh ~/Descargas/huc-pilot-with-weights.tar.gz"
    echo "  bash huc-deploy.sh \"https://drive.google.com/file/d/1aBc.../view\""
    exit 1
fi

ARG="$1"
PILOT_REPO="${PILOT_REPO:-${HOME}/huc-tfm-pilot}"
DOWNLOAD_DIR="${PILOT_REPO}"

log() { echo "[$(date +%H:%M:%S)] $*"; }

# ---------------------------------------------------------------------------
# Detección de modo: fichero local vs Drive
# ---------------------------------------------------------------------------
#
# Regla:
# - Si el argumento existe como fichero → modo LOCAL.
# - Si contiene "drive.google.com" → modo DRIVE (URL).
# - Si es alfanumérico de >20 caracteres y no contiene `/` → modo DRIVE (ID).
# - Otro caso → error.

MODE=""
TARBALL=""

if [[ -f "${ARG}" ]]; then
    MODE="local"
    TARBALL="${ARG}"
elif [[ "${ARG}" == *"drive.google.com"* ]]; then
    MODE="drive_url"
elif [[ "${ARG}" =~ ^[A-Za-z0-9_-]{20,}$ ]]; then
    MODE="drive_id"
else
    log "ERROR: no reconozco \"${ARG}\" ni como fichero local ni como URL/ID de Drive."
    log "       Verifica que el fichero existe o que el ID/URL está bien formado."
    exit 1
fi

# ---------------------------------------------------------------------------
# Pre-checks: docker y repo
# ---------------------------------------------------------------------------

if ! command -v docker >/dev/null 2>&1; then
    log "ERROR: docker no está instalado. Avisa a Nasser."
    exit 1
fi

if ! docker info >/dev/null 2>&1; then
    log "ERROR: Docker daemon no responde."
    log "       Prueba: sudo systemctl start docker"
    exit 1
fi

if [[ ! -d "${PILOT_REPO}/pilot" ]]; then
    log "ERROR: no encuentro ${PILOT_REPO}/pilot"
    log "       Ajusta PILOT_REPO=/ruta/al/repo o clona el repo primero:"
    log "       git clone https://github.com/oddissea/huc-tfm-pilot.git ~/huc-tfm-pilot"
    exit 1
fi

# ---------------------------------------------------------------------------
# Descarga si es modo Drive
# ---------------------------------------------------------------------------

if [[ "${MODE}" != "local" ]]; then
    if ! command -v gdown >/dev/null 2>&1; then
        log "ERROR: gdown no está instalado (necesario para descargar de Drive)."
        log "       Instala con:"
        log "         sudo apt install -y python3-pip"
        log "         pip install gdown"
        exit 1
    fi

    TARBALL="${DOWNLOAD_DIR}/huc-pilot-with-weights.tar.gz"
    CHECKSUM="${TARBALL}.sha256"

    if [[ "${MODE}" == "drive_url" ]]; then
        DRIVE_REF="${ARG}"
    else
        DRIVE_REF="https://drive.google.com/file/d/${ARG}/view"
    fi

    log "Descargando ${TARBALL} desde Drive (5-15 min según red)..."
    gdown --fuzzy "${DRIVE_REF}" -O "${TARBALL}"

    # El .sha256 conviene tenerlo en la misma carpeta del .tar.gz para
    # que `sha256sum -c` funcione sin más. Si no existe, lo intentamos
    # bajar también: el nombre estándar es el del .tar.gz + ".sha256".
    if [[ ! -f "${CHECKSUM}" ]]; then
        log "AVISO: no encuentro ${CHECKSUM}."
        log "       Si está en la misma carpeta de Drive, descárgalo manualmente"
        log "       y vuelve a lanzar el script en modo local."
        log "       Continuamos sin verificar integridad (riesgo bajo si red no falló)."
    fi
else
    CHECKSUM="${TARBALL}.sha256"
fi

# ---------------------------------------------------------------------------
# Verificación + load + tag + up
# ---------------------------------------------------------------------------

if [[ -f "${CHECKSUM}" ]]; then
    log "Verificando integridad del fichero (30-60s)..."
    ( cd "$(dirname "${TARBALL}")" && sha256sum -c "$(basename "${CHECKSUM}")" )
else
    log "Saltando verificación sha256 (no hay fichero .sha256 disponible)."
fi

log "Cargando imagen al daemon Docker (2-5 min)..."
docker load -i "${TARBALL}"

log "Re-tagueando huc-pilot:dev-with-weights → huc-pilot:dev..."
docker tag huc-pilot:dev-with-weights huc-pilot:dev

log "Levantando el piloto desde ${PILOT_REPO}/pilot..."
cd "${PILOT_REPO}/pilot"
docker compose up -d

log "Esperando 10s a que Streamlit arranque..."
sleep 10

log "Últimas líneas del log:"
docker compose logs app --tail=10

log ""
log "✅ Despliegue completo."
log "   Abre el navegador en http://localhost:8501"
log ""
log "Para apagar al final de la jornada:"
log "   cd ${PILOT_REPO}/pilot && docker compose down"
