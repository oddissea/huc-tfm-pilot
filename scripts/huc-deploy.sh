#!/usr/bin/env bash
# huc-deploy.sh — wrapper de un solo comando para el primer despliegue
# (o actualización) de DualPath CRC en el ordenador del HUC.
#
# Ejecutar desde una terminal WSL2 / Git Bash apuntando al USB con la
# imagen + checksum. Hace los 4 pasos manuales del USER_GUIDE_EDUARDO.md
# §1-§2 en una sola invocación, con verificación intermedia.
#
# Uso:
#   bash huc-deploy.sh <ruta/al/tar.gz>
#
# Ejemplo:
#   bash huc-deploy.sh /mnt/d/huc-pilot-with-weights.tar.gz
#
# Asume que el repo del piloto ya está clonado en ~/huc-tfm-pilot.

set -euo pipefail

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

if [[ $# -ne 1 ]]; then
    echo "Uso: bash huc-deploy.sh <ruta/al/huc-pilot-with-weights.tar.gz>"
    echo ""
    echo "Ejemplo:"
    echo "  bash huc-deploy.sh /mnt/d/huc-pilot-with-weights.tar.gz"
    exit 1
fi

TARBALL="$1"
CHECKSUM="${TARBALL}.sha256"
PILOT_REPO="${PILOT_REPO:-${HOME}/huc-tfm-pilot}"

# ---------------------------------------------------------------------------
# Pre-checks
# ---------------------------------------------------------------------------

log() { echo "[$(date +%H:%M:%S)] $*"; }

if [[ ! -f "${TARBALL}" ]]; then
    log "ERROR: no encuentro ${TARBALL}"
    exit 1
fi

if [[ ! -f "${CHECKSUM}" ]]; then
    log "ERROR: no encuentro ${CHECKSUM} (necesario para verificar integridad)"
    exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
    log "ERROR: docker no está instalado. Avisa a IT del HUC o a Nasser."
    exit 1
fi

if ! docker info >/dev/null 2>&1; then
    log "ERROR: Docker daemon no responde. Abre Docker Desktop primero."
    exit 1
fi

if [[ ! -d "${PILOT_REPO}/pilot" ]]; then
    log "ERROR: no encuentro ${PILOT_REPO}/pilot"
    log "       Ajusta PILOT_REPO=/ruta/al/repo o clona el repo primero:"
    log "       git clone https://github.com/oddissea/huc-tfm-pilot.git ~/huc-tfm-pilot"
    exit 1
fi

# ---------------------------------------------------------------------------
# Despliegue
# ---------------------------------------------------------------------------

log "Verificando integridad del fichero (30-60s)..."
( cd "$(dirname "${TARBALL}")" && sha256sum -c "$(basename "${CHECKSUM}")" )

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
