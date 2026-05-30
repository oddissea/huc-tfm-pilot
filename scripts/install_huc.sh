#!/usr/bin/env bash
# install_huc.sh — instalador/actualizador autónomo de DualPath CRC para
# el ordenador del HUC. Hace TODO el ciclo en un solo comando:
#
#   1. para y elimina el container actual (los datos se conservan),
#   2. (opcional) borra imágenes viejas del piloto para liberar disco,
#   3. descarga la imagen nueva desde Google Drive (por FILE_ID),
#   4. verifica el sha256,
#   5. carga la imagen, la re-taguea y lanza el container en el puerto 80,
#   6. comprueba que la app responde.
#
# Uso normal (Eduardo):
#   bash install_huc.sh
#
# Opciones:
#   --purge-images   borra huc-pilot:dev y :dev-with-weights antes de cargar
#                    la nueva (recomendado al actualizar, libera ~14 GB).
#   --port N         publica en el puerto N del host en vez del 80
#                    (p. ej. --port 8080 si el 80 está ocupado).
#   --keep-tar       no borra el .tar.gz descargado al terminar.
#   --allow-lan      publica el piloto en TODA la red (no solo en esta
#                    máquina). NO recomendado en redes inseguras: por
#                    defecto el acceso queda restringido a localhost.
#
# SEGURIDAD: por defecto el piloto solo es accesible desde el propio PC
# (127.0.0.1). En una red insegura (p. ej. la de la ULL) cualquier equipo
# de la red podría abrir la app si se publicara en todas las interfaces;
# por eso el binding por defecto es loopback. Usa --allow-lan solo si de
# verdad necesitas entrar desde otra máquina del grupo.
#
# Requisitos (ya instalados en el HUC PC): Docker CE + NVIDIA Container
# Toolkit + gdown. Si falta gdown:  sudo apt install -y pipx && pipx install gdown
#
# ── FILE_IDs de la versión a instalar (Google Drive) ───────────────────
# Nasser los rellena con cada release y te pasa el script ya listo.
TAR_FILE_ID="${TAR_FILE_ID:-1FVnFE0SU0QowQZwWIX0pzQ6Q1nGdmWov}"
SHA_FILE_ID="${SHA_FILE_ID:-1Ue8UB1nkJTnBOCnNWfLTYLTcWKIPf07y}"
VERSION_LABEL="${VERSION_LABEL:-v1.0.4}"
# ───────────────────────────────────────────────────────────────────────

set -euo pipefail

CONTAINER="huc-pilot"
HOST_PORT=80
PURGE_IMAGES=0
KEEP_TAR=0
DATA_DIR="${HOME}/huc-pilot-data"
TARBALL="${HOME}/huc-pilot-with-weights.tar.gz"
CHECKSUM="${TARBALL}.sha256"

log()  { echo "[$(date +%H:%M:%S)] $*"; }
die()  { echo "[$(date +%H:%M:%S)] ERROR: $*" >&2; exit 1; }

# ── Parseo de opciones ────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --purge-images) PURGE_IMAGES=1; shift ;;
        --keep-tar)     KEEP_TAR=1; shift ;;
        --port)         HOST_PORT="${2:?--port necesita un número}"; shift 2 ;;
        -h|--help)      grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)              die "opción no reconocida: $1 (usa --help)" ;;
    esac
done

if [[ "${TAR_FILE_ID}" == PEGAR_* || "${SHA_FILE_ID}" == PEGAR_* ]]; then
    die "FILE_IDs sin rellenar. Edita TAR_FILE_ID y SHA_FILE_ID arriba, o
       expórtalos:  TAR_FILE_ID=... SHA_FILE_ID=... bash install_huc.sh"
fi

# ── Pre-checks ────────────────────────────────────────────────────────
command -v docker >/dev/null 2>&1 || die "docker no está instalado. Avisa a Nasser."
docker info >/dev/null 2>&1        || die "el daemon de Docker no responde. Prueba: sudo systemctl start docker"
command -v gdown  >/dev/null 2>&1 || die "gdown no instalado. Ejecuta: sudo apt install -y pipx && pipx install gdown && export PATH=\"\$HOME/.local/bin:\$PATH\""

log "Instalando DualPath CRC ${VERSION_LABEL} en el HUC PC."

# ── 1. Parar y eliminar el container actual (datos a salvo) ────────────
if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER}"; then
    log "Parando y eliminando el container actual '${CONTAINER}' (los datos de ${DATA_DIR} se conservan)…"
    docker stop "${CONTAINER}" >/dev/null 2>&1 || true
    docker rm   "${CONTAINER}" >/dev/null 2>&1 || true
else
    log "No hay container '${CONTAINER}' previo (primera instalación)."
fi

# ── 2. (opcional) Borrar imágenes viejas ──────────────────────────────
if [[ "${PURGE_IMAGES}" -eq 1 ]]; then
    log "Borrando imágenes viejas del piloto para liberar disco…"
    docker rmi huc-pilot:dev huc-pilot:dev-with-weights >/dev/null 2>&1 || true
fi

# ── 3. Descargar la imagen + checksum desde Drive ─────────────────────
mkdir -p "${DATA_DIR}/archive" "${DATA_DIR}/queue"
log "Descargando la imagen (~4-5 GB, 5-15 min según la red del HUC)…"
gdown "${TAR_FILE_ID}" -O "${TARBALL}"
log "Descargando el checksum…"
gdown "${SHA_FILE_ID}" -O "${CHECKSUM}"

# ── 4. Verificar integridad ───────────────────────────────────────────
log "Verificando integridad (sha256, 30-60 s)…"
( cd "${HOME}" && sha256sum -c "$(basename "${CHECKSUM}")" ) \
    || die "el sha256 NO coincide → descarga corrupta. Vuelve a lanzar el script."

# ── 5. Cargar, re-taguear y lanzar ────────────────────────────────────
log "Cargando la imagen en Docker (2-5 min)…"
docker load -i "${TARBALL}"
docker tag huc-pilot:dev-with-weights huc-pilot:dev

if [[ -n "${BIND_ADDR}" ]]; then
    log "Lanzando el container en http://localhost:${HOST_PORT} (solo esta máquina)…"
else
    log "Lanzando el container en el puerto ${HOST_PORT} ABIERTO A LA RED (--allow-lan)…"
fi
docker run -d \
    --name "${CONTAINER}" \
    --gpus all \
    -p "${BIND_ADDR}${HOST_PORT}:80" \
    -v "${DATA_DIR}/archive:/var/archive" \
    -v "${DATA_DIR}/queue:/tmp/queue" \
    --restart unless-stopped \
    huc-pilot:dev >/dev/null

# ── 6. Comprobar que responde ─────────────────────────────────────────
log "Esperando a que la app arranque…"
ok=0
for i in $(seq 1 40); do
    code="$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:${HOST_PORT}/_stcore/health" 2>/dev/null || true)"
    if [[ "${code}" == "200" ]]; then ok=1; log "App lista (health 200) tras ${i}s."; break; fi
    sleep 1
done
[[ "${ok}" -eq 1 ]] || log "AVISO: la app no respondió al health en 40 s. Revisa: docker logs ${CONTAINER} --tail 50"

# ── 7. Limpieza del tar.gz ────────────────────────────────────────────
if [[ "${KEEP_TAR}" -eq 0 ]]; then
    log "Borrando el .tar.gz descargado (la imagen ya vive en Docker)…"
    rm -f "${TARBALL}" "${CHECKSUM}"
fi

docker ps --filter "name=${CONTAINER}" --format '  {{.Names}}  {{.Status}}  {{.Ports}}'
log ""
log "✅ Listo. Abre el navegador en  http://localhost$( [[ "${HOST_PORT}" != "80" ]] && echo ":${HOST_PORT}" )"
log "   Apagar al final del día:  docker stop ${CONTAINER}"
log "   Volver a arrancar:        docker start ${CONTAINER}"
