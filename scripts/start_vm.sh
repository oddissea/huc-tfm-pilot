#!/usr/bin/env bash
#
# Arranca la VM `huc-tfm-pilot-vm` iterando las zonas de la región hasta
# encontrar una con capacidad. Maneja automáticamente:
#
#   1. snapshot + migración del disco si la VM no está en la zona donde
#      hay capacidad
#   2. fallback a Spot si todas las zonas on-demand están stockout
#   3. limpieza de intentos fallidos (discos huérfanos, snapshots)
#   4. reasignación de la IP estática y del schedule de auto-stop
#
# Uso:
#   ./scripts/start_vm.sh                     # on-demand, todas las zonas
#   ./scripts/start_vm.sh --spot              # solo Spot (más barato, preemptible)
#   ./scripts/start_vm.sh --on-demand-then-spot   # on-demand primero, Spot fallback
#   ./scripts/start_vm.sh --region=europe-west1   # cambiar región (no implementado, queda como placeholder)
#
# Variables de entorno requeridas:
#   PROJECT_ID, REGION, VM_NAME, STATIC_IP_NAME
#
# Output:
#   En éxito: imprime el comando exacto para conectar por SSH y exporta ZONE.
#   En fallo: deja todo limpio y devuelve 1.

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults y parsing de args
# ---------------------------------------------------------------------------

MODE="on-demand"   # on-demand | spot | on-demand-then-spot
ZONES=("europe-west4-a" "europe-west4-b" "europe-west4-c")

# Nombre real del disco. Se setea en ensure_disk_in() a partir del nombre
# encontrado (puede ser "${VM_NAME}" o "${VM_NAME}-<sufijo-zona>" heredado
# de migraciones anteriores). Sin esta variable, el script asumía que el
# disco se llama igual que la VM y fallaba al buscarlo si tenía sufijo.
DISK_NAME=""

for arg in "$@"; do
    case "$arg" in
        --spot) MODE="spot" ;;
        --on-demand) MODE="on-demand" ;;
        --on-demand-then-spot) MODE="on-demand-then-spot" ;;
        --region=*)
            REGION="${arg#--region=}"
            echo "WARN: --region todavía no implementado por completo (DNS no se actualiza). Avísame si lo necesitas."
            ;;
        *) echo "Arg desconocido: $arg" >&2; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Validación de entorno
# ---------------------------------------------------------------------------

: "${PROJECT_ID:?ERROR: export PROJECT_ID=...}"
: "${REGION:?ERROR: export REGION=...}"
: "${VM_NAME:?ERROR: export VM_NAME=...}"
: "${STATIC_IP_NAME:?ERROR: export STATIC_IP_NAME=...}"

# ---------------------------------------------------------------------------
# Funciones auxiliares
# ---------------------------------------------------------------------------

log() { echo "[$(date +%H:%M:%S)] $*"; }

vm_zone() {
    # Si la VM existe, imprime su zona; si no, vacío.
    gcloud compute instances list --filter="name=${VM_NAME}" --format="value(zone)" 2>/dev/null \
        | head -1 | awk -F/ '{print $NF}'
}

disk_zone() {
    # Si el disco existe, imprime su zona. Acepta el nombre exacto
    # "${VM_NAME}" o variantes "${VM_NAME}-<sufijo>" (legacy de
    # migraciones anteriores donde el disco recreado se llamaba con
    # sufijo de zona).
    gcloud compute disks list \
        --filter="name=${VM_NAME} OR name~^${VM_NAME}-" \
        --format="value(zone)" 2>/dev/null \
        | head -1 | awk -F/ '{print $NF}'
}

disk_real_name() {
    # Si el disco existe, imprime su nombre real (puede tener sufijo).
    # Si no existe, vacío. Usado para acertar los gcloud disks {snapshot,
    # delete} sobre el disco preexistente.
    gcloud compute disks list \
        --filter="name=${VM_NAME} OR name~^${VM_NAME}-" \
        --format="value(name)" 2>/dev/null | head -1
}

find_existing_snapshot() {
    # Si hay un snapshot huc-pilot-snap-* del disco, lo devuelve.
    # Reconoce snapshots de discos con o sin sufijo.
    gcloud compute snapshots list \
        --filter="name~^huc-pilot-snap- AND (sourceDisk~/${VM_NAME}$ OR sourceDisk~/${VM_NAME}-[^/]+$)" \
        --format="value(name)" 2>/dev/null | head -1
}

ensure_disk_in() {
    local target_zone="$1"
    local current_zone current_name
    current_zone=$(disk_zone)
    current_name=$(disk_real_name)

    if [[ "${current_zone}" == "${target_zone}" ]]; then
        log "Disco ya está en ${target_zone} ✓ (nombre real: ${current_name})"
        DISK_NAME="${current_name}"
        return 0
    fi

    # Caso 1: el disco vive en otra zona → snapshot + restore en target_zone.
    # Aprovechamos para normalizar el nombre a "${VM_NAME}" (sin sufijo)
    # tras la migración, ya que el script local crea siempre con ese
    # nombre. Discos con sufijo "${VM_NAME}-<zona>" son legacy.
    if [[ -n "${current_zone}" ]]; then
        log "Migrando disco ${current_name} (${current_zone}) → ${VM_NAME} (${target_zone})"
        local snap_name="huc-pilot-snap-roam-$(date +%s)"

        # Si el disco está atado a una VM, hay que borrar la VM primero (sin perder el disco)
        local vm_z
        vm_z=$(vm_zone)
        if [[ -n "${vm_z}" ]]; then
            log "  borrando VM en ${vm_z} (manteniendo disco)…"
            gcloud compute instances delete "${VM_NAME}" --zone="${vm_z}" --keep-disks=boot --quiet
        fi

        log "  snapshot del disco actual (${current_name})…"
        gcloud compute disks snapshot "${current_name}" --zone="${current_zone}" --snapshot-names="${snap_name}"

        log "  borrando disco viejo ${current_name} en ${current_zone}…"
        gcloud compute disks delete "${current_name}" --zone="${current_zone}" --quiet

        log "  creando disco nuevo ${VM_NAME} en ${target_zone}…"
        gcloud compute disks create "${VM_NAME}" --zone="${target_zone}" --source-snapshot="${snap_name}" --type=pd-ssd

        log "  borrando snapshot temporal…"
        gcloud compute snapshots delete "${snap_name}" --quiet

        DISK_NAME="${VM_NAME}"
        return 0
    fi

    # Caso 2: no hay disco en ningún sitio. Buscamos snapshot del que restaurar.
    local snap_name
    snap_name=$(find_existing_snapshot)
    if [[ -z "${snap_name}" ]]; then
        log "ERROR: no hay disco en ningún sitio ni snapshot del que restaurar."
        return 1
    fi

    log "Restaurando disco ${VM_NAME} en ${target_zone} desde snapshot existente '${snap_name}'…"
    gcloud compute disks create "${VM_NAME}" --zone="${target_zone}" --source-snapshot="${snap_name}" --type=pd-ssd
    DISK_NAME="${VM_NAME}"
    return 0
}

cleanup_orphaned_disk() {
    # Limpia el disco creado en una zona si fallamos creando la VM ahí.
    local zone="$1"
    log "Limpiando disco huérfano en ${zone}…"
    gcloud compute disks delete "${VM_NAME}" --zone="${zone}" --quiet 2>/dev/null || true
}

cleanup_old_snapshots() {
    # Borra cualquier snapshot huc-pilot-snap-* tras éxito (para no acumular basura).
    local snaps
    snaps=$(gcloud compute snapshots list --filter="name~^huc-pilot-snap-" --format="value(name)" 2>/dev/null)
    if [[ -n "${snaps}" ]]; then
        log "Borrando snapshots usados:"
        for snap in ${snaps}; do
            log "  - ${snap}"
            gcloud compute snapshots delete "${snap}" --quiet 2>/dev/null || true
        done
    fi
}

try_create_vm() {
    local zone="$1"
    local provisioning="$2"   # "on-demand" o "spot"

    # Construimos el array completo (no vacío) para evitar el error
    # 'unbound variable' cuando set -u + array vacío.
    local cmd_args=(
        --machine-type=g2-standard-4
        --address="${STATIC_IP_NAME}"
        --tags=http-server,https-server
        --maintenance-policy=TERMINATE
        --disk="name=${DISK_NAME:-${VM_NAME}},boot=yes,auto-delete=yes"
    )
    if [[ "${provisioning}" == "spot" ]]; then
        cmd_args+=(--provisioning-model=SPOT --instance-termination-action=STOP)
    fi

    log "Intentando crear VM en ${zone} (${provisioning})…"
    if gcloud compute instances create "${VM_NAME}" --zone="${zone}" "${cmd_args[@]}" 2>&1 | tee /tmp/vm_create.log; then
        log "✅ VM creada en ${zone} (${provisioning})"
        return 0
    fi

    if grep -q "ZONE_RESOURCE_POOL_EXHAUSTED" /tmp/vm_create.log; then
        log "❌ Stockout en ${zone} (${provisioning})"
        return 2   # stockout específicamente
    fi

    log "❌ Error inesperado en ${zone} (${provisioning}); pegándolo:"
    cat /tmp/vm_create.log
    return 1
}

apply_post_create() {
    local zone="$1"
    log "Reasociando schedule de auto-stop…"
    gcloud compute instances stop "${VM_NAME}" --zone="${zone}" --quiet || true
    gcloud compute instances add-resource-policies "${VM_NAME}" --zone="${zone}" --resource-policies=auto-stop-2000
    gcloud compute instances start "${VM_NAME}" --zone="${zone}"
    log "VM en ${zone} arrancada con schedule auto-stop-2000 ✓"
}

# ---------------------------------------------------------------------------
# Lógica principal
# ---------------------------------------------------------------------------

log "Estado inicial:"
log "  VM: $(vm_zone || echo '(no existe)')"
log "  Disco: $(disk_real_name || echo '(no existe)') en $(disk_zone || echo '(n/a)')"

# Si la VM ya existe y está RUNNING, no hacemos nada
existing_zone=$(vm_zone)
if [[ -n "${existing_zone}" ]]; then
    status=$(gcloud compute instances describe "${VM_NAME}" --zone="${existing_zone}" --format="value(status)" 2>/dev/null || echo "UNKNOWN")
    if [[ "${status}" == "RUNNING" ]]; then
        log "VM ya está RUNNING en ${existing_zone}. Nada que hacer."
        echo
        echo "ZONE=${existing_zone}"
        echo "  gcloud compute ssh ${VM_NAME} --zone=${existing_zone}"
        exit 0
    fi
    if [[ "${status}" == "TERMINATED" ]]; then
        # Probar a arrancar primero antes de migrar
        log "VM existe en ${existing_zone}, parada. Intentando start directo primero…"
        if gcloud compute instances start "${VM_NAME}" --zone="${existing_zone}" 2>&1 | tee /tmp/vm_start.log; then
            log "✅ Start exitoso en ${existing_zone}"
            echo
            echo "ZONE=${existing_zone}"
            exit 0
        fi
        if grep -q "ZONE_RESOURCE_POOL_EXHAUSTED" /tmp/vm_start.log; then
            log "Start falló por stockout. Probando otras zonas…"
            # Antes del loop de zonas: borrar la VM TERMINATED (manteniendo
            # el disco). Si no la borramos, try_create_vm() del loop fallará
            # con "instance already exists" porque tratará de recrearla en
            # cualquier zona aunque siga viva en ${existing_zone}.
            log "Borrando VM en ${existing_zone} (manteniendo disco) para liberar el nombre…"
            gcloud compute instances delete "${VM_NAME}" --zone="${existing_zone}" --keep-disks=boot --quiet || true
        fi
    fi
fi

# Determinar orden de pruebas según modo
case "${MODE}" in
    on-demand) PROVISIONING_LIST=("on-demand") ;;
    spot) PROVISIONING_LIST=("spot") ;;
    on-demand-then-spot) PROVISIONING_LIST=("on-demand" "spot") ;;
esac

for prov in "${PROVISIONING_LIST[@]}"; do
    for zone in "${ZONES[@]}"; do
        # Asegurar disco en esta zona (migra si hace falta)
        if ! ensure_disk_in "${zone}"; then
            log "No se pudo poner el disco en ${zone}, salto"
            continue
        fi

        # Intentar crear VM
        rc=0
        try_create_vm "${zone}" "${prov}" || rc=$?
        if [[ $rc -eq 0 ]]; then
            apply_post_create "${zone}"
            cleanup_old_snapshots
            echo
            echo "ZONE=${zone}"
            echo "  gcloud compute ssh ${VM_NAME} --zone=${zone}"
            exit 0
        fi
        # rc=2 → stockout, dejamos el disco donde está (la siguiente iteración
        # lo migrará vía snapshot+restore en ensure_disk_in). rc=1 → abortar.
        if [[ $rc -eq 1 ]]; then
            log "Error inesperado, abortando."
            exit 1
        fi
        log "Stockout. El disco se quedará en ${zone} hasta que ensure_disk_in lo mueva."
    done
    log "Todas las zonas probadas para ${prov}, sin éxito."
done

log "❌ No se ha podido arrancar la VM en ninguna zona/provisioning. Toca esperar o migrar región."
exit 1
