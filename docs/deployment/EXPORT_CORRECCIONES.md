# Export de correcciones del patólogo a GCS

Hito 0 del módulo de aprendizaje. Garantiza que ningún
`corrections.jsonl` desaparezca por el TTL de 24h del worker sin
haberse replicado a `gs://huc-tfm-pilot-corrections/<job_id>/`.

## Arquitectura: dos redes superpuestas

1. **Hook en el worker** (red primaria). Cada
   `PRUNE_INTERVAL_SECONDS` (5 min, ver `src/jobs/worker.py`) el
   worker llama a `JobManager.prune()`. Antes de borrar cada
   `job_dir` expirado se invoca `export_job_safe(job_dir)`. Si la
   exportación falla, el dir NO se borra y se reintenta en el
   siguiente prune. Cubre el caso normal (app corriendo).

2. **Cron en el host** (red secundaria). Cubre el caso patológico
   "app caída cuando expira un TTL". Ejecuta el CLI
   `scripts.export_corrections` desde dentro del container.

Ambas usan el mismo módulo `src.corrections.export` y son
idempotentes vía sha256 guardado en `metadata.local_sha256` del
blob remoto.

## Requisitos en el host de producción (HUC)

- VM o equipo HUC con Docker Compose y `gcloud` instalado.
- Service Account asociada al equipo (o ADC vía
  `gcloud auth application-default login`) con permiso
  `roles/storage.objectUser` sobre `huc-tfm-pilot-corrections`.
- `google-cloud-storage` ya está en `requirements.txt` del
  container; no hace falta nada en el host.
- `docker-compose.yml` ya inyecta `PILOT_QUEUE_DIR=/tmp/queue` en el
  container, así que el CLI localiza la cola automáticamente cuando
  se invoca vía `docker compose exec`. No hace falta pasar
  `--queue-dir` explícitamente.

## Migración de permisos GCP pre-deploy (¡no saltarse!)

Durante QA, el rol `roles/storage.objectUser` sobre el bucket lo
tiene el Service Account de la **VM de Google Cloud** que usa el
alumno para pruebas:

```
622373334442-compute@developer.gserviceaccount.com
```

Al desplegar en el ordenador del HUC ese SA ya no debe poder
escribir en el bucket (principio de mínimo privilegio). Hay que:

1. **Identificar el SA del HUC.** Si el ordenador del HUC usa
   Application Default Credentials vía
   `gcloud auth application-default login`, el "principal" será una
   cuenta de usuario (`user:huc-operator@...`). Si en su lugar se
   crea una SA dedicada (recomendado), apunta su email aquí:

   ```bash
   # Crear SA dedicada para el host HUC (opción recomendada)
   gcloud iam service-accounts create huc-tfm-pilot-host \
     --display-name="HUC TFM Pilot — host del HUC"

   # Generar key JSON y guardarla en el host HUC en una ruta segura
   gcloud iam service-accounts keys create ~/huc-tfm-pilot-host.json \
     --iam-account=huc-tfm-pilot-host@PROJECT_ID.iam.gserviceaccount.com
   ```

2. **Conceder permiso al SA del HUC** sobre el bucket:

   ```bash
   gcloud storage buckets add-iam-policy-binding \
     gs://huc-tfm-pilot-corrections \
     --member="serviceAccount:huc-tfm-pilot-host@PROJECT_ID.iam.gserviceaccount.com" \
     --role="roles/storage.objectUser"
   ```

3. **Revocar el permiso al SA de la VM de QA**:

   ```bash
   gcloud storage buckets remove-iam-policy-binding \
     gs://huc-tfm-pilot-corrections \
     --member="serviceAccount:622373334442-compute@developer.gserviceaccount.com" \
     --role="roles/storage.objectUser"
   ```

4. **Verificar la política final** (debe aparecer solo el SA del
   HUC en `objectUser`):

   ```bash
   gcloud storage buckets get-iam-policy gs://huc-tfm-pilot-corrections
   ```

5. **Probar desde el host HUC** que la subida funciona con la nueva
   identidad antes de borrar la VM de QA:

   ```bash
   docker compose exec app python -m scripts.export_corrections --dry-run --verbose
   ```

   Debería resolver el bucket sin errores de auth.

## Limpieza pre-deploy (¡no saltarse!)

Durante el desarrollo y QA, el bucket
`gs://huc-tfm-pilot-corrections/` acumula **correcciones de prueba**
del propio alumno (no son del patólogo). Antes de instalar el
piloto en el ordenador del HUC hay que dejar el bucket vacío para
que las primeras entradas reales del patólogo estén en un estado
limpio y bien identificado.

Como el bucket tiene **versioning** activo, un `gsutil rm` normal
deja las versiones anteriores como soft-deleted (recuperables).
Para una limpieza de verdad:

```bash
# Listar todo lo que hay (incluyendo versiones anteriores)
gsutil ls -a gs://huc-tfm-pilot-corrections/

# Borrar TODAS las versiones de TODOS los objetos. Irreversible.
gsutil -m rm -a gs://huc-tfm-pilot-corrections/**

# Verificar que el bucket está completamente vacío
gsutil ls -a gs://huc-tfm-pilot-corrections/
```

Si el bucket aparece vacío en el último `gsutil ls -a`, todo
limpio. A partir de ese momento, la primera corrección que se
suba será del primer slide procesado en el HUC en producción.

## Setup del cron en HUC

Editar el crontab del usuario que corre el container:

```bash
crontab -e
```

Añadir:

```cron
# Hito 0 — red de seguridad: exportar correcciones huérfanas cada 6h.
# El worker ya hace esto cada 5 min vía hook en prune(); este cron
# solo cubre el caso "app caída durante muchas horas".
0 */6 * * * cd /home/huc/huc-tfm-pilot && /usr/bin/docker compose exec -T app python -m scripts.export_corrections >> /var/log/huc-pilot/export_corrections.log 2>&1
```

Ajustar `/home/huc/huc-tfm-pilot` a la ruta real del repo en el
ordenador del HUC, y crear el directorio de log:

```bash
sudo mkdir -p /var/log/huc-pilot
sudo chown huc:huc /var/log/huc-pilot
```

Rotación de logs (opcional pero recomendado), crear
`/etc/logrotate.d/huc-pilot`:

```
/var/log/huc-pilot/*.log {
    weekly
    rotate 8
    compress
    missingok
    notifempty
    copytruncate
}
```

## Verificación

```bash
# Dry-run desde el container (no sube nada, solo lista lo que subiría)
docker compose exec app python -m scripts.export_corrections --dry-run --verbose

# Subida real
docker compose exec app python -m scripts.export_corrections --verbose

# Verificar contenido en GCS
gsutil ls -l gs://huc-tfm-pilot-corrections/
```

## Recuperación ante errores

El bucket tiene **versioning activo**, por lo que sobreescrituras
accidentales son recuperables vía
`gsutil ls -a gs://huc-tfm-pilot-corrections/<job_id>/` (muestra
todas las versiones, incluyendo generaciones).

Si el cron emite errores recurrentes en el log:

1. Comprobar auth desde dentro del container:
   `docker compose exec app python -c "from google.cloud import storage; print([b.name for b in storage.Client().list_buckets()])"`
2. Comprobar permisos del SA sobre el bucket.
3. Comprobar conectividad de red (firewall hospitalario).
