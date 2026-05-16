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
