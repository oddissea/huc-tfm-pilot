# Archive local de correcciones del patólogo + features 512-d

Hitos 0+1 del módulo de aprendizaje de **DualPath CRC** (el piloto, by
Lumen Network). Garantiza que ningún
`corrections.jsonl` ni el `features.npy` asociado desaparezcan por el
TTL de 24 h del worker sin haber sido copiados a un archive persistente
del host.

**Diseño "local-only" (decidido en la sesión #64)**: el archive vive en
el filesystem del host (bind mount Docker), NO en la nube. En el HUC los
datos del paciente no salen del hospital. La misma implementación
funciona en QA y en producción HUC — el path que se valida es el path
que se despliega.

## Arquitectura: dos redes superpuestas

1. **Hook en el worker** (red primaria). Cada `PRUNE_INTERVAL_SECONDS`
   (5 min, ver `src/jobs/worker.py`) el worker llama a
   `JobManager.prune()`. Antes de borrar cada `job_dir` expirado se
   invoca `archive_job_safe(job_dir)`. Si el archivado falla, el dir
   **NO se borra** y se reintenta en el siguiente prune. Cubre el caso
   normal (app corriendo).

2. **Cron en el host** (red secundaria). Cubre el caso patológico "app
   caída cuando expira un TTL". Ejecuta el CLI `scripts.archive_jobs`
   desde dentro del container.

Ambas usan el mismo módulo `src.corrections.archive` y son idempotentes
vía comparación directa de sha256 entre `job_dir` y el archive.

## Qué se archiva

Por cada `job_dir` con al menos una corrección no vacía en
`corrections.jsonl`, se copian al archive estos cuatro ficheros:

| Fichero            | Contenido                                                                     | Bloqueante |
|--------------------|-------------------------------------------------------------------------------|------------|
| `corrections.jsonl`| Etiquetas que el patólogo corrigió post-inferencia (solo parches modificados) | sí         |
| `features.npy`     | Embeddings 512-d (post-ReLU) por parche del slide                             | sí         |
| `patch_eval.npz`   | Predicciones patch-level del modelo (`pred_index`, `pred_probs`) por parche   | sí         |
| `meta.json`        | Filename, n_patches, predicted_class, timestamps                              | no         |

Bloqueante = si la copia falla, el `job_dir` NO se borra; se reintenta.

Jobs sin `corrections.jsonl` o con el fichero vacío se ignoran: no
existe señal supervisada que justifique conservarlos.

**Por qué `patch_eval.npz` también**: en Hito 2 (reentrenamiento del
head F4 con replay buffer), las correcciones explícitas del patólogo
son típicamente decenas de parches por slide, pero el slide tiene
cientos. Para reentrenar bien necesitamos el target de **todos** los
parches, no solo los corregidos. Las predicciones del modelo
guardadas en `patch_eval.npz` permiten reconstruir el target de los
parches "no tocados" bajo distintas políticas (asumir "no corregido
= aprobado" como ground truth fuerte, o con peso reducido tipo soft
label). La política exacta se decide en Hito 2; archivamos siempre
el fichero para no cerrar opciones.

## Requisitos en el host de producción (HUC)

- Equipo HUC con Docker Compose.
- Disco con espacio razonable para el archive. Estimación:
  `~2 MB/slide × N slides/año` (features 512-d × N parches × float32 es
  el componente dominante). Para 5 slides/día × 250 días/año = ~2.5
  GB/año. Manejable en cualquier disco moderno.
- El bind mount `./archive:/var/archive` ya está configurado en
  `docker-compose.yml`. Al levantar el container la primera vez se crea
  automáticamente el directorio `./archive/` en el host.
- `docker-compose.yml` ya inyecta `PILOT_ARCHIVE_DIR=/var/archive` y
  `PILOT_QUEUE_DIR=/tmp/queue` en el container, así que el CLI localiza
  ambas rutas sin necesidad de flags.

> **Nota sobre los pesos del modelo (frente independiente)**: el
> container sigue descargando los pesos F4 + AttnMIL al arranque desde
> `gs://huc-tfm-pilot-models/` vía `src/inference/weights.py`. Eso NO
> afecta a los datos del paciente (los pesos son artefactos públicos del
> piloto), pero sí requiere conectividad de salida a GCS al primer
> arranque del container. Pre-deploy en HUC: o (a) garantizar
> conectividad al arrancar la primera vez y dejar los pesos cacheados en
> el bind mount `./weights`, o (b) pre-cargar manualmente los pesos en
> `./weights/` antes del primer `docker compose up`. Esto es trabajo
> aparte del archivado de correcciones.

## Recogida de correcciones para reentrenar

La transferencia desde el HUC al entorno de reentrenamiento es manual y
queda a criterio del operador (Eduardo y/o el alumno):

```bash
# Desde el host del HUC, copia el archive entero a USB cifrado:
rsync -a /ruta/del/repo/archive/ /Volumes/USB_cifrado/huc-archive-YYYY-MM-DD/

# O tar comprimido + cifrado para transferencia por red:
tar czf - /ruta/del/repo/archive/ | \
    openssl enc -aes-256-cbc -pbkdf2 -salt -out huc-archive.tar.gz.enc
```

Cada subdirectorio del archive es autónomo:
`archive/<job_id>/{corrections.jsonl, features.npy, meta.json}`. Se
puede mover, copiar o borrar carpeta por carpeta sin afectar al resto.

## TTL del archive

El archive **no tiene TTL programático**. La política de retención queda
a criterio operacional (Eduardo decide cuándo limpiar tras recoger una
tanda). Para borrar manualmente correcciones ya recogidas:

```bash
# Borrar un job concreto del archive
rm -rf /ruta/del/repo/archive/<job_id>/

# Borrar todo el archive (con confirmación)
read -p "Borrar todo el archive? (sí/no) " ok && \
    [ "$ok" = "sí" ] && rm -rf /ruta/del/repo/archive/*
```

## Setup del cron en HUC

Editar el crontab del usuario que corre el container:

```bash
crontab -e
```

Añadir:

```cron
# Hito 0+1 — red de seguridad: archivar correcciones huérfanas cada 6h.
# El worker ya hace esto cada 5 min vía hook en prune(); este cron solo
# cubre el caso "app caída durante muchas horas".
0 */6 * * * cd /home/huc/huc-tfm-pilot && /usr/bin/docker compose exec -T app python -m scripts.archive_jobs >> /var/log/huc-pilot/archive_jobs.log 2>&1
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
# Dry-run desde el container (no copia nada, solo lista lo que archivaría).
docker compose exec app python -m scripts.archive_jobs --dry-run --verbose

# Copia real.
docker compose exec app python -m scripts.archive_jobs --verbose

# Verificar el archive en el host.
ls -la archive/
du -sh archive/
```

## Recuperación ante errores

Si el log del cron muestra errores recurrentes:

1. Comprobar que el bind mount está montado:

   ```bash
   docker compose exec app ls -la /var/archive
   ```

2. Comprobar permisos del directorio en el host. El UID dentro del
   container debe poder escribir en `/var/archive`:

   ```bash
   docker compose exec app touch /var/archive/.write_test && \
       docker compose exec app rm /var/archive/.write_test
   ```

3. Comprobar espacio en disco del host:

   ```bash
   df -h .
   ```

Si una copia se interrumpió a mitad (proceso matado, host reiniciado),
queda un fichero `*.tmp` huérfano en el archive. La idempotencia detecta
esto en el siguiente prune: el fichero final no existe o tiene sha256
distinto del job_dir → se reintenta la copia atómica.
