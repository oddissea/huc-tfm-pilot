# DualPath CRC — by Lumen Network

## Release v1.0.1 — 2026-05-26 (patch)

Fix del visor OpenSeadragon en el despliegue HUC PC (deploy "sin
compose / sin nginx"). v1.0 ya está en producción con Eduardo desde
las 21:53 del 2026-05-26 — esta v1.0.1 sustituye la imagen para
restaurar el visor de slides, sin tocar el resto del pipeline.

### Fix principal

- **Visor de slides OpenSeadragon ahora funciona en el deploy HUC PC**.
  En v1.0 el visor mostraba `Unable to open [object Object]: Unable
  to load TileSource` porque el código construía URLs relativas
  `/dzi/<job_id>/slide.dzi` que dependían de un nginx delante (sólo
  presente en el despliegue cloud). En HUC PC, sin nginx, esas URLs
  no resolvían a ningún servidor.

  Solución: un mini HTTP server Python (`scripts/serve_dzi.py`)
  embebido en el mismo container que Streamlit, sirviendo
  `/tmp/queue/` (donde JobManager guarda los DZIs) en el puerto 8888
  con headers CORS habilitados. El `entrypoint.sh` del container
  arranca ambos procesos (`serve_dzi.py` en background + `streamlit`
  en foreground).

  Compatibilidad cloud: el código lee `DZI_BASE_URL` (env var). Por
  defecto en la imagen va `http://localhost:8888` (apto para HUC PC).
  En cloud, `docker-compose.yml` lo sobreescribe a `/dzi` para que
  nginx siga sirviendo los tiles vía su location block.

### Sin cambios funcionales

- Modelo F4 + 5 AttnMIL ensemble: idéntico a v1.0.
- Métricas: 92,37% accuracy, 95,9/100 Safety Score, etc.
- Página `⚙️ Configuración`: igual.
- Flujo de archive (corrections + features + patch_eval + meta): igual.

### Distribución — Google Drive (Shared Drive "Lumen Network")

Ruta: `Releases/DualPath-CRC/v1.0.1/`.

| Fichero | FILE_ID | URL |
|---|---|---|
| `huc-pilot-with-weights-v1.0.1.tar.gz` | `1n03IPOxLsoyyAWxai2bc7W4l2dWIG5rY` | https://drive.google.com/file/d/1n03IPOxLsoyyAWxai2bc7W4l2dWIG5rY/view |
| `huc-pilot-with-weights-v1.0.1.tar.gz.sha256` | `1AT4RHKKA79RtmJESEyUdnAp3ppO4UhOM` | https://drive.google.com/file/d/1AT4RHKKA79RtmJESEyUdnAp3ppO4UhOM/view |

### Cambio para el `docker run` de Eduardo (vs v1.0)

Hay que añadir `-p 8888:8888` para exponer el puerto del servidor de
tiles. Comando completo actualizado:

```bash
docker run -d \
  --name huc-pilot \
  --gpus all \
  -p 8501:8501 \
  -p 8888:8888 \
  -v ~/huc-pilot-data/archive:/var/archive \
  -v ~/huc-pilot-data/queue:/tmp/queue \
  --restart unless-stopped \
  huc-pilot:dev
```

Si Eduardo olvida el `-p 8888:8888`, el piloto arranca pero el visor
falla con el mismo `Unable to load TileSource` que pasaba en v1.0
(visible en la tabla de errores típicos de `USER_GUIDE_EDUARDO.md`
sección 4).

### Procedimiento de upgrade desde v1.0 para Eduardo

1. Parar y eliminar el container v1.0:

   ```
   docker stop huc-pilot && docker rm huc-pilot
   ```

   (Los datos en `~/huc-pilot-data/` se conservan automáticamente).

2. Descargar nueva versión desde el Shared Drive:

   ```
   cd ~
   gdown "1n03IPOxLsoyyAWxai2bc7W4l2dWIG5rY" -O huc-pilot-with-weights.tar.gz
   gdown "1AT4RHKKA79RtmJESEyUdnAp3ppO4UhOM" -O huc-pilot-with-weights.tar.gz.sha256
   sha256sum -c huc-pilot-with-weights.tar.gz.sha256
   ```

3. Recargar y relanzar (con el `-p 8888:8888` nuevo):

   ```
   docker load -i huc-pilot-with-weights.tar.gz
   docker tag huc-pilot:dev-with-weights huc-pilot:dev
   docker run -d --name huc-pilot --gpus all -p 8501:8501 -p 8888:8888 \
     -v ~/huc-pilot-data/archive:/var/archive \
     -v ~/huc-pilot-data/queue:/tmp/queue \
     --restart unless-stopped huc-pilot:dev
   ```

### Archivos modificados (versión técnica)

- `pilot/Dockerfile`: `EXPOSE 8501 8888` + `ENV DZI_BASE_URL` + `CMD entrypoint.sh`.
- `pilot/scripts/serve_dzi.py`: NUEVO — HTTP server CORS-enabled.
- `pilot/scripts/entrypoint.sh`: NUEVO — lanza serve_dzi + streamlit.
- `pilot/src/viz/slide_detail.py`: línea 489 lee `DZI_BASE_URL` env var.
- `pilot/docker-compose.yml`: añade `DZI_BASE_URL: /dzi` para que cloud + nginx sigan funcionando.
- `pilot/docs/USER_GUIDE_EDUARDO.md`: añadido `-p 8888:8888` en los `docker run` + entrada en tabla de errores.

### Limitaciones conocidas (sin cambios vs v1.0)

- Tarea ternaria a nivel de parche (NOR/ADE/CAR); slide-level MIL
  implementado pero no expuesto en UI (v1.1 planificado).
- Sin reentrenamiento online (Hito 2 post-defensa).
- N=1 HUC.
- Modo CPU fallback no incluido (planificado v1.1).
