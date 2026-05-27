# DualPath CRC — by Lumen Network

## Release v1.0.2 — 2026-05-27 (patch)

Fix sutil del visor OpenSeadragon en el despliegue HUC PC. v1.0.1
implementó correctamente el sidecar `serve_dzi.py` pero el visor seguía
fallando con `HTTP 0 attempting to load TileSource`. Causa: el código
JavaScript de OpenSeadragon tenía `ajaxWithCredentials: true`,
incompatible con el header `Access-Control-Allow-Origin: *` del
sidecar según especificación CORS.

### Fix

Cambiar `ajaxWithCredentials: true` → `false` en los dos sitios donde
aparece:

- `pilot/src/viz/slide_detail.py:653` (visor inline).
- `pilot/src/viz/osd_component/index.html:351` (custom component con
  click capture).

### Por qué ocurría

La especificación CORS dice que cuando una respuesta tiene
`Access-Control-Allow-Origin: *` (wildcard), el browser NO puede
aceptarla si la request lleva credentials (cookies, Authorization
header). Eso produce un fallo silencioso: la request técnicamente
funciona (curl la ve sin problema), pero el browser rechaza la
respuesta antes de que el JavaScript pueda usarla. En la consola
aparece como `HTTP 0`.

En v1.0 / v1.0.1 con deploy cloud nginx+BasicAuth, este flag tenía
sentido: las credentials del Basic Auth se enviaban por cookies del
dominio. En deploy HUC PC sin nginx, no hay credentials que enviar,
y el flag solo introduce este choque CORS innecesario.

### Validación previa al fix

`curl -i http://localhost:8888/` desde el HUC PC respondió:

```
HTTP/1.0 200 OK
Server: SimpleHTTP/0.6 Python/3.11.13
Access-Control-Allow-Origin: *
Cache-Control: public, max-age=86400
<directory listing with d18be72e-..., d7301fb4-..., f4b8e616-...>
```

Confirmó que el sidecar funciona, que CORS está habilitado, y que
los 3 jobs procesados tienen sus DZIs en disco. El problema era
exclusivamente del cliente browser por el conflicto wildcard +
credentials.

### Sin cambios funcionales

- Modelo F4 + 5 AttnMIL ensemble: idéntico a v1.0/v1.0.1.
- Métricas: 92,37% accuracy, 95,9/100 Safety Score, etc.
- Sidecar serve_dzi.py: idéntico (sigue sirviendo /tmp/queue en 8888
  con Allow-Origin: *).
- Volúmenes persistentes (`~/huc-pilot-data/archive`, `queue`):
  idénticos.
- `docker run` command: idéntico al de v1.0.1 (con `-p 8888:8888`).

Solo cambia el comportamiento del JavaScript de OpenSeadragon dentro
del browser. Inferencia, archive, métricas — todo invariante.

### Distribución — Google Drive (Shared Drive "Lumen Network")

Ruta: `Releases/DualPath-CRC/v1.0.2/`.

| Fichero | FILE_ID | URL |
|---|---|---|
| `huc-pilot-with-weights-v1.0.2.tar.gz` | `1Tikt1qYaA6h-ks_aX6mzRWu4DY8Qh0aG` | https://drive.google.com/file/d/1Tikt1qYaA6h-ks_aX6mzRWu4DY8Qh0aG/view |
| `huc-pilot-with-weights-v1.0.2.tar.gz.sha256` | `1zoOsIb9Vjgst_NtY-PYzUCSO05T54xC3` | https://drive.google.com/file/d/1zoOsIb9Vjgst_NtY-PYzUCSO05T54xC3/view |

### Procedimiento de upgrade desde v1.0.1 para Eduardo

Idéntico a v1.0.1, solo cambian los FILE_IDs. Una vez Eduardo tenga
los nuevos:

1. Parar y eliminar el container actual (los datos en
   `~/huc-pilot-data/` se conservan, incluyendo el archive con jobs
   ya procesados):

   ```
   docker stop huc-pilot && docker rm huc-pilot
   ```

2. Descargar la nueva versión:

   ```
   cd ~
   gdown "1Tikt1qYaA6h-ks_aX6mzRWu4DY8Qh0aG" -O huc-pilot-with-weights.tar.gz
   gdown "1zoOsIb9Vjgst_NtY-PYzUCSO05T54xC3" -O huc-pilot-with-weights.tar.gz.sha256
   sha256sum -c huc-pilot-with-weights.tar.gz.sha256
   ```

3. Cargar y relanzar (mismas flags que v1.0.1):

   ```
   docker load -i huc-pilot-with-weights.tar.gz
   docker tag huc-pilot:dev-with-weights huc-pilot:dev
   docker run -d --name huc-pilot --gpus all -p 8501:8501 -p 8888:8888 \
     -v ~/huc-pilot-data/archive:/var/archive \
     -v ~/huc-pilot-data/queue:/tmp/queue \
     --restart unless-stopped huc-pilot:dev
   ```

4. Esperar 15 segundos. Abrir `http://localhost:8501` (Ctrl+Shift+R
   si la pestaña ya estaba abierta para evitar cache). Subir un slide
   o reprocesar uno ya hecho. **El visor OpenSeadragon ya debería
   cargar los tiles correctamente**.

### Cómo verificar visualmente que v1.0.2 funcionó

Si después del upgrade el visor ya muestra el slide:
- Aparece la imagen del portaobjetos (no el error "Unable to load
  TileSource" ni HTTP 0).
- Funciona el zoom (rueda del ratón) y el pan (arrastrar).
- Aparecen los rectángulos coloreados sobre cada parche según la
  predicción del modelo.

Si todavía no funciona, capturar:
- DevTools F12 → Console (errores en rojo).
- DevTools F12 → Network → filtrar por `slide.dzi` o `_files`.

### Limitaciones conocidas (sin cambios vs v1.0.1)

- Tarea ternaria a nivel de parche (NOR/ADE/CAR); slide-level MIL
  implementado pero no expuesto en UI (v1.1 planificado).
- Sin reentrenamiento online (Hito 2 post-defensa).
- N=1 HUC.
- Modo CPU fallback no incluido (planificado v1.1).
