# DualPath CRC — by Lumen Network

## Release v1.0.3 — 2026-05-27 (arquitectura)

**Cambio arquitectónico**: el container del piloto ahora incluye
**nginx interno** que sirve tanto Streamlit como los DZIs bajo un
único puerto. Replica el setup de la VM cloud (que lleva meses
funcionando en producción), eliminando de raíz cualquier problema
CORS y simplificando el `docker run` para Eduardo.

### Por qué este cambio

v1.0/v1.0.1/v1.0.2 intentaron iterar parches sobre el visor sin
volver a una arquitectura con nginx:

- v1.0: sin nginx → visor falló porque nadie servía los DZIs.
- v1.0.1: añadió sidecar `serve_dzi.py` (Python http.server en puerto
  8888) → resolvió la falta del servidor pero introdujo problema CORS
  por orígenes distintos (8501 vs 8888).
- v1.0.2: quitó `ajaxWithCredentials: true` de OpenSeadragon para
  intentar resolver el CORS → diagnóstico probable pero no
  garantizado (probabilidad ~70% empíricamente).

**v1.0.3 elimina la incertidumbre**: replica el setup VM cloud (un
nginx delante de Streamlit, ambos sirviendo bajo el mismo origen)
pero embebido dentro del mismo container. Sin docker compose, sin
certbot, sin TLS — solo nginx ligero (`nginx-light` de Ubuntu) con
una configuración mínima que hace proxy a Streamlit y sirve los DZIs
como ficheros estáticos.

### Cambios técnicos

**Dockerfile**:
- `apt install nginx-light` añadido a las system deps.
- `COPY nginx-internal/nginx.conf /etc/nginx/nginx.conf` para la
  configuración mínima.
- `EXPOSE 80` en lugar de `EXPOSE 8501 8888`.
- `ENV DZI_BASE_URL=/dzi` (URL relativa, servida por nginx interno).
- `serve_dzi.py` se mantiene en el repo pero **no se usa en v1.0.3+**.
  Lo dejamos como artefacto histórico.

**Nuevo `nginx-internal/nginx.conf`**:
- `location /` → `proxy_pass http://127.0.0.1:8501` con WebSocket
  upgrade headers (esencial para Streamlit reactivo).
- `location /dzi/` → `alias /tmp/queue/` (DZIs y tiles como static).
- `types { application/xml dzi; }` para que Firefox no descargue el
  `.dzi` (lo trata como XML inline).
- `client_max_body_size 4G` para uploads de TIFF grandes.
- `proxy_read_timeout 86400` para evitar 504s en cargas largas
  (modelos, batches grandes).

**`scripts/entrypoint.sh`** reescrito:
- Lanza `nginx -g "daemon off;" &` en background.
- Lanza `streamlit run app.py --server.address=127.0.0.1` (¡solo
  loopback!).
- Streamlit ya **NO es accesible** directamente desde fuera del
  container — el único punto de entrada es nginx en puerto 80.

**`src/viz/slide_detail.py`** y **`osd_component/index.html`**:
- `ajaxWithCredentials: false` (heredado de v1.0.2).
- Sin más cambios — el código ya usa `DZI_BASE_URL` env var, ahora
  con valor `/dzi` que nginx resuelve internamente.

**`USER_GUIDE_EDUARDO.md`**:
- `docker run` con `-p 80:80` (en lugar de `-p 8501:8501 -p
  8888:8888`).
- Acceso por `http://localhost` (sin `:8501`).
- Tabla de errores actualizada (puerto 80 ocupado → fallback a
  `-p 8080:80`).

### Acceso para Eduardo

Cambia de:

```
http://localhost:8501
```

A:

```
http://localhost
```

(puerto 80 por defecto). Si el puerto 80 del HUC PC está ocupado por
otro servicio (cosa rara pero posible), usar `-p 8080:80` en el
docker run y acceder por `http://localhost:8080`.

### Beneficios

1. **Cero problemas CORS**: Streamlit y DZIs bajo el mismo origen.
2. **Arquitectura idéntica a la VM cloud** que funciona en producción
   desde hace meses. Cualquier debugging futuro tiene precedente
   probado.
3. **Un solo puerto en `docker run`**: simplifica el comando para
   Eduardo.
4. **nginx maneja correctamente** MIME types, range requests, caching
   — cosas que `python http.server` no hacía bien.

### Sin cambios funcionales

- Modelo F4 + 5 AttnMIL: idéntico.
- Métricas: 92,37% accuracy, 95,9 Safety Score, etc.
- Volúmenes persistentes (`~/huc-pilot-data/{archive,queue}/`):
  idénticos. Las correcciones de Eduardo y los jobs procesados se
  conservan en el upgrade.

### Distribución — Google Drive (Shared Drive "Lumen Network")

Ruta: `Releases/DualPath-CRC/v1.0.3/`.

| Fichero | FILE_ID | URL |
|---|---|---|
| `huc-pilot-with-weights-v1.0.3.tar.gz` | `1BLKKtv2qftUx0maX7h-p512b53nE4LRD` | https://drive.google.com/file/d/1BLKKtv2qftUx0maX7h-p512b53nE4LRD/view |
| `huc-pilot-with-weights-v1.0.3.tar.gz.sha256` | `1nTHE7jCcbxl1W86lfNYpFu5MkObUdGQB` | https://drive.google.com/file/d/1nTHE7jCcbxl1W86lfNYpFu5MkObUdGQB/view |

### Procedimiento de upgrade para Eduardo (desde cualquier versión previa)

1. Parar y eliminar el container actual (datos en `~/huc-pilot-data/`
   se conservan):

   ```
   docker stop huc-pilot && docker rm huc-pilot
   ```

2. Descargar la nueva versión:

   ```
   cd ~
   gdown "1BLKKtv2qftUx0maX7h-p512b53nE4LRD" -O huc-pilot-with-weights.tar.gz
   gdown "1nTHE7jCcbxl1W86lfNYpFu5MkObUdGQB" -O huc-pilot-with-weights.tar.gz.sha256
   sha256sum -c huc-pilot-with-weights.tar.gz.sha256
   ```

3. Cargar y relanzar (un solo puerto, mucho más simple):

   ```
   docker load -i huc-pilot-with-weights.tar.gz
   docker tag huc-pilot:dev-with-weights huc-pilot:dev
   docker run -d --name huc-pilot --gpus all -p 80:80 \
     -v ~/huc-pilot-data/archive:/var/archive \
     -v ~/huc-pilot-data/queue:/tmp/queue \
     --restart unless-stopped huc-pilot:dev
   ```

4. Abrir Firefox/Chrome en `http://localhost` (sin puerto). El visor
   debería cargar correctamente al procesar un slide.

### Limitaciones conocidas (sin cambios vs versiones previas)

- Tarea ternaria a nivel de parche (NOR/ADE/CAR); slide-level MIL
  implementado pero no expuesto en UI (v1.1 planificado).
- Sin reentrenamiento online (Hito 2 post-defensa).
- N=1 HUC.
- Modo CPU fallback no incluido (v1.1).

### Para la VM cloud — sin impacto

Esta versión NO afecta al despliegue cloud. El `docker-compose.yml`
sigue exactamente igual (con nginx + certbot + Streamlit en
containers separados). Solo cambia el flujo HUC PC.
