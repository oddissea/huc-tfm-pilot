# Changelog — DualPath CRC Pilot

## v1.0.4 (2026-05-29) — subida multi-archivo estable (WebSocket ping)

### 🎯 Qué cambia para ti (Eduardo)

1 cosa:

1. **Ya puedes subir varios portaobjetos a la vez.** Antes, al
   seleccionar 2 o 3 slides juntos, solo se cargaba uno y los demás
   daban un error (`AxiosError 400`). Ahora se cargan todos.

El resto funciona exactamente igual que en la v1.0.3.

### Para arrancar (igual que antes, con la imagen nueva)

```bash
docker run -d --name huc-pilot --gpus all -p 80:80 huc-pilot:v1.0.4
```

### 🔧 Qué cambió por dentro (técnico)

- **Causa raíz**: Streamlit 1.40.2 fija `websocket_ping_interval=1` en
  `web/server/server.py`. Tornado 6.5 **capa el `ping_timeout` al
  `ping_interval`**, así que el timeout efectivo del WebSocket es **1 s**
  (no los 30 s que el código pretendía). Durante el primer run pesado
  (carga de modelos, que retiene el GIL) sumado al parseo de los cuerpos
  de los TIFF grandes, el IOLoop de Tornado se queda sin servir el
  ping/pong durante más de 1 s → Tornado da el WebSocket por muerto →
  `disconnect_session` → la sesión sale del registro de sesiones activas
  → los `PUT /_stcore/upload_file` siguientes responden
  **`400: Invalid session_id`**. Por eso, al subir varios slides a la
  vez, "sobrevivía" solo uno por tanda (el que caía tras la reconexión
  del WebSocket).
- **No es** el caso multi-réplica de los issues conocidos
  (#4173/#6224/#2936): el HUC PC corre **una sola instancia**, así que
  las *sticky sessions* no aplican. Es una carrera de timing en
  instancia única.
- **Fix**: réplica del arreglo oficial de Streamlit
  ([PR #11693](https://github.com/streamlit/streamlit/pull/11693)) —
  subir `websocket_ping_interval` de **1 a 30**. Como en 1.40.2 el valor
  está hardcodeado y no existe opción de `config.toml`, se aplica como
  un parche a `server.py` dentro del `Dockerfile` (tras el
  `pip install`), con un `assert` que **rompe el build** si el patrón
  desaparece, para no enviar nunca una imagen sin parchear.
- **Descartados** (no atacan la causa): `proxy_request_buffering off`
  (podía incluso empeorarlo) y el patrón `map $http_upgrade` de nginx
  (higiene correcta, pero ortogonal al bug). La config de `nginx` no se
  toca en esta versión.

### ⚠️ Estado de validación — honesto

**El bug está confirmado en producción por Eduardo** (HUC PC): cuerpo del
400 = `Invalid session_id` en las DevTools, y el `access.log` muestra el
patrón exacto (varios `PUT … 400` y un `204` que sobrevive tras la
reconexión del WebSocket). Esa es la evidencia primaria de que el bug
existe y de cuál es su firma.

**La causa raíz** está establecida leyendo el código de Streamlit 1.40.2
y Tornado 6.5 (ver sección técnica) y **coincide con el fix oficial**
([streamlit#11693](https://github.com/streamlit/streamlit/pull/11693)).

**Lo que NO se ha podido hacer: un A/B local que reproduzca el fallo.**
Se montó un repro mínimo (`python:3.11-slim` + `nginx-light` +
`streamlit==1.40.2` + el `nginx.conf` interno EXACTO) y se intentó forzar
la caída del WebSocket bloqueando el IOLoop en el primer run. **No
reprodujo**: tanto con `ping_interval=1` (stock) como con `=30` (fix), las
subidas concurrentes salieron todas `204` y el WebSocket no se cayó. El
motivo probable es que un bloqueo *total* del IOLoop (busy-loop reteniendo
el GIL) se comporta distinto a la contención *parcial* real de Eduardo
(IOLoop corriendo pero con retraso >1 s por la lectura de cuerpos grandes
+ la carga de modelos cediendo el GIL a ratos); además Docker Desktop en
Mac añade su propia capa de red. **Conclusión honesta: no tengo prueba
local de que el fix resuelva el bug; tengo evidencia de código + el fix
oficial upstream + la firma exacta del log de Eduardo.**

**Validación pendiente: la hace Eduardo** en el entorno donde el bug sí se
manifiesta (es además quien lo reprodujo). La imagen `huc-pilot:v1.0.4`
se construyó y arranca correctamente en local (health 200, `ping_interval`
= 30 verificado dentro de la imagen); eso confirma que el parche se aplica
y la app levanta, **no** que el bug desaparezca.

### 📦 Artefactos en Drive

Shared Drive "Lumen Network", carpeta `Releases/DualPath-CRC/v1.0.4/`:

| Fichero | FILE_ID | URL |
|---|---|---|
| `huc-pilot-with-weights-v1.0.4.tar.gz` | `1FVnFE0SU0QowQZwWIX0pzQ6Q1nGdmWov` | https://drive.google.com/file/d/1FVnFE0SU0QowQZwWIX0pzQ6Q1nGdmWov/view |
| `huc-pilot-with-weights-v1.0.4.tar.gz.sha256` | `1Ue8UB1nkJTnBOCnNWfLTYLTcWKIPf07y` | https://drive.google.com/file/d/1Ue8UB1nkJTnBOCnNWfLTYLTcWKIPf07y/view |

- `sha256`: `42593581238bec74464c808b2afd2479ac1763104071ae172acd42b5396bedf0`
  (referencia el nombre canónico `huc-pilot-with-weights.tar.gz`).
- Imagen `linux/amd64`, apta para el HUC PC (Ubuntu x86 + RTX 5070).
- Descarga + verificación validadas (`gdown` + `sha256sum -c` → `…OK`).

### 🚀 Instalación / actualización en el HUC PC

Nuevo script `scripts/install_huc.sh` — hace todo el ciclo en un comando
(para+elimina el container conservando datos, descarga por FILE_ID,
verifica sha256, carga, lanza en `:80`, comprueba salud):

```bash
bash install_huc.sh                 # instala/actualiza a esta versión
bash install_huc.sh --purge-images  # además borra imágenes viejas (~14 GB)
bash install_huc.sh --port 8080     # si el puerto 80 está ocupado
```

Los FILE_IDs de esta versión ya vienen escritos en el script.
