# Guía de despliegue del piloto en el HUC

Eduardo, esta guía te explica paso a paso cómo arrancar el piloto
**DualPath CRC — by Lumen Network** en el ordenador del HUC. Está
pensada para que la sigas tú solo; si en algún paso te atascas,
llámame.

## Antes de empezar — prerrequisito (instalado en sesión previa)

El ordenador del HUC ya debe tener instalado (validado contigo el
2026-05-25):

- **Ubuntu 24.04** nativo.
- **Docker CE 29.4.3** + Compose v5.1.3.
- **Driver NVIDIA 580.159.03** + CUDA 13.0.
- **NVIDIA Container Toolkit 1.19.1**.

### Cómo verificar que todo sigue OK

```
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

→ Debe imprimir una tabla con tu **RTX 5070**, `Driver Version:
580.xx`, `CUDA Version: 13.0` y procesos vacíos.

Si en lugar de la tabla sale `Could not select device driver "nvidia"`
o `'docker' no se reconoce` → **avísame** antes de continuar.

## 1. Deploy inicial — un único bloque copy-paste

Abre terminal y pega los comandos siguientes (todos a la vez o uno a
uno, ambas formas funcionan):

```
# 1.1. Instalar gdown (para descargar la imagen desde Google Drive)
sudo apt install -y pipx
pipx install gdown
export PATH="$HOME/.local/bin:$PATH"

# 1.2. Crear directorios de datos persistentes
#      Aquí guardará el piloto las correcciones del patólogo y la cola
#      de jobs procesados. Sobreviven aunque pares/elimines el container.
mkdir -p ~/huc-pilot-data/archive ~/huc-pilot-data/queue

# 1.3. Descargar la imagen Docker y su checksum desde el Shared Drive
#      de Lumen Network. La imagen son ~4,4 GB → tarda 5-15 min según
#      la conexión del HUC. El .sha256 son 96 bytes, instantáneo.
cd ~
gdown "1iR8AHCIofHCfOwQilkD3q3z7mmCs3Fu0" -O huc-pilot-with-weights.tar.gz
gdown "1A9B1xTTN_A1l5MGpHnloqhMsIJaEL6OI" -O huc-pilot-with-weights.tar.gz.sha256

# 1.4. Verificar integridad del fichero (tarda 30-60 segundos)
sha256sum -c huc-pilot-with-weights.tar.gz.sha256
```

→ La última línea debe imprimir: `huc-pilot-with-weights.tar.gz: OK`

Si dice `FAILED`, vuelve a ejecutar la descarga (la red del HUC pudo
haber cortado en medio).

```
# 1.5. Cargar la imagen al daemon Docker (tarda 2-5 minutos)
docker load -i huc-pilot-with-weights.tar.gz
docker tag huc-pilot:dev-with-weights huc-pilot:dev

# 1.6. Lanzar el container
docker run -d \
  --name huc-pilot \
  --gpus all \
  -p 80:80 \
  -v ~/huc-pilot-data/archive:/var/archive \
  -v ~/huc-pilot-data/queue:/tmp/queue \
  --restart unless-stopped \
  huc-pilot:dev

# 1.7. Confirmar que arrancó
docker ps
```

→ Debe aparecer una línea con `huc-pilot` en estado `Up X seconds` y
el puerto 80 mapeado: `0.0.0.0:80->80/tcp`. Dentro del container,
nginx multiplexa Streamlit y los tiles del visor — solo hay un único
puerto externo, mucho más simple que en versiones anteriores.

### 1.8 (opcional) — Liberar 4,4 GB del home

Después de `docker ps` confirme que el container está `Up`, el
`.tar.gz` ya no se necesita: la imagen vive en el almacén interno de
Docker (`/var/lib/docker/...`), independiente del archivo de
descarga. Si quieres liberar espacio del home:

```
rm ~/huc-pilot-with-weights.tar.gz
rm ~/huc-pilot-with-weights.tar.gz.sha256
```

Libera ~4,4 GB. **No es urgente** — en un PC con disco de 500 GB-1 TB
no notarás la diferencia, y mantener el `.tar.gz` te ahorra una
descarga si en algún momento necesitaras reinstalar la imagen sin
internet. Decisión tuya.

## 2. Abrir y usar el piloto

Abre Firefox o Chrome y ve a:

```
http://localhost
```

A partir de aquí, sigue el flujo habitual que ya conoces:

1. Sidebar → **"Cargar modelos"** (debería ser inmediato; los pesos
   ya están dentro de la imagen).
2. Subir un slide TIFF.
3. Esperar inferencia.
4. Hacer correcciones en el visor.

Para la **página `⚙️ Configuración`**, consulta `QA_EDUARDO.md` —
explica TTL, estado del archive y acciones (puedo enviártelo por
WhatsApp).

## 3. Comandos diarios

### Apagar al final de la jornada

```
docker stop huc-pilot
```

Solo para el container, no borra nada. Las correcciones y jobs
quedan a salvo en `~/huc-pilot-data/`.

### Volver a arrancar al día siguiente

```
docker start huc-pilot
```

Reanuda el container con todo su estado. La app vuelve a estar
accesible en `http://localhost` en pocos segundos (nginx interno
arranca casi instantáneo, Streamlit detrás tarda ~5-10 s en estar
listo para servir requests).

### Ver logs en tiempo real

```
docker logs -f huc-pilot
```

Sale con `Ctrl+C`. Útil si la app no responde y quieres ver qué
está pasando.

### Reiniciar el container si se queda colgado

```
docker restart huc-pilot
```

Equivalente a `stop` + `start` en un comando.

## 4. Si algo falla — errores típicos

| Mensaje exacto | Qué pasa | Qué hacer |
|---|---|---|
| `'docker' no se reconoce` o `command not found` | Docker no instalado | Avisar a Nasser |
| `gdown: command not found` | gdown no instalado | Reintentar `pipx install gdown && export PATH="$HOME/.local/bin:$PATH"` |
| `error: externally-managed-environment` al hacer `pip install gdown` | Ubuntu 24.04 protege Python sistema | Usa `pipx install gdown` en lugar de `pip install gdown` |
| `sha256sum: FAILED` | Descarga corrupta | Volver a descargar el `.tar.gz` |
| `Cannot connect to the Docker daemon` | Docker daemon no arrancado | `sudo systemctl start docker` |
| `Could not select device driver "nvidia"` | NVIDIA Container Toolkit mal configurado | Avisar a Nasser |
| `bind: address already in use` (puerto 80) | Otro proceso usa el puerto 80 (puede ser otro nginx, apache, o algún servicio web del HUC PC) | Cambiar a un puerto alto: `docker run ... -p 8080:80 ...` y acceder a `http://localhost:8080`. O identificar qué ocupa el 80 con `sudo lsof -i :80` y pararlo. |
| `Unable to open [object Object]: Unable to load TileSource` en el visor | nginx interno no está sirviendo los DZIs correctamente | Verificar con `docker exec huc-pilot ls /tmp/queue` que hay carpetas de jobs procesados. Si están vacías, el problema es del worker. Si están, capturar `docker logs huc-pilot --tail 50` y enviar a Nasser. |
| `out of disk space` | Sin espacio para la imagen (~14 GB tras `docker load`) | Liberar espacio en `~` |
| `docker: Error response from daemon: ... already in use by container` | Hay un container viejo con ese nombre | `docker rm huc-pilot` y reintentar el `docker run` |
| `Permission denied (write)` al hacer `mkdir` | Permisos en `~` | Verifica que el directorio home es tuyo: `ls -ld ~` |

Si el mensaje **no es ninguno de estos**, hazme una captura entera
del terminal (con el comando que ejecutaste arriba y el error abajo)
y mándamela por WhatsApp.

## 5. Cuando llegue una versión nueva del modelo (post-defensa)

Después de la defensa, cuando reentrenemos el modelo con las
correcciones acumuladas (Hito 2 del módulo de aprendizaje), te
mandaré por WhatsApp:

- Los nuevos **FILE_IDs** del `.tar.gz` y `.sha256` (cambian con cada
  versión).
- Un link al `CHANGELOG.md` del Shared Drive con qué incluye la
  versión nueva.

El procedimiento de actualización es:

```
# Eliminar el container viejo (las correcciones en ~/huc-pilot-data/
# se conservan automáticamente).
docker stop huc-pilot
docker rm huc-pilot

# Descargar nueva versión (sustituye <NUEVO_TAR_ID> y <NUEVO_SHA_ID>
# por los que te pasaré).
cd ~
gdown "<NUEVO_TAR_ID>" -O huc-pilot-with-weights.tar.gz
gdown "<NUEVO_SHA_ID>" -O huc-pilot-with-weights.tar.gz.sha256
sha256sum -c huc-pilot-with-weights.tar.gz.sha256

# Cargar y relanzar
docker load -i huc-pilot-with-weights.tar.gz
docker tag huc-pilot:dev-with-weights huc-pilot:dev
docker run -d \
  --name huc-pilot \
  --gpus all \
  -p 80:80 \
  -v ~/huc-pilot-data/archive:/var/archive \
  -v ~/huc-pilot-data/queue:/tmp/queue \
  --restart unless-stopped \
  huc-pilot:dev
```

Los volúmenes `archive/` y `queue/` se reusan automáticamente entre
versiones, así que las correcciones que hiciste antes siguen ahí.

## Contacto

Si en cualquier paso algo no es claro o aparece un error que no
sabes interpretar, llámame o mándame WhatsApp con la captura. La
mayoría de problemas se resuelven en 10 minutos por videollamada.

— Nasser
