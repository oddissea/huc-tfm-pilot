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

## 1. Instalar o actualizar el piloto

Tienes dos caminos. **El A (recomendado) es el script automático**; el
B es el manual de siempre por si prefieres ver cada paso.

---

### Camino A (recomendado) — script automático

Te paso el fichero **`install_huc.sh`** por WhatsApp (también está en el
repo, en `pilot/scripts/install_huc.sh`). Guárdalo en tu carpeta
personal (`~`) y ejecuta:

```
# A.1. (solo la primera vez) instalar gdown, que descarga de Google Drive
sudo apt install -y pipx
pipx install gdown
export PATH="$HOME/.local/bin:$PATH"

# A.2. lanzar el instalador (ya trae dentro los FILE_IDs de esta versión)
bash install_huc.sh
```

Eso es todo. El script hace el ciclo completo por ti:

1. Para y elimina el container anterior (**tus correcciones y jobs en
   `~/huc-pilot-data/` se conservan**).
2. Descarga la imagen desde el Drive de Lumen Network (~4,4 GB, 5-15 min).
3. Verifica la integridad con `sha256` (si la descarga se corta, te
   avisa y para — no instala nada corrupto).
4. Carga la imagen y arranca el piloto en el **puerto 80**.
5. Comprueba que la app responde y te lo confirma.

Opciones útiles:

```
# Además borra las imágenes viejas del piloto y libera ~14 GB de disco:
bash install_huc.sh --purge-images

# Si el puerto 80 del HUC PC está ocupado por otro servicio:
bash install_huc.sh --port 8080      # luego abre http://localhost:8080
```

Cuando termine, abre el navegador en **`http://localhost`** y salta al
punto **2** de esta guía.

> Si el script se detiene con un mensaje en rojo (`ERROR: …`), léelo: te
> dice exactamente qué falta (Docker parado, gdown sin instalar, sha
> incorrecto…). La tabla de la sección **4** cubre los casos típicos.

---

### Camino B (manual) — paso a paso

Si prefieres ejecutar cada comando a mano, pega lo siguiente (todo a la
vez o uno a uno, ambas formas funcionan):

```
# B.1. Instalar gdown (para descargar la imagen desde Google Drive)
sudo apt install -y pipx
pipx install gdown
export PATH="$HOME/.local/bin:$PATH"

# B.2. Crear directorios de datos persistentes
#      Aquí guardará el piloto las correcciones del patólogo y la cola
#      de jobs procesados. Sobreviven aunque pares/elimines el container.
mkdir -p ~/huc-pilot-data/archive ~/huc-pilot-data/queue

# B.3. Descargar la imagen Docker y su checksum desde el Shared Drive
#      de Lumen Network. La imagen son ~4,4 GB → tarda 5-15 min según
#      la conexión del HUC. El .sha256 son 96 bytes, instantáneo.
cd ~
gdown "1FVnFE0SU0QowQZwWIX0pzQ6Q1nGdmWov" -O huc-pilot-with-weights.tar.gz
gdown "1Ue8UB1nkJTnBOCnNWfLTYLTcWKIPf07y" -O huc-pilot-with-weights.tar.gz.sha256

# B.4. Verificar integridad del fichero (tarda 30-60 segundos)
sha256sum -c huc-pilot-with-weights.tar.gz.sha256
```

→ La última línea debe imprimir: `huc-pilot-with-weights.tar.gz: OK`

Si dice `FAILED`, vuelve a ejecutar la descarga (la red del HUC pudo
haber cortado en medio).

```
# B.5. Cargar la imagen al daemon Docker (tarda 2-5 minutos)
docker load -i huc-pilot-with-weights.tar.gz
docker tag huc-pilot:dev-with-weights huc-pilot:dev

# B.6. Lanzar el container
#      OJO al -p 127.0.0.1:80:80 -> el piloto solo será accesible desde
#      ESTE ordenador, no desde otros equipos de la red. Es lo que quieres
#      en una red compartida como la de la ULL. Si algún día necesitaras
#      entrar desde otra máquina del grupo, cambia a "-p 80:80".
docker run -d \
  --name huc-pilot \
  --gpus all \
  -p 127.0.0.1:80:80 \
  -v ~/huc-pilot-data/archive:/var/archive \
  -v ~/huc-pilot-data/queue:/tmp/queue \
  --restart unless-stopped \
  huc-pilot:dev

# B.7. Confirmar que arrancó
docker ps
```

→ Debe aparecer una línea con `huc-pilot` en estado `Up X seconds` y
el puerto 80 mapeado: `0.0.0.0:80->80/tcp`. Dentro del container,
nginx multiplexa Streamlit y los tiles del visor — solo hay un único
puerto externo, mucho más simple que en versiones anteriores.

#### B.8 (opcional) — Liberar 4,4 GB del home

Después de que `docker ps` confirme que el container está `Up`, el
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
2. Subir un slide TIFF (**ya puedes subir varios a la vez** — esta
   versión arregla el fallo de la subida múltiple).
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
| `sha256sum: FAILED` | Descarga corrupta | Volver a descargar el `.tar.gz` (o relanzar `install_huc.sh`) |
| `Cannot connect to the Docker daemon` | Docker daemon no arrancado | `sudo systemctl start docker` |
| `Could not select device driver "nvidia"` | NVIDIA Container Toolkit mal configurado | Avisar a Nasser |
| `bind: address already in use` (puerto 80) | Otro proceso usa el puerto 80 (otro nginx, apache, o un servicio web del HUC PC) | Con el script: `bash install_huc.sh --port 8080` y abrir `http://localhost:8080`. A mano: `docker run ... -p 8080:80 ...`. O ver qué ocupa el 80 con `sudo lsof -i :80` y pararlo. |
| `Unable to open [object Object]: Unable to load TileSource` en el visor | nginx interno no sirve los DZIs | Verificar con `docker exec huc-pilot ls /tmp/queue` que hay carpetas de jobs. Si están vacías, es el worker. Si están, capturar `docker logs huc-pilot --tail 50` y enviar a Nasser. |
| `out of disk space` | Sin espacio para la imagen (~14 GB tras `docker load`) | Liberar espacio en `~`, o relanzar con `bash install_huc.sh --purge-images` |
| `docker: Error response from daemon: ... already in use by container` | Container viejo con ese nombre | `docker rm huc-pilot` y reintentar (el script lo hace solo) |
| `Permission denied (write)` al hacer `mkdir` | Permisos en `~` | Verifica que el home es tuyo: `ls -ld ~` |

Si el mensaje **no es ninguno de estos**, hazme una captura entera
del terminal (con el comando que ejecutaste arriba y el error abajo)
y mándamela por WhatsApp.

## 5. Cuando llegue una versión nueva del modelo

El procedimiento de actualización es el **mismo** que el de instalación
— vuelve a ejecutar el **Camino A**:

```
bash install_huc.sh --purge-images
```

(El `--purge-images` borra la imagen vieja antes de cargar la nueva, así
no se te acumulan 14 GB por versión. Tus correcciones en
`~/huc-pilot-data/` se conservan siempre.)

Si te mando una versión con **FILE_IDs distintos** y un `install_huc.sh`
nuevo, simplemente usa el script nuevo. Si prefieres no cambiar de
script, también puedes pasárselos por variables de entorno:

```
TAR_FILE_ID="<NUEVO_TAR_ID>" SHA_FILE_ID="<NUEVO_SHA_ID>" bash install_huc.sh --purge-images
```

Y si quieres hacerlo **a mano** (Camino B), repite los pasos B.1–B.7
sustituyendo los dos FILE_IDs del paso B.3 por los que te pase.

## Contacto

Si en cualquier paso algo no es claro o aparece un error que no
sabes interpretar, llámame o mándame WhatsApp con la captura. La
mayoría de problemas se resuelven en 10 minutos por videollamada.

— Nasser
