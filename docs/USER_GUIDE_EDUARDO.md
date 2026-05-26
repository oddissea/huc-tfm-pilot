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

**Cómo verificar que todo está listo**: abre terminal y ejecuta:

```
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

→ Si responde con una tabla mostrando tu **RTX 5070**, todo OK, sigue
a la sección 1.

→ Si dice `Could not select device driver "nvidia"` o `'docker' no se
reconoce` → **avísame** antes de continuar.

## 1. Descargar la imagen del piloto desde Google Drive

La imagen Docker (`huc-pilot-with-weights.tar.gz`, ~4,4 GB) está
alojada en el Shared Drive de Lumen Network. Te he compartido la
subcarpeta `Releases/DualPath-CRC/v1.0/` con permiso de lector. Tienes
el link en el correo (o lo pides por WhatsApp).

Hay dos formas de descargarla: la **A** es la más sencilla (clic en
navegador), la **B** es scripteable y la recomendada si vas a
actualizar versiones en el futuro.

### Opción A — Descarga manual desde navegador

1. Abre el link de Google Drive en Firefox o Chrome.
2. Sobre el fichero `huc-pilot-with-weights.tar.gz`, clic derecho →
   **"Descargar"**.
3. Google avisa de "No se puede analizar virus, ¿continuar?" → confirma.
4. Espera 5-15 min (depende de la red del HUC; son ~4,4 GB).
5. Descarga también el fichero `huc-pilot-with-weights.tar.gz.sha256`
   de la misma carpeta (~100 bytes, instantáneo).
6. Mueve ambos a `~/huc-tfm-pilot/`:

```
mv ~/Descargas/huc-pilot-with-weights.tar.gz* ~/huc-tfm-pilot/
cd ~/huc-tfm-pilot/
```

### Opción B — Descarga scripteable con `gdown`

Útil si vas a actualizar a versiones futuras o no quieres usar
navegador. Requiere `gdown` (instalación una vez, 30 s):

```
sudo apt install -y python3-pip
pip install gdown
```

Luego, con el **ID del fichero** que te paso por separado:

```
cd ~/huc-tfm-pilot/
gdown --fuzzy "https://drive.google.com/file/d/<FILE_ID>/view" -O huc-pilot-with-weights.tar.gz
gdown --fuzzy "https://drive.google.com/file/d/<SHA_ID>/view" -O huc-pilot-with-weights.tar.gz.sha256
```

(Te enviaré `<FILE_ID>` y `<SHA_ID>` por separado, no van en este doc.)

### Verificar integridad

Una vez descargado, asegúrate de que el fichero no se corrompió:

```
sha256sum -c huc-pilot-with-weights.tar.gz.sha256
```

→ Debe imprimir: `huc-pilot-with-weights.tar.gz: OK`

Si dice `FAILED`, vuelve a descargar (la red del HUC puede haber
cortado).

## 2. Despliegue: opción atajo o paso a paso

### Atajo — un único comando

Si todo lo anterior está correcto, ejecuta el wrapper:

```
bash ~/huc-tfm-pilot/pilot/scripts/huc-deploy.sh ~/huc-tfm-pilot/huc-pilot-with-weights.tar.gz
```

El script hace los 4 pasos manuales (verificación + load + tag +
`docker compose up`) en una sola invocación, con progreso visible.
Termina con un mensaje verde **"✅ Despliegue completo"** y la app
accesible en `http://localhost:8501`.

**Alternativa moderna del script** — si quieres que el script también
descargue de Drive (no solo cargue un fichero local):

```
bash ~/huc-tfm-pilot/pilot/scripts/huc-deploy.sh "https://drive.google.com/file/d/<FILE_ID>/view"
```

El script detecta que es un link de Drive, descarga + verifica +
carga. Útil para actualizaciones futuras: un solo comando, recibes
link nuevo, lo lanzas. Requiere `gdown` instalado (ver §1.B).

### Paso a paso (versión larga, recomendada la primera vez)

Si prefieres ir comando por comando para entender qué hace cada uno:

```
# 2.1. Cargar la imagen al daemon Docker (tarda 2-5 min)
cd ~/huc-tfm-pilot/
docker load -i huc-pilot-with-weights.tar.gz
```

→ Debe terminar con: `Loaded image: huc-pilot:dev-with-weights`

```
# 2.2. Renombrar la imagen (para que docker compose la reconozca)
docker tag huc-pilot:dev-with-weights huc-pilot:dev
```

→ Sin salida. Es normal.

```
# 2.3. Levantar todos los servicios en segundo plano
cd ~/huc-tfm-pilot/pilot
docker compose up -d
```

→ Tres líneas tipo `Container huc-pilot-XXX Started`.

```
# 2.4. Comprobar que arrancó bien (espera 10 segundos antes)
sleep 10
docker compose logs app --tail=20
```

→ Debes ver al final del log: `You can now view your Streamlit app in
your browser.` y `URL: http://0.0.0.0:8501`.

Si el log muestra **errores rojos**, hazme una captura entera y
mándamela.

## 3. Abrir y usar el piloto

Abre Firefox y ve a:

```
http://localhost:8501
```

A partir de aquí, sigue el flujo habitual que ya conoces:

1. Sidebar → **"Cargar modelos"** (debería ser inmediato; los pesos
   ya están dentro de la imagen).
2. Subir un slide TIFF.
3. Esperar inferencia.
4. Hacer correcciones en el visor.

Para la **página `⚙️ Configuración`**, consulta `QA_EDUARDO.md` —
explica TTL, estado del archive y acciones.

## 4. Apagar al final de la jornada (opcional pero recomendado)

Para no dejar el container corriendo durante la noche:

```
cd ~/huc-tfm-pilot/pilot
docker compose down
```

La próxima vez que quieras usar el piloto, solo necesitas repetir
**`docker compose up -d`** (la imagen ya está cargada en Docker desde
el primer deploy).

## 5. Si algo falla — errores típicos

| Mensaje exacto | Qué pasa | Qué hacer |
|---|---|---|
| `'docker' no se reconoce` o `command not found` | Docker no instalado | Avisar a Nasser |
| `sha256sum: FAILED` | Descarga corrupta | Volver a descargar |
| `Cannot connect to the Docker daemon` | Docker daemon no arrancado | `sudo systemctl start docker` |
| `Could not select device driver "nvidia"` | NVIDIA Container Toolkit mal configurado | Avisar a Nasser |
| `bind: address already in use` | Puerto 80/443/8501 ocupado | `docker compose down` y reintentar |
| `out of disk space` | Sin espacio para la imagen (~14 GB) | Liberar espacio en `~` |
| `gdown: command not found` | `gdown` no instalado | `pip install gdown` |
| `Permission denied (write)` al hacer `mv` | Permisos en `~/huc-tfm-pilot/` | Verifica que la carpeta es tuya: `chown -R eduardo:eduardo ~/huc-tfm-pilot/` |

Si el mensaje **no es ninguno de estos**, hazme una captura entera
del terminal (con el comando que ejecutaste arriba y el error abajo)
y mándamela.

## 6. Cuando llegue una versión nueva del modelo (post-defensa)

Después de la defensa, cuando reentrenemos el modelo con las
correcciones acumuladas (Hito 2 del módulo de aprendizaje), te
mandaré por WhatsApp el **link nuevo de Drive** apuntando a la nueva
versión (`Releases/DualPath-CRC/v1.1/`, etc.).

El procedimiento de actualización es **idéntico al de la primera
vez**: descargas (manual o con `gdown`), verificas sha256, ejecutas
`huc-deploy.sh` apuntando al nuevo `.tar.gz`. Docker reemplaza la
imagen vieja automáticamente; no hace falta desinstalar nada.

**Atajo moderno** — si tienes `gdown` instalado, una sola línea:

```
bash ~/huc-tfm-pilot/pilot/scripts/huc-deploy.sh "<LINK_NUEVO_DRIVE>"
```

## Contacto

Si en cualquier paso algo no es claro o aparece un error que no
sabes interpretar, llámame o mándame WhatsApp con la captura. La
mayoría de problemas se resuelven en 10 minutos por videollamada.

— Nasser
