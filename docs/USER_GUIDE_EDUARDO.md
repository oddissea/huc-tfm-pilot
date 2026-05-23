# Guía de despliegue del piloto en el HUC

Eduardo, esta guía te explica paso a paso cómo arrancar el piloto
**DualPath CRC — by Lumen Network** en el ordenador del HUC, una vez
recibido el USB con el paquete. Está pensada para que la sigas tú
solo; si en algún paso te atascas, llámame.

## Antes de empezar — prerrequisito (lo hace IT del HUC o Nasser)

Esta guía asume que el ordenador del HUC ya tiene instalado:

- Docker Desktop (o Docker CE en WSL2).
- NVIDIA Container Toolkit (para que Docker hable con la GPU RTX 5070).
- Acceso a un terminal (PowerShell o WSL2 bash).

**Cómo saber si está listo**: abre PowerShell y escribe:

```bash
docker info
```

- Si responde con un montón de información sobre versión, GPU, etc.
  → **listo**, sigue a la sección 1.
- Si dice `'docker' no se reconoce como un comando...` o sale otro
  error → **avísame** (o a IT del HUC) antes de continuar. La
  instalación inicial requiere reiniciar Windows y configurar
  permisos; no es algo que debas hacer tú solo.

## 1. Cargar la imagen del piloto desde el USB

Conecta el USB cifrado al ordenador. Asume que Windows lo monta como
unidad **`D:`** (puede ser `E:`, `F:`, etc.; ajusta los comandos si es
otra letra).

### Atajo: un único comando

Si prefieres no hacer los 4 pasos manuales, hay un wrapper que los
encadena con verificación intermedia. Abre WSL2 / Git Bash y ejecuta:

```bash
bash ~/huc-tfm-pilot/pilot/scripts/huc-deploy.sh /mnt/d/huc-pilot-with-weights.tar.gz
```

Te imprime el progreso de cada paso. Si todo va bien, termina con un
mensaje verde **"✅ Despliegue completo"** y la app accesible en
`http://localhost:8501`.

Si prefieres ir paso a paso (recomendado la primera vez para entender
qué hace cada comando), sigue las secciones 1.1 a 2.3 a continuación.

### Paso a paso (versión larga)

Abre PowerShell y ejecuta, comando por comando:

```bash
# 1.1. Cambiar al directorio del USB
cd D:\

# 1.2. Verificar integridad del fichero (debe tardar 30-60 segundos)
sha256sum -c huc-pilot-with-weights.tar.gz.sha256
```

→ Debe imprimir: `huc-pilot-with-weights.tar.gz: OK`

Si dice `FAILED`, el USB se corrompió en la transferencia. Llámame.

```bash
# 1.3. Cargar la imagen al daemon Docker (tarda 2-5 minutos)
docker load -i huc-pilot-with-weights.tar.gz
```

→ Debería terminar con: `Loaded image: huc-pilot:dev-with-weights`

```bash
# 1.4. Renombrar la imagen (para que docker-compose la reconozca)
docker tag huc-pilot:dev-with-weights huc-pilot:dev
```

→ Sin salida. Es normal.

## 2. Lanzar el piloto

```bash
# 2.1. Cambiar al directorio del repositorio del piloto (donde lo
#      tengas clonado o copiado del USB; ajusta la ruta).
cd C:\Users\PC\huc-tfm-pilot\pilot
```

```bash
# 2.2. Levantar todos los servicios en segundo plano
docker compose up -d
```

→ Tres líneas tipo `Container huc-pilot-XXX Started`.

```bash
# 2.3. Comprobar que arrancó bien (espera 10 segundos antes)
docker compose logs app --tail=20
```

→ Debes ver al final del log: `You can now view your Streamlit app in
your browser.` y `URL: http://0.0.0.0:8501`.

Si el log muestra **errores rojos**, hazme una captura entera y
mándamela.

## 3. Abrir y usar el piloto

Abre el navegador (Edge o Firefox) y ve a:

```
http://localhost:8501
```

A partir de aquí, sigue el flujo habitual que ya conoces:

1. Sidebar → **"Cargar modelos"** (debería ser inmediato; los pesos
   ya están dentro de la imagen).
2. Subir un slide TIFF.
3. Esperar inferencia.
4. Hacer correcciones en el visor.

Para la **página `⚙️ Configuración`** (nueva), consulta
`QA_EDUARDO.md` — explica TTL, estado del archive y acciones.

## 4. Apagar al final de la jornada (opcional pero recomendado)

Para no dejar el container corriendo durante la noche:

```bash
cd C:\Users\PC\huc-tfm-pilot\pilot
docker compose down
```

La próxima vez que quieras usar el piloto, solo necesitas repetir
**`docker compose up -d`** (el `docker load` de la sección 1 es
único; la imagen ya está cargada en Docker desde entonces).

## 5. Si algo falla — los errores típicos

| Mensaje exacto | Qué pasa | Qué hacer |
|---|---|---|
| `'docker' no se reconoce como un comando` | Docker no instalado | Avisar IT del HUC o Nasser |
| `sha256sum: FAILED` | USB corrupto | Pedir un USB nuevo |
| `Cannot connect to the Docker daemon` | Docker no arrancó | Abrir Docker Desktop; esperar a que la ballenita verde aparezca |
| `Could not select device driver "nvidia"` | NVIDIA Container Toolkit no configurado | Avisar IT del HUC |
| `bind: address already in use` | Puerto 80/443 ocupado | Avisar a Nasser |
| `out of disk space` | Sin espacio para la imagen (~14 GB) | Liberar espacio en `C:` |

Si el mensaje **no es ninguno de estos**, hazme una captura entera
del terminal (con el comando que ejecutaste arriba y el error abajo)
y mándamela.

## 6. Cuando llegue una versión nueva del modelo (post-defensa)

Después de la defensa, cuando reentrenemos el modelo con las
correcciones acumuladas (Hito 2 del módulo de aprendizaje), te
mandaré un USB nuevo con un `.tar.gz` actualizado. El procedimiento
para actualizar es **idéntico al paso 1**: cargar la nueva imagen,
renombrarla, y `docker compose up -d` lo recoge automáticamente.

No hace falta desinstalar la versión vieja; Docker mantiene ambas y
puedes alternar si fuera necesario.

## Contacto

Si en cualquier paso algo no es claro o aparece un error que no
sabes interpretar, llámame o mándame WhatsApp con la captura. La
mayoría de problemas se resuelven en 10 minutos por video-llamada.

— Nasser
