# Despliegue de la imagen Docker en HUC (self-contained con pesos)

Procedimiento para desplegar **DualPath CRC** (el piloto, by Lumen
Network) en el ordenador del HUC usando una imagen Docker
self-contained que incluye los pesos del modelo F4 + 5 AttnMIL
ensemble bakeados dentro. Sin necesidad de pre-cargar pesos sueltos
ni de credenciales GCS.

Decisión de la sesión #65 (2026-05-23): empaquetar el modelo en la
propia imagen y transferirla vía USB como un único fichero
`.tar.gz`, en lugar de mantener pesos sueltos. Ventajas: cero error
humano de copia de paths, sello sha256 único cubre código + modelo,
reproducibilidad fuerte.

## Estructura de la imagen

`huc-pilot:dev-with-weights` extiende `huc-pilot:dev` añadiendo una
sola capa con los pesos:

```
/app/
├── src/
├── pages/
├── scripts/
├── app.py
└── weights/                                  ← capa nueva
    ├── F4/final_inference_model.pth          (~114 MB)
    └── attnmil_production/
        ├── seed_42/model.pth                 (~1.5 MB)
        ├── seed_123/model.pth                (~1.5 MB)
        ├── seed_456/model.pth                (~1.5 MB)
        ├── seed_789/model.pth                (~1.5 MB)
        └── seed_2026/model.pth               (~1.5 MB)
```

Tamaño total: ~14 GB sin comprimir, ~5-6 GB comprimida con gzip.
La diferencia gigante respecto a los 137 MB de pesos es PyTorch 2.7
+ CUDA 12.8 + cuDNN runtime, que ocupa la mayor parte.

`src/inference/weights.py` tiene un **fast path offline**: si los 6
ficheros ya están en `/app/weights/`, no importa `google.cloud.storage`
ni intenta descargar de GCS. Con esta imagen, ese fast path siempre
se activa.

## Procedimiento operativo

### 1. En el entorno del alumno (con GCS y Docker)

```bash
cd pilot

# Descargar pesos desde GCS (solo primera vez o si cambian)
mkdir -p weights/F4 weights/attnmil_production
gsutil cp gs://huc-tfm-pilot-models/F4/final_inference_model.pth \
  weights/F4/
for seed in 42 123 456 789 2026; do
  mkdir -p weights/attnmil_production/seed_$seed
  gsutil cp gs://huc-tfm-pilot-models/attnmil_production/seed_$seed/model.pth \
    weights/attnmil_production/seed_$seed/
done

# Build de la imagen principal (huc-pilot:dev)
docker compose build app

# Build de la variante self-contained
docker build -f Dockerfile.huc -t huc-pilot:dev-with-weights .

# Save + compresión
docker save huc-pilot:dev-with-weights | gzip > huc-pilot-with-weights.tar.gz

# Verificar tamaño
ls -lh huc-pilot-with-weights.tar.gz

# Checksum para verificar integridad post-transferencia
shasum -a 256 huc-pilot-with-weights.tar.gz > huc-pilot-with-weights.tar.gz.sha256
```

### 2. Transferencia al HUC

```bash
rsync -av --progress \
  huc-pilot-with-weights.tar.gz \
  huc-pilot-with-weights.tar.gz.sha256 \
  /Volumes/USB_CIFRADO/
```

### 3. En el host del HUC (Windows 11 con Docker en WSL2)

```bash
# Verificar integridad post-transferencia
cd /mnt/d/USB_CIFRADO    # o donde monte WSL2 el USB
sha256sum -c huc-pilot-with-weights.tar.gz.sha256

# Cargar la imagen al daemon Docker (tarda 2-5 min descomprimiendo)
docker load -i huc-pilot-with-weights.tar.gz

# Verificar que está cargada
docker images huc-pilot:dev-with-weights

# Re-tagear como huc-pilot:dev para que el docker-compose.yml estándar
# la encuentre sin modificaciones (alternativa: editar el image: del
# compose para que apunte a la variante with-weights).
docker tag huc-pilot:dev-with-weights huc-pilot:dev

# Clonar el repo (si no se hizo antes)
git clone https://github.com/oddissea/huc-tfm-pilot.git ~/huc-tfm-pilot
cd ~/huc-tfm-pilot

# Levantar
docker compose up -d

# Verificar arranque
docker compose logs app --tail=20
```

Al primer arranque, el log debería mostrar:

```
pesos ya cacheados en /app/weights, no se contacta con GCS
```

Si en su lugar muestra `downloading gs://...`, significa que algo en
la imagen no incluyó los pesos correctamente — la imagen incorrecta
no es la `dev-with-weights`.

## Actualización de la imagen (post-Hito 2 reentrenamiento)

Cuando los pesos se actualicen (Hito 2 produce un nuevo modelo F4
fine-tuneado con replay buffer):

1. Repetir paso 1 con los pesos nuevos.
2. Bumpear tag (ej. `huc-pilot:v2-with-weights`).
3. Transferir nuevo `.tar.gz` por USB.
4. En HUC: `docker load` + retag + `docker compose up -d`.

El histórico de versiones queda local en el host HUC; `docker images`
muestra todas las imágenes cargadas históricamente. Limpiar las
viejas con `docker rmi` cuando ya no se necesiten.

## Tamaño del USB necesario

- Imagen comprimida `.tar.gz`: ~5-6 GB.
- Buffer + checksums + otros artefactos: ~10 GB total razonable.
- **Recomendado: USB cifrado de 16 GB+** con formato exFAT
  (compatible macOS + Windows 11 nativo).

## Por qué no usamos un registry remoto (Docker Hub, ghcr.io)

Evaluado en sesión #65 y descartado:

- **Docker Hub público**: gratis pero los pesos del modelo quedan
  accesibles para descarga universal. Choca con la idea de Lumen
  Network como IP comercial futura.
- **ghcr.io privado**: tiene cuota free de solo 500 MB / 1 GB de
  bandwidth/mes para repos privados. Una imagen de 14 GB excede de
  largo el plan free; costaría ~$10-20/mes por una sola
  transferencia al HUC.
- **USB físico**: gratis, privado, sin recurring cost, sin
  dependencias de red en HUC. Una sola transferencia manual al
  desplegar.

Si en el futuro Lumen Network se comercializa o se despliega a más
hospitales, **conviene re-evaluar**: un registry privado con plan
pago tiene sentido cuando hay N hospitales recibiendo updates.

## Por qué no descargamos pesos vía `gsutil` desde GCS al primer arranque

El piloto antes descargaba pesos con `gsutil`/`google.cloud.storage`
al primer arranque del container. Eso requiere:

- Credenciales `gcloud auth application-default login` en el host.
- Permiso de IT del HUC para abrir HTTPS hacia `*.googleapis.com`.

Ambos pueden ser bloqueantes en HUC. Pre-bakeando los pesos en la
imagen y transfiriendo por USB, el container arranca **sin tocar
GCS ni Internet** una vez cargada la imagen.
