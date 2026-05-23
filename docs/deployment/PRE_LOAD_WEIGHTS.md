# Pre-carga de pesos del modelo (deploy HUC offline)

Por defecto, `src/inference/weights.py` descarga los pesos del modelo F4
+ los 5 AttnMIL desde `gs://huc-tfm-pilot-models/` al primer arranque
del container. En el HUC, donde el host no tiene credenciales GCS ni
necesariamente conectividad de salida, los pesos se pre-cargan
manualmente.

El módulo `weights.py` tiene un **fast path offline**: si los 6 ficheros
ya están en `WEIGHTS_DIR`, NO se inicializa el cliente GCS (ni siquiera
se importa `google.cloud.storage`). El piloto arranca sin red.

## Estructura esperada

El bind mount `./weights:/app/weights` del `docker-compose.yml` debe
contener:

```
weights/
├── F4/
│   └── final_inference_model.pth                 (~71 MB)
└── attnmil_production/
    ├── seed_42/model.pth                         (~14 MB)
    ├── seed_123/model.pth                        (~14 MB)
    ├── seed_456/model.pth                        (~14 MB)
    ├── seed_789/model.pth                        (~14 MB)
    └── seed_2026/model.pth                       (~14 MB)
```

Total ~150 MB. Cabe de sobra en un USB.

## Procedimiento de pre-carga (alumno antes de entrar al HUC)

### Paso 1: descargar pesos desde una máquina con acceso GCS

Desde un entorno con `gcloud` autenticado y permisos sobre
`gs://huc-tfm-pilot-models/`:

```bash
mkdir -p /tmp/huc-weights/F4
mkdir -p /tmp/huc-weights/attnmil_production

# F4
gsutil cp gs://huc-tfm-pilot-models/F4/final_inference_model.pth \
  /tmp/huc-weights/F4/

# AttnMIL ensemble (5 semillas)
for seed in 42 123 456 789 2026; do
  mkdir -p /tmp/huc-weights/attnmil_production/seed_$seed
  gsutil cp gs://huc-tfm-pilot-models/attnmil_production/seed_$seed/model.pth \
    /tmp/huc-weights/attnmil_production/seed_$seed/
done
```

### Paso 2: verificar checksums (opcional pero recomendado)

Calcular sha256 antes de copiar al USB y dejarlo apuntado para verificar
en HUC:

```bash
find /tmp/huc-weights -name "*.pth" -exec sha256sum {} \; \
  > /tmp/huc-weights/SHA256SUMS.txt
```

### Paso 3: transferir al USB cifrado

```bash
rsync -av /tmp/huc-weights/ /Volumes/USB_cifrado/huc-weights/
```

### Paso 4: copiar al host HUC

En el host del HUC, antes de levantar el container por primera vez:

```bash
# Crear estructura en el bind mount del repo del piloto:
mkdir -p /ruta/del/repo/pilot/weights

# Copiar desde USB
rsync -av /media/USB_cifrado/huc-weights/ /ruta/del/repo/pilot/weights/

# Verificar
cd /ruta/del/repo/pilot/weights && sha256sum -c SHA256SUMS.txt
```

### Paso 5: levantar el container

```bash
cd /ruta/del/repo/pilot
docker compose up -d
```

Si todo está bien pre-cargado, el log del container debería mostrar:

```
pesos ya cacheados en /app/weights, no se contacta con GCS
```

Si en su lugar muestra `downloading gs://...`, significa que algún
fichero falta o tiene path incorrecto.

## Verificación dentro del container

```bash
docker compose exec app python -c "
from src.inference.weights import ensure_weights, WEIGHTS_DIR
import google.cloud
import sys; sys.modules['google.cloud'] = None  # forzar fast path
res = ensure_weights()
print('F4:', res['f4'])
print('AttnMIL:', [(s, str(p)) for s, p in res['attnmil']])
"
```

Debe imprimir las 6 rutas sin errores.

## ¿Qué pasa si la conexión a Internet del HUC sí permite GCS?

Funciona, pero requiere credenciales `gcloud auth application-default
login` en el host antes del primer arranque. Si no hay credenciales y
el container tampoco encuentra los pesos pre-cargados, `ensure_weights`
fallará con un `AuthError` o similar al inicializar `storage.Client()`.

**Recomendación**: pre-cargar siempre por USB. Evita dependencia de
red al primer arranque y simplifica la conversación con IT del HUC
(no hace falta solicitar permisos de salida HTTPS a `*.googleapis.com`).
