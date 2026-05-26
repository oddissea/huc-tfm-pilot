# DualPath CRC — by Lumen Network

## Release v1.0 — 2026-05-26

Primera versión distribuida para despliegue en el HUC (PC del
servicio de Anatomía Patológica, Hospital Universitario de Canarias).

### Qué incluye

- **Modelo**: DualPath CRC F4 (BiT-M ResNet-50 estándar con
  arquitectura dual-stream, fusión por concatenación, Focal Loss
  calibrada con pesos raíz cuadrada).
- **Pesos preentrenados** del encoder y del clasificador, embebidos
  en la imagen Docker (no requiere descarga aparte).
- **Aplicación Streamlit** con flujo completo de inferencia +
  corrección por patólogo + visor OpenSeadragon con cuadrícula de
  parches.
- **Página `⚙️ Configuración`** para administración del piloto
  (TTL editable, estado del archive, prune manual, descarga
  archive.zip).
- **Pipeline de archivado**: cada job procesado guarda
  `corrections.jsonl` + `features.npy` + `patch_eval.npz` +
  `meta.json` (Hito 1).

### Métricas del modelo (sobre test set HUC, 80 portaobjetos)

| Métrica | Valor |
|---|---|
| Accuracy | 92,37% |
| Sensibilidad carcinoma | 98,8% |
| Error grave (carcinoma → normal) | 0,2% (5/2412) |
| Clinical Safety Score | 95,9/100 |

### Hardware soportado

- **GPU**: NVIDIA con CUDA 12.4+ (validado con RTX 5070 Blackwell,
  driver 580.159.03). Soporta cualquier GPU de >= 8 GB VRAM.
- **CPU fallback**: no incluido en esta versión (planificado para
  v1.1).

### Software soportado

- **SO**: Ubuntu 24.04 LTS validado. Otros Linux modernos
  probablemente compatibles.
- **Docker**: CE >= 28.0 con Buildx + Compose v2.
- **NVIDIA Container Toolkit**: >= 1.16.

### Despliegue

Ver `USER_GUIDE_EDUARDO.md` (este mismo Shared Drive lo aloja en la
carpeta `Documentation/`, o también está en el repo público
`huc-tfm-pilot`).

Resumen rápido del flujo sin clonar el repo (autosuficiente con esta
imagen):

1. Instalar `gdown` con `pipx install gdown`.
2. Descargar `huc-pilot-with-weights.tar.gz` + `.sha256` desde el
   Shared Drive con `gdown` (FILE_IDs abajo).
3. Verificar integridad con `sha256sum -c`.
4. `docker load -i ...` + `docker tag ... huc-pilot:dev`.
5. `docker run -d --gpus all -p 8501:8501 -v ~/huc-pilot-data/archive:/var/archive -v ~/huc-pilot-data/queue:/tmp/queue --restart unless-stopped --name huc-pilot huc-pilot:dev`.
6. Abrir `http://localhost:8501` en navegador.

### Tamaños

- `huc-pilot-with-weights.tar.gz`: ~4,4 GB (comprimido).
- Imagen Docker descomprimida: ~14 GB.
- Requiere ~20 GB libres en disco para el load inicial.

### Distribución — Google Drive (Shared Drive "Lumen Network")

Ruta: `Releases/DualPath-CRC/v1.0/`.

| Fichero | FILE_ID | URL |
|---|---|---|
| `huc-pilot-with-weights.tar.gz` | `1iR8AHCIofHCfOwQilkD3q3z7mmCs3Fu0` | https://drive.google.com/file/d/1iR8AHCIofHCfOwQilkD3q3z7mmCs3Fu0/view |
| `huc-pilot-with-weights.tar.gz.sha256` | `1A9B1xTTN_A1l5MGpHnloqhMsIJaEL6OI` | https://drive.google.com/file/d/1A9B1xTTN_A1l5MGpHnloqhMsIJaEL6OI/view |

Comando recomendado para Eduardo (HUC PC, Ubuntu, sin clonar repo):

```bash
sudo apt install -y pipx
pipx install gdown
export PATH="$HOME/.local/bin:$PATH"
mkdir -p ~/huc-pilot-data/archive ~/huc-pilot-data/queue
cd ~
gdown "1iR8AHCIofHCfOwQilkD3q3z7mmCs3Fu0" -O huc-pilot-with-weights.tar.gz
gdown "1A9B1xTTN_A1l5MGpHnloqhMsIJaEL6OI" -O huc-pilot-with-weights.tar.gz.sha256
sha256sum -c huc-pilot-with-weights.tar.gz.sha256
docker load -i huc-pilot-with-weights.tar.gz
docker tag huc-pilot:dev-with-weights huc-pilot:dev
docker run -d --name huc-pilot --gpus all -p 8501:8501 \
  -v ~/huc-pilot-data/archive:/var/archive \
  -v ~/huc-pilot-data/queue:/tmp/queue \
  --restart unless-stopped huc-pilot:dev
```

### Limitaciones conocidas

- **Tarea ternaria a nivel de parche** (normal / adenoma / carcinoma).
  La inferencia a nivel de portaobjetos (slide aggregator MIL) está
  implementada pero **no expuesta** en la UI de esta versión — está
  planificada para v1.1.
- **Sin agregación de múltiples patólogos**: el archive guarda
  correcciones por job, pero no compara entre patólogos. Para Hito 2
  está previsto.
- **Sin reentrenamiento online**: las correcciones acumuladas se
  recogen para reentrenamiento offline (Hito 2 post-defensa, requiere
  SSD TIME).
- **Dataset N=1 (HUC)**: las métricas son sobre una sola partición del
  HUC. La generalización a otros hospitales queda como trabajo futuro.

### Próximas versiones (roadmap orientativo)

- **v1.1** — Modo CPU fallback (para defensa sin GPU) + UI slide
  aggregator MIL.
- **v1.2** — Reentrenamiento head F4 con correcciones acumuladas
  (Hito 2).
- **v2.0** — Validación multi-centro (segundo hospital).

### Contacto

Soporte técnico: Fernando Nasser (Lumen Network).
