# Hito 2 — Fine-tune del head F4 con replay buffer

Hito del módulo de aprendizaje de **DualPath CRC** (el piloto, by Lumen
Network). Documento de diseño operativo. **NO implementado todavía**:
este doc
sirve para que la sesión que arranque la implementación parta de un
plan razonado, no de cero.

## Motivación

Hito 0 + Hito 1 garantizan que cada job procesado por el patólogo deja
en el archive del host un trío `(features.npy, corrections.jsonl,
meta.json)`. Hito 2 cierra el círculo: convertir esas correcciones en
una nueva versión del modelo F4 que sea **estrictamente mejor** que la
anterior, sin perder lo que el modelo ya sabía hacer bien sobre los 91
slides del §5.9.

El riesgo central que ataca este Hito es el **catastrophic forgetting**:
si re-entrenamos el head solo con las correcciones del patólogo, el
modelo se ajusta a ese subconjunto pequeño y degrada en datos
históricos. El **replay buffer** (mezclar correcciones nuevas con
muestras del §5.9 original) es la mitigación estándar.

## Componentes del problema

### Lo que ya existe

| Componente | Dónde vive |
|---|---|
| Pesos F4 actuales (`F4-v1`) | `gs://huc-tfm-pilot-models/F4/final_inference_model.pth` (descarga inicial) |
| Encoder BiT-M R50x1 congelado | parte del `F4Bundle` (no se toca en Hito 2) |
| Head F4 (Linear 4096→512 + ReLU + Dropout + Linear 512→3) | `src/models/classifier.py::MLPClassifier` |
| Features 512-d post-ReLU por parche | `archive/<job_id>/features.npy` (~2 KB/parche) |
| Correcciones del patólogo | `archive/<job_id>/corrections.jsonl` (índices de parches + clase nueva) |
| Metadatos del slide | `archive/<job_id>/meta.json` |
| §5.9 dataset (91 slides clínicos) | `data_remote/` (SSD TIME) o reconstruible desde HDF5 originales |

### Lo que NO existe todavía

- Embeddings 4096-d del encoder (pre-Linear(4096→512)) por parche. **NO los persistimos** — `worker.py` guarda solo los 512-d post-ReLU. Decisión abierta (ver §"Alcance del fine-tune").
- Estructura de versionado de pesos (`F4-v2`, `F4-v3`, …).
- Loop de fine-tune.
- Detector de catastrophic forgetting.
- UI que muestre versión activa del modelo (eso es Hito 5).

## Alcance del fine-tune: decisión de diseño abierta

El "head F4" en el TFM se refiere al MLP `Linear(4096→512) + ReLU +
Dropout + Linear(512→3)`. Hay dos opciones de alcance, con tradeoffs
distintos:

### Opción A: fine-tune solo de la última capa `Linear(512→3)`

**Lo que entrena**: 1.539 parámetros (512×3 + 3 bias).

**Por qué es atractiva**:
- Los features 512-d del archive son exactamente la entrada de esta
  capa. **Cero coste de IO extra**, cero cambios en `worker.py`.
- Tan pocos parámetros caben en CPU sin problemas: el fine-tune entero
  podría correrse en el propio HUC (i9-14900KF) sin GPU.
- Menos parámetros = menos riesgo de catastrophic forgetting con datos
  limitados.

**Limitación**:
- Las features post-ReLU del head original son una representación
  cristalizada por el F4-v1. Si la primera capa `Linear(4096→512)` no
  estaba bien calibrada para los casos de borde corregidos por el
  patólogo, la última capa por sí sola no puede arreglarlo. El techo
  del fine-tune es bajo.

### Opción B: fine-tune del head completo

**Lo que entrena**: `512×4096 + 512 + 3×512 + 3 = 2.099.715` parámetros.

**Por qué es atractiva**:
- Más expresividad. La primera capa puede re-mapear los embeddings
  4096-d del encoder hacia regiones nuevas del espacio latente.

**Limitación**:
- Requiere persistir embeddings 4096-d en el archive. ~16 KB/parche ×
  1.000 parches = 16 MB/slide. A 5 slides/día × 250 días/año = 20
  GB/año. Manejable pero multiplica × 8 la huella del archive.
- Más parámetros = más datos necesarios para que el fine-tune no
  sobreajuste. El replay buffer 1:3 puede no ser suficiente.

### Recomendación

**Empezar por Opción A** y mantener el archive ligero. Si la validación
muestra que el techo es demasiado bajo (el fine-tune no mueve la aguja
sobre §5.9), escalamos a Opción B en un Hito 2.1 que añadiría
`features_4096.npy` al `worker.py` y al archive.

Esta decisión condiciona el resto del doc: a partir de aquí asumo
**Opción A**.

## Diseño técnico

### Replay buffer 1:3 (corrección : §5.9)

Cada batch de fine-tune mezcla muestras de dos fuentes:

- **Correcciones nuevas** (del archive): por cada parche corregido, una
  muestra `(features_512, clase_nueva)`. Pesa **1** en la ratio.
- **§5.9 replay** (los 91 slides clínicos): por cada corrección, **3
  muestras** del §5.9, muestreadas con un sampler que mantenga la
  distribución de clase del conjunto original.

Con 30 correcciones en el archive y un batch effective de 128, cada
batch tendría ~32 correcciones (sub-muestrear con repetición si hay
menos) + 96 muestras §5.9.

### Pipeline de datos

```
archive/<job_id>/ →
  features.npy (N_patches × 512) +
  corrections.jsonl (lista de {patch_idx, label_new}) →
    extraer (features[patch_idx], label_new) por corrección.
```

Acumulado de todos los jobs del archive → `Dataset[Tensor 512, int]`.

`§5.9 dataset` cargado en memoria entera (~250 MB para 91 slides
patch-level): construido una sola vez al arrancar el fine-tune. Si el
SSD TIME no está conectado, fallback a un dump pre-generado en
`pilot/data_remote/section_5_9.npz` (a generar como sub-tarea).

### Modelo y loss

- Cargar pesos del `Linear(512→3)` actual desde el `F4-v1` activo
  (path discoverable vía `src/inference/model_versions.py`, nuevo).
- Congelar resto del head (`Linear(4096→512)` y `ReLU`).
- Loss: Focal Loss con `gamma=2.0` y `alpha = inverse-sqrt-frequency`
  recalculada sobre el merged dataset (correcciones + §5.9).
- Optimizer: Adam con `lr=1e-4`, `weight_decay=0`.
- Scheduler: ninguno al principio; añadir cosine si hace falta.
- Epochs: 20, con early stopping basado en validación §5.9 hold-out.

### Detección de catastrophic forgetting

**ANTES del fine-tune**:
- Calcular `metrics_before` sobre `§5.9` con el `F4-v1` activo:
  accuracy global, recall ADE/NOR/CAR, CAR→NOR rate, ACSS.

**DESPUÉS del fine-tune**:
- Calcular `metrics_after` con la nueva versión candidata.

**Decisión de aceptación**:
- Aceptar la nueva versión solo si **todas** estas condiciones se
  cumplen:
  1. `accuracy_after >= accuracy_before - 1.0pp` (no perder más de 1pp).
  2. `recall_CAR_after >= recall_CAR_before - 0.5pp` (recall CAR es la
     clase clínicamente más crítica).
  3. `CAR→NOR_after <= CAR→NOR_before` (NO empeorar el error grave).
  4. `ACSS_after >= ACSS_before - 0.5` (no degradar la métrica clínica).
- Si **cualquiera falla**, la versión se descarta y el archive sigue
  como estaba. El operador recibe un log con qué métrica falló.

### Versionado de pesos

```
weights/F4/
├── v1/
│   ├── model.pth                # head completo de F4-v1
│   └── meta.json                # {version, source, accuracy_§5.9, ...}
├── v2-2026-06-15/
│   ├── model.pth                # head con Linear(512→3) fine-tuneado
│   └── meta.json                # {version, parent: v1, n_corrections,
│                                #   accuracy_§5.9, accepted_at, ...}
└── current → v2-2026-06-15/     # symlink que el piloto lee
```

El piloto al arrancar resuelve `current/model.pth`. Si no existe, cae a
`v1/`. La UI (Hito 5) leerá `current/meta.json` para mostrar versión
activa.

### Cuándo se dispara el fine-tune

**Fase inicial (manual)**: el alumno (o Eduardo) lanza desde el host:

```bash
docker compose exec app python -m scripts.finetune_head --dry-run
docker compose exec app python -m scripts.finetune_head
```

`--dry-run` lista qué correcciones se usarían sin ejecutar nada.

**Fase posterior (semi-automática)**: cron diario que comprueba si
`n_corrections_acumuladas >= UMBRAL` (sugerencia: 30). Si sí, dispara
el fine-tune. Si no, log y siguiente día.

**Nunca automático sin red de seguridad**: el detector de forgetting
debe estar entre fine-tune y aceptación.

## APIs propuestas

### Nuevos módulos

```
src/learning/
├── __init__.py
├── replay_buffer.py             # mezcla correcciones + §5.9 con ratio 1:3
├── dataset.py                   # Dataset PyTorch sobre archive + §5.9
├── finetune.py                  # loop de entrenamiento
├── validation.py                # cálculo de métricas + detector forgetting
└── model_versions.py            # discovery + selección + activación

scripts/
└── finetune_head.py             # CLI wrapper
```

### Firmas (preliminares)

```python
# src/learning/replay_buffer.py
def build_replay_dataset(
    archive_dir: Path,
    section_59_path: Path,
    ratio: int = 3,
) -> torch.utils.data.Dataset: ...

# src/learning/finetune.py
def finetune_head(
    archive_dir: Path,
    section_59_path: Path,
    base_model_version: str,
    output_dir: Path,
    *,
    epochs: int = 20,
    lr: float = 1e-4,
    batch_size: int = 128,
    dry_run: bool = False,
) -> dict: ...
# devuelve dict con metrics_before, metrics_after, accepted, output_version

# src/learning/validation.py
def evaluate_on_section_59(
    head_weights: Path,
    section_59_path: Path,
) -> dict: ...

def is_acceptance_safe(
    metrics_before: dict,
    metrics_after: dict,
) -> tuple[bool, str]: ...
# devuelve (accepted, reason)

# src/inference/model_versions.py
def list_available_versions(weights_dir: Path) -> list[dict]: ...
def get_current_version(weights_dir: Path) -> str: ...
def activate_version(weights_dir: Path, version: str) -> None: ...
```

## Riesgos conocidos y mitigaciones

| Riesgo | Probabilidad | Mitigación |
|---|---|---|
| Catastrophic forgetting | Alta sin replay | Replay 1:3 + detector + rollback automático |
| Sobreajuste a correcciones | Media | Threshold mínimo de correcciones antes de fine-tune (30+); regularización Adam default |
| Pocos datos en archive al principio | Alta | UMBRAL mínimo + log informativo "necesitas N correcciones más" |
| Drift entre versiones (v2 mejor que v1 pero peor que el modelo "ideal" si tuviéramos más datos) | Media | Aceptar versión solo si **mejora**; documentar trayectoria en `weights/F4/*/meta.json` |
| Corrupted archive (job sin features pero con corrections) | Baja | El dataset loader debe filtrar jobs incompletos, log warning |
| §5.9 no accesible (SSD TIME desconectado) | Media | Dump pre-generado en `pilot/data_remote/section_5_9.npz` |
| Pesos del head F4-v1 cargados mal | Baja | Test de smoke: reproducir `accuracy §5.9` antes de empezar a fine-tunear |

## Plan de implementación (cuando se aborde)

Estimación realista: **6-10 horas en una sesión dedicada** (no las
asíncronas como hoy).

1. **Pre-trabajo (1h)**: confirmar SSD TIME conectado; generar
   `section_5_9.npz` dump si no existe; verificar que se pueden cargar
   los pesos `F4-v1` desde `weights/F4/v1/model.pth`.
2. **`replay_buffer.py` + `dataset.py` (1.5h)**: leer archive + §5.9,
   mezclar 1:3, devolver `Dataset` + `DataLoader`.
3. **`finetune.py` (2h)**: cargar pesos, congelar `Linear(4096→512)`,
   loop de entrenamiento con Focal Loss + Adam.
4. **`validation.py` (1h)**: cálculo de métricas + detector de
   aceptación.
5. **`model_versions.py` (0.5h)**: discovery + symlink.
6. **`scripts/finetune_head.py` (0.5h)**: CLI wrapper.
7. **Validación end-to-end (1.5h)**: smoke test con 30 correcciones
   sintéticas (alumno actuando como patólogo) + verificar que el
   fine-tune se ejecuta y la versión se acepta/rechaza correctamente.
8. **Doc + commits (0.5h)**.

## Decisiones que requieren confirmación antes de implementar

- [ ] **Opción A (solo última capa) vs Opción B (head completo)**.
- [ ] **Umbral mínimo de correcciones** para disparar fine-tune (30
      sugerido).
- [ ] **Threshold de aceptación**: ¿1pp accuracy o más estricto?
- [ ] **Generación del `section_5_9.npz` dump** dentro o fuera del repo
      del piloto.
- [ ] **Lr y epochs** finales (los del doc son punto de partida
      sensato pero requieren ajuste experimental).

## Hitos siguientes (referencia)

- **Hito 3**: fine-tune AttnMIL slide-level sobre los nuevos
  embeddings. Mismo patrón replay buffer + detector pero a nivel slide.
- **Hito 4**: active learning + mina de oro §5.9. El piloto sugiere al
  patólogo qué slides revisar primero según incertidumbre del modelo.
- **Hito 5**: release versionado + UI. El patólogo ve qué versión está
  en uso, qué cambió, y puede rollback manual.
