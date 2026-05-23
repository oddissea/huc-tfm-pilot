# Despliegue del piloto en el HUC

Documento operativo. Doble propósito: (a) referencia técnica del
deploy en el HUC PC, (b) agenda de decisiones para la reunión con
Eduardo Salido (2026-05-28).

Última actualización: 2026-05-23 (sesión #65).

## 1. Estado de preparación

| Pieza | Estado |
|---|---|
| Refactor sink GCS → archive local | ✅ commit `5fba4ab` |
| Dockerfile PyTorch 2.7 + CUDA 12.8 (Blackwell) | ✅ commit `ac87a8c` |
| `TTL_HOURS` configurable + fast-path offline pesos | ✅ commit `e87436b` |
| Página `⚙️ Configuración` en sidebar | ✅ commit `b5df2d6` |
| Parche `start_vm.sh` para naming del disco | ✅ commit `c5945f6` |
| Hardware HUC confirmado | ✅ ver §2 |
| Validación end-to-end del software | ⚠️ pendiente VM disponible o HUC |

## 2. Hardware HUC confirmado (factura RCT, 07/07/2025)

| Componente | Spec |
|---|---|
| CPU | Intel Core i9-14900KF (24 cores: 8P + 16E) |
| GPU | NVIDIA GeForce RTX 5070 12 GB GDDR7 (Blackwell SM 12.0) |
| RAM | 128 GB DDR5 6000 MHz (4×32 GB Corsair Fury) |
| SSD | 8 TB NVMe WD Black SN850X PCIe 4 |
| MB / PSU | ASUS ROG Strix B760-F / Corsair RM1000E 1000W |
| SO | Windows 11 Pro |

**Confirmación de driver y CUDA** (Eduardo, 2026-05-23):

- Driver Version: `576.88` (≥ 572 requerido para Blackwell ✓)
- CUDA Version: `12.9` (≥ 12.8 requerido por el Dockerfile ✓)
- Modo WDDM (GPU compartida con escritorio Windows; ~400 MB consumidos
  por el sistema, ~11.8 GB libres para inferencia).
- VRAM total: 12.227 MiB.

**Holgura**: F4 + 5 AttnMIL pican < 4 GB durante inferencia → margen
sobrado en los 11.8 GB disponibles.

## 3. Software preparado en el repo del piloto

### Refactor para "local-only" en HUC

Decidido en sesión #64: en el HUC los datos del paciente NO salen del
hospital, así que el sink remoto GCS se eliminó. Una sola
implementación, idéntica en QA y producción:

```
queue/<job_id>/             → efímero, sometido al TTL del prune
  ├── corrections.jsonl
  ├── features.npy
  ├── meta.json
  └── (otros artefactos: result.json, attention.npy, etc.)

archive/<job_id>/            → persistente, sobrevive al TTL
  ├── corrections.jsonl
  ├── features.npy
  └── meta.json
```

El bind mount Docker `./archive:/var/archive` garantiza que el archive
vive en el filesystem del host, fuera del ciclo de vida del container.

### Configuración runtime editable

La página `⚙️ Configuración` del sidebar permite a Eduardo:

- Ajustar el TTL del prune en días (default 1, recomendado 7 para HUC).
- Ver estado del archive (N jobs, MB, fechas first/last, correcciones).
- Ejecutar "prune ahora" (acción manual; archive primero, borrado
  después).
- Vaciar archive completo (doble confirmación) tras una recogida.

Cambios desde la UI surten efecto en el siguiente ciclo del worker
(máx. 5 min) sin reiniciar el container.

## 4. Decisiones pendientes con Eduardo (agenda reunión)

### 4.1 Ruta Docker en Windows 11

El piloto está hecho para Docker en Linux. En Windows 11 hay dos
opciones realistas:

| Opción | Pros | Contras |
|---|---|---|
| **Docker Desktop + WSL2** | Setup trivial vía installer; GPU passthrough automático | Licencia comercial requerida para FIISC/HUC (~5 USD/mes/usuario) |
| **Docker CE en WSL2 nativo** | Free Software; mismo GPU passthrough | Curva de instalación mayor; sin GUI (todo CLI) |

**Recomendación**: Docker CE en WSL2 nativo. Una sola tarde de setup,
cero coste recurrente, suficiente para uso operacional del piloto
(Eduardo no necesita la GUI de Docker Desktop para el día a día).

**A decidir con Eduardo + IT del HUC**: cuál se prefiere y quién lo
instala.

### 4.2 Política de transferencia de correcciones

El archive vive en el host del HUC. Cada cierto tiempo el alumno (o
Eduardo) lleva su contenido al entorno de reentrenamiento. Tres
opciones:

| Vía | Pros | Contras |
|---|---|---|
| **USB cifrado, presencial** | Más seguro; cero infra | Lento; requiere visita |
| **VPN al HUC + rsync** | Cómodo; remoto | Necesita infra VPN; permisos IT HUC |
| **Eduardo envía `.tar.gz.enc`** | Asíncrono | Le carga trabajo a Eduardo |

**A decidir con Eduardo**: cuál encaja con la operativa del HUC.

### 4.3 Acceso de red al piloto

- ¿La máquina debe ser accesible desde otras máquinas de la LAN, o
  solo desde el propio equipo? Esto define si necesitamos IP fija
  + nginx + acceso HTTPS dentro de LAN.
- Si solo localhost: `http://localhost:8501` y no necesitamos nginx.
- Si LAN: necesitamos certificado interno (Let's Encrypt no aplica sin
  dominio público; el certificado autofirmado da warnings; lo ideal
  sería un cert interno del HUC si IT lo provee).

**A decidir con Eduardo + IT HUC**.

### 4.4 Política de retención del archive

Hoy el archive **no tiene TTL programático**. Crece hasta que alguien
lo limpie. Con 5 slides/día × 250 días/año = ~2.5 GB/año, en 8 TB de
disco la presión es nula.

**Recomendación**: limpieza manual tras cada recogida (botón "Vaciar
archive" de la página Configuración). Si en algún futuro se quiere
automatizar, añadir un cron que detecte archive > N meses y avise por
log (no que borre automáticamente — el patólogo decide).

### 4.5 Frecuencia de recogida de correcciones

¿Cada cuánto recoge el alumno las correcciones para reentrenamiento?

- **Semanal**: tanda chica (5×7 = 35 correcciones). Reentrenar tan
  poco frecuentemente no aporta mucho si el modelo no cambia.
- **Mensual**: tanda mediana (~100-150 correcciones).
- **Trimestral**: tanda grande (~400 correcciones), suficiente para
  un fine-tune significativo del head F4 (Hito 2).

**Recomendación**: empezar mensual los primeros meses para
calibrar; pasar a trimestral cuando el flujo esté estable.

### 4.6 Decisiones del frente MedIA (publicación)

- **Corresponding author**: ¿Eduardo (ULL, CRUE-Elsevier APC=0) o
  cerrado/suscripción? El alumno (UNED alumno máster) está en zona
  gris según lección no-obvia 4.
- **OA vs cerrado**: reabierto desde la salida de Luis. Eduardo (ULL)
  mantiene Gold OA; cerrado lo descarta a cambio de paywall + arXiv
  preprint.
- **Número CEIm**: necesario para el Ethics statement del paper. Es
  el `Comité de Ética de la Investigación con Medicamentos del HUC`
  que aprobó el uso retrospectivo de las 76+91 slides. Eduardo
  facilita número de referencia + fecha de aprobación.

## 5. Checklist primer arranque en HUC

Asumiendo decisiones cerradas en la reunión.

### 5.1 Pre-arranque (alumno antes de ir al HUC)

- [ ] Pre-cargar pesos del modelo en USB cifrado siguiendo
      `PRE_LOAD_WEIGHTS.md`.
- [ ] Repo del piloto clonado en USB también (por si el HUC no tiene
      acceso GitHub directo).
- [ ] Verificar versión de Docker que se va a instalar (CE WSL2 o
      Docker Desktop), tener el installer descargado offline.

### 5.2 En el HUC, primer arranque

- [ ] Instalar Docker (según decisión §4.1).
- [ ] Clonar repo desde USB o `git clone https://github.com/oddissea/huc-tfm-pilot.git`.
- [ ] Copiar pesos pre-cargados a `pilot/weights/` siguiendo
      `PRE_LOAD_WEIGHTS.md` paso 4-5.
- [ ] Verificar checksums: `cd pilot/weights && sha256sum -c SHA256SUMS.txt`.
- [ ] Configurar TTL inicial editando docker-compose.yml o vía la
      página Configuración tras el primer arranque (recomendado 7 días).
- [ ] `cd pilot && docker compose up -d`.
- [ ] Esperar ~15s, verificar logs: `docker compose logs app --tail=30`.
- [ ] Abrir navegador en `http://localhost:8501`.
- [ ] Sidebar → ⚙️ Configuración: ajustar TTL a 7 días.
- [ ] Sidebar → "Cargar modelos" (debe ser instantáneo si pre-carga
      OK; nada de descarga GCS).
- [ ] Subir un slide de prueba (TIFF pequeño), confirmar que la
      inferencia funciona y el visor OpenSeadragon carga.

### 5.3 Validación end-to-end (con Eduardo presente)

- [ ] Eduardo sube un slide real.
- [ ] Eduardo hace al menos una corrección (multi-select o lasso).
- [ ] Esperar ~5 min y verificar en sidebar → ⚙️ Configuración que el
      archive muestra "1 job con N correcciones".

## 6. Plan de contingencia

### 6.1 Si la GPU no es reconocida por el container

- Verificar `nvidia-smi` desde dentro del container:
  `docker compose exec app nvidia-smi`.
- Si falla: NVIDIA Container Toolkit no está bien configurado en
  WSL2. Reinstalar siguiendo guía oficial NVIDIA.
- Verificar `docker info | grep -i nvidia`.

### 6.2 Si la imagen Docker falla al construir

- Posible: incompatibilidad de alguna dep con PyTorch 2.7. Revisar
  `pip install` log. Soluciones puntuales:
  - Bump versión específica (timm ya está en 1.0.22, debería ir bien).
  - Si pyvips falla en wheel, instalar `libvips-dev` del host (poco
    probable, ya está en el Dockerfile).

### 6.3 Si el piloto no encuentra los pesos al arrancar

- Verificar bind mount: `docker compose exec app ls -la /app/weights/`.
- Si vacío o estructura incorrecta: revisar `PRE_LOAD_WEIGHTS.md`
  paso 1 (estructura esperada).
- Si correcto y aún así intenta GCS: verificar logs por error
  intermedio.

### 6.4 Si tras N horas de uso aparecen errores de prune

- Revisar página ⚙️ Configuración → Estado del archive.
- Si hay `archive_errors > 0` en logs (`docker compose logs app | grep
  archive`): typically permisos del bind mount.
- Verificar permisos del host: `ls -la archive/` y que el UID del
  container puede escribir.

## 7. Limitaciones conocidas / TODOs pendientes

- **Validación end-to-end del refactor no se ha podido hacer en VM L4
  por stockout persistente**. Primera validación real ocurrirá en HUC.
  Si algo falla, hay que diagnosticar in-situ.
- **Hito 2 (fine-tune head F4 con replay buffer)** diseñado en
  `pilot/docs/learning/HITO_2_FINE_TUNE_F4.md` pero NO implementado.
  El piloto en HUC solo hace inferencia con `F4-v1`; las correcciones
  se acumulan en el archive hasta que se reentrene en una sesión
  posterior.
- **Sin export del archive como `.zip` descargable** desde la UI;
  Eduardo o el alumno deben copiar la carpeta manualmente. Marcado
  como "cositas próximas" en la página Configuración.
- **Toggle modo debug** y **versión modelo activa** mencionados en
  Configuración como futuras opciones; sin implementar.
- **Disco de la VM QA (`huc-tfm-pilot-vm-c`) con sufijo de zona**
  legacy. El script `start_vm.sh` está parcheado para soportarlo, pero
  el "renombrado" a `huc-tfm-pilot-vm` (sin sufijo) queda para una
  ventana de capacidad L4 disponible. No afecta a HUC, solo a QA.

## 8. Referencias internas

- `docs/deployment/ARCHIVE_CORRECCIONES.md` — arquitectura del archive
  local.
- `docs/deployment/PRE_LOAD_WEIGHTS.md` — procedimiento de pre-carga
  de pesos por USB.
- `docs/learning/HITO_2_FINE_TUNE_F4.md` — plan operativo Hito 2.
- `docs/HUC-PC.pdf` — factura del equipo (untracked, decisión
  pendiente de versionar).
