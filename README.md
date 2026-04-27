# HUC TFM Pilot

Demo interactiva del modelo F4 (BiT-M dual-stream + AttnMIL ternario 512-d) sobre WSI colorrectales, desplegada como container Docker con GPU (NVIDIA L4 en GCP, RTX 5070 en el HUC PC tras la defensa).

## Stack

- **App**: Streamlit + PyTorch 2.5 + CUDA 12.4
- **Reverse proxy + HTTPS**: Nginx + Let's Encrypt (Certbot)
- **Auth**: BasicAuth con `htpasswd` por usuario nominal
- **Container runtime**: Docker Compose v2 + NVIDIA Container Toolkit

## Estructura

```
.
├── Dockerfile              # imagen de la app (PyTorch + Streamlit)
├── docker-compose.yml      # orquesta app + nginx + certbot
├── requirements.txt        # deps Python (sobre la imagen base)
├── app.py                  # Streamlit (smoke test por ahora)
├── nginx/
│   └── nginx.conf          # reverse proxy + BasicAuth + SSL
├── certbot/                # data runtime: certs Let's Encrypt + ACME challenges
│   ├── conf/
│   └── www/
└── scripts/
    ├── init-letsencrypt.sh # bootstrap inicial de certificados
    ├── add_user.sh         # añade/actualiza un usuario en .htpasswd
    └── rotate_guest.sh     # rota la contraseña del usuario 'guest'
```

## Despliegue inicial (en la VM GCP)

Tras tener Docker + NVIDIA Container Toolkit instalados (ver bitácora):

```bash
git clone https://github.com/oddissea/huc-tfm-pilot.git ~/huc-tfm-pilot
cd ~/huc-tfm-pilot

# 1. Crear los 5 usuarios BasicAuth iniciales y guardar sus contraseñas en un
#    fichero local gitignored (.credentials.local). Después de distribuirlas
#    manualmente por canal seguro, borra ese fichero.
./scripts/bootstrap_users.sh

# Si solo quieres añadir uno suelto en otro momento:
# ./scripts/add_user.sh nombre_usuario

# 2. Bootstrap inicial de Let's Encrypt (genera el primer certificado).
#    Recomendado primero en STAGING para confirmar que todo va antes de gastar rate limit.
STAGING=1 ./scripts/init-letsencrypt.sh

# 3. Una vez confirmado en staging, repetir en producción:
./scripts/init-letsencrypt.sh

# 4. Arrancar el stack completo (si no está ya corriendo).
docker compose up -d
```

Verificar desde fuera:

```bash
curl -I https://huc-tfm-pilot.oddissea.com
# Espera código HTTP 401 (sin auth) → la BasicAuth está activa.

curl -u eduardo:CONTRASEÑA -I https://huc-tfm-pilot.oddissea.com
# Espera código HTTP 200 (con auth) → todo funciona.
```

Y desde un navegador, abrir `https://huc-tfm-pilot.oddissea.com` (te pedirá usuario y contraseña).

## Operaciones comunes

### Ver logs

```bash
docker compose logs -f         # todos los servicios
docker compose logs -f app     # solo la app Streamlit
docker compose logs -f nginx   # solo nginx
```

### Añadir un usuario nuevo

```bash
./scripts/add_user.sh nombre_usuario
docker compose exec nginx nginx -s reload
```

### Quitar un usuario

```bash
sed -i '/^nombre_usuario:/d' nginx/.htpasswd
docker compose exec nginx nginx -s reload
```

### Rotar la contraseña del invitado

```bash
./scripts/rotate_guest.sh
docker compose exec nginx nginx -s reload
```

### Recargar la app tras un cambio

```bash
git pull
docker compose up -d --build app
```

### Renovación de certificados

El container `certbot` corre en background y reintenta renovar cada 12 h. Nginx se recarga cada 6 h para tomar los certs renovados. No suele requerir intervención manual.

Si quieres forzar una renovación:

```bash
docker compose run --rm certbot certbot renew --force-renewal
docker compose exec nginx nginx -s reload
```

## Estado de fases

- [x] **Fase 1**: infraestructura GCP (VM con L4, IP estática, DNS, firewall, schedule)
- [x] **Fase 2**: container Docker (PyTorch + CUDA + Streamlit smoke test)
- [ ] **Fase 3**: reverse proxy + HTTPS + BasicAuth ← *en curso*
- [ ] **Fase 4**: app real (F4 + AttnMIL ternario + tiff_to_h5)

## Documentación operativa

La bitácora completa con todos los pasos, comandos y decisiones vive en el repo TFM, en `docs/deployment/BITACORA_STREAMLIT_PILOT.md`. Ese fichero está gitignored por contener IDs de billing e IPs.
