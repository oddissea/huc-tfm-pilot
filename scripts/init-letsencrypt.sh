#!/usr/bin/env bash
#
# Inicializa los certificados de Let's Encrypt para huc-tfm-pilot.oddissea.com.
#
# Adaptado del patrón estándar nginx + certbot (https://github.com/wmnnd/nginx-certbot).
# Ejecutar UNA VEZ tras montar la VM y antes de arrancar el stack normal.
#
# Lo que hace:
#  1. Descarga las opciones SSL recomendadas (options-ssl-nginx.conf y ssl-dhparams.pem).
#  2. Crea un certificado dummy temporal para que nginx pueda arrancar sin error.
#  3. Arranca solo el container nginx (con el cert dummy).
#  4. Borra el cert dummy.
#  5. Lanza certbot con webroot challenge para obtener el cert real.
#  6. Recarga nginx y arranca el resto del stack.
#
# Uso: ./scripts/init-letsencrypt.sh
#
# Variables:
#  STAGING=1  →  usa el endpoint de staging de Let's Encrypt (no rate-limited; útil para pruebas).
#                Recomendado en la primera ejecución para confirmar que todo funciona, luego a producción.
#  STAGING=0  →  endpoint de producción (rate limit: 5 certs/semana por dominio + variantes).

set -euo pipefail

DOMAIN="huc-tfm-pilot.oddissea.com"
EMAIL="oddissea@gmail.com"
RSA_KEY_SIZE=4096
DATA_PATH="./certbot"
STAGING="${STAGING:-0}"

if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: docker no está instalado o no está en el PATH" >&2
    exit 1
fi

if [ -d "${DATA_PATH}/conf/live/${DOMAIN}" ]; then
    read -r -p "Ya existen certificados para ${DOMAIN}. ¿Sobrescribir? (y/N) " decision
    if [ "${decision}" != "y" ] && [ "${decision}" != "Y" ]; then
        echo "Abortado por el usuario."
        exit 1
    fi
fi

# 1. Descargar opciones SSL recomendadas si no existen.
if [ ! -e "${DATA_PATH}/conf/options-ssl-nginx.conf" ] || [ ! -e "${DATA_PATH}/conf/ssl-dhparams.pem" ]; then
    echo "### Descargando opciones SSL recomendadas..."
    mkdir -p "${DATA_PATH}/conf"
    curl -sS "https://raw.githubusercontent.com/certbot/certbot/master/certbot-nginx/src/certbot_nginx/_internal/tls_configs/options-ssl-nginx.conf" \
        > "${DATA_PATH}/conf/options-ssl-nginx.conf"
    curl -sS "https://raw.githubusercontent.com/certbot/certbot/master/certbot/certbot/ssl-dhparams.pem" \
        > "${DATA_PATH}/conf/ssl-dhparams.pem"
fi

# 2. Crear certificado dummy temporal.
echo "### Creando certificado dummy para ${DOMAIN}..."
PATH_LIVE="/etc/letsencrypt/live/${DOMAIN}"
mkdir -p "${DATA_PATH}/conf/live/${DOMAIN}"
docker compose run --rm --entrypoint "\
    openssl req -x509 -nodes -newkey rsa:${RSA_KEY_SIZE} -days 1 \
        -keyout '${PATH_LIVE}/privkey.pem' \
        -out '${PATH_LIVE}/fullchain.pem' \
        -subj '/CN=localhost'" certbot

# 3. Arrancar nginx con el cert dummy.
echo "### Arrancando nginx con certificado dummy..."
docker compose up -d nginx

# 4. Borrar cert dummy (certbot necesita tener el directorio limpio).
echo "### Borrando certificado dummy..."
docker compose run --rm --entrypoint "\
    rm -Rf /etc/letsencrypt/live/${DOMAIN} && \
    rm -Rf /etc/letsencrypt/archive/${DOMAIN} && \
    rm -Rf /etc/letsencrypt/renewal/${DOMAIN}.conf" certbot

# 5. Pedir cert real a Let's Encrypt.
echo "### Solicitando certificado real a Let's Encrypt..."
STAGING_ARG=""
if [ "${STAGING}" -ne 0 ]; then
    STAGING_ARG="--staging"
fi

docker compose run --rm --entrypoint "\
    certbot certonly --webroot -w /var/www/certbot \
        ${STAGING_ARG} \
        --email ${EMAIL} \
        -d ${DOMAIN} \
        --rsa-key-size ${RSA_KEY_SIZE} \
        --agree-tos \
        --force-renewal \
        --non-interactive" certbot

# 6. Recargar nginx para que lea el cert real.
echo "### Recargando nginx con el cert real..."
docker compose exec nginx nginx -s reload

echo ""
echo "### ✅ Listo. Verificar:"
echo "    curl -I https://${DOMAIN}"
echo ""
echo "Si STAGING=1, el cert es de staging (válido pero no confiable). Re-ejecutar con STAGING=0 para producción."
