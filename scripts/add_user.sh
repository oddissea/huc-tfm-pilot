#!/usr/bin/env bash
#
# Añade o actualiza un usuario en nginx/.htpasswd.
#
# Genera una contraseña aleatoria robusta y la imprime UNA SOLA VEZ.
# Apuntala el hash bcrypt en el htpasswd.
#
# Uso:
#   ./scripts/add_user.sh <username>
#
# Tras ejecutar:
#   - Comparte la contraseña con el usuario por canal seguro (Signal, Telegram, etc.).
#   - Si nginx ya está corriendo: docker compose exec nginx nginx -s reload
#
# Eliminar un usuario:
#   sed -i '/^username:/d' nginx/.htpasswd
#   docker compose exec nginx nginx -s reload

set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Uso: $0 <username>" >&2
    exit 1
fi

USER="$1"
HTPASSWD_FILE="./nginx/.htpasswd"

if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: docker no está instalado o no está en el PATH" >&2
    exit 1
fi

# Generar contraseña aleatoria de 16 caracteres alfanuméricos.
# Nota: usamos Python's `secrets` (criptográficamente seguro) en vez de
# `tr -dc ... < /dev/urandom | head -c N` porque ese pipeline produce
# SIGPIPE en `tr` y, con `set -o pipefail`, el script muere en silencio.
PASSWORD=$(python3 -c "import secrets, string; print(''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(16)))")

# Generar hash bcrypt con httpd:alpine (evita depender de apache2-utils en el host).
HASH=$(docker run --rm httpd:alpine htpasswd -nbB "${USER}" "${PASSWORD}" | tr -d '\r')

mkdir -p "$(dirname "${HTPASSWD_FILE}")"

# Si el usuario ya existe, eliminar la línea anterior antes de añadir la nueva.
if [ -f "${HTPASSWD_FILE}" ] && grep -q "^${USER}:" "${HTPASSWD_FILE}"; then
    grep -v "^${USER}:" "${HTPASSWD_FILE}" > "${HTPASSWD_FILE}.tmp"
    mv "${HTPASSWD_FILE}.tmp" "${HTPASSWD_FILE}"
    echo "(Usuario '${USER}' ya existía, contraseña rotada.)"
fi

echo "${HASH}" >> "${HTPASSWD_FILE}"

cat <<EOF

✅ Usuario '${USER}' añadido a ${HTPASSWD_FILE}.

   Contraseña: ${PASSWORD}

⚠️  Esta contraseña NO se almacena en ningún sitio salvo el hash en htpasswd.
   Cópiala AHORA y compártela con el usuario por canal seguro (Signal/Telegram/etc.).

Para que nginx lea el cambio sin reiniciar:
    docker compose exec nginx nginx -s reload
EOF
