#!/usr/bin/env bash
#
# Crea los 5 usuarios iniciales del piloto y guarda sus credenciales en
# `.credentials.local` (gitignored).
#
# Uso:
#   ./scripts/bootstrap_users.sh
#
# Cada llamada a add_user.sh genera una contraseña aleatoria nueva, así que
# si re-ejecutas este script las contraseñas existentes se rotan. Úsalo solo
# en el setup inicial salvo que quieras rotar todo de golpe.
#
# Comparte cada par usuario:contraseña por canal seguro y BORRA el fichero
# `.credentials.local` cuando ya hayas distribuido todas (queda solo el hash
# bcrypt en `nginx/.htpasswd`, irrecuperable).

set -euo pipefail

USERS=(eduardo nasser andres luis carlos)
OUTPUT_FILE=".credentials.local"

if [ -f "${OUTPUT_FILE}" ]; then
    read -r -p "Ya existe ${OUTPUT_FILE}. ¿Sobrescribir? (y/N) " decision
    if [ "${decision}" != "y" ] && [ "${decision}" != "Y" ]; then
        echo "Abortado por el usuario."
        exit 1
    fi
fi

# Cabecera con timestamp.
{
    echo "# DualPath CRC — by Lumen Network — credenciales BasicAuth iniciales"
    echo "# Generadas: $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "# Formato: usuario:contraseña"
    echo "#"
    echo "# Comparte cada línea con el usuario correspondiente por canal seguro"
    echo "# (Signal, Telegram, gestor de contraseñas) y borra este fichero después."
    echo ""
} > "${OUTPUT_FILE}"

for user in "${USERS[@]}"; do
    # Capturar la salida del add_user.sh para extraer la contraseña impresa.
    output=$("$(dirname "$0")/add_user.sh" "${user}")
    password=$(echo "${output}" | grep -E '^\s+Contraseña:' | sed -E 's/^\s+Contraseña:\s*//')

    if [ -z "${password}" ]; then
        echo "ERROR: no se pudo extraer la contraseña de '${user}'." >&2
        echo "Salida cruda de add_user.sh:" >&2
        echo "${output}" >&2
        exit 1
    fi

    echo "${user}:${password}" >> "${OUTPUT_FILE}"
    echo "✓ ${user}"
done

cat <<EOF

✅ Los 5 usuarios creados y registrados en nginx/.htpasswd.

Credenciales guardadas en ${OUTPUT_FILE} (gitignored).

Pasos siguientes:
 1. Comparte cada línea de ${OUTPUT_FILE} con el usuario correspondiente por canal seguro.
 2. Cuando hayas distribuido todas, borra ${OUTPUT_FILE}:
       shred -u ${OUTPUT_FILE}     # elimina sobrescribiendo el contenido
    o al menos:
       rm ${OUTPUT_FILE}
 3. Si nginx ya está corriendo, recarga para que lea el nuevo htpasswd:
       docker compose exec nginx nginx -s reload
EOF
