#!/usr/bin/env bash
#
# Rota la contraseña del usuario 'guest' en nginx/.htpasswd.
#
# Útil para sesiones puntuales (demos en congresos, accesos a colaboradores
# que no quieres dar de alta nominalmente). Tras ejecutar, comparte la nueva
# contraseña con quien necesite acceso temporal y rota cuando quieras
# revocar.
#
# Uso:
#   ./scripts/rotate_guest.sh

set -euo pipefail

# Reutilizamos add_user.sh: hace exactamente lo mismo (rota si existe).
exec "$(dirname "$0")/add_user.sh" guest
