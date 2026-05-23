# QA del piloto — qué probar esta vez

Eduardo, en preparación para la reunión del jueves 28, te pido un
pequeño QA de la app para cazar problemas antes de meternos a hablar
de despliegue. Te enumero por orden lo que hay nuevo desde la última
demo y qué me interesa que pruebes y reportes.

## Acceso

- URL: <https://huc-tfm-pilot.oddissea.com>
- Usuario y contraseña: los que ya tienes (las que te pasé en su día,
  no las cambio).
- Si el navegador queda colgado tras el login, recarga (`Cmd+R` /
  `Ctrl+R`). El primer arranque de la VM tarda un poco en estabilizar.

## Lo nuevo desde la última demo

1. **Nombre nuevo**: ahora se llama **DualPath CRC — by Lumen
   Network** (antes "HUC TFM Pilot"). Solo es marca; la funcionalidad
   sigue siendo la misma del piloto que ya conoces.
2. **Página `⚙️ Configuración` en el sidebar** (es la novedad
   principal a probar). Te permite administrar la app sin que tengas
   que llamarme: ajustar cuánto tiempo se guardan los slides
   procesados, ver cuántas correcciones se han archivado, borrar la
   cola si quieres, etc.
3. **Almacenamiento local seguro**: tus correcciones y los embeddings
   del modelo se guardan en una carpeta del propio servidor, **sin
   salir a la nube de Google**. Esto es lo que permitirá llevarte el
   piloto al ordenador del HUC sin GCS de por medio.

## Cosas que me interesa que pruebes

### (A) Página de Configuración

1. En el sidebar (panel izquierdo) entra en **⚙️ Configuración**.
2. **Sección "Retención de jobs"**:
   - Verifica que aparece "TTL actual" con un número.
   - Cambia el valor a, por ejemplo, **7 días**.
   - Pulsa **"Guardar nuevo TTL"**.
   - Recarga la página entera con `Cmd+R` / `Ctrl+R`. El valor
     guardado debería seguir ahí.
3. **Sección "Estado del archive"**: si todavía no hay correcciones
   acumuladas, verás "Archive vacío". Es correcto.
4. **Sección "Acciones"**: prueba a marcar la caja de selección de
   "Ejecutar prune ahora" (el botón se desbloquea). Si hay 0 jobs en
   DONE no pasará nada al pulsarlo; es esperado.

### (B) Flujo normal (lo que ya conoces)

5. Vuelve a la página principal (clic en el nombre **DualPath CRC**
   del sidebar).
6. Pulsa **"Cargar modelos"** (sidebar). En ~25 segundos debería
   aparecer "Modelos cargados ✓".
7. **Sube un slide TIFF** de los que sueles usar para QA. Verifica:
   - Que la conversión + inferencia terminan sin error.
   - Que aparece la matriz de confusión (si tiene GT) y el visor
     OpenSeadragon.
   - Que puedes hacer al menos una **corrección** (click en parche,
     o `Cmd+click` para acumular, o **Shift+drag** para lasso).
   - Que la corrección se registra (debería aparecer en la lista de
     parches corregidos).

### (C) Ver que la corrección llegó al archive

8. Tras hacer la corrección, espera **5 minutos** (el worker hace
   limpieza cada 5 min). Alternativa rápida: ve a **⚙️
   Configuración** y pulsa **"Ejecutar prune ahora"** tras marcar
   la confirmación.
9. Vuelve a **⚙️ Configuración** y verifica que el "Estado del
   archive" ahora muestra:
   - "Jobs archivados: 1" (o más).
   - "Correcciones totales": las que hayas hecho.
   - "Tamaño en disco": algo distinto de 0 MB.
   - Fechas de "Último archivado" / "Más antiguo": deberían ser de
     hoy.

## Qué reportar si algo va mal

Para cualquier problema, lo más útil para mí es:

1. **Captura de pantalla** del error (idealmente la pantalla entera
   con la URL visible, no solo el mensaje).
2. **El paso exacto que estabas haciendo** cuando apareció
   (¿qué pulsaste, qué subiste, en qué sección?).
3. **El nombre del slide** que estabas procesando (si aplica).

Si la app entera deja de responder o ves un error genérico tipo
"Internal server error", avísame para que reinicie el servidor.

## Sin presión

Esto es QA, no hay datos clínicos reales en juego. Si pruebas con un
slide TIFF de los habituales, la corrección se queda en el archive
del servidor de pruebas (Google Cloud) y se borra en los plazos que
configuremos. No se publica nada, no se manda a nadie. Es solo para
que cacemos bugs antes de la reunión.

Cualquier cosa, me dices.
