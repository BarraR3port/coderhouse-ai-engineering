# Convenciones de la API pública de Nimbus

## Versionado

La API pública se versiona en la URL: `https://api.nimbus.example/v2/...`. Una versión mayor se mantiene al menos 12 meses después de publicada la siguiente. Los cambios incompatibles (eliminar un campo, cambiar un tipo, volver obligatorio un parámetro) solo se introducen en una versión mayor nueva; agregar campos opcionales o endpoints nuevos se puede hacer dentro de la misma versión.

Cuando un endpoint queda obsoleto, las respuestas incluyen el encabezado `Deprecation` y el encabezado `Sunset` con la fecha en que dejará de funcionar.

## Autenticación

Los clientes se autentican con tokens OAuth 2.0 obtenidos por el flujo de credenciales de cliente (client credentials). Los tokens de acceso duran 1 hora y no se pueden renovar: al vencer, el cliente pide uno nuevo. Las claves de API estáticas se discontinuaron en la versión v2.

## Límites de uso (rate limiting)

Cada cliente puede hacer hasta 600 solicitudes por minuto en el plan estándar y 3000 en el plan empresarial. Al superar el límite, la API responde `429 Too Many Requests` con el encabezado `Retry-After`, que indica cuántos segundos esperar. Los clientes deben reintentar con backoff exponencial y no antes de lo que indica `Retry-After`.

## Paginación y errores

Los listados se paginan por cursor: la respuesta trae `next_cursor` y el cliente lo envía en el parámetro `cursor` para obtener la página siguiente. El tamaño de página por defecto es 50 elementos y el máximo es 200.

Los errores siguen el formato RFC 9457 (Problem Details): un objeto JSON con `type`, `title`, `status`, `detail` y, cuando aplica, `errors` con el detalle por campo. Los errores 5xx incluyen un `request_id` que el cliente debe enviar a soporte para que se pueda rastrear la solicitud.
