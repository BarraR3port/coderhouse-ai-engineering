# Política de despliegues de la plataforma Nimbus

## Ventanas de despliegue

Los despliegues a producción se hacen de lunes a jueves entre las 10:00 y las 16:00 (hora de Buenos Aires). Los viernes, fines de semana y feriados solo se despliegan correcciones de incidentes de severidad SEV1 o SEV2, con aprobación del líder de guardia.

Durante la semana de cierre contable de cada mes (los últimos tres días hábiles) rige un congelamiento: no se despliegan cambios en los servicios de facturación ni en la base de datos de pagos.

## Flujo de despliegue

Todo cambio llega a producción a través de un pull request aprobado por al menos dos revisores. El pipeline de integración continua corre los tests, el análisis estático y el escaneo de dependencias; si alguno falla, el merge queda bloqueado.

Después del merge, el cambio se despliega primero en el entorno de staging, donde corren las pruebas de humo durante 15 minutos. Si pasan, el despliegue a producción es canario: el 5 % del tráfico va a la versión nueva durante 30 minutos, luego el 25 % durante otros 30 minutos y finalmente el 100 %.

## Criterios de rollback

El rollback es automático si durante el canario la tasa de errores 5xx supera el 1 % o la latencia p99 aumenta más de un 20 % respecto de la versión anterior. También se puede pedir un rollback manual con el comando `nimbus deploy rollback <servicio>`, que vuelve a la última versión estable en menos de 5 minutos.

Las migraciones de base de datos deben ser compatibles hacia atrás: primero se despliega el código que soporta el esquema viejo y el nuevo, después la migración y recién en un tercer despliegue se elimina el soporte del esquema viejo.

## Feature flags

Las funcionalidades nuevas que cambian el comportamiento para el usuario se publican detrás de un feature flag. Un flag no puede quedar activo más de 90 días: pasado ese plazo, el equipo dueño debe eliminarlo del código o justificar su permanencia en la revisión trimestral de arquitectura.
