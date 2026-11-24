# Observabilidad en Nimbus

## Logs

Los servicios escriben logs estructurados en JSON a la salida estándar; nunca a archivos locales. Cada línea incluye `timestamp`, `level`, `service`, `trace_id` y `message`. Está prohibido loguear datos personales, tokens o números de tarjeta: el pipeline de logs los enmascara, pero el enmascarado es una red de seguridad, no la primera defensa.

Los logs se retienen 30 días en el almacenamiento caliente, donde se pueden buscar, y 1 año en el almacenamiento frío, donde solo se pueden restaurar a pedido.

## Métricas y SLO

Cada servicio expone métricas en formato Prometheus en el endpoint `/metrics`. Las métricas mínimas son las cuatro señales doradas: latencia, tráfico, errores y saturación.

El SLO de disponibilidad de la API pública es 99,9 % mensual, lo que equivale a unos 43 minutos de indisponibilidad por mes. El SLO de latencia es que el 95 % de las solicitudes respondan en menos de 300 milisegundos. Cuando un servicio consume más del 50 % de su presupuesto de error en una semana, el equipo congela las funcionalidades nuevas y prioriza la confiabilidad hasta recuperarlo.

## Trazas

Las trazas distribuidas se instrumentan con OpenTelemetry y se muestrean al 10 % en producción. Las solicitudes que terminan con error se conservan siempre, sin muestreo, para poder investigarlas.

## Alertas

Las alertas se definen sobre síntomas que percibe el usuario (errores, latencia, consumo del presupuesto de error), no sobre causas como el uso de CPU. Toda alerta que despierta a alguien debe tener un runbook enlazado que explique cómo diagnosticarla y mitigarla. Una alerta que se dispara más de tres veces en una semana sin requerir acción se revisa y se ajusta o se elimina.
