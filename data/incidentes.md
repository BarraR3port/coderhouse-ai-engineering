# Gestión de incidentes en Nimbus

## Niveles de severidad

- **SEV1**: la plataforma está caída para todos los clientes o hay pérdida o exposición de datos. Tiempo de respuesta: 5 minutos, las 24 horas.
- **SEV2**: una funcionalidad crítica (pagos, inicio de sesión o la API pública) está degradada para una parte de los clientes. Tiempo de respuesta: 15 minutos, las 24 horas.
- **SEV3**: una funcionalidad secundaria falla o hay un problema con un rodeo conocido. Se atiende en horario laboral, dentro del día hábil siguiente.

## Guardias

Cada equipo de producto tiene una guardia semanal que rota los lunes a las 10:00. La persona de guardia recibe las alertas por PagerDuty y tiene que confirmar la alerta (ack) dentro del tiempo de respuesta de la severidad. Si no la confirma, la alerta escala automáticamente al líder de guardia y, 10 minutos después, al gerente de ingeniería.

La guardia se compensa con un día libre por cada semana completa de guardia, a tomar dentro de los 30 días siguientes.

## Durante un incidente

Para un SEV1 o SEV2 se abre un canal de Slack con el formato `#inc-AAAAMMDD-descripcion` y se designa un comandante de incidente, que coordina y decide; no es necesariamente quien arregla el problema. El comandante publica una actualización de estado cada 30 minutos en el canal y, si hay clientes afectados, en la página de estado pública.

La prioridad es mitigar, no encontrar la causa raíz: si un despliegue reciente coincide con el inicio del problema, primero se hace rollback y después se investiga.

## Post-mortem

Todo SEV1 y SEV2 requiere un post-mortem escrito dentro de los 5 días hábiles posteriores al cierre del incidente. El documento es sin culpables (blameless): describe la línea de tiempo, el impacto, la causa raíz y las acciones correctivas, cada una con un responsable y una fecha. Las acciones correctivas de un SEV1 tienen prioridad sobre cualquier trabajo de producto planificado.
