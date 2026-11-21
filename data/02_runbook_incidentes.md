# Runbook de gestión de incidentes

> Documento interno de Nimbus Cloud S.A. (empresa ficticia usada como dataset de ejemplo).
> Última revisión: febrero de 2026. Responsable: equipo de SRE.

## 1. Niveles de severidad

| Severidad | Definición | Tiempo de respuesta | Ejemplo |
|---|---|---|---|
| SEV1 | Caída total del servicio o pérdida de datos de clientes | 5 minutos | `ledger-api` responde 5xx a más del 50% de las solicitudes |
| SEV2 | Degradación grave que afecta a muchos clientes | 15 minutos | La autoridad tributaria de un país rechaza todos los documentos por un error nuestro |
| SEV3 | Degradación parcial con alternativa disponible | 4 horas hábiles | Los webhooks llegan con más de 10 minutos de retraso |
| SEV4 | Problema menor o cosmético | 3 días hábiles | Un texto mal traducido en el portal de clientes |

Ante la duda entre dos severidades, se declara siempre la más alta. Bajar la severidad más tarde no tiene costo; subestimar un incidente sí.

## 2. Roles durante un incidente

- **Incident Commander (IC):** coordina la respuesta, decide y delega. No escribe código ni ejecuta comandos durante el incidente. En SEV1 y SEV2 el IC es la persona de guardia de SRE hasta que asigne el rol a otra persona.
- **Líder técnico:** investiga la causa y ejecuta la mitigación. Suele ser alguien del equipo dueño del servicio afectado.
- **Responsable de comunicaciones:** actualiza la página de estado (`status.nimbus.example`) y el canal de Slack `#incidentes-clientes`. En SEV1 debe publicar una actualización cada 30 minutos como máximo; en SEV2, cada 60 minutos.
- **Escriba:** registra la línea de tiempo en el documento del incidente. Es obligatorio en SEV1 y opcional en SEV2.

## 3. Guardia (on-call)

La guardia de SRE rota semanalmente, de lunes a las 10:00 al lunes siguiente a las 10:00, hora de Santiago. Cada rotación tiene una persona primaria y una secundaria. Si la primaria no reconoce la alerta en PagerDuty en 10 minutos, la alerta escala automáticamente a la secundaria, y a los 20 minutos al Engineering Manager de turno.

Compensación: cada semana de guardia primaria se compensa con 1,5 días libres, y cada incidente atendido fuera del horario laboral suma 2 horas libres adicionales. Los días libres deben tomarse dentro de los 60 días siguientes.

Durante la guardia es obligatorio:

1. Tener el notebook corporativo y acceso a internet a menos de 15 minutos de distancia.
2. No consumir alcohol ni sustancias que afecten el juicio.
3. Mantener la batería del teléfono por encima del 30% y PagerDuty con notificaciones críticas activadas.

## 4. Procedimiento de respuesta

### Paso 1: Reconocer y declarar

Reconoce la alerta en PagerDuty. Si el impacto se confirma, declara el incidente con el comando de Slack `/incident declare` en el canal `#incidentes`. El bot crea un canal dedicado con el formato `#inc-AAAAMMDD-descripcion-corta` y un documento de seguimiento en Confluence.

### Paso 2: Evaluar el impacto

Revisa el tablero "Ledger — Golden Signals". Responde tres preguntas: ¿qué porcentaje de solicitudes falla?, ¿qué países o planes están afectados?, ¿desde cuándo? Con esas respuestas, confirma o ajusta la severidad.

### Paso 3: Mitigar antes de investigar

La prioridad es restablecer el servicio, no encontrar la causa raíz. Las mitigaciones preaprobadas, que cualquier persona de guardia puede aplicar sin pedir autorización, son:

- **Rollback del último despliegue** con `nimbusctl rollback <servicio>`. Es la primera opción si hubo un despliegue en las últimas 2 horas.
- **Escalar réplicas** de un servicio hasta el doble de su máximo configurado.
- **Activar el modo degradado** de `ledger-api` con el feature flag `ledger.degraded_mode`, que acepta documentos en estado `QUEUED` sin esperar a la autoridad tributaria.
- **Pausar consumidores de Kafka** de un servicio que esté corrompiendo datos.

Cualquier otra acción en producción durante un SEV1 o SEV2 requiere la aprobación explícita del IC en el canal del incidente.

### Paso 4: Failover a la región de respaldo

El failover a `us-east-1` solo lo puede autorizar el IC junto con el CTO o, en su ausencia, el VP de Ingeniería. Se considera cuando la región principal lleva más de 30 minutos sin servicio y AWS no ha dado una estimación de recuperación. El procedimiento está automatizado con el workflow `region-failover` de GitHub Actions y tarda unos 40 minutos.

### Paso 5: Cerrar el incidente

El incidente se cierra cuando las métricas vuelven a la normalidad durante al menos 30 minutos seguidos. El IC anuncia el cierre en el canal del incidente y el responsable de comunicaciones actualiza la página de estado.

## 5. Postmortem

Todo SEV1 y SEV2 requiere un postmortem escrito en un plazo máximo de 5 días hábiles. Los SEV3 requieren postmortem solo si se repiten más de dos veces en un mes.

El postmortem es *blameless*: describe fallas de sistemas y procesos, nunca culpa a personas. Debe incluir:

1. Resumen del impacto (duración, clientes afectados, documentos perdidos o retrasados).
2. Línea de tiempo con horas en UTC.
3. Causa raíz, usando la técnica de los "5 porqués".
4. Qué funcionó bien y qué no.
5. Acciones correctivas, cada una con dueño y fecha, registradas como tickets en Jira con la etiqueta `postmortem`.

La revisión del postmortem se hace en la reunión semanal de confiabilidad, los jueves a las 15:00.

## 6. Incidentes frecuentes y su solución

### Agotamiento de conexiones de PgBouncer

**Síntoma:** errores `no more connections allowed` en `ledger-api` y latencias superiores a 5 segundos.
**Mitigación:** reiniciar los pods de `billing-worker`, que suelen dejar transacciones abiertas durante el cobro masivo del día 1 de cada mes. Si no basta, aumentar temporalmente `default_pool_size` de PgBouncer de 40 a 60.

### Lag alto en Kafka

**Síntoma:** el tablero "Kafka — Lag de consumidores" muestra más de 50.000 mensajes pendientes en el grupo `tax-gateway-consumers`.
**Mitigación:** escalar `tax-gateway` a 24 réplicas (el máximo normal es 12). El tópico tiene 24 particiones, así que más réplicas no aportan.

### Certificado digital vencido

**Síntoma:** la autoridad tributaria rechaza los documentos de un cliente con el error `CERT_EXPIRED`.
**Mitigación:** no es un incidente de plataforma. Se deriva al equipo de Soporte, que contacta al cliente para que suba un certificado vigente desde el portal.
