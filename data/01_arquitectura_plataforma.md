# Arquitectura de la plataforma Nimbus Cloud

> Documento interno de Nimbus Cloud S.A. (empresa ficticia usada como dataset de ejemplo).
> Última revisión: marzo de 2026. Responsable: equipo de Plataforma.

## 1. Visión general

Nimbus Cloud ofrece "Nimbus Ledger", un servicio de facturación electrónica para pymes de Chile, Perú y Colombia. La plataforma procesa en promedio 1,8 millones de documentos tributarios por día, con picos de 4,2 millones durante los cierres de mes.

La arquitectura sigue un modelo de microservicios desplegados en un clúster de Kubernetes administrado (EKS) en la región `sa-east-1` (São Paulo), con una región de respaldo pasiva en `us-east-1` (Virginia del Norte). La región de respaldo solo se activa ante un desastre declarado por el Incident Commander (ver el runbook de incidentes).

Los servicios principales son:

| Servicio | Lenguaje | Responsabilidad | Equipo dueño |
|---|---|---|---|
| `ledger-api` | Go 1.23 | API pública REST para emitir y consultar documentos | Equipo Core |
| `tax-gateway` | Kotlin | Integración con las autoridades tributarias (SII, SUNAT, DIAN) | Equipo Integraciones |
| `pdf-renderer` | Python 3.12 | Generación de PDF y XML firmados | Equipo Documentos |
| `billing-worker` | Python 3.12 | Cobro de suscripciones a los clientes de Nimbus | Equipo Finanzas Tech |
| `notifier` | Node.js 22 | Envío de correos y webhooks | Equipo Core |
| `admin-portal` | TypeScript (Angular) | Portal interno de soporte | Equipo Soporte Tech |

## 2. Flujo de emisión de un documento

1. El cliente llama a `POST /v2/documents` en `ledger-api`. La petición pasa primero por el API Gateway, que valida la API key y aplica el *rate limit* del plan contratado.
2. `ledger-api` valida el payload contra el esquema del país correspondiente y guarda el documento en estado `PENDING` en la base de datos principal.
3. Se publica un evento `document.created` en el bus de eventos (Apache Kafka, tópico `ledger.documents.v2`).
4. `tax-gateway` consume el evento, firma el documento con el certificado digital del cliente y lo envía a la autoridad tributaria. La respuesta de la autoridad cambia el estado a `ACCEPTED` o `REJECTED`.
5. `pdf-renderer` genera la representación impresa y la sube al bucket `nimbus-documents-prod`.
6. `notifier` envía el correo al receptor y dispara los webhooks que el cliente haya configurado.

El tiempo objetivo de punta a punta (desde el `POST` hasta el estado `ACCEPTED`) es de 8 segundos en el percentil 95. Si la autoridad tributaria está caída, los documentos quedan en estado `QUEUED` y `tax-gateway` los reintenta con backoff exponencial durante un máximo de 72 horas.

## 3. Almacenamiento de datos

- **Base de datos principal:** PostgreSQL 16 en Amazon Aurora, con una instancia *writer* y dos *readers*. El pool de conexiones lo administra PgBouncer en modo `transaction`, con un máximo de 400 conexiones hacia Aurora.
- **Caché:** Redis 7 (ElastiCache) para sesiones del portal, *rate limiting* y la caché de folios. El TTL por defecto de la caché de folios es de 15 minutos.
- **Archivos:** los PDF y XML se guardan en S3 durante 7 años, que es el plazo legal de conservación tributaria más largo de los tres países. Después de 90 días los objetos pasan a la clase de almacenamiento Glacier Instant Retrieval.
- **Analítica:** una réplica diaria de los datos anonimizados se carga en BigQuery a las 03:00 hora de Santiago mediante un job de Airflow llamado `ledger_daily_export`.

## 4. Observabilidad

Todas las métricas se envían a Grafana Cloud (Prometheus administrado) y los logs a Grafana Loki. Las trazas distribuidas usan OpenTelemetry y se visualizan en Grafana Tempo.

Los tableros oficiales son:

- **"Ledger — Golden Signals"**: latencia, tráfico, errores y saturación de `ledger-api`.
- **"Tax Gateway — Autoridades"**: tasa de aceptación y latencia por autoridad tributaria.
- **"Kafka — Lag de consumidores"**: retraso de cada grupo de consumidores.

Las alertas se enrutan a PagerDuty. Una alerta de severidad SEV1 o SEV2 despierta a la persona de guardia; las alertas SEV3 y SEV4 solo crean un ticket en Jira en el proyecto `OPS`.

## 5. Límites y cuotas por plan

| Plan | Documentos por mes | Rate limit de la API | Webhooks |
|---|---|---|---|
| Emprende | 500 | 5 solicitudes por segundo | No incluye |
| Pyme | 10.000 | 20 solicitudes por segundo | Hasta 3 endpoints |
| Corporativo | Ilimitado | 100 solicitudes por segundo | Hasta 20 endpoints |

Cuando un cliente supera el rate limit, el API Gateway responde `429 Too Many Requests` con el encabezado `Retry-After`. Los documentos que excedan la cuota mensual del plan no se rechazan: se cobran como excedente a 0,02 USD por documento.

## 6. Seguridad

- Los certificados digitales de los clientes se guardan cifrados en AWS KMS y nunca salen del servicio `tax-gateway`.
- El acceso a producción requiere VPN corporativa (WireGuard) y autenticación con llave de hardware (YubiKey). No se permite el acceso por SSH directo a los nodos: se usa AWS Systems Manager Session Manager.
- Los secretos de las aplicaciones se gestionan con AWS Secrets Manager y se rotan cada 90 días de forma automática.
- Se realiza un pentest externo cada seis meses, a cargo de la empresa ficticia "Andes Security Labs".

## 7. Decisiones de arquitectura relevantes

- **ADR-014 (2024):** se eligió Kafka en lugar de RabbitMQ por la necesidad de reprocesar eventos históricos durante las auditorías tributarias. La retención del tópico `ledger.documents.v2` es de 14 días.
- **ADR-021 (2025):** se migró `pdf-renderer` de Java a Python para usar la biblioteca WeasyPrint, lo que redujo el tiempo de renderizado de 1,4 a 0,6 segundos por documento.
- **ADR-027 (2026):** se descartó usar una base de datos multi-región activa-activa por su costo; se mantiene el esquema activo-pasivo con un RPO de 5 minutos y un RTO de 1 hora.
