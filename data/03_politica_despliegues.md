# Política de despliegues y control de cambios

> Documento interno de Nimbus Cloud S.A. (empresa ficticia usada como dataset de ejemplo).
> Última revisión: enero de 2026. Responsable: equipo de Plataforma.

## 1. Principios

1. Todo cambio en producción pasa por un *pull request* revisado y por el pipeline de CI/CD. No existen cambios manuales en producción fuera de las mitigaciones preaprobadas del runbook de incidentes.
2. Desplegamos cambios pequeños y frecuentes. Un *pull request* no debería superar las 400 líneas modificadas, sin contar archivos generados ni migraciones automáticas.
3. Cada despliegue debe poder revertirse en menos de 10 minutos.

## 2. Flujo de ramas

Usamos *trunk-based development* con la rama `main` como tronco. Las ramas de trabajo se nombran `tipo/TICKET-descripcion`, por ejemplo `feat/LED-482-notas-de-credito`. Las ramas no deben vivir más de 3 días; si una funcionalidad tarda más, se integra detrás de un feature flag.

Los mensajes de commit siguen *Conventional Commits* (`feat:`, `fix:`, `chore:`, etc.). Los *pull requests* se integran con *squash merge*.

## 3. Revisión de código

- Todo *pull request* necesita al menos **una aprobación** de alguien del equipo dueño del servicio.
- Los cambios en `tax-gateway`, en migraciones de base de datos o en la configuración de seguridad necesitan **dos aprobaciones**, y una debe ser de una persona con rol de *Staff Engineer* o superior.
- La persona autora no puede aprobar su propio *pull request*, aunque sea administradora del repositorio.
- El pipeline de CI debe pasar completo: lint, tests unitarios, tests de contrato y escaneo de dependencias con Trivy. La cobertura mínima de tests es del 80% en código nuevo.

## 4. Entornos

| Entorno | Uso | Despliegue | Datos |
|---|---|---|---|
| `dev` | Pruebas de cada equipo | Automático con cada push a cualquier rama | Sintéticos |
| `staging` | Validación integrada y pruebas de QA | Automático con cada merge a `main` | Copia anonimizada semanal de producción |
| `prod` | Clientes | Manual, con aprobación en GitHub Actions | Reales |

Los ambientes de prueba de las autoridades tributarias (por ejemplo, el ambiente de certificación del SII) solo se usan desde `staging`.

## 5. Ventanas de despliegue

Los despliegues a producción se permiten de **lunes a jueves, entre las 10:00 y las 16:00 hora de Santiago**. Los viernes, fines de semana y feriados de Chile solo se permiten despliegues de *hotfix* para resolver un incidente SEV1 o SEV2.

Además, existe un **congelamiento de cambios** (*change freeze*) durante los últimos 2 días hábiles y los primeros 3 días hábiles de cada mes, porque coinciden con los cierres tributarios y el mayor volumen de documentos. Durante el congelamiento solo se permiten *hotfixes* aprobados por el Engineering Manager del equipo y por el IC de guardia.

El congelamiento anual de fin de año va del 20 de diciembre al 5 de enero.

## 6. Estrategia de despliegue progresivo

Los servicios críticos (`ledger-api` y `tax-gateway`) se despliegan con *canary* mediante Argo Rollouts:

1. **Canary 5%:** la nueva versión recibe el 5% del tráfico durante 15 minutos.
2. **Canary 25%:** si los indicadores son sanos, sube al 25% durante 15 minutos más.
3. **100%:** se completa el despliegue.

Argo Rollouts revierte automáticamente si durante cualquier etapa la tasa de errores 5xx supera el 1% o la latencia p95 aumenta más de un 20% respecto de la versión estable. El resto de los servicios usa *rolling update* estándar de Kubernetes con `maxUnavailable: 0` y `maxSurge: 25%`.

## 7. Migraciones de base de datos

Las migraciones se gestionan con Atlas y siguen el patrón *expand and contract*:

1. **Expandir:** agregar la columna o tabla nueva sin eliminar la antigua. El código escribe en ambas.
2. **Migrar:** copiar los datos históricos con un job en segundo plano, en lotes de máximo 10.000 filas para no bloquear tablas.
3. **Contraer:** eliminar la estructura antigua en un despliegue posterior, como mínimo una semana después.

Está prohibido ejecutar en un mismo despliegue una migración que elimine columnas y el código que deja de usarlas. Cualquier migración que bloquee una tabla con más de 1 millón de filas debe ejecutarse fuera del horario de mayor tráfico, entre las 02:00 y las 05:00, y con un DBA presente.

## 8. Feature flags

Los feature flags se gestionan con Unleash. Las reglas son:

- Todo flag tiene un dueño y una fecha de expiración, como máximo 90 días después de su creación.
- Los flags de lanzamiento se eliminan del código cuando la funcionalidad llega al 100% de los clientes.
- Los nombres siguen el formato `servicio.nombre_en_snake_case`, por ejemplo `ledger.degraded_mode`.

## 9. Rollback

El comando estándar es `nimbusctl rollback <servicio>`, que vuelve a la última versión estable registrada en Argo CD. El rollback no revierte migraciones de base de datos; por eso el patrón *expand and contract* es obligatorio.

Después de un rollback en producción, el equipo dueño debe abrir un ticket en Jira con la etiqueta `rollback` y explicar la causa antes de volver a intentar el despliegue.
