# Guía de onboarding para ingenieros

> Documento interno de Nimbus Cloud S.A. (empresa ficticia usada como dataset de ejemplo).
> Última revisión: abril de 2026. Responsable: People & Engineering Enablement.

## 1. Bienvenida

Esta guía resume lo que necesitas durante tus primeras semanas como ingeniero o ingeniera en Nimbus Cloud. Ante cualquier duda, tu *onboarding buddy* es el primer contacto: es una persona de otro equipo asignada para acompañarte durante tus primeros 60 días.

## 2. Primera semana

### Día 1

- Retira tu equipo en la oficina o recíbelo por courier. Los ingenieros reciben un MacBook Pro de 14 pulgadas con 36 GB de RAM, o un notebook Lenovo ThinkPad con Ubuntu si lo pides antes de tu ingreso.
- Activa tu cuenta de Google Workspace y configura la autenticación en dos pasos.
- Solicita tu YubiKey al equipo de IT en el canal `#it-ayuda`. Sin YubiKey no podrás acceder a la VPN ni a producción.

### Días 2 y 3

- Completa los cursos obligatorios en la plataforma de aprendizaje: seguridad de la información, protección de datos personales y código de conducta. Debes aprobarlos con al menos 80% dentro de tus primeros 10 días.
- Configura tu entorno de desarrollo siguiendo el repositorio `nimbus/dev-setup`, que instala las herramientas con un único script (`./bootstrap.sh`).
- Pide acceso de lectura a los repositorios de tu equipo en GitHub mediante el formulario "Acceso a repos" de Jira Service Management.

### Días 4 y 5

- Haz tu primer *pull request*. Por tradición, consiste en agregar tu nombre y tu comida favorita al archivo `TEAM.md` del repositorio `nimbus/handbook`.
- Asiste a la sesión "Nimbus en 90 minutos", donde el equipo de Plataforma explica la arquitectura. Se realiza el primer miércoles de cada mes.

## 3. Primer mes

- Acompaña a la persona de guardia durante una semana completa en modo *shadow*, sin responsabilidad sobre las alertas. Esto ocurre normalmente en tu tercera o cuarta semana.
- Resuelve al menos dos tickets etiquetados `good-first-issue` en el backlog de tu equipo.
- Ten una reunión 1:1 con tu Engineering Manager cada semana, y con el *Staff Engineer* de tu área al menos una vez.

Te sumarás a la rotación de guardia como persona secundaria al cumplir 3 meses, y como persona primaria al cumplir 6 meses, siempre que hayas hecho al menos una semana de *shadow*.

## 4. Herramientas principales

| Herramienta | Para qué se usa |
|---|---|
| GitHub | Código, *pull requests* y GitHub Actions para CI/CD |
| Jira | Tickets de producto (proyectos `LED`, `TAX`, `DOC`) y de operaciones (`OPS`) |
| Confluence | Documentación, ADRs y postmortems |
| Slack | Comunicación diaria; los canales de incidentes empiezan con `#inc-` |
| PagerDuty | Alertas y rotaciones de guardia |
| Grafana Cloud | Métricas, logs y trazas |
| 1Password | Gestión de contraseñas personales y compartidas del equipo |

## 5. Beneficios para el equipo de ingeniería

- **Presupuesto de aprendizaje:** 1.200 USD anuales para cursos, libros o conferencias. No es acumulable entre años y se solicita con el formulario "Aprendizaje" en Jira Service Management, con aprobación del Engineering Manager.
- **Trabajo híbrido:** se espera presencia en la oficina de Santiago (Providencia) los martes y jueves. Quienes viven a más de 100 km de la oficina pueden trabajar en modo 100% remoto.
- **Equipamiento para casa:** un aporte único de 400 USD para silla, monitor u otros elementos de ergonomía, disponible desde el primer mes.
- **Semana de hackathon:** dos veces al año (abril y octubre) se dedica una semana completa a proyectos libres. Los tres mejores proyectos se presentan a toda la empresa.

## 6. Cultura de ingeniería

- **Escribimos antes de construir:** cualquier cambio que afecte a más de un equipo empieza con un *RFC* en Confluence, que queda abierto a comentarios al menos 5 días hábiles.
- **Documentamos las decisiones:** las decisiones de arquitectura se registran como ADRs numerados (por ejemplo, ADR-014).
- **Aprendemos de los errores sin culpar:** los postmortems son *blameless*.
- **Medimos:** los equipos revisan sus métricas DORA (frecuencia de despliegue, tiempo de entrega, tasa de fallos y tiempo de recuperación) en la retrospectiva mensual.

## 7. Contactos útiles

| Tema | Canal o persona |
|---|---|
| Problemas con el equipo o accesos | `#it-ayuda` |
| Dudas sobre la plataforma | `#plataforma` |
| Dudas de recursos humanos, sueldos o vacaciones | `#people` |
| Seguridad (reportar un incidente o un correo sospechoso) | `#seguridad` o seguridad@nimbus.example |

## 8. Periodo de prueba

El periodo de prueba dura 3 meses. Al final del segundo mes tendrás una conversación de retroalimentación formal con tu Engineering Manager y tu *onboarding buddy*, basada en los objetivos acordados durante la primera semana.
