# Sanity check — Embeddings (`compare.py`)

Comprobación mínima de que el pipeline de embeddings funciona end-to-end y que
los vectores discriminan razonablemente entre textos cercanos y lejanos. **No es
una validación formal del modelo**: es el mínimo aceptable para demostrar que
`OpenAIEmbedder` + similitud coseno se comportan como cabe esperar.

- Modelo: `text-embedding-3-small`
- Script: `scripts/compare.py` (similitud coseno calculada a mano: producto escalar / producto de normas)
- Fecha del run: 2026-07-07

## Resultados

| Pareja | Expectativa | Similitud coseno | ¿Encaja? |
|---|---|---|---|
| A — semánticamente cercanos | alta (> 0.6 orientativo) | **0.5958** | Casi (justo por debajo del umbral) |
| B — no relacionados | baja (< 0.4 orientativo) | **0.1920** | Sí, con holgura |
| C — genéricos / ambiguos | sin expectativa fija | **0.5407** | — (a comentar) |

### Pareja A — cercanos
- Texto 1: `OAuth 2.0 authentication backend with JWT tokens for fintech mobile app`
- Texto 2: `Authorization service using JSON Web Tokens for a banking application`

### Pareja B — no relacionados
- Texto 1: `OAuth 2.0 authentication backend with JWT tokens for fintech mobile app`
- Texto 2: `Database migration from MySQL to PostgreSQL with zero downtime`

### Pareja C — genéricos / ambiguos
- Texto 1: `Backend services`
- Texto 2: `API development`

## Comentario

Los resultados encajan con la intuición en lo esencial: **B (0.19) queda muy por
debajo de A (0.60) y de C (0.54)**, así que el embedding separa con claridad un
par no relacionado de los demás — que es justo lo que este sanity check quiere
demostrar. Lo que llama la atención es que **A se queda en 0.5958, un pelo por
debajo del 0.6 orientativo** pese a ser un par claramente sinónimo (OAuth/JWT/
banca ≈ autorización/JWT/banco): con `text-embedding-3-small` el rango de coseno
está comprimido y valores ~0.6 ya representan "muy parecido", por lo que no lo
leería como un fallo sino como una nota sobre calibrar umbrales al modelo, no a
la intuición humana. El caso más discutible es **C (0.5407)**: dos frases muy
cortas y genéricas puntúan casi tan alto como el par sinónimo A, porque comparten
el mismo dominio ("backend"/"API") y, al tener poco contenido, el vector se apoya
en ese vocabulario compartido. Buen material para el directo: **el coseno alto en
textos vagos no implica equivalencia semántica**, y refuerza por qué un umbral de
caché semántica (~0.85) se fija bastante por encima de estos valores.
