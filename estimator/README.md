# Estimator — Servicio IA de estimación de software

Servicio IA en FastAPI que estima proyectos de software a partir de un formulario tipado. Es la pieza Python del programa **Master en AI Engineering**: un endpoint pensado para ser consumido por un backend de negocio (Rails, Streamlit u otro), no por un usuario final.

A partir de la **Sesión 04** el contrato es deliberadamente estrecho:
- entrada tipada (`description` + tres enums),
- salida en texto libre,
- prompt fuera del código en templates Jinja2 versionados (`app/foundation/prompts/<use_case>/<version>/`).

La inteligencia adicional (output estructurado, guardrails, cache semántico) se construye encima de esta base en directo.

## Cómo levantar

### Con Docker (recomendado)

```bash
cd estimator
cp .env.example .env  # añade al menos OPENAI_API_KEY o ANTHROPIC_API_KEY
docker compose up --build
```

El servicio queda en `http://localhost:8000` (Swagger en `/docs`, health en `/health`). Redis arranca como servicio vecino para el cache exact-match del wrapper.

### Sin Docker

```bash
cd estimator
uv sync
uv run uvicorn app.main:app --reload
```

### Probar el endpoint

```bash
curl -X POST http://localhost:8000/api/v1/estimate \
  -H "Content-Type: application/json" \
  -d '{
    "description": "A small B2B SaaS to manage employee equipment loans across teams. Role-based access, audit trail, weekly digest.",
    "project_type": "web_saas",
    "detail_level": "medium",
    "output_format": "phases_table"
  }'
```

Respuesta:

```json
{
  "text": "| phase | duration_weeks | cost_eur | confidence_pct | …",
  "prompt_version": "v1"
}
```

### Cliente Streamlit

El cliente Streamlit es un formulario que construye el JSON y muestra el `text` recibido. Corre fuera de Docker y consume la API por HTTP:

```bash
cd estimator
uv run streamlit run streamlit_app.py
# Abrir http://localhost:8501
```

La URL del servicio se lee de `ESTIMATOR_API_BASE_URL` (default `http://localhost:8000`).

## Cómo testar

```bash
cd estimator
uv run pytest
```

La batería corre en milisegundos sin tocar APIs externas. Cubre cuatro categorías:

- `tests/test_schemas.py` — validaciones del `EstimationRequest` (longitudes, enums, campos obligatorios).
- `tests/test_prompts.py` — render del template `v1`: `description` aparece dentro de `<project_description>`, los bloques condicionales por `output_format` y `detail_level` solo se incluyen cuando aplica, y `StrictUndefined` falla early ante variables faltantes.
- `tests/test_estimate_endpoint.py` — endpoint con el wrapper LLM mockeado vía `app.dependency_overrides`: comprueba el contrato 200/422, que `system_prompt` y `user_message` viajan separados, y que la respuesta lleva `prompt_version="v1"`.
- `tests/test_llm_wrapper.py` y `tests/test_cache.py` — wrapper y cache de la Sesión 03, intactos.

## Estructura del proyecto

```
estimator/
├── app/
│   ├── main.py                        # FastAPI app, CORS, lifespan, /health
│   ├── config.py                      # Settings (Pydantic Settings, .env)
│   ├── dependencies.py                # Singletons cacheados: cache + LLMWrapper
│   ├── routers/
│   │   └── estimations.py             # POST /api/v1/estimate
│   ├── schemas/
│   │   └── estimation.py              # EstimationRequest, EstimationResponse, enums
│   ├── prompts/
│   │   ├── loader.py                  # Environment Jinja2 + render_estimation_prompt
│   │   └── estimation/
│   │       └── v1/
│   │           ├── system.j2          # rol + reglas + bloques condicionales + include
│   │           ├── user.j2            # bloque <project_description>
│   │           └── examples.j2        # few-shot examples
│   └── services/
│       ├── llm_wrapper.py             # LiteLLM Router con fallback y cost tracking
│       └── cache.py                   # Redis exact-match cache
├── tests/
│   ├── test_schemas.py
│   ├── test_prompts.py
│   ├── test_estimate_endpoint.py
│   ├── test_llm_wrapper.py
│   └── test_cache.py
├── streamlit_app.py                   # Formulario que consume /api/v1/estimate
├── Dockerfile                         # Multi-stage con uv
├── docker-compose.yml                 # Servicio IA + Redis
└── pyproject.toml
```

### Versionado de prompts

La estructura `app/foundation/prompts/<use_case>/<version>/` no es opcional: `v1/` ya existe desde el primer día porque versionar un prompt es la forma más barata de habilitar A/B testing y rollback en producción. Cuando una iteración del prompt se cocina, se crea `v2/` al lado y `render_estimation_prompt(request, version="v2")` lo recoge sin tocar router ni schemas.

Lo que vive **fuera** del template (en código): el contrato (`EstimationRequest`), el switch de versión y el wrapper. Todo lo demás (rol del modelo, reglas, ejemplos, formatos de salida, niveles de detalle) vive dentro del `.j2`. Si para cambiar el comportamiento del modelo hay que tocar Python, la separación está rota.

## Variables de entorno

| Variable | Default | Notas |
|---|---|---|
| `OPENAI_API_KEY` | — | Requerido al menos uno de los dos |
| `ANTHROPIC_API_KEY` | — | Requerido al menos uno de los dos |
| `PRIMARY_MODEL` | `gpt-4o-mini` | Modelo principal del Router |
| `FALLBACK_MODEL` | `claude-haiku-4-5-20251001` | Se usa si el primario falla |
| `REDIS_URL` | `redis://localhost:6379` | Cache exact-match |
| `CACHE_TTL` | `86400` | Segundos |
| `APP_ENV` | `development` | Controla el renderer de structlog |
| `ESTIMATOR_API_BASE_URL` | `http://localhost:8000` | Lo lee el cliente Streamlit |

`get_settings()` es un singleton cacheado con `lru_cache`: cualquier cambio en `.env` requiere reiniciar uvicorn (no basta con `--reload`). **Excepción: los modelos LLM** — ver la sección siguiente.

## Configuración de modelos en runtime

Los knobs de modelo (`PRIMARY_MODEL`, `FALLBACK_MODEL`, `CRITIC_MODEL`, `METADATA_EXTRACTOR_MODEL`, `COMPRESSION_MODEL`, `PROPOSITIONAL_CHUNKER_MODEL`, `CONTEXTUAL_CHUNKER_MODEL`) se pueden **sobreescribir en caliente** sin tocar `.env` ni recrear contenedores — pensado para cambiar de modelo en mitad de un directo (la pestaña *Ajustes* del cliente Rails usa este endpoint).

```
GET /api/v1/config/models
  → {"models": {KEY: {"effective", "default", "overridden"}},
     "available_models": [...], "embedding_model": "..."}

PUT /api/v1/config/models
  Body: {"models": {"PRIMARY_MODEL": "gpt-4o", "CRITIC_MODEL": null}}   # null = reset
  → mismo shape que el GET (snapshot fresco)
  422 key desconocida / modelo fuera de catálogo · 400 modelo sin API key · 503 Redis caído
```

Cómo funciona (`app/foundation/llm/runtime_config.py`):

- Los overrides viven en un hash de Redis (`estimator:runtime_config`): **sobreviven a `--reload` y reinicios**, y todos los workers los ven al instante. `.env` sigue siendo la capa de defaults.
- El wrapper y el servicio resuelven el modelo **por llamada** (properties), así que el cambio aplica en la siguiente petición. El catálogo (`AVAILABLE_MODELS`) se filtra por las API keys configuradas.
- Con un override de primario activo no hay fallback automático de provider (misma semántica que `model_override`: llamada directa, sin Router).
- Las caches se particionan por modelo (la exacta ya lo hacía; la semántica incluye el modelo en su bucket desde este cambio), así que cambiar de modelo nunca sirve respuestas generadas por otro.
- `EMBEDDING_MODEL` queda fuera a propósito: cambiarlo invalidaría todos los vectores almacenados.

```bash
http PUT :8000/api/v1/config/models models:='{"PRIMARY_MODEL": "gpt-4o"}'
http PUT :8000/api/v1/config/models models:='{"PRIMARY_MODEL": null}'     # volver al .env
```

---

## Sesión 5 — Memoria conversacional y adjuntos

A partir de la Sesión 05 el estimator deja de ser puramente transaccional y soporta **sesiones conversacionales**: el cliente puede refinar el alcance del proyecto a lo largo de varios turnos, subir documentos (PDF/Word) y el sistema recuerda el proyecto en curso entre llamadas. El endpoint `POST /api/v1/estimate` original se mantiene intacto para compatibilidad y para la demo transaccional.

### Endpoints nuevos

```
POST /sessions                              → 201 {"session_id": "<uuid>"}
GET  /sessions/{session_id}                 → 200 {session_id, message_count, max_turns, metadata}
POST /sessions/{session_id}/estimate        → 200 EstimationResponse
   (multipart/form-data: transcript, project_type, detail_level, output_format, attachments[])
```

Ejemplo end-to-end con httpie:

```bash
http POST :8000/sessions
# {"session_id": "abc-123"}

http -f POST :8000/sessions/abc-123/estimate \
  transcript="Queremos estimar un CRM llamado Nimbus en React + Postgres para el equipo de ventas." \
  project_type=web_saas detail_level=medium output_format=phases_table \
  attachments@spec.pdf

http GET :8000/sessions/abc-123
# Inspecciona el ProjectMetadata acumulado y el tamaño del historial.
```

Y un segundo turno reutilizando el mismo `session_id` sin repetir el contexto:

```bash
http -f POST :8000/sessions/abc-123/estimate \
  transcript="Añade un módulo de facturación con Stripe." \
  project_type=web_saas detail_level=medium output_format=phases_table
```

La respuesta del segundo turno integra Nimbus + React + Postgres + facturación porque el `<project_metadata>` se inyecta en el system prompt y el historial reciente viaja en el array `messages`.

### Decisiones de diseño

1. **Camino B para los adjuntos.** Extraemos el texto del PDF/Word **dentro del servicio IA** con `pypdf` y `python-docx`, lo recortamos a `MAX_ATTACHMENT_CHARS` y lo concatenamos al transcript con fences explícitos (`--- attachment: spec.pdf ---`). La alternativa (Camino A: subir el binario a la Files API de OpenAI o Anthropic) habría sido más corta de implementar pero acopla el wrapper a un proveedor multimodal concreto. Camino B mantiene `complete_structured_chat` agnóstico de proveedor (texto en, texto fuera vía LiteLLM Router + Instructor) y prepara el terreno para el chunking real de RAG en el módulo 3. La extracción es robusta a páginas corruptas (fallos por página se loguean y se ignoran) y a archivos vacíos.

2. **`project_metadata` con extractor LLM, no heurística.** Tras cada respuesta del estimador, una **segunda llamada** al LLM (modelo barato configurable vía `METADATA_EXTRACTOR_MODEL`, por defecto `gpt-4o-mini`) lee el último turno y devuelve un `ProjectMetadata` parcial vía Instructor. Lo fusionamos con el previo: campos escalares sobrescriben si vienen no-nulos, la lista de tecnologías se une case-insensitively. Se eligió el extractor LLM frente a una heurística regex porque el coste de una llamada con prompt corto es marginal y la robustez frente a paráfrasis del usuario es mucho mejor — y porque el curso enseña precisamente cómo construir estos pasos con LLMs. Si la llamada falla, se loguea y se conserva la metadata previa: la conversación no se cae por una extracción rota.

3. **Memoria en proceso, no Redis ni Postgres.** El `SessionStore` es un `dict` en memoria del worker FastAPI. La volatilidad (estado perdido al reiniciar el contenedor) es **intencional** para esta fase y está documentada en el docstring del store. La persistencia entre reinicios entra en el directo cuando hablemos de compresión de memoria con anclas.

4. **Cachés desactivadas en el path conversacional.** Cada turno depende del historial + metadata + adjuntos: dos transcripciones idénticas en sesiones distintas **no** son la misma llamada. El método nuevo `EstimationService.estimate_conversational` por tanto no consulta ni el cache exact-match ni el semántico, y `EstimationResponse.cached` siempre es `false` en este path. El endpoint transaccional original `POST /api/v1/estimate` sigue usando las dos cachés sin cambios.

5. **Ventana deslizante con `MAX_CONVERSATION_TURNS=6` por defecto.** El system prompt se regenera fresco cada turno desde el `ProjectMetadata` actual, así que no consume slot. Lo que llega al LLM en el turno N es: `[system_v2] + últimos N pares (user, assistant) + nuevo user`. Cuando el historial supera el tope, los pares más antiguos se descartan en bloque para preservar la alternancia de roles. El siguiente paso (resumen acumulativo + anclas) lo construimos en el directo.

### Variables de entorno nuevas

| Variable | Default | Notas |
|---|---|---|
| `MAX_CONVERSATION_TURNS` | `6` | Pares user+assistant que mantiene la ventana. |
| `MAX_ATTACHMENT_CHARS` | `60000` | Corte por archivo extraído. Trunca, no rechaza. |
| `METADATA_EXTRACTOR_MODEL` | `gpt-4o-mini` | Modelo de la segunda llamada por turno. |

### Tests del Paso 7

```bash
uv run pytest tests/test_sessions_metadata.py tests/test_sessions_attachments.py tests/test_sessions_window.py -v
```

Los tres tests son de integración con `TestClient`, un `FakeLLMWrapper` que captura cada llamada y devuelve resultados scripted, y un `SessionStore` aislado por test (sin singleton). Cubren los tres criterios del enunciado: dos turnos acumulan metadata, el contenido de un PDF llega al `messages` del LLM, y enviar más turnos que `MAX_CONVERSATION_TURNS` nunca infla el array de mensajes más allá del límite.

### Cliente Rails

El cliente Rails (`estimator-web/`) se adaptó al flujo conversacional con un nuevo controller `ChatSessionsController` (rutas `/chat_sessions`, root re-apuntado aquí), un panel lateral con el `ProjectMetadata` actual, multipart vía `faraday-multipart` y un botón "Nueva conversación" que destruye el mirror local y arranca una sesión limpia. El endpoint transaccional `EstimationsController` se mantiene operativo para la demo histórica.

## Sesión 7 — Pipeline de embeddings

Primer paso hacia la búsqueda semántica: convertir presupuestos históricos (JSON) en vectores. El módulo nuevo vive en `app/generation/rag/` y expone un único endpoint. En la Sesión 07 **no se persistía nada** — los vectores se generaban en memoria y se devolvían por HTTP. La **[Sesión 08](#sesión-8--persistencia-vectorial-y-búsqueda-semántica)** cambió eso: ahora la ingesta persiste en Postgres + pgvector y se añade un endpoint de búsqueda semántica.

Piezas:

- `chunker.py` (`JSONStructuralChunker`) — chunking **estructural**: un componente del presupuesto = un chunk. A cada chunk se le antepone una cabecera de contexto del presupuesto padre (proyecto, sector, tecnología) para que no pierda la pista de a quién pertenece. Cuenta tokens con `tiktoken`.
- `embedder.py` (`OpenAIEmbedder`) — invoca `text-embedding-3-small` (1536 dims) en **batches** de 100, con reintento exponencial (1s/2s/4s) ante `RateLimitError` y logging por batch.
- `router.py` — orquesta `chunk → embed → stats`.

### Endpoint (forma de la Sesión 07, refactorizado en S08)

En la Sesión 07 `POST /embeddings/ingest` recibía `{"budgets": [<Budget>, ...]}` y **devolvía** los `chunks` embebidos + `stats` por HTTP, sin persistir (con el sample: 15 presupuestos → 52 chunks → ~4.1k tokens → ~$0.00008). La **[Sesión 08](#sesión-8--persistencia-vectorial-y-búsqueda-semántica)** cambió el contrato: ahora recibe **un** documento (`source_path` + `document_type` + `content`), lo persiste en una transacción y devuelve solo identificadores y métricas. El contrato vigente está más abajo.

### Sanity check de embeddings

El `scripts/compare.py` de la Sesión 07 (similitud coseno entre dos textos sueltos, calculada a mano) fue **reemplazado en S08** por [`scripts/query_examples.py`](#sesión-8--persistencia-vectorial-y-búsqueda-semántica), que ejercita el corpus real vía `POST /search`. Los resultados de las parejas de validación originales siguen en [`app/generation/rag/SANITY_CHECK.md`](app/generation/rag/SANITY_CHECK.md).

### Comparativa de estrategias de chunking (sesión en vivo)

Ocho estrategias de chunking tras una interfaz común (`app/generation/rag/chunking/base.py::Chunker`): `structural`, `fixed_size`, `recursive`, `sentence_window`, `semantic`, `propositional`, `contextual_retrieval`, `hierarchical`. Viven en `app/generation/rag/chunking/strategies/` (el estructural en `structural.py`).

```
POST /embeddings/compare
  Input:  {"budgets": [...], "queries": [...], "strategies": [...], "top_k": 3}
  Output: {"stats_per_strategy": {...}, "queries_per_strategy": {...}}
```

CLI del comparador (la herramienta de las demos), que carga `data/budgets_sample.json` + `data/test_queries.json`:

```bash
# Estadísticos + coste de todas las estrategias
uv run python scripts/compare_chunkers.py --strategies all --queries all --show-stats --show-cost

# Top-k de una consulta para dos estrategias
uv run python scripts/compare_chunkers.py --strategies sentence-window,structural \
  --queries "OAuth authentication for fintech mobile app" --show-top-k 3

# Comparar dimensiones del modelo (1536 vs 768 / Matryoshka)
uv run python scripts/compare_chunkers.py --models small-1536,small-768

# Generar el reporte de respaldo
uv run python scripts/compare_chunkers.py --strategies all --queries all \
  --show-stats --show-cost --output app/generation/rag/COMPARISON_REPORT.md
```

Las estrategias `semantic`, `propositional` y `contextual_retrieval` llaman a APIs externas durante la ingesta (necesitan `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`) y reportan su coste en `chunking_done`. `sentence_window` usa NLTK (`punkt`/`punkt_tab`, descarga perezosa). El comparador **no persiste nada** (es una herramienta de análisis); la persistencia vectorial llegó en la **[Sesión 08](#sesión-8--persistencia-vectorial-y-búsqueda-semántica)** para el flujo de ingesta + búsqueda.

### Dependencias y scope

- Dependencias del pre-ejercicio: `tiktoken>=0.7.0` (`openai` ya estaba desde Sesión 01).
- Dependencias de la sesión en vivo: `langchain-text-splitters`, `langchain-experimental`, `langchain-openai`, `nltk` (`anthropic` ya estaba). No se añade numpy/scikit-learn ni `sentence-transformers`; la coseno y los percentiles son stdlib.
- **Late chunking** se trata como concepto en el directo (no hay código ejecutable: requiere modelos con token-level embeddings que no son el del proyecto).
- **Ya en la Sesión 08** (ver abajo): persistencia vectorial (pgvector) y búsqueda semántica real (`POST /search`). **Sigue fuera de scope**: índice ANN (HNSW) y métricas formales de retrieval (recall@k, NDCG).
- El guion del directo está en `guides/session-7-live-guide.md` (git-ignored, material de instructor).

## Sesión 8 — Persistencia vectorial y búsqueda semántica

La ingesta deja de ser en memoria: los chunks embebidos se **persisten** en Postgres + pgvector y se consultan por **similitud semántica**. La migración `0002_rag_documents_chunks` activa la extensión `vector` y crea dos tablas; el código nuevo vive en `app/generation/rag/store/` (escritura) y `app/generation/rag/retriever.py` (lectura).

```
POST /embeddings/ingest   {source_path, document_type, content:<Budget>}
  200 → {document_id, chunks_created, embedding_dimension, ingestion_time_ms}
  409 → {detail: "Document already ingested", document_id}   # source_path repetido
  Chunk + embed + persist ocurren en UNA transacción: si el embedder falla, no queda documento huérfano.

POST /search              {query, k}
  200 → {query, k, search_time_ms, results:[{chunk_id, document_id, chunk_type, content, distance, metadata}]}
```

`scripts/query_examples.py` (sustituye al `compare.py` de S07) lanza cinco queries representativas contra `/search` e imprime el top-k, para observar el ranking desde ángulos distintos (match directo, paráfrasis, dominio ajeno, consulta ambigua, consulta muy específica).

### Decisiones de esquema (y por qué)

**(a) Dos tablas (`documents` / `chunks`), no una.** Un presupuesto produce N chunks (1:N). Una sola tabla plana duplicaría los metadatos del documento (`source_path`, `document_type`, `ingested_at`) en cada fila de chunk: almacenamiento desperdiciado y anomalías de actualización. Con dos tablas y `FOREIGN KEY ... ON DELETE CASCADE`, borrar un documento elimina atómicamente sus chunks — sin chunks huérfanos ni identidad de documento repetida. La separación es también conceptual: `documents` guarda **procedencia**, `chunks` guarda **contenido + vector**.

**(b) `metadata` como `JSONB`, no columnas.** Los campos estables y de forma fija van en columnas tipadas (`document_type`, `chunk_type`, `source_path`, fechas). Los campos **variables o que el chunker enriquece** (`budget_id`, `client_sector`, `main_technology`, `complexity`, y a futuro tags/scope/tecnologías) van en `JSONB`: su forma cambia según el tipo de documento y la estrategia de chunking, y el chunker puede añadir claves **sin una migración cada vez**. El índice **GIN** sobre `chunks.metadata` permite filtrar por claves arbitrarias del JSONB. El coste asumido: se pierde el chequeo de tipos a nivel de columna, aceptable porque estos campos son para filtrado flexible, no la vía de acceso principal.

**(c) `cosine_distance` (`<=>`), no L2 ni inner product.** La similitud semántica de texto es **angular**: importa la dirección del vector, no su magnitud. La distancia coseno es invariante a la escala e interpretable (rango `[0, 2]`, `0` = misma dirección). El *inner product* (`<#>`) premia además la magnitud, así que solo equivale a coseno si los vectores están perfectamente normalizados — una suposición frágil (dimensiones Matryoshka recortadas, futuros modelos). L2 (`<->`) mezcla magnitud y dirección. Coseno es la convención para embeddings de texto y la opción robusta aunque un vector no venga con norma exacta 1. (Fija además la *operator class* del futuro índice: `vector_cosine_ops`.)

**(d) Sin índice vectorial, a propósito.** Sin índice, Postgres hace *sequential scan* completo: vecinos **exactos**, recall 100%, cero aproximación ni knobs que tunear. Para el corpus de ejemplo (decenas de documentos, cientos de chunks) eso resuelve en pocos cientos de ms — perfectamente aceptable. Un índice ANN (HNSW/IVFFlat) cambia exactitud por velocidad: es **aproximado** (recall < 100%), exige tuning (`m`, `ef_construction`, `ef_search`…) y solo compensa a escala. Empezar sin índice permite **observar** la latencia del scan en el directo y verla desplomarse al añadir HNSW — no se aprecia el índice sin sentir antes su ausencia. El slot está reservado en el roadmap de `ARCHITECTURE.md`.

---

> Este proyecto forma parte del **Master en AI Engineering** y es la base sobre la que se construye en directo el resto de la Sesión 04 (output estructurado, guardrails, cache semántico) y de la Sesión 05 (compresión avanzada de memoria con anclas, tier dinámico, patrón Actor-Critic-Boss).
