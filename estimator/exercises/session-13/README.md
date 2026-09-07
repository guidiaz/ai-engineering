# Sesión 13 — De bucle agéntico a grafo explícito (LangGraph)

En la Sesión 12 construiste un **agente a mano**: un bucle sobre la Responses API que decidía en
cada iteración qué tool llamar (`search_budgets`, `calculate_estimate`, `validate_estimate`) y
cuántas veces. Funciona, y su virtud es precisamente esa: **decide**.

Pero esa virtud tiene un precio. El control de flujo vive dentro de un `while`, así que no puedes
mirarlo, no puedes testear un paso aislado, no puedes decir "estos dos pasos son independientes,
que corran a la vez" y no puedes reanudar una ejecución que se cayó a la mitad.

En este ejercicio ese bucle pasa a ser un **grafo explícito**.

> Un grafo no te hace el sistema más listo. Te hace la topología **visible**, **testeable** y
> **paralelizable**. Primero se declara la forma; sólo después se decide qué puede correr a la vez.

**De puertas afuera no cambia nada.** El servicio IA sigue recibiendo una transcripción y
devolviendo una estimación estructurada con su campo `status`. El backend de negocio no se entera
de que por debajo hay un grafo. El grafo vive **dentro** del servicio IA.

## El grafo que construyes

Cinco nodos, de momento **en secuencia**:

```
START
  → extract_requirements      transcripción            → lista de requisitos
  → classify_components       requisitos               → componentes con categoría
  → search_budgets            cada componente          → presupuestos de referencia
  → generate_estimate         presupuestos             → estimación consolidada
  → validate_and_consolidate  estimación               → revisión + `status` de salida
END
```

Que sea secuencial es deliberado: es la línea base que la siguiente iteración paraleliza.

## Dónde vive

```
app/generation/agentic/graph/
├── schemas.py    contratos Pydantic que viajan en el estado (incluye GraphEstimate.status)
├── state.py      el TypedDict con los reducers          ← punto 1 del enunciado
├── nodes.py      las cinco funciones puras              ← punto 2
├── builder.py    cableado + compile()                   ← punto 3
└── __init__.py
```

Los tests están en `tests/generation/agentic/test_graph.py` y **no tocan la red**.

## 1. El estado y sus reducers

En LangGraph el estado no es un objeto que mutas: es un conjunto de **canales**. Cada nodo devuelve
una **actualización parcial** y LangGraph la fusiona, canal por canal, con:

- el **reducer por defecto** — gana la última escritura, el valor nuevo sustituye al viejo; o
- un **reducer explícito** declarado con `Annotated[T, fn]` — el valor nuevo se combina con lo
  acumulado.

`operator.add` sobre una lista es "añade lo que ha producido este nodo a lo que produjeron los
anteriores". Eso es justo lo que necesita un campo que crece a lo largo del flujo:

```python
class EstimationGraphState(TypedDict, total=False):
    transcript: str

    # acumuladores
    budgets: Annotated[list[BudgetHit], operator.add]
    errors:  Annotated[list[str], operator.add]

    # última escritura gana
    requirements: list[Requirement]
    components:   list[Component]
    estimate:     GraphEstimate
```

`errors` es el que de verdad acumula **entre nodos** hoy: cualquier nodo puede añadir un fallo sin
pisar los de los demás. `budgets` acumula dentro de `search_budgets` y, sobre todo, es lo que
permitirá abrir la búsqueda en paralelo **sin tocar el estado**: con `operator.add` dos ramas que
escriben `budgets` a la vez se fusionan; con un canal normal saltaría `InvalidUpdateError`.

> ⚠️ Los canales sin reducer son seguros aquí **sólo porque las aristas son secuenciales**: ningún
> par de nodos escribe la misma clave. En cuanto haya fan-out, toda clave que escriban dos ramas o
> se vuelve acumulador o se separa por rama.

## 2. Los nodos son funciones puras

Reciben el estado, devuelven una actualización parcial. Dos reglas que el grafo da por hechas:

1. **No mutes el estado que recibes.** Construye un valor nuevo y devuélvelo.
2. **No devuelvas el estado entero.** Devolver un canal acumulado vuelve a aplicar `operator.add`
   y **duplica la lista en silencio**. Es el error más común del ejercicio, y es invisible en los
   números finales: sólo lo pilla un test que mire la longitud
   (`test_accumulator_channels_do_not_double`).

**Cada nodo reutiliza lógica que ya existe** — no se reimplementa nada:

| Nodo | Reutiliza |
|---|---|
| `extract_requirements` | `LLMWrapper.complete_structured` (Instructor, S3) |
| `classify_components` | `LLMWrapper.complete_structured` |
| `search_budgets` | `agent_tools.default_retrieval_backend` → el `retrieve()` híbrido de S9/S10 |
| `generate_estimate` | `agent_tools.calculate_estimate` (determinista, sin LLM) |
| `validate_and_consolidate` | `agent_tools.validate_estimate` (guardrails estilo S4) |

Los fallos se **acumulan, no se lanzan**: un componente cuya búsqueda revienta añade una línea a
`errors` y el resto de la estimación sigue adelante. Lo único que cambia es el `status` final.

## 3. El contrato de salida: `status`

Lo decide **un solo nodo**, `validate_and_consolidate`, para que haya un único sitio donde mirar
cuando el backend recibe algo inesperado:

| `status` | Significa |
|---|---|
| `ok` | Pasó todos los guardrails. Los números son utilizables. |
| `needs_review` | Hay estimación, pero algo saltó: un componente sin referencia histórica, horas fuera del rango que implican sus referencias, un total que no cuadra… |
| `insufficient_context` | No se recuperó nada en que apoyarse. Totales a 0 y **no** se debe leer como "proyecto barato", sino como "pide más información". |

## Cómo ejecutarlo

```bash
# offline, sin base de datos: el stub de retrieval de S12 sustituye al pipeline real
uv run python scripts/run_graph_s13.py \
    exercises/session-12/sample_transcript_complex.txt --stub

# la ejecución real, contra el retrieval de S9/S10
docker compose exec estimator python scripts/run_graph_s13.py \
    exercises/session-12/sample_transcript_complex.txt
```

Se reutiliza la transcripción compleja de S12 (cuatro componentes que no tienen nada que ver entre
sí) porque es exactamente el caso que justifica el grafo. El stub de recuperación también es el de
S12: no se duplica material.

La ejecución real necesita el stack levantado **y el corpus de tareas ingerido**:

```bash
docker compose exec estimator python scripts/build_task_corpus.py --ingest
```

Sin ese corpus, `search_budgets` filtra por `chunk_type='historical_task'`, no encuentra nada y el
grafo devuelve —correctamente— `status=insufficient_context` con todo a cero.

> **La descomposición varía entre ejecuciones.** `extract_requirements` y `classify_components` son
> llamadas a un LLM: la misma transcripción puede dar 5 componentes en una ejecución y 9 en otra, y
> el `status` final depende de si alguno se queda sin referencia histórica (`needs_review`) o no
> (`ok`). `example_run.txt` es una ejecución concreta, no un resultado que debas reproducir clavado:
> si tu recuento no coincide, no has roto nada.

---

# Paso 2 — Persistencia y observabilidad

## 4. Checkpointing sobre el Postgres del proyecto

Un checkpointer convierte el grafo de una llamada de usar y tirar en una **reanudable**: LangGraph
escribe el estado después de cada nodo, indexado por el `thread_id` que pasas en la invocación.

```python
async with postgres_checkpointer() as saver:            # checkpointing.py
    compiled = compile_estimation_graph(checkpointer=saver)
    await compiled.ainvoke(payload, config={"configurable": {"thread_id": estimation_id}})
```

**El `thread_id` es el identificador de la estimación.** El mismo valor va como `thread_id`, como
canal `estimation_id` del estado y como atributo de todos los spans, así que la traza de Logfire y
las filas de checkpoint de una misma ejecución se unen por la misma clave. Se acuña uno si no lo
pasas, igual que hace `rag.estimator._current_request_id`.

Reutiliza el Postgres de pgvector que ya tienes: no hay servicio nuevo. Dos cosas que conviene
saber antes de tocarlo:

- **La URL hay que convertirla.** `settings.DATABASE_URL` viene en forma de dialecto SQLAlchemy
  (`postgresql+psycopg://`) y psycopg la rechaza. De eso se encarga `psycopg_conn_string`.
- **Las cuatro tablas de checkpoint NO las gestiona alembic.** `saver.setup()` crea y versiona
  `checkpoints`, `checkpoint_writes`, `checkpoint_blobs` y `checkpoint_migrations` por su cuenta
  (con su propia tabla `checkpoint_migrations`). Están en la misma base de datos que las tablas de
  alembic pero **deliberadamente fuera** de su historial: no las metas en una migración, y no
  ejecutes `alembic revision --autogenerate` sin excluirlas o generará una migración que las borra.

También se declara un allowlist de serialización (`allowed_msgpack_modules`) con los modelos que
viajan en el estado. Sin él, LangGraph avisa —«Deserializing unregistered type … will be blocked in
a future version»— y en una versión futura el resume dejaría de funcionar.

### Reanudar de verdad

Ojo con una trampa: **re-invocar un hilo que ya llegó a `END` no reanuda, re-ejecuta**. Reanudar va
de continuar una ejecución *inacabada*. Por eso la demo interrumpe a propósito:

```bash
docker compose exec estimator python scripts/run_graph_s13.py     exercises/session-12/sample_transcript_complex.txt --demo-resume
```

Pase 1 se compila con `interrupt_before=["generate_estimate"]` y se para ahí — como haría una caída
o una puerta de aprobación humana. Pase 2 invoca el **mismo** `thread_id` con `input=None`, que
significa "continúa desde el checkpoint", no "empieza de nuevo":

```
  pass 1 executed nodes     : extract_requirements, classify_components, search_budgets, __interrupt__
  pass 2 executed nodes     : generate_estimate, validate_and_consolidate
  checkpoints after pass 1  : 5
  checkpoints after pass 2  : 7
  components  before/after  : 5 / 5
  budgets     before/after  : 23 / 23
```

Que los recuentos coincidan es la prueba: `extract_requirements` es una llamada a LLM, así que
obtener el mismo número tras reiniciar sólo es posible si el estado se **replicó**, no se re-infirió.

## 5. Observabilidad con Logfire

Un span por nodo, todos anidados bajo un span raíz de la ejecución:

```
19:02:30.299 estimation_graph
19:02:30.320   node.extract_requirements
19:02:47.099   node.classify_components
19:02:59.745   node.search_budgets
19:03:03.033   node.generate_estimate
19:03:03.035   node.validate_and_consolidate
```

La traza completa está en **`example_trace.txt`**, con la demo de resume al final.

**La instrumentación vive en `builder.py`, no en los nodos.** Un único `_instrumented(name, fn)`
envuelve los cinco al añadirlos, así que los nodos siguen siendo funciones puras que no saben nada
de trazas, y `NODE_SEQUENCE` sigue siendo la única fuente de nombres. Cada span lleva el
`estimation_id` y un **resumen** de lo que devolvió el nodo (listas → contadores), nunca el payload:
un span con transcripciones enteras y todos los chunks es ilegible y filtra contenido al backend de
trazas.

No hace falta cuenta de Logfire: `send_to_logfire="if-token-present"` imprime los spans por consola
y, sólo si defines `LOGFIRE_TOKEN`, los manda además a la UI. Sin `logfire.configure()` los spans
son *no-ops* silenciosos, y `[tool.logfire] ignore_no_config = true` en `pyproject.toml` evita el
aviso al ejecutar la app o los tests sin token.

> `logfire.instrument_litellm()` **no** se usa: se probó y no genera ningún span, porque los nodos
> llegan a LiteLLM a través de Instructor (`LLMWrapper.complete_structured`), que esa
> instrumentación no engancha. Los eventos structlog `llm_structured_call_*` ya dan modelo y
> latencia, y salen dentro del span del nodo al que pertenecen.

## Lo que sigue quedando fuera

- **Paralelismo.** `search_budgets` recorre los componentes uno tras otro. El estado ya está
  preparado para el fan-out; falta cambiar las aristas.
- **Endpoint HTTP.** Igual que en S12, el previo se queda en grafo + script.
