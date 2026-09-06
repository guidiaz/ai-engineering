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

## Lo que queda fuera de este primer paso

Está instalado y decidido, pero **no cableado todavía**, a propósito:

- **Checkpointing** (`langgraph-checkpoint-postgres` sobre el Postgres de pgvector del proyecto).
  Persistir cambia el contrato de invocación —toda llamada pasa a necesitar un `thread_id`— y añade
  una migración de tablas. La costura está lista: `compile()` acepta `checkpointer=`.
- **Paralelismo.** `search_budgets` recorre los componentes uno tras otro. El estado ya está
  preparado para el fan-out; falta cambiar las aristas.
- **Endpoint HTTP.** Igual que en S12, el previo se queda en grafo + script.
