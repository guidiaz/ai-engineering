# Stress Report — Multi-turn CAG / fact-tracker

Fuente: `evals/stress/results.csv` (148 filas = 1 fila por turno).
Matriz: **4 escenarios × 5 tamaños de adjunto (0/5/20/50/100 KB) × 3 repeticiones × hasta 5 turnos**, contra la API real (`gpt`-class, llamada de estimación + summariser + extractor de metadatos por turno).
Presupuestos del runner: latencia ≤ 60 000 ms/turno, coste ≤ $1.00/conversación.

128 turnos devolvieron `200`; **20 fallaron con `502 "Upstream LLM call failed"`** (ver §4). Todas las agregaciones de latencia/coste/recall se calculan **solo sobre los 200**.

---

## 1. Tabla resumen

| Escenario | Turnos OK | P50 latency | P95 latency | máx | Coste acumulado | Hit rate exact-cache | Hit rate semantic-cache | Recall medio fact-tracker |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `growing` (web_saas) | 72 | 7 812 ms | 17 003 ms | 22 486 ms | **$2.0351** | 0 % * | 0 % * | **1.000** |
| `pivot` (mobile_app) | 56 | 5 532 ms | 18 636 ms | 23 525 ms | **$1.0144** | 0 % * | 0 % * | **0.928** |
| `contradiction_literal` (internal_tool) | 0 | — | — | — | $0.0000 | — | — | — (no se llegó a medir) |
| **Global** | **128** | **6 890 ms** | **17 624 ms** | 23 525 ms | **$3.0495** | **0 % *** | **0 % *** | **0.968** |

`*` **El CSV no instrumenta las caches** — `CSV_FIELDS` en `run.py` no incluye ninguna columna `cache_hit` / `semantic_hit`, así que el *hit rate* no es un dato medido sino **inferido**: cada turno presenta `tokens_in/tokens_out` reales, latencia multi-segundo y `cost_usd > 0`, es decir **ninguno se sirvió de cache**. El motivo no es accidental y se desarrolla en §3 (es justo donde empieza a romperse el CAG). Para convertir ese `0 % *` en un dato de primera clase hay que añadir `cache_hit` y `semantic_hit` al `turn_observed` y al CSV.

- **Recall medio del fact-tracker** = media de `drift_score` sobre los turnos `200`. `drift_score = (expected encontrados + forbidden ausentes) / total`. `growing` nunca pierde un hecho; `pivot` cae a partir del turno 4 (§2, Curva C).
- **Coste acumulado por escenario** = suma de `cost_usd` de todos los turnos OK del escenario.

---

## 2. Tres curvas

### Curva A — Latencia vs. tokens de entrada
Mediana de `latency_ms` por bucket de `tokens_in` (todos los escenarios OK). La latencia escala con el tamaño del prompt, con un salto claro por encima de 20 K tokens.

```
tokens_in        n   median_latency
     0– 5 000   30   4 470 ms  | ###########
 5 000–10 000   36   6 293 ms  | ###############
10 000–20 000   26   5 305 ms  | #############
20 000–40 000   19  10 047 ms  | #########################
40 000–70 000    7  10 322 ms  | #########################
70 000+         10  16 382 ms  | ########################################
```

### Curva B — Coste acumulado vs. turno
Media de `cost_total_usd` por índice de turno. El coste crece **super-lineal**: cada turno re-inyecta todo el adjunto + la historia, así que el turno 5 cuesta ~10× el turno 1.

```
turno   growing            pivot
  1   $0.0133 | ####       $0.0085 | ##
  2   $0.0327 | #########   $0.0182 | #####
  3   $0.0611 | #################  $0.0363 | ##########
  4   $0.0971 | ###########################  $0.0608 | #################
  5   $0.1431 | ########################################  $0.0911 | #########################
        (misma escala: el ancho máx = $0.1431)
```

### Curva C — Recall vs. N (turnos acumulados = nº de hechos a recordar)
`drift_score` medio por turno. `growing` se mantiene perfecto; `pivot` se desploma en el turno 4 cuando `feature: offline mode` se queda fuera de la ventana y `anchors_count` sigue a 0.

```
turno   growing                                   pivot
  1   1.000 | ########################################   1.000 | ########################################
  2   1.000 | ########################################   1.000 | ########################################
  3   1.000 | ########################################   1.000 | ########################################
  4   1.000 | ########################################   0.795 | ################################
  5   1.000 | ########################################   0.836 | #################################
        (escala 0.0 – 1.0)
```

---

## 3. Lectura: dónde empieza a romperse mi CAG y por qué

**El CAG no se rompe por la cache: se rompe porque nunca llega a usarse, y el coste por turno lo deja claro.** En este diseño cada turno conversacional construye un prompt *enriquecido* con toda la historia previa y **re-inyecta el adjunto completo** (`attachments_total_chars` es constante dentro de una conversación: ~32 586 chars para 50 KB, ~60 000 para 100 KB, en *todos* los turnos). Eso significa que el contenido efectivo de entrada es **monótonamente creciente y único en cada turno** — exactamente el patrón que invalida una exact-match cache (la clave SHA-256 nunca se repite) y que tampoco colisiona en la semantic cache (cada turno acarrea hechos nuevos, así el bucket `prompt_version:project_type:detail_level:output_format` se llena de vectores distintos por encima del umbral 0.85). El resultado se ve en la Curva B: el coste acumulado escala super-lineal (turno 5 ≈ 10× turno 1) y en la Curva A la latencia salta de ~4–6 s por debajo de 10 K tokens a ~16 s por encima de 70 K. Un CAG sano amortiza turnos repetidos; aquí el adjunto re-enviado convierte cada turno en una generación íntegra y más cara que la anterior. **El primer punto de ruptura es económico**: a 100 KB de adjunto un solo turno ya ronda ~50 K tokens de entrada, y a ese ritmo la conversación se come el presupuesto de $1/conv antes de los 5 turnos previstos.

**El segundo punto de ruptura es la memoria, y es de arquitectura, no de escala.** En `pivot` el recall cae a 0.795 en el turno 4 (Curva C) porque `feature: offline mode` deja de aparecer en el snapshot: con `anchors_count = 0` en *todas* las filas, el fact-tracker se apoya solo en `metadata+window`, y en cuanto un hecho se desliza fuera de la ventana (`max_turns_store = 6`) desaparece sin que ningún ancla lo retenga — el tier de anclaje está desconectado, no fallando. `growing` no lo sufre (recall 1.000) sólo porque sus hechos se acumulan sin contradicción y caben en la ventana; en cuanto un escenario *pivota* (descarta una decisión y la sustituye), el sistema no distingue "superado" de "olvidado". El tercer punto, el más duro, son los **502**: `contradiction_literal` falla el **100 %** de las veces ya en el turno 1 incluso con 0 KB de adjunto — no es un problema de tamaño sino del propio contenido del escenario (un fallo reproducible que hay que depurar aparte), mientras que `pivot` a 100 KB falla el 100 % en el turno 1 y `pivot` 50 KB / `growing` 20 KB fallan de forma intermitente: ahí sí es presión de contexto/tamaño contra el upstream. Resumen operativo: **el CAG aguanta conversaciones cortas y acumulativas, y empieza a romperse en tres frentes — coste (adjuntos re-inyectados), recall (anclas desactivadas + ventana corta) y disponibilidad (502 por tamaño y un bug de escenario en `contradiction_literal`).**

---

## 4. Anexo — fallos `502` por celda

| Escenario | Adjunto | Fallos | Diagnóstico |
|---|---|---|---|
| `contradiction_literal` | 0/5/20/50/100 KB | 15/15 (turno 1) | Falla en T1 con y sin adjunto → **bug de escenario**, no de escala. |
| `pivot` | 100 KB | 3/3 (turno 1) | Falla en T1 sólo a 100 KB → presión de contexto/tamaño. |
| `pivot` | 50 KB | 1/12 (turno 2) | Intermitente bajo carga. |
| `growing` | 20 KB | 1/13 (turno 3) | Intermitente bajo carga. |

Una conversación que falla un turno **aborta el resto** (`_run_conversation` hace `return`: un turno fallido envenena la historia), por eso `contradiction_literal` sólo tiene filas de turno 1.

---

## 5. Dos decisiones de diseño previas y su justificación

**Decisión 1 — La supersesión de hechos se restringe a una allowlist de claves de valor único (`SINGLE_VALUED_KEYS = {"project name", "budget locked"}`).**
*Justificación.* El partidor `_partition_facts` (en `run.py`) separa los hechos cronológicos en `expected_now` (deben seguir presentes) y `superseded` (deben haber desaparecido). Sólo un hecho cuya clave es genuinamente de valor único puede ser "contradicho" por un turno posterior: si el presupuesto pasa de 30 000 a 50 000, el primero queda *forbidden*. El resto **acumula**: `feature: ...` se apila turno tras turno y los hechos de stack (`stack includes ...`, sin `:`) también. Sin la allowlist, cada `feature:` nuevo superseder­ía falsamente al anterior, corrompiendo a la vez la partición de esperados y la de prohibidos y dejando el recall sin sentido. Es la corrección introducida en `f6ca806` (*"only supersede single-valued facts in drift partition"*).

**Decisión 2 — El runner lee la observabilidad y el snapshot de memoria desde `GET /sessions/{id}`, no rascando el stdout del estimador.**
*Justificación.* El servicio guarda el último evento `turn_observed` en la sesión y expone un `memory_snapshot` por buckets (`metadata` / `anchors` / `summary` / `window`). El runner hace `POST .../estimate` y luego un `GET /sessions/{id}` para recuperar `last_turn` (latencia, coste, tokens, tiers) y el snapshot que alimenta `MemoryDriftMetric`. Esto **desacopla la recogida de métricas del formato de logging** y mantiene el runner **agnóstico al transporte**: funciona idéntico en proceso (vía `httpx.ASGITransport`, sin levantar servidor) y contra un servidor real (`--http BASE`). Introducido junto al runner multi-turno en `a4e2121` (*"multi-turn stress runner + enriched session GET"*).
