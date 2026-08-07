# Resultados — Golden set y medición (Sesión 10)

Comparativa de las cuatro configuraciones de recuperación sobre un *golden set* de **15
consultas** anotadas a mano, midiendo **precisión@5** y **latencia** por consulta.

## Metodología

- **Golden set**: `evals/golden_retrieval.json` — 15 descripciones de proyecto representativas
  del dominio, cada una anotada a mano con los presupuestos históricos genuinamente relevantes
  (`relevant_budget_ids`). Cubre los 4 sectores (finanzas, ecommerce, salud, industrial) y los 17
  presupuestos del corpus base aparecen como relevantes en alguna consulta.
- **Corpus**: `data/budgets_sample.json`, `chunk_type='budget_component'` (17 presupuestos → 60
  *chunks*, uno por componente).
- **Métrica**: precisión@5 a **nivel de chunk** — un chunk recuperado cuenta como relevante si su
  `budget_id` está en `relevant_budget_ids`. Con un umbral de distancia permisivo (`2.0`) el top-5
  nunca se trunca, así que el **techo** de una consulta es `min(chunks_relevantes, 5) / 5`: un
  presupuesto de 3 componentes topa en **0,6**; uno de 4, en **0,8**; dos o más presupuestos
  alcanzan **1,0**. El techo se reporta por consulta para que un 0,60 sea interpretable (es el
  100 % de lo alcanzable, no un fallo).
- **Latencia**: media de 3 corridas cronometradas por consulta (más 1 de calentamiento descartada).
  El *embedding* de la consulta se hace una sola vez y se excluye del cronometraje; se mide el coste
  que añaden la recuperación y el *reranking*. Es tiempo de pared en CPU y depende de la máquina.
- **Idioma de las consultas**: **inglés**, a propósito. El corpus está en inglés y la columna de
  texto completo usa `to_tsvector('spanish', content)`; consultas en español no casarían léxicamente
  con el contenido inglés (`lexical_hits ≈ 0`) y colapsarían la rama híbrida sobre la vectorial,
  invalidando la comparación. Solo la **tabla de resultados** es en español.

## Tabla comparativa

| Configuración | Búsqueda | Reranking | Precisión@5 | Latencia (ms) |
| --- | --- | --- | --- | --- |
| A | Vectorial | No | 0,81 | 57,5 |
| B | Híbrida | No | 0,79 | 77,8 |
| C | Vectorial | Sí | 0,81 | 7047,1 |
| D | Híbrida | Sí | 0,81 | 7440,9 |

## Precisión@5 por consulta

| Consulta | Techo | A | B | C | D | Relevantes |
| --- | --- | --- | --- | --- | --- | --- |
| Q1 | 1,0 | 1,00 | 0,80 | 0,80 | 0,80 | banca 001+003 |
| Q2 | 1,0 | 1,00 | 0,80 | 1,00 | 1,00 | storefront 005+006+007+017 |
| Q3 | 1,0 | 0,80 | 0,80 | 0,80 | 0,80 | telemedicina 009+010 |
| Q4 | 1,0 | 0,80 | 0,80 | 1,00 | 1,00 | telemetría industrial 013+015 |
| Q5 | 1,0 | 1,00 | 1,00 | 1,00 | 1,00 | pagos 003+001 |
| Q6 | 0,8 | 0,80 | 0,80 | 0,80 | 0,80 | lending 002 |
| Q7 | 0,6 | 0,60 | 0,60 | 0,60 | 0,60 | robo-advisor 004 |
| Q8 | 0,6 | 0,60 | 0,60 | 0,60 | 0,60 | farmacia 011 |
| Q9 | 0,6 | 0,60 | 0,60 | 0,60 | 0,60 | ensayos clínicos 012 |
| Q10 | 0,6 | 0,60 | 0,60 | 0,60 | 0,60 | devoluciones moda 008 |
| Q11 | 1,0 | 1,00 | 1,00 | 1,00 | 1,00 | PHI salud 009+012 |
| Q12 | 1,0 | 1,00 | 1,00 | 1,00 | 1,00 | logística 007+014 |
| Q13 | 0,8 | 0,80 | 0,80 | 0,80 | 0,80 | marketplace 006 |
| Q14 | 0,6 | 0,60 | 0,60 | 0,60 | 0,60 | monitorización remota 010 |
| Q15 | 1,0 | 1,00 | 1,00 | 1,00 | 1,00 | core banking 001+003+016 |

## Análisis

- **La híbrida sin rerank (B) queda ligeramente por debajo de la vectorial (A)** en agregado
  (0,79 vs 0,81). La fusión RRF introduce algún chunk que casa léxicamente pero es distractor en el
  top-5: se ve en **Q1** (1,00 → 0,80) y **Q2** (1,00 → 0,80), donde añadir la rama léxica expulsa un
  chunk relevante. Es el mismo patrón que ya vimos: en este corpus el señal denso ya es fuerte y el
  léxico aporta ruido tanto como señal.
- **El reranker recupera la híbrida** (D, 0,81 = A) reordenando esos candidatos, y **cambia qué
  consultas ganan o pierden** en lugar de subir la media: **Q4** sube de 0,80 a **1,00** con rerank
  (C y D) — el cross-encoder flota el chunk industrial correcto —, mientras **Q1** se queda en 0,80
  bajo rerank. En este corpus pequeño el reranker no supera la precisión agregada de la vectorial
  base; su valor aquí es **estabilizar** (rescata la caída de la híbrida) más que mejorar el techo.
- **Coste**: el reranking multiplica la latencia unas **~120×** (≈57 ms → ≈7047 ms). Es el
  cross-encoder puntuando 50 pares por consulta en CPU. Este es el compromiso coste/calidad central
  de la decisión: sin ganancia agregada en este dataset, el rerank solo se justifica si el reordenado
  por consulta (p. ej. Q4) importa para el caso de uso.
- **Techos por diseño**: Q7–Q10 y Q14 tienen techo 0,6 (un único presupuesto de 3 componentes); su
  0,60 es el **máximo alcanzable**, no un error de recuperación — todas las configuraciones lo tocan.
  **Q3** (techo 1,0) se queda en 0,80 en las cuatro: ninguna configuración logra colar el quinto
  chunk relevante de telemedicina en el top-5.

## Reproducción

Con el *stack* levantado (`estimator-postgres` + `estimator-redis`), el corpus base ingerido
(`uv run python scripts/query_examples.py`) y `OPENAI_API_KEY` en `.env`:

```bash
cd estimator
REDIS_URL=redis://localhost:6379 uv run python scripts/eval_retrieval_s10.py
```

La primera corrida con *reranking* descarga los pesos del cross-encoder
(`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`); verificar antes con
`python -m app.generation.rag.retrieval.verify_reranker`. La tabla comparativa y el desglose por
consulta se imprimen en español directamente desde el *script* (no se transcriben a mano).

## Conclusiones

**¿Por qué ampliamos el golden set a 15 muestras?** Porque con 5 consultas cada una pesa 0,20 en la
media, así que una diferencia agregada como 0,79 (B) frente a 0,81 (A) cabe entera dentro del ruido de
una sola consulta y no permite decidir nada. Quince reduce la varianza del estimador, cubre los cuatro
sectores y hace aparecer los 17 presupuestos como relevantes en alguna consulta, con distractores
cross-sector deliberados (p. ej. Q14: *sensor telemetry* industrial vs. *wearable ingestion* clínico)
que son justo el modo de fallo que la búsqueda híbrida y el reranking dicen atacar; además reparte los
techos entre 0,6 y 1,0 para no medirlo todo en saturación. Dicho esto, 15 sigue siendo una muestra
pequeña: sirve para decidir con criterio, no como prueba estadística concluyente.

**¿Qué configuración usaríamos?** Como *baseline* de producción, **A (vectorial pura)**: iguala en
precisión@5 agregada a C y D (0,81) con la mínima latencia (~57 ms) y la mínima complejidad, y la
híbrida sin rerank (B) se descarta directamente porque es *peor* en precisión y *más* lenta que A. Ahora
bien, mantendríamos el **reranking como opción activable** (ya es un *toggle* sin tocar código) en
lugar de eliminarlo: su valor no se aprecia con 60 chunks pero crece con el corpus. La postura contraria
también es defendible: en el flujo real `estimate_from_transcript`, que ya espera segundos por la
generación del LLM, los ~7 s del reranker son marginales y su reordenado por consulta (Q4 sube de 0,80 a
1,00) reduce el riesgo de anclar la estimación en un presupuesto irrelevante; con ese criterio, dejar C
o D activado por defecto se justifica.

**¿La ganancia de relevancia del reranking justifica su latencia en este caso?** En este dataset y por
precisión@5 agregada, **no**: 0,81 con y sin reranker, pagando ~120× de latencia (≈57 ms → ≈7047 ms)
por cero ganancia media; el reranker solo redistribuye qué consultas ganan o pierden, no eleva el techo.
El matiz honesto es doble: (1) un corpus de 60 chunks infravalora al reranker, cuyo beneficio aumenta
con el tamaño del *recall* y el número de distractores densos que la etapa barata deja pasar; y (2) esos
7 s son de un cross-encoder en CPU y su coste *relativo* se diluye dentro de un pipeline que ya invoca un
LLM lento. Conclusión: no lo activaríamos por defecto con estos datos, pero lo dejaríamos disponible y
volveríamos a medir en cuanto el corpus crezca.
