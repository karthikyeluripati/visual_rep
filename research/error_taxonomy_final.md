# Final Error Taxonomy (Phase 1, test_id, Qwen3-VL-4B @ ebb281ec)

Source: `results/phase1/gate2_analysis.json` (`categories`) and `results/phase1/final_analysis.json`.
Counts are over orig + cf examples. Category definitions are in `src/relbind/gate2.py::classify`.
**Categories label the wrong answer post hoc; they are not observations of the model's internal process.**
Observations and hypothesised causes are kept separate below.

## 1. Primary categories (mutually exclusive, precedence order)

| category | definition |
|---|---|
| invalid_output | answer is not one of the candidate ids (e.g. returns decision text "auth?") |
| recognition | the crop probe misread the annotation text or the gold target's label |
| unsupported_relation | a2e: predicted entity has **no** annotation at all. e2a: names a non-annotation text, or any text when gold is NONE |
| missed_relation | e2a: answers NONE although a relation exists |
| connector_interpretation | the predicted entity is the end of *another* connector, or an entity the gold connector crosses |
| proximity_binding | e2a only: the predicted annotation is the one nearest to the entity, and not a stuck answer |
| other | none of the above |

## 2. Counts

### a2e (annotation → entity), 1000 examples

| category | A direct | B reasoning | E oracle boxes |
|---|---|---|---|
| wrong / total | 457 / 1000 | 392 / 1000 | 450 / 1000 |
| connector_interpretation | 270 (59%) | 266 (68%) | 266 (59%) |
| unsupported_relation | 150 (33%) | 78 (20%) | 148 (33%) |
| recognition | 34 (7%) | 24 (6%) | 35 (8%) |
| invalid_output | 3 (1%) | 24 (6%) | 1 (0%) |
| tags (overlapping): proximity-nearest / crossed by gold / stuck on old target | 241 / 146 / 88 | 171 / 72 / 99 | 243 / 127 / 86 |

### e2a answerable (entity → value), 1000 examples

| category | A direct | B reasoning | E oracle boxes |
|---|---|---|---|
| wrong / total | 467 / 1000 | 498 / 1000 | 405 / 1000 |
| proximity_binding | 166 | 134 | 167 |
| connector_interpretation | 153 | 147 | 167 |
| unsupported_relation (non-annotation text) | 100 | 103 | 30 |
| recognition | 35 | 30 | 29 |
| missed_relation (said NONE) | 13 | 84 | 12 |

### e2a unanswerable (entity with no annotation; gold NONE), 500 examples

| | A | B | E |
|---|---|---|---|
| wrong (named something) | 302 | 248 | 271 |
| named an annotation linked to another entity | 248 | 209 | 247 |
| named non-annotation text (e.g. "N.T.S.") | 54 | 39 | 24 |

## 3. The four requested groups (A direct, a2e+e2a answerable, orig+cf, n = 2000; wrong = 924)

| group | count | share of errors | basis |
|---|---|---|---|
| Recognition failures | 69 | 7.5% | crop probe |
| Connector interpretation failures | 423 | 45.8% | wrong answer is another connector's end or a crossed entity |
| Annotation-to-entity binding failures (proximity prior, e2a) | 166 | 18.0% | nearest annotation chosen, no connector support |
| Unsupported relation predictions | 150 + 100 = 250 | 27.1% | a2e: entity that has no leader at all (150); e2a: non-annotation text (100) |
| invalid / missed | 16 | 1.7% | |
| (unanswerable items, not in the 924) | 302 of 500 | — | a link asserted for an entity that has none |

Overlap warning: 241 of the a2e connector/unsupported errors are *also* the proximity-nearest entity.
"Connector interpretation" and "proximity binding" are not separable from the answer alone (§4).

## 4. Observations vs hypothesised causes

Observations:
- O1. Errors are rare without crossings (a2e pair accuracy 0.77, n = 43) and dominant with leader–leader
  crossings (0.15, n = 211) and in held-out zigzag routes (0.08, n = 120).
- O2. On counterfactual images, 88 a2e errors repeat the original target exactly ("stuck").
- O3. When the target is the nearest entity in neither image, pair accuracy is 0.13; when it is the nearest
  in one image, 0.39.
- O4. Zoom, OCR, boxes and reasoning do not change the pattern (PCA within 0.24–0.28).
- O5. With oracle relations stated, a2e errors appear only when the image is present (F vs blank image:
  −17.6 pp), and 90 of 99 match the model's no-oracle answer.

Hypotheses (not established by these data):
- H-a. The model does not follow a thin line through crossings. It switches to a locally similar path,
  consistent with "local competition" reported for path following. Testable with the path-following control
  (gate2 report §8.3a).
- H-b. Without a confident trace, the model falls back to a proximity prior. Supported indirectly by O3 and the
  proximity tag rate. Not separable from H-a with answer-level labels.
- H-c. The visual estimate is held with enough confidence to override stated relations (O5). Whether the
  connectivity information is present but mis-read, or absent from the representation, is unknown. Testable
  with the frozen-feature linkage probe.
