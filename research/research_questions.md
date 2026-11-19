# Research Questions, Hypotheses and Task Definition

Status: Phase 0 (pre-registration draft, frozen before any model is run). Revised framing per
`novelty_audit.md` §6: *"Do VLMs bind annotations by connectivity or by prior?"*

## 1. Central question

When the text and the candidate entities in a diagram stay fixed and **only the connector that links
an annotation to an entity changes**, does a vision-language model change its answer accordingly? And
can post-training make a model follow connectivity *causally* (under held-out templates and
topologies), rather than fitting the layout priors of the training generator?

## 2. Hypotheses (independent; none is evidence for another)

**H1 — Relational binding deficit.** An open-weight VLM (Qwen3-VL-4B-Instruct) makes annotation→entity
binding errors on diagrams whose text and entities it can read, and these errors are *connectivity
insensitive*: under a connectivity swap the prediction often does not move.
- Operationalisation: on `test_id`, condition A, paired counterfactual accuracy (PCA) is low
  and the cf-stuck rate is high **while** text-recognition accuracy on the same items is high
  (probe P_read), and the oracle-crop condition C does not remove the deficit.
- H1 is *not* supported by a low raw accuracy alone. See the diagnostic gate in `experimental_protocol.md` §4.

**H2 — Relational learning.** Training with explicitly supervised positive and hard-negative
annotation–entity relations improves PCA on **held-out templates and topologies**, beyond
answer-only fine-tuning at matched data and compute.
- Sub-question H2a (data): does training on *counterfactually paired* examples (both members of a swap
  pair in the same batch) improve held-out PCA over the same number of unpaired examples?
- Sub-question H2b (objective): does an explicit pairwise/contrastive relation objective add
  anything over relation-structured SFT on the *same* paired data?

**H3 — Grounding-aware post-training (conditional).** RL with verifiable relational rewards improves
held-out PCA beyond (i) the best supervised model from H2 and (ii) inference-time crop/verification at
a matched inference budget. Only tested if Gate 3 passes.

## 3. Task definition

**Input:** a diagram image I (rendered raster), a query q of one of two types:
- `a2e`: "Which ⟨entity noun⟩ does annotation ⟨value⟩ point to?" (answer: entity id)
- `e2a`: "Which annotation is attached to ⟨entity id⟩?" (answer: annotation value, or `NONE`)

Optional (separately labelled diagnostic protocols only): oracle text boxes (D), oracle
annotation/candidate boxes (E), oracle relation graph (F). Gold edges are **never** given at inference
in any non-oracle condition.

**Output:** JSON `{"answer": ..., ["annotation_box": [...], "entity_box": [...]]}`; boxes in 0–1000
relative coordinates (Qwen-VL convention), converted to canonical canvas coordinates for scoring.

**Recognition vs. association.** An example is a *relational grounding error* if the model's answer
is wrong while, on the same image, it correctly reads both the queried annotation text and the gold
target's label (probe P_read). If it misreads either, the error is a *perception error*.

**Abstention.** `e2a` queries on entities without an annotation have gold answer `NONE`. In this
prototype, every *answerable* query has exactly one verified answer; "ambiguous" items are excluded
by construction (verifier) and checked by blind human audit.

## 4. Ground-truth representation

Each scene JSON (`data/<build>/<split>/scenes/<scene_id>.json`) contains: image id, family (domain),
template, topology (route class), font, seed, canvas size, coordinate convention, entity nodes
(stable ids, bboxes, geometry), annotation nodes (stable ids, text, bboxes), connector polylines with
terminator type, directed relations `annotation → entity` (with connector id), distractors, and
`meta` (group id, variant, swapped annotation pair, whether terminator positions were held fixed).
Coordinates: canonical canvas pixels, origin top-left, x right, y down, bbox `[x0,y0,x1,y1]`.

## 5. Primary outcome and comparisons (pre-registered)

- **Primary outcome:** paired counterfactual accuracy (PCA): fraction of flipping queries answered
  correctly on *both* the original and the counterfactual image. Any strategy that ignores connector
  pixels scores exactly 0 (tested in `tests/test_build_pipeline.py`).
- **Secondary:** acc_orig, cf-stuck rate, conditional flip accuracy, invariance under style and
  reroute controls, abstention accuracy, false-NONE rate, localisation hit@0.5, all stratified by
  crossings, connector length, proximity rank of the target, entity count, text density.
- **H1 primary split:** `test_id`. **H2/H3 primary split:** `test_unseen_topology` ∪
  `test_unseen_template` (pooled, pre-registered); `test_cross_domain` and real-world sets are
  secondary transfer tests and are never used for selection.
- **Minimum practically meaningful improvement:** +5 percentage points PCA (absolute).
