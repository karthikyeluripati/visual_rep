# Experimental Protocol (pre-registration, Phase 0)

Frozen on 2026-11-19 before any VLM output was observed. Changes after this date must be logged in
§10 with the date and the reason, and must not be motivated by test-set results.

## 1. Benchmark: RelBind-Bench

**Generation.** Procedural scenes, each with a relation graph built *before* rendering
(`src/relbind/generation/`). Three families × two templates:

| family | templates | entity | annotation | connector |
|---|---|---|---|---|
| schematic | elevation, plan | column lines / column markers (repeated, identical) | dimension values (e.g. 4'-0") | drafting leader, arrowhead |
| flowchart | tb, lr | process boxes S1..Sn | branch labels at decision exits | branch arrow |
| circuit | ladder, grid | resistors / capacitors with designators | component values | thin leader, dot |

Route classes (topology): `straight`, `elbow` (shoulder leader / orthogonal routing),
`zigzag` (two-bend dog-leg / oblique). Fonts: DejaVu sans, mono, serif (vendored).
Distractors: level lines + confusable level text, grid lines + bubbles, unlinked notes, flow arrows,
rails, wires, junction dots.

**Verification** (`src/relbind/verify.py`) — a scene is accepted only if: each connector starts at its
own annotation and ends on its own target within 2 px, ≥ 16 px from every other entity; crossings with
other strokes are at ≥ 30° and ≥ 16 px from any endpoint; non-crossing strokes are ≥ 7 px apart;
connectors never touch other text; texts never overlap each other or entities; each annotation has
exactly one relation, each entity at most one, values unique. Rejection sampling records yields.

**Counterfactual groups.** Each group = 4 images of one base scene:
- `orig`: base scene.
- `cf`: connectivity swap of two annotations (a1→e1, a2→e2 ⇒ a1→e2, a2→e1). Text, entities,
  annotation positions, distractors, terminator types are identical; terminator **positions** are
  identical when feasible (`attach_fixed=True`, the strict variant), otherwise re-sampled on the
  same two targets (logged; analysed as a stratum). A pixel check asserts that `orig` and `cf` differ
  only inside a 2-px dilation of the swapped connectors (zero tolerance).
- `reroute` (negative control): every connector gets a different path shape (route class drawn from
  the split's own topologies) to the same endpoint; relations unchanged; pixel check as above.
- `style` (negative control): identical geometry; different ink, background tint, stroke widths,
  terminator glyph, dash pattern.

**Queries per group** (asked on all 4 images): `a2e` for a1, a2; `e2a` for e1, e2; `e2a` for one
unannotated entity (gold `NONE`). So 20 examples per group, of which 16 flip under the swap.

**Splits** (`configs/prototype.json`, frozen). Seeds are split-specific; groups never straddle splits.

| split | families | templates | topologies | fonts |
|---|---|---|---|---|
| train / val / test_id | schematic, flowchart | elevation, tb | straight, elbow | sans, mono |
| test_unseen_template | schematic, flowchart | plan, lr | straight, elbow | sans, mono |
| test_unseen_topology | schematic, flowchart | elevation, tb | zigzag | sans, mono |
| test_unseen_font | schematic, flowchart | elevation, tb | straight, elbow | serif |
| test_cross_domain | circuit | ladder, grid | straight, elbow | sans, mono |

Each OOD split changes exactly one factor relative to train. Leakage audit
(`src/relbind/splits.py`): split-rule conformity, group straddling, identical images, seed overlap,
cross-split near-duplicates (16×16 dHash), recurring relation signatures (reported, expected to be
rare but non-zero because value pools are finite).

**Real-world validation (planned, Phase 1b).** Candidate sources: AI2D-RST connectivity layer
(CC-BY-4.0), Enginuity callout diagrams (CC-BY-4.0), DRAGON Circuit-VQA/AI2D subsets. An adapter is
written only for link types verified to be annotation→target connector relations; target ≥ 150
human-verified items; never used for development or selection.

## 2. Human audit (before Gate 2 is evaluated)

`python -m relbind.viz audit` samples items across splits and variants into a blind sheet (no
answers). Two reviewers answer independently. Pass criterion: reviewer accuracy ≥ 98% and ≤ 2%
items flagged ambiguous, per split (`python -m relbind.validation score`). Failing items are inspected
and the verifier tightened; the test splits are then regenerated *before* any model is evaluated.

## 3. Diagnostic conditions (Phase 1, inference only)

Model: `Qwen/Qwen3-VL-4B-Instruct`, revision `ebb281ec70b05090aa6165b016eac8ec08e71b17`, bf16, SDPA,
greedy decoding, default processor resolution (input 1024×≈720 px ≈ 0.74 MP, within default limits),
transformers 5.23.0, torch 2.8.0+cu129, single H100 80GB.

| id | condition | isolates |
|---|---|---|
| A | full image, direct answer | baseline behaviour |
| B | full image, trace-the-line reasoning prompt, ≤768 new tokens | inference-time reasoning |
| C | oracle *pair-region* crop (same crop for all 4 variants), ≤2.5× upsampled | resolution / context selection |
| D | full image + all text strings with boxes, no relations | OCR / text perception |
| E | full image + annotation and candidate boxes, no relations | entity localisation |
| F | full image + gold relation graph as text | reading/format ceiling (sanity) |
| G | heuristics: random, text-prior, proximity, connected-component tracer | shortcut audit |
| H | strong external VLM (optional, if API access is approved) | frontier reference |
| P_read | read the text in a given box (queried annotation; gold target label) | recognition vs association |

**Error taxonomy** for wrong `a2e` answers on `orig`/`cf` (computed per example):
1. *Perception*: P_read wrong for the annotation or the gold target label.
2. *Proximity capture*: predicted entity = proximity-nearest entity ≠ gold.
3. *Stuck/partner capture*: on `cf`, predicted = the `orig` gold (did not follow the rewiring).
4. *Crossing switch*: predicted entity is crossed by the gold connector (path-switching at crossings).
5. *Other association*: none of the above, recognition correct.
6. *Format/parse*: no parsable answer.

## 4. Gate 2 — problem existence (decision rule, evaluated on `test_id`, condition A)

Let PCA_A, PCA_C, Read be measured with 95% cluster-bootstrap CIs.

- **STOP H1 (no deficit):** PCA_A ≥ 0.90.
- **REVISE (perception-dominated):** Read < 0.90, or ≥ 50% of wrong answers are perception errors.
- **REVISE (resolution/context problem, not binding):** PCA_C − PCA_A ≥ 0.25 and PCA_C ≥ 0.85.
- **REVISE (benchmark):** human-audit failure, or style/reroute invariance of the *heuristics* < 1.0
  (indicates rendering artefacts), or heuristics with no access to connector pixels achieving PCA > 0.
- **GO to Phase 2:** PCA_A ≤ 0.70, Read ≥ 0.95, cf-stuck rate ≥ 0.15, PCA_C ≤ 0.85, audit passes.
- Otherwise: report as inconclusive and discuss; no training compute is spent.

## 5. Phase 2 — supervised comparison (only if Gate 2 = GO)

Same base model, LoRA r=16 on the language model + merger (vision tower frozen in the main runs),
identical number of training images and optimiser steps across arms, 3 seeds each for the final
comparison.

| arm | supervision |
|---|---|
| B3 | answer-only SFT on unpaired examples (each group contributes only one random variant) |
| B4 | structured relation SFT (answer + annotation box + target box + short connector description) on unpaired examples |
| B4-pair | B4 on counterfactually paired batches (orig and cf of a group in the same batch), same example count |
| B5 | B4-pair + pairwise relation objective: for each query, the log-likelihood margin between the gold answer and the counterfactual partner's answer (hard negative), plus margins vs. proximity-nearest and random distractors (hinge, margin m tuned on `val`) |
| ablations | B5 without counterfactual negatives (only proximity/random negatives); B5 with negatives but unpaired data |

B5 is a *discriminative post-training objective on the VLM's answer likelihood*, not a separate
representation-learning module; it will be described as such. A relation-scoring head over pooled
multimodal features is a separate optional arm only if B5 shows signal.

**Gate 3:** the best of B4-pair/B5 beats B4 by ≥ 5 pp PCA on the pooled held-out split
(`test_unseen_topology` ∪ `test_unseen_template`), with the 95% CI of the paired difference excluding 0
in each of 3 seeds, and is not worse than B4 by > 2 pp on `test_id`. Else stop method claims;
report the benchmark/diagnostic paper.

**Gate 4 (transfer):** improvement on `test_cross_domain` and on the real-world set, CI excluding 0.

## 6. Phase 3 — RL (only if Gate 3 passes)

R0 best SFT; R1 GRPO with answer reward; R2 GRPO with relation reward; R3 B5 → R2. Rewards
(per rollout y for query q on image I):

- r_ans = 1[norm(answer(y)) = norm(gold)]
- r_loc = ½·1[IoU(box_ann(y), gold_ann) ≥ 0.5] + ½·1[IoU(box_ent(y), gold_ent) ≥ 0.5]
- r_abs = +1 if gold = NONE and answer = NONE; −1 if gold ≠ NONE and answer = NONE
- r_pair (R2 only, computed per group): +1 only if the rollouts for the same query on `orig` and
  `cf` are both correct (sampled jointly in one GRPO group)
- R1 = r_ans + r_abs; R2 = r_ans + 0.5·r_loc + r_abs + 0.5·r_pair. No format reward beyond parse
  validity (unparsable ⇒ total 0). Candidate order is randomised in prompts; answers are scored after
  normalisation only (no partial string credit).

Gate 5: R2/R3 beat R0 and inference-time crop+verify (condition C with self-verification, same
number of generated tokens) by ≥ 5 pp PCA on the pooled held-out split. Otherwise RL is dropped from
the paper.

## 7. Statistics

- Resampling unit: group (cluster bootstrap, B = 2000) for every CI.
- Paired model comparisons: exact McNemar on per-query PCA outcomes + cluster-bootstrap CI of the
  difference in percentage points. Report absolute pp differences.
- Multiple comparisons: Holm correction across the pre-registered Phase-2 contrasts
  (B4-pair−B4, B5−B4-pair, B5−B4).
- **Planned sample sizes.** Prototype test splits (40 groups ≈ 160 flipping queries) are for pipeline
  validation only. Phase-1 diagnostic: `test_id` regenerated at 250 groups (1,000 flipping queries).
  With an expected discordance of ~20% between arms, 1,000 paired queries give > 90% power for a
  5 pp difference at α = 0.05 (McNemar, normal approximation); 250 groups give a PCA CI half-width of
  about ±4 pp at PCA≈0.5 assuming an intra-group correlation of 0.3.
- Final test sets are generated once from a held seed, hashed in the manifest, and evaluated only for
  the pre-registered comparisons. Development uses `val` only.

## 8. Compute plan and budget (H100 80GB; price assumed $2.5/GPU-h until the provider rate is confirmed)

| step | estimate | gate |
|---|---|---|
| Phase 0 CPU prototype (this) | 0 GPU-h | — |
| Phase 1 diagnostic: 6 conditions + P_read on 250-group test_id (≈5k examples/condition; B is ≈8× the cost of A) | 1.5–2.5 GPU-h (≈$4–6) | Gate 2 |
| Phase 2 pilot: B4 LoRA on 4k examples, 1 seed | ~1 GPU-h | gradient/data sanity |
| Phase 2 full: 5 arms × 3 seeds × ~1.5 GPU-h + eval | ~25–30 GPU-h (≈$70) | Gate 3 |
| Phase 3 RL pilot + 4 arms | ~40–60 GPU-h | Gate 5 |

## 9. Threats to validity (tracked)

Synthetic-only signal; generator-specific shortcuts (terminator glyph, stroke width, rejection-sampling
selection bias); answer-format brittleness; resolution confounds (crossings are thin strokes at 1 px
scale); contamination of the base model with similar public diagrams; prompts tuned on test (forbidden).

## 10. Change log

- 2026-11-19: initial version.
