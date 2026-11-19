# Gate 2 — Final Report (Phase 1 diagnostic)

Date: 2026-11-19. Status: **decision = REVISE** (no training authorised or run).
Human audit: **PENDING**. Benchmark validity is therefore classified as **pending**.

All numbers below are recomputed from preserved artifacts by `python -m relbind.final_analysis`
(output: `results/phase1/final_analysis.json`) and `python -m relbind.gate2`
(`results/phase1/gate2_analysis.json`). Format: value [95% CI] (numerator/denominator). CIs are cluster
bootstraps over **generation groups**. A group is one base scene plus its cf/reroute/style variants and all of
its queries, so counterfactual pairs are never split. B = 2000 unless noted.

## 1. What was evaluated

| item | status | artifact |
|---|---|---|
| Diagnostic set diag_v1: 430 groups (test_id 250, unseen template 60, unseen topology 60, cross-domain 60) | done; all examples.jsonl SHA-256 match the manifest; leakage audit ok (0 near-duplicates, min cross-split dHash distance 6) | `data/builds/diag_v1/` |
| Qwen3-VL-4B-Instruct @ `ebb281ec`, conditions A (direct), C (oracle crop) on all splits; B (reasoning), D (OCR), E (boxes), F (oracle graph) on test_id | done | `experiments/phase1_diag/*/` |
| F_text_only (F prompt + blank image), test_id | done (audit control, added Phase 1b) | `experiments/phase1_diag/F_text_only/` |
| P_read v0 (pixel-coordinate boxes) | **invalid**, kept, not used | `experiments/phase1_diag/P_read_v0_INVALID_pixelcoords/` |
| P_read corrected: crop (pure recognition) and box (0–1000 locate-and-read) modes | done, all splits | `experiments/phase1_diag/P_read/` |
| Heuristics: proximity, text prior (fitted on prototype train), connected-component tracer, ray tracer, all on all diag_v1 splits | done | `results/b0_diag/` |
| Two-reviewer blind audit package (172 items: 71 orig/cf pairs + 30 controls) | **prepared, not performed** | `data/validation/audit_diag_v1/` |
| Pod | results pulled and checksum-verified, then pod stopped; `EXITED` confirmed via RunPod API | `experiments/pod_logs/` |

Total GPU time for Phase 1: about 1.6 h at $4.39/h (≈ $7).

## 2. Recognition vs grounding (test_id unless noted)

| quantity | estimate |
|---|---|
| A. Text recognition: annotation text, crop probe | **0.954** [0.934, 0.972] (477/500) |
| A'. Annotation text, box probe (locate by 0–1000 box, then read) | 0.454 [0.406, 0.500] (227/500) |
| B. Entity recognition: entity *label*, crop probe | **1.000** [1.000, 1.000] (500/500) |
| B'. Entity label, box probe | 0.232 [0.194, 0.270] (116/500) |
| B''. Visual recognition of the entity's shape/stroke | **unavailable** (not probed) |
| C. Relational accuracy (a2e, orig+cf) given annotation text **and** gold-target label read correctly | **0.539** [0.508, 0.571] (514/954) |
| D1. End-to-end a2e accuracy (orig+cf) | 0.543 [0.514, 0.572] (543/1000) |
| D2. End-to-end value extraction (e2a answerable, orig+cf) | 0.533 [0.503, 0.561] (533/1000) |
| E. Pair accuracy (a2e) given recognition of the annotation and both targets | **0.260** [0.217, 0.302] (124/477) |
| Connector interpretation (did the model trace the connector?) | **unavailable** as a measurement; only post-hoc labels of wrong answers exist |

Caveats: probes ran on original images only. E uses original-image recognition as a proxy for the
counterfactual image, whose text pixels are identical (verified by the build's zero-tolerance pixel check).
Box-mode reading is poor: the model cannot reliably use coordinates in prompts. So condition E's
null effect does **not** show that localisation is irrelevant. It shows that the oracle boxes, as given, were not usable.

**Conclusion.** Recognition explains very little. In 228 of 250 groups all relevant texts are read correctly,
and on exactly those groups pair accuracy is 0.236 [0.202, 0.271], indistinguishable from the overall 0.241.

## 3. Headline results (test_id)

Primary pre-registered metric: paired counterfactual accuracy (PCA) over all flipping queries (a2e + e2a).

| condition | PCA | acc orig | cf-stuck rate |
|---|---|---|---|
| A direct | 0.241 [0.208, 0.274] | 0.594 | 0.512 |
| B reasoning (≈8× tokens) | 0.270 [0.237, 0.303] | 0.612 | 0.434 |
| C oracle pair-region crop | 0.259 [0.223, 0.293] | 0.602 | 0.560 |
| D oracle OCR | 0.247 [0.212, 0.282] | 0.588 | 0.534 |
| E oracle boxes | 0.283 [0.247, 0.322] | 0.624 | 0.526 |
| F oracle relations in prompt | 0.659 [0.622, 0.694] | 0.840 | 0.181 |
| F_text_only (blank image) | 0.707 [0.676, 0.737] | 0.740 | 0.236 |
| proximity / text prior | 0.000 / 0.000 | 0.285 / 0.062 | 1.0 / 1.0 |
| ray tracer (oracle text & node masks) | see §5 | | |

## 4. Where it fails (a2e pair accuracy, test_id; observations only)

| stratum | A direct | B reasoning | ray tracer |
|---|---|---|---|
| no crossings at all (n=43) | 0.767 [0.644, 0.881] | 0.605 | 0.977 |
| crosses entity lines only (n=210) | 0.305 [0.246, 0.376] | 0.519 | 0.933 |
| the two leaders cross each other (n=211) | 0.152 [0.110, 0.202] | 0.166 | 0.393 |
| leaders cross each other + entity line (n=36) | 0.139 [0.031, 0.243] | 0.083 | 0.889 |
| strict swap (arrowheads identical; n=334) | **0.153** [0.117, 0.189] | 0.192 | 0.575 |
| non-strict swap (n=166) | 0.500 [0.422, 0.579] | 0.657 | — |
| route straight / elbow (test_id) | 0.310 / 0.189 | 0.405 / 0.286 | 0.666 / 0.853 |
| zigzag (unseen-topology split) | 0.083 [0.033, 0.142] (10/120) | n/a | 0.583 |
| target is not the nearest entity in either image (n=236) | 0.131 [0.088, 0.178] | 0.237 | 0.674 |
| target is nearest in one image (n=264) | 0.390 [0.330, 0.456] | 0.443 | 0.735 |
| schematic / flowchart family | 0.352 / 0.152 | 0.479 / 0.162 | 0.938 / 0.386 |

The "distractor strokes > 4" clutter stratum coincides exactly with the flowchart family, so it cannot be
separated from family. Splits, condition A: unseen template 0.233 (28/120), unseen topology 0.083 (10/120),
cross-domain 0.192 (23/120).

**Unlinked entities.** Abstention accuracy for A is 0.396 [0.340, 0.452] (198/500). When the model answers
anyway, it usually names an annotation that is linked to a different entity (492 of 608 non-NONE answers
across orig/cf/style/reroute), and sometimes an unlinked note such as "N.T.S." (87).

**Meaning-preserving controls.** For A, the answer stays the same under a style change in 0.886 (886/1000)
of cases, and accuracy drops by 3.6 pp [1.7, 5.6] (paired, p = 1.6e-4). For B: 0.732 unchanged, −6.6 pp
[4.1, 9.5]. Reroute (same relations, different path shape): A accuracy +3.0 pp [0.5, 5.4] on the reroute
image.

## 5. Tracer vs VLM (same examples, same ground truth, same metric)

Paired differences in a2e pair accuracy (A minus tracer), exact McNemar:

| subset | A direct | ray tracer | diff (pp) | discordant (A only / tracer only) | p |
|---|---|---|---|---|---|
| test_id | 0.268 | 0.706 | −43.8 [−49.6, −38.0] | 26 / 245 | 7.7e-46 |
| test_id, strict pairs | 0.153 | 0.575 | −42.2 [−50.3, −34.1] | 24 / 165 | 4.5e-27 |
| unseen topology | 0.083 | 0.583 | −50.0 [−60.0, −40.0] | 6 / 66 | 7.3e-14 |
| cross-domain (circuit) | 0.192 | 0.975 | −78.3 [−86.7, −70.0] | 2 / 96 | 3.1e-26 |

The tracer uses privileged ground-truth masks: all text boxes and node shapes. It is an
**oracle-assisted raster baseline**, not an image-only competitor. A tracer on the source vector paths
would be correct by construction and is not informative. Its errors are mostly parsing failures
(test_id a2e: 190 "no entity reached" vs 46 wrong entity), concentrated where flowchart arrows cross each
other. Full comparison: `research/tracer_vs_vlm.md`.

## 6. Condition F audit

- **a2e (entity lookup):** with a blank image, the stated relations give 0.996 (498/500) pair accuracy.
  With the diagram it is 0.820 (410/500): −17.6 pp [−21.6, −13.8], 90 vs 2 discordant pairs, p = 1.7e-24. Of
  the 99 wrong a2e answers with the image, 90 equal the model's own no-oracle (A) answer. This is
  **paired evidence of visual–text conflict** for a2e: the image pulls the answer toward the model's own
  (wrong) visual reading.
- **e2a (value lookup, incl. NONE):** text-only pair accuracy is only 0.418 (209/500), and the image *improves*
  it to 0.498 (+8.0 pp [0.8, 15.6]). e2a errors in F are dominated by **task/format misunderstanding**
  (reverse lookup, deciding NONE), not by visual conflict.
- The earlier draft claim that "the image overrides text" generally is withdrawn. It holds for a2e only, under the paired control.

## 7. Benchmark validity

Checks that pass (automated):
- Relations are created before rendering. The verifier enforces unambiguous geometry.
- Every cf/reroute image differs from its original only within the changed connectors (zero-tolerance pixel check, 430/430 groups).
- Text-only and proximity strategies score 0 PCA. Text prior acc_orig 0.062 ⇒ text does not reveal answers.
- Split rules hold, no seed overlap, no near-duplicates, dataset hashes verified.

Open concerns:
1. **Human answerability not established.** The audit package is ready; no reviews have been done.
2. **Two swap regimes.** In strict swaps (arrowhead positions identical, 67%), the swapped leaders almost
   always cross each other. Non-strict swaps (33%) move arrowheads and rarely do. At equal crossing counts the
   tracer also scores higher on non-strict pairs (e.g. 1 crossing: 0.97 vs 0.55), so the regime gap reflects
   crossing type at least partly. A residual arrowhead-position cue cannot be excluded either. Only strict pairs
   are shortcut-proof by construction.
3. **Crossing type confounds family.** Leader–leader crossings without entity crossings are mostly flowcharts.
4. **Rejection-sampling selection.** Low-yield configurations are filtered toward easier geometry.
5. **Small strata.** No-crossing pairs n = 43.
6. **No real-world set** (unavailable).
7. **e2a/NONE questions** mix binding with task-format competence (F_text_only e2a 0.42).

## 8. Decision against the stated criteria

- *Errors remain after controlling for recognition*: **yes** (§2, E = 0.26 given recognition).
- *Not already solved by simple baselines*: **partially**. A crude deterministic tracer with oracle masks
  beats every VLM condition by 36–50 pp on synthetic data, and solves the circuit domain (0.975).
  The synthetic task is therefore not hard *per se*. What remains interesting is that the VLM fails at
  something a simple procedure does.
- *Failures primarily recognition-related*: no. *Benchmark-related*: possibly in part (audit pending,
  swap-regime and crossing confounds). *Largely resolved through deterministic geometry*: largely, for the
  synthetic distribution.
- *Distinguishes the hypothesised failure*: it distinguishes "reads but does not follow the connector"
  from recognition. It does **not** yet distinguish *binding* from generic *path following*. The failure
  concentrates at crossings, the regime already reported by path-following work (Trace Without Tracking,
  TraversalBench).

**Recommendation: REVISE.** No Phase-2 training yet. Required before reconsidering GO:

1. Two-reviewer blind audit of `audit_diag_v1`. Pass: ≥ 98% accuracy, ≤ 2% ambiguous per reviewer, ≥ 95%
   of pairs correct by both.
2. Rebuild test sets with **strict swaps only** and **crossing type as a controlled, balanced factor**
   (none / entity-line only / leader–leader), with ≥ 100 pairs per cell, plus the e2a/a2e split reported
   separately.
3. Run the smallest decisive experiment (`research/phase2_go_no_go.md`):
   - (a) a **path-following control** on the same images: "which entity does the line starting at the marked
     dot end on?", with no annotation text. This separates binding from tracing.
   - (b) a **frozen-feature linkage probe**: a linear read-out of connectivity from Qwen3-VL hidden states on
     strict held-out pairs. This tests whether the information is represented at all.

   Cost < 1.5 GPU-h; needs approval because the probe fits a small classifier.

## 9. Example failures (ids in diag_v1)

- Correct OCR, wrong binding (stuck after swap): `test_id-00000-a2e-a0-cf`. Gold G4, A answers G1 (the
  original target). The crop probe read "13'-10"" correctly.
- Proximity capture: `test_id-00002-a2e-a0-orig` (gold C5, pred C3 = nearest column).
- Crossing switch: `test_id-00009-a2e-a1-cf` (gold A, pred C, an entity the gold leader crosses).
- Visual–text conflict in F: `test_id-00000-a2e-a0-cf`. Told `"13'-10"" is connected to G4`, F answers G1
  (= its A answer). With a blank image it answers G4.
- Unlinked entity: `test_id-00000-e2a-G3-orig`. Gold NONE, F answers "N.T.S." (an unlinked note).
- Format failure: `test_id-00207-a2e-a2-orig`. Pred "auth?" (decision text instead of a step id).
- Correct binding: see the panels in `results/figures/diag_A_test_id.png` (GOLD = PRED rows).
