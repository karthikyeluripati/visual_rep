# Failure Analysis — Phase 1 diagnostic (DRAFT, in progress)

Model: Qwen/Qwen3-VL-4B-Instruct @ ebb281ec, bf16, greedy, native resolution (1024×700 → 704 visual
tokens), transformers 5.23.0. Data: `diag_v1` (configs/diag_v1.json): test_id 250 groups (5,000
examples; 1,000 flipping a2e/e2a queries), OOD splits 60 groups each. CIs: 95% cluster bootstrap
over groups (B = 2000). Status of pending items is marked **[pending]**.

**Important caveat.** The blind human audit (`data/validation/audit_v0`) has **not** been done.
No result here counts as Gate-2 evidence until reviewers confirm that the items are uniquely answerable.

## 1. Headline numbers (test_id)

| condition | PCA | acc orig | acc cf | stuck (cf) | style invariance | abstention acc |
|---|---|---|---|---|---|---|
| A direct | 0.241 [0.208, 0.274] | 0.594 | see json | 0.512 | 0.879 | 0.392 |
| B reasoning | **[pending]** | | | | | |
| C oracle pair-region crop | 0.259 [0.223, 0.293] | 0.602 | | 0.560 | 0.893 | 0.440 |
| D oracle OCR text+boxes | 0.247 [0.212, 0.282] | 0.588 | | 0.534 | 0.894 | 0.503 |
| E oracle annotation/candidate boxes | 0.283 [0.247, 0.322] | 0.624 | | 0.526 | 0.890 | 0.462 |
| F oracle relation graph (as text) | 0.659 [0.622, 0.694] | 0.840 | | 0.181 | 0.910 | 0.697 |
| G proximity / text prior | 0.000 / 0.000 | | | 1.0 | 1.0 | |
| G connected-component tracer (oracle boxes) | 0.079 [0.056, 0.104] | 0.401 | | 0.741 | 0.865 | 0.868 |
| P_read (crop = pure recognition) | **[pending]** | | | | | |

OOD splits, condition A: unseen template 0.238 [0.171, 0.308]; unseen topology (zigzag)
0.096 [0.050, 0.146]; cross-domain (circuit) 0.213 [0.154, 0.271]. Condition C on the same splits:
0.221 / 0.129 / 0.213.

## 2. Observations (provisional)

1. **Binding is far from solved, and the answer often ignores the rewiring.** On test_id the model
   answers both members of a swap pair correctly for only 24% of queries. In 51% of flipping queries
   its answer on the counterfactual image is identical to its answer on the original, even though the
   only change between the images is the two connector paths.
2. **The usual inference-time remedies barely move the result.** An oracle crop of exactly the
   relevant region, upsampled up to 2.5× (C, +1.8 pp), oracle OCR (D, +0.6 pp) and oracle boxes
   (E, +4.2 pp) all leave PCA below 0.30. None of the gate's "revise" conditions about
   resolution/context (PCA_C − PCA_A ≥ 25 pp) is close to triggering.
3. **The visual prior overrides a verified textual graph.** In F the correct edges are stated in the
   prompt, yet acc_orig is only 0.84 and PCA 0.66. Typical failure: told `"11'-0"" is connected to G4`,
   the model answers "N.T.S." (an unlinked note drawn near G4). F was pre-registered as a
   reading/format ceiling. Because it is clearly below 1, we report it as evidence of a strong
   image-based prior that competes with explicit relational information, and do **not** use it as a
   ceiling.
4. **Abstention is weak.** For entities with no annotation, the model names some nearby text
   ~60% of the time (abstention acc 0.39 in A).
5. **Held-out zigzag routes are hardest** (PCA 0.10), consistent with path-following failure at bends.

## 3. Error taxonomy **[pending: needs corrected P_read]**

Categories (precedence): parse → perception → stuck → proximity → crossing → other
(definitions in `src/relbind/analysis.py`).

## 4. Qualitative examples **[pending]**

- Correct OCR, wrong binding.
- Correct binding.
- Failure after a counterfactual flip (stuck on the old target).
- Visual prior overriding the oracle graph (F).

## 5. Protocol deviations

- P_read v0 used pixel coordinates and was invalid (the model read the wrong text region). It was replaced
  by crop/box probes in 0–1000 coordinates. Conditions D and E were run *after* the same coordinate fix.
  See the protocol change log.
- F does not behave as a ceiling (see §2.3); it is reported as an observation.
