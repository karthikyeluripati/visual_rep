# Template/family transfer run (strict_v3): M1 / M2R / M3, 1 seed

## Run record

- **Date and code:** run 2026-11-20 at commit 381b902 (exact; the pod's COMMIT file matches).
- **Pod:** uux57hw32tdm1j (H100), confirmed `EXITED` through the provider API.
- **GPU uptime:** 11,779 s (3.27 h), about $14.40.
- **Aborted attempts 1–2:** about $1.45 more. They were caused by my own logging bug, not by training (see the protocol change log, 2026-11-20 correction).
- **Data checks:**
  - strict_v3's train, val, test_id and test_unseen_topology are byte-identical to strict_v2 (checked on the pod).
  - The leakage report has 0 errors and the same 7 cross-split near-duplicate warnings.
- **Reload check:** M2R smoke reload is exact (weights bit-equal; Δlogit 0.0 for reload vs reload and for in-memory vs reload).
- **Evaluation:** every arm is evaluated on its reloaded saved adapter.
- **Adapters:** not preserved; their sha256 values are in `experiments/transfer/logs/adapter_sha256.txt`.
- **Raw outputs:** `experiments/transfer/{M1,M2R,M3,M0_base}/` and `results/transfer/transfer_analysis.json`.
- **Decision rule:** pre-registered in `src/relbind/transfer_analysis.py` before any output existed.

## Results: pair accuracy over all flipping queries (95% cluster-bootstrap CI; 600 pairs per test split)

| split | M0 untrained | ray tracer | **M1 answer-SFT** | M2R rationale-first | M3 relation-SFT + head | M3 head read-out |
|---|---|---|---|---|---|---|
| val (in-template) | — | 0.608 | **0.983** [0.967, 0.997] | 0.811 | 0.897 | 0.939 |
| unseen topology (zigzag) | 0.048 | 0.583 | **0.815** [0.768, 0.857] | 0.472 | 0.598 | 0.753 |
| unseen template (plan, lr) | 0.197 | 0.657 | **0.857** [0.812, 0.898] | 0.345 | 0.575 | 0.563 |
| unseen family (circuit) | 0.247 | **0.967** | 0.900 [0.867, 0.930] | 0.617 | 0.767 | 0.933 |

Paired differences against M1, in pp (all p < 1e−7, exact McNemar):

| comparison | val | unseen topology | unseen template | unseen family |
|---|---|---|---|---|
| M3 − M1 | −8.6 | −21.7 | −28.2 | −13.3 |
| M2R − M1 | −17.2 | −34.3 | −51.2 | −28.3 |

## Pre-registered outcome: `drop_or_reframe_method_claim`

| held-out split | M1 drop from val | verdict | method helps? |
|---|---|---|---|
| unseen template | 12.7 pp | transfers (≤ 15) | no |
| unseen family | 8.3 pp | transfers (≤ 15) | no |

## What this means

1. **Plain answer-only SFT transfers to unseen templates and to an unseen diagram family.**
   - Training used one template each from two families (schematic/elevation and flowchart/tb, 600 groups).
   - The fine-tuned model reaches 0.86 on unseen templates and 0.90 on circuits, a family it never saw.
   - It holds on leader-leader crossings: 0.82 on unseen templates and 0.88 on circuits.
   - The swaps are strict (arrowheads fixed), so these scores cannot come from arrowhead or layout priors.
   - Within this benchmark, the Phase-1 "binding failure" is mostly a supervision gap, not an architectural limit.
2. **The relational methods hurt, at every level of distribution shift.**
   - Ranking is M1 > M3 > M2R on all four splits.
   - Emitting boxes and path before the answer (M2R) is worst: −51 pp on unseen templates.
   - So the explicit textual coordinate rationale is actively harmful as implemented. Its 3 of 5,400 unparsed outputs rule out truncation as the cause.
   - The M3 head lifts M3 above M2R but never above M1.
3. **The remaining hard case is flowchart/lr, and the tracer still wins on circuits.**
   - On flowchart/lr, M1 scores 0.77 and the tracer 0.38.
   - On circuits, the tracer scores 0.967 and M1 0.900 (M1 − tracer = −6.7 pp [−10.0, −3.3]). The tracer uses oracle text and node masks, so it is not a fair VLM baseline, but it shows circuits are geometrically easy.
4. **Replication across runs** (same seed and data, different run; GPU nondeterminism only):
   - M1: val 0.969 → 0.983 and unseen topology 0.842 → 0.815. Consistent.
   - M3 on unseen topology: 0.655 → 0.598. A 5.7 pp swing from nondeterminism alone, about the size of the CI half-width. Single-seed differences under about 6 pp between runs should not be interpreted.

## Caveats

- One seed per arm. There was no tuning of lr, λ, margin or steps; the relational arms might do better with tuning. The data gives no sign of a regime where they would beat M1.
- Everything is procedurally generated, from one renderer and one style system. "Unseen family" means unseen within this generator, not real diagrams. The human audit of benchmark validity is still pending.
- Style and reroute invariance is not measured.
- The 7 near-duplicate pairs cannot explain effects of this size.

## Recommendation

**Stop the method track (RelBind-RL as proposed: relational head, contrastive grounding, and the planned RL stage).** The pre-registered transfer rule says the method claim should be dropped or reframed, and every comparison points the same way. No 3-seed run of M2/M3 is warranted.

The defensible paper is diagnostic, with three results:
1. Strong VLMs fail strict counterfactual connector binding zero-shot. The Phase-1 and Exp-1 evidence covers the stuck rate, crossing types and the oracle-graph gap.
2. A small amount of answer-only supervision on strict counterfactual pairs largely fixes it, and the fix transfers across template and family.
3. Explicit relational or coordinate supervision hurts.

Each of these needs PI approval before more GPU spend:

| option | what it is | approx. cost |
|---|---|---|
| (a) data-efficiency curve | M1 trained on 50 / 150 / 600 groups, plus an ablation without counterfactual pairs. Is it the cf pairs or just data? | about $10 |
| (b) robustness | M1 × 3 seeds on the transfer splits | about $8 |
| (c) external validity | M1 on a small set of real or hand-drawn diagrams; needs human-labelled data, which is the PI's call | — |
