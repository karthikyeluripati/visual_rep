# Phase-2 pilot report (M1 / M2 / M3, 1 seed)

Run 2026-11-19, code commit 804d5ea, pod uux57hw32tdm1j (H100). The pod is confirmed `EXITED` through the provider API.
GPU uptime was 11,417 s (3.17 h), about $13.90. The data was strict_v2, rebuilt on the pod; its examples sha256 is identical to the Exp-1 build.
The 4-step M3 smoke test passed its reload check (max |Δlogit| = 0.0).
The adapters were not preserved (deviation h). Their sha256 values are in `experiments/pilot/logs/adapter_sha256.txt`.
Raw outputs: `experiments/pilot/{M1,M2,M3}/`, `results/pilot/pilot_analysis.json`, `results/pilot/pilot_breakdown.txt`.

## Results: pair accuracy, all flipping queries (orig and cf both correct; 95% cluster-bootstrap CI)

| split | M0 untrained | ray tracer | M1 answer-SFT | M2 relation-SFT | M3 relation-SFT + head | M3 head read-out (a2e) |
|---|---|---|---|---|---|---|
| val | — | 0.608 | **0.969** [0.939, 0.994] | 0.883 [0.839, 0.922] | 0.900 [0.853, 0.942] | 0.961 |
| test_id | 0.165 | 0.657 | **0.969** [0.953, 0.983] | 0.882 [0.858, 0.907] | 0.917 | 0.957 |
| test_unseen_topology | 0.048 | 0.583 | **0.842** [0.800, 0.878] | 0.470 [0.425, 0.517] | 0.655 [0.605, 0.703] | 0.793 |

Paired comparisons on test_unseen_topology (n = 600 pairs, exact McNemar):

- M3 − M2 = **+18.5 pp** [13.5, 23.3], p = 7e−14
- M3 − M1 = **−18.7 pp** [−23.0, −14.3], p = 1e−19
- M2 − M1 = −37.2 pp [−41.8, −32.2], p = 8e−48
- M1 − M0 = +79.3 pp

## Pre-registered pilot rule

The rule was: proceed to 3 seeds if M3 − M2 ≥ +3 pp on test_unseen_topology, OR the gap to the tracer narrows by ≥ 10 pp.
**Formally met:** +18.5 pp, and the gap narrowed by 18.5 pp. `proceed_to_3_seeds = true`.

**I recommend not running the 3-seed study as designed, even though the rule passes.** The rule took M2 as the comparator and implicitly assumed M1 ≤ M2. That assumption failed badly.

- Answer-only SFT (M1) is the best arm on every split.
- M3 beats M2 only by recovering part of the damage that relation-SFT did.
- The core hypothesis is that explicit relational supervision or grounding improves binding over plain SFT. It is **not supported** against the simplest baseline: M3 − M1 = −18.7 pp.
- A 3-seed run would only tighten the CI on "M3 > M2", which does not answer the research question.

## Findings

1. **The Phase-1 binding failure is largely fixed by plain SFT, at least within template.**
   - 600 training groups take the model from 0.16 to 0.97 on test_id, and from 0.05 to 0.84 on unseen route shapes (zigzag).
   - That is well above the tracer.
   - It holds on leader-leader crossings (M1 0.82 on unseen topology, tracer 0.62).
   - Strict swaps keep arrowheads fixed, so arrowhead or layout priors cannot produce correct answers on both images. The model must be reading the connector.
   - The 7 near-duplicate pairs (out of 1,140 groups) cannot account for an effect this large.
2. **Relation-SFT as implemented hurts generalization (M2).** Possible causes, none tested yet:
   - The target is generated answer-first, so the boxes and path cannot causally inform the answer. They only compete for capacity.
   - Coordinate and path regression in text is hard.
   - The path supervision comes from straight/elbow routes only, and may impose the wrong prior on zigzag routes.
3. **The head knows more than the model's generated answer uses.**
   - M3's head read-out gets 0.79 on unseen topology, against 0.655 for M3's generated answer.
   - It is still below M1's generated answer (0.84).
4. **Scope limit.** strict_v2 has a single template per family (schematic/elevation, flowchart/tb) in every split.
   - "Generalization" here means unseen route shape only.
   - Unseen template and unseen family are untested. That is where plain SFT is most likely to break, and where the method could still matter.

## Caveats

- One seed per arm. 600 optimizer steps, with no tuning of lr, λ or margin.
- Logging defect: with accumulation 2, the logged `rel`/`pair` came from the second micro-batch only. That batch is reroute+style, which has no orig/cf pairs.
  - So `pair = 0.0` in all logs is an artifact.
  - The pair term was computed on the orig/cf micro-batches; all 600 train groups contain orig/cf a2e pairs.
  - Its magnitude is unrecorded.
  - Fixed after the run: `train.py` now logs per-step means. The pilot itself is unaffected, because the loss was correct and only the logging was wrong.
- Not measured in the pilot: invariance to style and reroute.

## Proposed next step (needs PI approval; not started)

Test whether M1's success transfers beyond the training template. This decides whether there is any gap left for the method to close.

- **Build:** a held-out-template/family split. For example, train on schematic/elevation and flowchart/tb; test on schematic/plan, flowchart/lr and circuit.
- **Arms:** retrain M1 and M3, plus an "M2-rationale-first" variant that emits the path/boxes before the answer.
- **Cost:** about 2.5–3 GPU-h, roughly $11–13.
- **Decision:**
  - If M1 also transfers, plain SFT closes the gap and the relational-method claim should be dropped or reframed. The finding becomes a negative/diagnostic result.
  - If M1 collapses on unseen templates, that is the regime where a 3-seed method comparison is worth running.
