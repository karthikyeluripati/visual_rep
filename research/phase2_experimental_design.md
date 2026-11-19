# Phase 2 Experimental Design (conditional; NOT authorised)

Runs only after `phase2_go_no_go.md` §1 preconditions are met and the user approves this design and
`phase2_compute_budget.md`. RL is out of scope.

## 1. Question

Does an explicit, candidate-level relational objective with hard (counterfactual) negatives improve
counterfactual annotation→entity binding and transfer to unseen connector topologies **beyond
relation-supervised SFT (M2)** at matched data, parameters and compute?

## 2. Data (strict-only rebuild, frozen before training)

- Generator as in `configs/prototype.json`, plus two changes: **strict swaps only** and **crossing type
  balanced** (none / entity-only / leader–leader).
- Train: schematic/elevation + flowchart/tb; straight + elbow; sans + mono. Pilot 1,000 groups; full 2,000.
- Val: 100 groups (same distribution). It is used for early stopping and probe/margin hyperparameters only.
- Tests: test_id (250), unseen topology (zigzag, 150), unseen template (150), cross-domain circuit (150),
  unseen style (150: dashed leaders, alternate terminators, ink), each strict and crossing-balanced.
  Generated from a held seed and hashed once.
- Each group yields orig/cf/reroute/style images × (2 a2e + 2 e2a + 1 NONE) queries = 20 examples.

## 3. Arms (same base checkpoint Qwen3-VL-4B-Instruct @ ebb281ec, native resolution ≈ 704 visual tokens)

Identical for all arms: the same images and questions, an equal number of optimizer steps, LoRA r = 16 /
α = 32 / dropout 0.05 on all LM attention and MLP projections, the vision-language merger trainable, and the
vision encoder frozen. Batches are built so that orig and cf of a group co-occur (paired batching) in every
arm, so pairing is not a confound. An unpaired variant is an ablation.

| arm | supervision (target text) | extra objective |
|---|---|---|
| M0 | none (base model) + ray tracer reference | — |
| M1 answer-only SFT | `{"answer": X}` | — |
| M2 relation SFT | `{"answer": X, "annotation_box": [...], "entity_box": [...], "path": [p0, ..., pk]}`. Path = connector polyline in 0–1000 coordinates, ≤ 5 points | — |
| M3 relation SFT + contrastive relational scoring | same as M2 | L_rel (below) |

Supervision differences are explicit and reported. M2/M3 receive geometry targets that M1 does not.
M1 vs M2 measures *relation supervision*; M2 vs M3 measures *the representation objective*. M2 vs M3 is the
primary comparison.

**M3 objective.** For each a2e query on image I, with annotation box b_a and candidate entity boxes {b_e}:
- h_a = mean of last-layer LM hidden states at visual-token positions inside b_a (merger output grid → token map).
- h_e = the same for b_e. g = mean over all visual tokens.
- score s(a, e | I) = MLP([h_a ⊙ h_e, |h_a − h_e|, g]) → scalar. Head ≈ 2.1 M parameters. M1/M2 get an
  untrained head of the same size (parameter parity at inference: unused).
- L_rel = InfoNCE over candidates with the gold entity positive. Hard negatives: (i) the counterfactual
  partner's target (the entity the same annotation points to in the paired image), (ii) the proximity-nearest
  entity, (iii) entities the gold connector crosses.
- L_pair = hinge(m − [s(a, e₁|I_orig) − s(a, e₂|I_orig)]) + hinge(m − [s(a, e₂|I_cf) − s(a, e₁|I_cf)]),
  with m tuned on val.
- L = L_SFT + λ (L_rel + L_pair), with λ ∈ {0.1, 0.5} chosen on val.
- **Trained representations:** LoRA adapters in the LM, the merger, and the scoring head. Gradients from L_rel
  flow into the merger and LM adapters, which is the representation being shaped. The vision encoder stays
  frozen; unfreezing its last 4 blocks is an optional ablation.

**Outputs reported for every arm:**
- (i) generated-answer metrics with no oracle input, the primary endpoint;
- (ii) relationship-prediction accuracy: argmax_e s(a, e). For M1/M2 the head is fitted post hoc as a frozen
  probe for comparability. It needs candidate boxes, so it is reported as an oracle-box protocol.

## 4. Endpoints (pre-registered before any pilot output is seen)

- **Primary:** strict-pair accuracy (a2e + e2a flipping queries, generated answers) on pooled
  unseen topology ∪ unseen template. Comparison: M3 vs M2.
- Minimum meaningful improvement: **+5 pp absolute**. Tests: exact McNemar on per-query pair outcomes per
  seed, and a cluster-bootstrap CI of the paired difference over groups. Success requires the CI to exclude 0 in
  ≥ 2 of 3 seeds and the pooled-seed estimate to be ≥ +5 pp. Holm correction across {M2−M1, M3−M2, M3−M1}.
- Secondary: per-split pair accuracy, crossing-type strata, style invariance, abstention accuracy,
  relationship-prediction accuracy, and the gap to the ray tracer.
- Guardrail: M3 must not be worse than M2 by > 2 pp on test_id.

## 5. Ablations (full phase only, 1 seed each unless a primary effect is found)

1. M3 without hard negatives (random negatives only).
2. M3 without counterfactual examples (train groups without cf images; same number of examples).
3. M3 without L_pair (L_rel only), and M3 with λ = 0 (equal to M2 plus an untrained head; sanity check).
4. Topology transfer: train straight-only, test elbow and zigzag.
5. Out-of-domain: circuit test and the unseen-style test.

## 6. Pilot (first step after approval)

- 1,000 train groups → 20,000 examples (orig/cf/reroute/style × 5 queries). 1 epoch, effective batch 32 →
  625 optimizer steps. LR 1e-4 for LoRA, 1e-4 for the head, cosine schedule, warm-up 30 steps. bf16, gradient
  checkpointing.
- M1, M2, M3, 1 seed each. Evaluate on val + unseen topology (150 groups), generated answers.
- Pilot checks before any full run: loss decreases, gradient norms are finite, nonzero gradients reach the
  head and merger (M3), checkpoints reload with identical outputs, and the image processor is identical to
  Phase 1 (704 tokens).
- Pilot decision: proceed to 3 seeds only if M3 − M2 ≥ +3 pp on the primary endpoint *or* the M2/M3 gap to
  the tracer narrows by ≥ 10 pp. Otherwise stop and report.
- Memory: 4B bf16 weights ≈ 9 GB; activations at ~1,000 tokens/example with checkpointing ≈ 2–3 GB per
  example; micro-batch 8 + grad-accum 4 fits in 80 GB (estimate; verified in the pilot).
