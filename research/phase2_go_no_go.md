# Phase 2 Go / No-Go

**Current decision: NO-GO for Phase-2 training. Gate 2 = REVISE** (`research/gate2_final_report.md`).
Nothing in this file is authorised to run. It specifies what must be true before training is requested.

**Update 2026-11-19, Experiment 1 done** (`research/exp1_report.md`):
- path control − binding = +9.2 pp [5.3, 13.0], "intermediate": mainly a path-following deficit, plus ≤ 9 pp binding cost;
- probe on unseen topology 0.313 [0.257, 0.370], "inconclusive" (test_id 0.442; answers 0.162).

Per the pre-registered rule, the next step is the **M2-vs-M3 pilot (with M1 as reference)**. It needs explicit
approval. P1 (human audit) is still pending and is required before any GO claim on benchmark validity.

## 1. Preconditions for requesting Phase-2 approval

| # | precondition | status |
|---|---|---|
| P1 | Two-reviewer blind audit of `audit_diag_v1` passes (≥ 98% acc, ≤ 2% ambiguous per reviewer, ≥ 95% pairs correct by both) | pending (needs two humans) |
| P2 | Strict-only rebuild with crossing type balanced (none / entity-only / leader–leader), ≥ 100 pairs per cell for test_id and unseen topology | not started (CPU only, ~20 min on a pod) |
| P3 | Decisive Experiment 1 (below) completed and pre-registered outcome read | not started; needs approval (≈ 1.5 GPU-h) |
| P4 | Novelty re-check against path-following work, given that the failure is crossing-driven | done in principle (see §4) |

## 2. Decisive Experiment 1 (smallest experiment that can falsify the Phase-2 premise)

The Phase-2 premise is that explicit relational *representation* learning is needed. That is false if (i) the
deficit is purely path following, which ordinary SFT on tracing data should fix, or (ii) connectivity is already
linearly decodable from the frozen model, in which case it is a read-out problem and SFT suffices.

**1a. Path-following control (inference only).** On the strict rebuild, re-render each orig/cf pair with
the annotation text replaced by a small filled marker at the leader's start. Ask "Which ⟨entity⟩ does the
line starting at the marked dot end on?" Same images otherwise. Compare pair accuracy with the a2e question
on the same pairs (paired McNemar).
- Path-control accuracy ≈ a2e accuracy (|Δ| < 5 pp) ⇒ the failure is path following. Binding adds nothing
  beyond it, so "binding" should not be claimed as a separate deficit.
- Path-control accuracy ≥ a2e + 10 pp ⇒ a binding-specific deficit exists beyond tracing.

**1b. Frozen-feature linkage probe (requires approval: fits a small classifier; base model frozen).**
- Features: last-layer and one mid-depth language-model layer (chosen on the train split, then fixed),
  taken from the base model's hidden states at the visual-token positions.
  Pool them over the annotation box (h_a) and over each candidate entity box (h_e); add the
  mean over the image (h_g).
- Probe: logistic regression on [h_a ⊙ h_e, |h_a − h_e|, h_g], L2-regularised. Train on the train split of the
  strict rebuild (1,000 groups). Evaluate strict-pair accuracy (argmax over candidates for both images)
  on the held-out unseen-topology split.
- Pre-registered reading:
  - probe pair accuracy ≥ 0.60 ⇒ information is present; failure is read-out ⇒ M1/M2 SFT is the right next step,
    and M3's "representation learning" claim is not motivated.
  - probe ≤ 0.30 (A-level) ⇒ connectivity is not linearly available ⇒ representation-level learning (M3)
    is motivated.
  - in between ⇒ report as inconclusive; run M2 vs M3 pilot only.
- Controls: the same probe on proximity-only features (box centres), whose strict-pair accuracy should be 0, and a
  shuffled-label probe.

Cost: feature extraction ≈ 0.5 GPU-h; path control ≈ 0.3 GPU-h; pod overhead (setup, data, model download)
≈ 0.5 h. **≈ 1.3–1.5 GPU-h ≈ $6–7.**

## 3. What Phase 2 would then test

See `research/phase2_experimental_design.md` (M1 answer-only SFT, M2 relation SFT, M3 relation SFT +
hard-negative contrastive relational scoring) and `research/phase2_compute_budget.md`.

## 4. Is explicit relational representation learning scientifically justified now?

**Not yet.**
- (+) Errors persist with near-perfect recognition. Prompting, zoom, OCR and boxes do not fix them.
- (+) With stated relations the model still follows its own reading for a2e (−17.6 pp vs blank image).
- (−) The failure is concentrated at crossings, which is the documented path-following failure mode.
  Knossos/Ariadne already show that plain relation SFT on synthetic diagrams fixes edge extraction
  in-distribution.
- (−) A simple geometric tracer with oracle masks solves most of the task, so the learning objective must
  prove value where geometry is brittle (styles, dashes, zigzags, real diagrams), which is not yet measured.
- (?) Whether the information is absent (representation) or present but unused (read-out) is unknown.
  Experiment 1b decides this.

## 5. The RL question (not to be implemented now)

RL would have an independent reason to exist only if all of the following hold after M1–M3:
1. a meaningful residual failure remains on strict held-out pairs (> 15 pp below the tracer);
2. actions change the observation or decision: crop/zoom along a connector, request a re-render at higher
   resolution, query a candidate, abstain. A "submit JSON" action is not enough;
3. rewards are verifiable from the scene graph without leaking it: a2e/e2a correctness, abstention
   correctness, paired-consistency;
4. a matched SFT baseline (same data, same number of model calls) and an inference-time verification
   baseline (crop + re-ask, same token budget) exist;
5. the hypothesis is falsifiable. Pre-register: RL must beat both baselines by ≥ 5 pp strict-pair accuracy
   on unseen topology at equal inference compute.

Current evidence argues against RL as a next step. Cropping (C) did not help, so an agent whose main action
is zooming has no demonstrated lever. Reasoning (B) helped by 3 pp at 8× cost and reduced style
robustness.
