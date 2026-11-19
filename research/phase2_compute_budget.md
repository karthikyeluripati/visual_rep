# Phase 2 Compute Budget (estimates; NOT authorised)

Rate: RunPod H100 80GB, **$4.39/h** (measured via API 2026-11-19). The pod has no persistent volume.
Every run ends with bundle → pull → checksum → stop (`experiments/pod/`).
Throughput assumptions are taken from Phase 1, where inference ran at ≈ 13 examples/s for short answers.
Training throughput is an *estimate* (≈ 4 examples/s with LoRA + checkpointing at ~1k tokens) to be
measured in the pilot. Re-estimate the budget after the pilot.

| step | GPU-h | cost | gate |
|---|---|---|---|
| Pod setup per session (venv, model download ≈ 9 GB, data build) | 0.3–0.5 | $1.3–2.2 | — |
| **Decisive Experiment 1** (path-following control + frozen linkage probe) | 1.0–1.5 | **$4.4–6.6** | needs approval now |
| Strict-only, crossing-balanced rebuild (CPU on pod) | 0.3 | $1.3 | with Exp. 1 |
| Phase-2 pilot: M1, M2, M3 × 1 seed, 20k examples, 625 steps each (≈ 1.4 h each) + eval (≈ 0.3 h) | ≈ 4.5–5 | **≈ $20–22** | Exp. 1 outcome + approval |
| Phase-2 full: 3 arms × 3 seeds on 2,000 groups (≈ 2.8 h each) + 5 ablations × 1 seed (≈ 2.8 h each) + evals (≈ 2 h) | ≈ 41 | **≈ $180** | pilot decision rule |
| Contingency (failed runs, re-evaluation), 20% | ≈ 9 | ≈ $40 | — |

Totals:
- Experiment 1 only: about **$7**.
- Through the pilot: about **$30**.
- Full Phase 2: about **$250**.

RL (Phase 3) is not budgeted. It will not be requested unless `phase2_go_no_go.md` §5 conditions hold.
