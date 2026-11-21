# Cross-family learnability: answer-only SFT on InternVL3-8B

## Run record

- **Date and code:** run 2026-11-20/21 at commit 992a9a4.
- **Pod:** uux57hw32tdm1j (H100), confirmed `EXITED`. Uptime 8,077 s, about $9.85.
- **Recipe:** the pilot recipe (LoRA r16 on the language model plus the projector, lr 1e-4, 1 epoch, 600 groups, seed 0).
- **Model:** InternVL3-8B-hf at revision 259a3b6 (the same revision as the sweep). Trainable parameters: 67.9M.
- **Smoke reload check:** exact.
- **Rule:** pre-registered in `src/relbind/xfam_analysis.py`.
- **Gemma-3-12B:** skipped. The PI's HF account is not yet authorized for this gated model.

## Results: pair accuracy

| split | InternVL3-8B zero-shot | **InternVL3-8B + SFT** | Qwen3-VL-4B + SFT (3-seed mean) |
|---|---|---|---|
| val (in-template) | — | 0.875 [0.828, 0.917] | 0.977 |
| unseen topology | — | 0.652 [0.595, 0.707] | 0.845 |
| unseen template | 0.078 | 0.468 [0.422, 0.515] | 0.833 |
| unseen family | 0.120 | 0.763 [0.722, 0.803] | 0.895 |
| **pooled template + family** | 0.099 | **0.616** [0.580, 0.652] | 0.864 |

## Pre-registered outcome: `transfers_across_families = true`

Fine-tuned minus base = **+51.7 pp** [47.9, 55.4] on pooled held-out pairs. The threshold was ≥ 40 pp with CI lower bound > 0.

## Interpretation

- **The fix is not Qwen-specific.** The same small answer-only SFT lifts a second architecture from 0.10 to 0.62 on held-out
  layouts.
- **How far it generalizes depends on the base model.** InternVL3-8B learns the training layouts (0.875 val) but transfers
  much less to unseen templates: 0.468, a 41 pp drop. Under the earlier transfer rule's thresholds (≥ 30 pp) that would count
  as "collapses". For Qwen the drop was 14 pp.
  - One possible explanation: InternVL's tiling at 448 px changes the visual-token geometry across layouts. This is not tested.
  - So the paper's claim must be **"learnable, with base-model-dependent transfer"**, not "solved by 600 examples".
  - This also partly answers the framing risk: the benchmark is not trivially solved for every model.
- **Caveat:** one seed. Per-split seed SD for Qwen was 1.5–3.2 pp, far smaller than these between-model gaps.
