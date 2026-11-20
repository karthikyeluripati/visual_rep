# Data-scale, counterfactual-pair ablation and seed robustness (M1, strict_v3)

## Run record

- **Date and code:** run 2026-11-20 at commit 884e539 (exact).
- **Pod:** uux57hw32tdm1j (H100), confirmed `EXITED` through the provider API.
- **GPU uptime:** 12,736 s, about $15.50.
- **Data:** the strict_v3 build is hash-identical to the transfer run's build.
- **Recipe:** the pilot recipe (answer-only SFT, LoRA r16 plus merger, lr 1e-4, 1 epoch). Each arm is evaluated on its reloaded saved adapter.
- **Rules:** pre-registered in `src/relbind/datascale_analysis.py` before any output existed.
- **Raw outputs:** `experiments/datascale/`, `results/datascale/datascale_analysis.json`.
- **Adapters:** not kept; sha256 values in `experiments/datascale/logs/adapter_sha256.txt`.

## Pair accuracy (all flipping queries)

| arm | train examples | val | unseen topology | unseen template | unseen family | **pooled unseen** |
|---|---|---|---|---|---|---|
| M1 600 groups, seed 0 | 12,000 | 0.983 | 0.815 | 0.857 | 0.900 | 0.857 |
| M1 600 groups, seed 1 | 12,000 | 0.964 | 0.862 | 0.797 | 0.878 | 0.846 |
| M1 600 groups, seed 2 | 12,000 | 0.983 | 0.858 | 0.845 | 0.907 | 0.870 |
| **seeds: mean ± sd** | | 0.977 ± 0.011 | 0.845 ± 0.026 | 0.833 ± 0.032 | 0.895 ± 0.015 | **0.858 ± 0.012** |
| M1 150 groups | 3,000 | 0.850 | 0.645 | 0.702 | 0.802 | 0.716 |
| M1 50 groups | 1,000 | 0.550 | 0.400 | 0.482 | 0.585 | 0.489 |
| noStyle (orig + **cf** + reroute) | 9,000 | 0.964 | 0.825 | 0.838 | 0.833 | 0.832 |
| noCF (orig + **style** + reroute) | 9,000 | 0.947 | 0.753 | 0.805 | 0.832 | 0.797 |

Reference points: the untrained model gets 0.05–0.25 on these splits (transfer report). The ray tracer gets 0.58 / 0.66 / 0.97.

## Pre-registered outcomes

**Q1, counterfactual pairs: `cf_pairs_matter`.**
- noStyle − noCF on pooled unseen = **+3.6 pp** [1.6, 5.5], McNemar p = 9e−5, n = 1,800 pairs.
- This passes all three conditions: ≥ 3 pp, CI excluding 0, and larger than the pooled seed range (2.4 pp).
- Per split:
  - unseen topology +7.2 [3.2, 11.0];
  - unseen template +3.3 [0.3, 6.3];
  - unseen family +0.2 [−2.8, 3.0];
  - val +1.7 [−1.7, 5.3].

**Q2, data efficiency: n\* = 600.**
- Pooled unseen accuracy is 0.49 at 50 groups, 0.72 at 150 and 0.86 at 600. The target was 0.808.
- The curve has not saturated at 600 groups, so more data would likely help further (untested).

**Q3, seed robustness.**
- The headline numbers are stable: pooled unseen sd is 1.2 pp, with a range of 0.846–0.870.
- Per-split sd is larger: up to 3.2 pp on unseen templates, with a 6 pp range.

## Interpretation and caveats

- **Answer-only SFT is a robust fix across seeds.** 0.858 ± 0.012 pooled on held-out topology, template and family,
  compared with 0.05–0.25 zero-shot. This is the paper's main learnability result.
- **Counterfactual pairing helps, but modestly, and not everywhere.** At matched data, seeing the answer-flipping twin
  beats seeing a style-only twin by 3.6 pp pooled. Almost all of the gain is on unseen topology (+7.2). On circuits
  there is none.
  - The comparison is **one seed per arm**. The decision threshold used the pooled seed range of the 600-group arm
    (2.4 pp), as pre-registered.
  - The per-split seed ranges are larger, up to 6 pp on unseen templates. So only the pooled and unseen-topology effects
    should be stated as findings; the template and family numbers should not.
  - A 3-seed replication of the two ablation arms would cost about $9 and would make Q1 robust. That is recommended
    before submission.
- **Scale matters more than pairing.** Going from 150 to 600 groups adds 14 pp. Pairing adds 3.6 pp at a fixed 9,000 examples.
