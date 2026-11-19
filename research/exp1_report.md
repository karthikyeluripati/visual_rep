# Decisive Experiment 1 — Report

Date: 2026-11-19. Inference + frozen-feature linear probe only; **no VLM weights were updated.**
Model Qwen3-VL-4B-Instruct @ `ebb281ec`, transformers 5.23.0, torch 2.8.0+cu129, 1× H100 (pod run 15:06–15:29 UTC;
≈ $2.5 including two aborted attempts). Code @ `602dfd1`. Artifacts: `experiments/exp1/`, `results/exp1/`,
`data/builds/strict_v2/` (manifest + leakage report; examples.jsonl kept in the gitignored bundle, hashes in the manifest).

## Data: strict_v2

1,140 groups (train 600, val 90, test_id 300, unseen topology 150). **All swaps strict** (arrowhead positions
identical between orig and cf; 1,140 / 1,140). Crossing cells per family: schematic entity_only / leader_leader,
flowchart leader_leader (836 leader_leader, 304 entity_only). The "none" cell is infeasible under strict swaps
(protocol change log). Every examples.jsonl SHA-256 matches the manifest. Leakage audit ok, **with a warning**:
7 cross-split base-image pairs at dHash distance ≤ 4 (min 3). They are not removed. This could marginally inflate
probe test accuracy (train → test).

## 1a. Binding vs path following (same strict pairs, same geometry)

Unit: a2e pair (orig and cf both correct). Paired exact McNemar + cluster bootstrap over groups (B = 2000).

| split / cell | n pairs | binding (A) | path control | path − binding | discordant (path only / binding only) | p | ray tracer |
|---|---|---|---|---|---|---|---|
| test_id, all | 600 | 0.162 [0.135, 0.192] | 0.253 [0.222, 0.287] | **+9.2 pp [5.3, 13.0]** | 110 / 55 | 2.2e-5 | 0.680 |
| test_id, entity_only | 150 | 0.213 | 0.353 | +14.0 [6.0, 22.0] | 36 / 15 | 0.005 | 0.913 |
| test_id, leader_leader | 450 | 0.144 | 0.220 | +7.6 [3.3, 11.8] | 74 / 40 | 0.002 | 0.602 |
| unseen topology, all | 300 | 0.057 [0.030, 0.087] | 0.127 [0.093, 0.163] | +7.0 [2.7, 11.3] | 34 / 13 | 0.003 | 0.600 |

Pre-registered reading: Δ ≥ 10 pp with CI > 0 ⇒ binding-specific deficit; |Δ| < 5 ⇒ pure path following.
**Observed Δ = 9.2 pp ⇒ "intermediate".**

Interpretation (observations):
- The dominant deficit is **path following**. With no text binding at all, the model follows the marked line
  correctly on both images in only 25% of pairs (tracer: 68%).
- Binding through the annotation text costs a further ≈ 9 pp. The path images also remove all annotation
  text (less clutter), so 9 pp is an **upper bound** on the binding-specific cost.

## 1b. Frozen-feature linkage probe

Linear candidate scorer on pooled visual-token hidden states. Layer and weight decay were chosen on val
only (selected: layer 18 of 36, wd 1e-3).

| split | probe pair acc | probe single acc | binding (A) on same pairs | shuffled-label control | geometry-only control |
|---|---|---|---|---|---|
| test_id | **0.442** [0.402, 0.483] | 0.682 | 0.162 | 0.032 | 0.000 |
| unseen topology | **0.313** [0.257, 0.370] | 0.550 | 0.057 | 0.027 | 0.000 |
| — by cell (test_id) | entity_only 0.613 (n = 150), leader_leader 0.384 (n = 450) | | | | |

Validation pair accuracy by depth (best wd): layer 0 = 0.28, 9 = 0.29, **18 = 0.41**, 27 = 0.25, 36 (last) = 0.14.

Pre-registered reading on unseen topology: ≥ 0.60 read-out deficit, ≤ 0.30 representation deficit, otherwise
inconclusive. **Observed 0.313 (CI 0.257–0.370) ⇒ inconclusive.**

Observations:
- Connectivity is **partly linearly decodable** from frozen features, well above what the model's answers
  use: +28 pp in-distribution, +26 pp on unseen topology. The controls rule out position and label leakage.
  The probe and answer comparison is between separate estimates on the same pair set. The per-pair probe
  outcomes were not saved, so **a paired test is unavailable**. The intervals are far apart, but that is not
  offered as a significance test.
- The information **peaks mid-depth and decays toward the output** (layer 36: 0.14 on val).
- It is weak where tracing is hard: leader–leader crossings 0.38, zigzag routes 0.31. All probe numbers are below
  the deterministic tracer.
- A linear probe on pooled features is a lower bound on what is encoded.

## What this means for the Phase-2 premise

- Both parts of the "relational representation deficit" story have some support, and neither is decisive.
  There is a read-out gap: information is present mid-network but not used, which favours ordinary SFT
  (M1/M2). There is also a weak encoding: even the best linear read-out stays far below the tracer at crossings
  and on unseen topology, which leaves room for a representation-shaping objective (M3).
- The pre-registered consequence of "inconclusive" is to run **only the M2-vs-M3 pilot** (plus M1 as a
  cheap reference). That pilot directly tests whether an explicit relational objective beats relation SFT.
- The paper's binding-specific claim must be modest: ≤ 9 pp beyond path following.

## Status of other blockers

- Human audit: **still pending.** Benchmark validity remains pending.
- Phase-2 pilot (≈ $22): **not started; requires explicit approval.**
