# Zero-shot multi-model sweep (strict_v3, A_direct)

## Run record

- **Date and code:** run 2026-11-20 (attempt 1 at commit 050ea75, attempt 2 at 6fb99bb).
- **Pod:** uux57hw32tdm1j (H100). Both attempts confirmed `EXITED` through the provider API.
- **GPU cost:** about $6 (attempt 1) + $11 (attempt 2).
- **Data and prompt:** strict_v3 test_id, test_unseen_template and test_unseen_family, orig+cf; 6,000 examples and 600 flipping pairs per model;
  the Phase-1 A_direct prompt; greedy decoding.
- **Rule:** H_general, pre-registered in `src/relbind/sweep_analysis.py`.
- **Raw outputs:** `experiments/sweep/<model>/`, `results/sweep/sweep_analysis.json`.
- **Model revisions:** the run did **not** record them (transformers 5 dropped `config._commit_hash`; the backend is fixed now).
  They were resolved post hoc from the HF API (`experiments/sweep/model_revisions_posthoc.json`). Every model's last
  modification predates the run, so these SHAs are exactly what was downloaded.

## Results: pooled pair accuracy (95% cluster-bootstrap CI), sorted

| model | pooled | leader-leader crossings | a2e stuck rate | valid-answer rate* |
|---|---|---|---|---|
| *fine-tuned M1 (Qwen3-VL-4B + 600 groups SFT)* | *0.924* [0.908, 0.940] | *0.909* | *0.03* | — |
| *ray tracer (oracle text/node masks)* | *0.734* [0.703, 0.766] | *0.645* | *0.15* | — |
| Qwen3-VL-8B | 0.195 [0.176, 0.215] | 0.153 | 0.60 | 0.94 |
| Qwen3-VL-4B | 0.194 [0.175, 0.213] | 0.144 | 0.60 | 0.95 |
| Qwen3-VL-2B | 0.101 [0.087, 0.114] | 0.074 | 0.69 | 0.85 |
| Qwen2.5-VL-7B | 0.091 [0.077, 0.104] | 0.067 | 0.75 | 0.88 |
| InternVL3-8B | 0.084 [0.072, 0.097] | 0.059 | 0.72 | 0.86 |
| Pixtral-12B | 0.078 [0.066, 0.090] | 0.053 | 0.70 | 0.88 |
| LLaVA-OneVision-7B | 0.028 [0.021, 0.035] | 0.018 | 0.80 | 0.68 |
| SmolVLM2-2.2B † | 0.016 [0.010, 0.021] | 0.010 | 0.80 | 0.34 |
| Idefics3-8B † | 0.013 [0.008, 0.018] | 0.011 | 0.75 | 0.38 |

- Chance pair accuracy (uniform over candidates) is about **0.054** (test_id).
- \* Valid-answer rate = the share of predictions that are a legal candidate. Measured on test_id only, because only that split's
  examples are available locally.
- † Low-validity runs: they often answer with a node's *text* instead of its ID, so they measure instruction-following, not binding.
- Gemma-3-12B was not evaluated (gated; no HF token). Qwen3-VL-32B was not evaluated (does not fit the pod's disk).

## Pre-registered outcome: **H_general holds**

- **All 9 models** are below the tracer on pooled leader-leader pairs (pre-registered threshold: ≥ 75%), and all 9 are below fine-tuned M1.
- **Excluding the two low-validity runs** (a post-hoc sensitivity check, not pre-registered), all 7 remaining models still
  fail: the best is 0.195, against the tracer's 0.734.

## Findings

1. **Zero-shot failure is general across families** (Qwen, InternVL, Pixtral/Mistral, LLaVA, Idefics, SmolVLM) and
   across sizes from 2B to 12B.
   - The best model gets 0.195. Four models are at or below chance (0.054).
   - Stuck rates of 0.60–0.80 mean most models give **the same answer on both twins**. They ignore the connector
     change entirely.
2. **Scale within a family helps a little and saturates early.** Qwen3-VL goes 2B → 4B → 8B as 0.101 → 0.194 → 0.195.
3. **Answer-only fine-tuning of the 4B model (0.924) beats every zero-shot model by more than 70 pp.** It also beats
   the oracle-input tracer.

## Limitations and next steps

- **Coverage:** open models up to 12B only. Frontier API models need a PI decision (keys; data sent to the providers).
  A larger model (≥ 30B) needs a pod with more disk.
- **Prompt:** one prompt format. A prompt-sensitivity check (e.g., B_reasoning for the top 3 models) would cost about $3.
