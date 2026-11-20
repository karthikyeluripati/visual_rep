# Following the Line: Strict Counterfactual Evaluation of Connector Binding in Vision-Language Models

*Draft v0, 2026-11-20. Target: NeurIPS 2027 Evaluations & Datasets track. Every number traces to a committed report in `research/`.
The bracketed tags **[PENDING]** mark sections that wait on human-annotated data. **Do not submit with them unresolved.***

## Abstract (draft)

Technical diagrams attach meaning through drawn connectors: a leader line ties the label "12" to one part, and a flowchart
arrow ties "> 5" to one step. We ask whether vision-language models (VLMs) actually follow these connectors, or whether they
guess from layout and text priors.

We introduce RelBind-Bench, a procedurally generated benchmark built from *strict counterfactual twins*. The two connectors'
paths are swapped while their arrowheads stay fixed, so the two images are pixel-identical outside the swapped strokes, yet
the correct answer flips. A model can get both twins right only by tracing the line.

We evaluate nine open VLMs from six families (2B–12B). All of them fail. The best scores 0.195 pair accuracy, against 0.73
for a simple ray-tracing baseline. Most models give the same answer on both twins 60–80% of the time, and step-by-step
prompting does not help.

The failure is not perceptual: models read the labels 95% of the time. Nor is it architectural. Answer-only fine-tuning of
a 4B model on 600 strict pairs reaches 0.858 ± 0.012 (3 seeds) on held-out topologies, templates and an unseen diagram
family. At matched data, the counterfactual pairing contributes +3.9 pp. In contrast, explicit relational supervision
(coordinate rationales, a contrastive relational head) *hurts* by 9–51 pp, a pre-registered negative result.
**[PENDING: real patent-drawing results; human audit.]**

## 1. Introduction

- **Motivating failure:** the model reads the label correctly and attaches it to the wrong part. In our diagnosis,
  relational accuracy given a correct reading is only 0.54.
- **Why standard evaluation cannot see this:** single-image accuracy rewards layout priors. Strict twins remove arrowhead,
  position and text cues by construction.
- **Contributions:**
  1. The benchmark: strict twins, controls, held-out splits, reproducible build.
  2. A cross-model diagnosis.
  3. Learnability and what does (not) help, with pre-registered decisions and a full protocol log, including our own errors.

## 2. Related work

To be written. Cover:
- diagram topology extraction and path following;
- minimal-pair and counterfactual VLM benchmarks;
- chart and diagram QA grounding;
- data- versus objective-centric fixes.

The novelty audit is in `research/novelty_audit.md`.

## 3. RelBind-Bench

### Generation

- Scene graphs are built before rendering, across three families:
  - schematic (elevation, plan);
  - flowchart (tb, lr);
  - circuit (ladder, grid).
- Connector routes are straight, elbow or zigzag.
- A geometric verifier rejects ambiguous scenes.

### Strict twins

- Two annotations' connector paths are exchanged and the arrowhead positions are held fixed.
- A pixel check requires zero difference outside the swapped strokes, dilated by 3 px (the Lanczos support).
- Controls in each group:
  - reroute: the path class changes, the answer does not;
  - style: rendering changes, the answer does not.

### Splits (strict_v3)

| split | groups | layouts | routes |
|---|---|---|---|
| train | 600 | schematic/elevation, flowchart/tb | straight, elbow |
| val | 90 | same | straight, elbow |
| test_id | 300 | same | straight, elbow |
| test_unseen_topology | 150 | same | zigzag |
| test_unseen_template | 150 | schematic/plan, flowchart/lr | straight, elbow |
| test_unseen_family | 150 | circuit/ladder, circuit/grid | straight, elbow |

- Crossing type is balanced over the cells that are feasible under strict swaps: leader-leader 1,025, entity-only 304, none 111.
- The leakage audit reports 0 errors and 7 near-duplicate warnings. A rebuild is byte-identical.
- The datasheet is in `research/datasheet.md`.
- **[PENDING] Human audit:** two blind reviewers, 172 items.

### Metric

- **Pair accuracy:** both twins correct. Chance is about 0.054.
- **Stuck rate:** the same answer on both twins.
- CIs come from a cluster bootstrap over generation groups. Paired comparisons use exact McNemar.

## 4. Zero-shot diagnosis

### 4.1 Nine models fail (`research/sweep_report.md`)

Pooled over test_id, unseen template and unseen family (600 pairs per model):

| model | pair acc. | leader-leader | stuck |
|---|---|---|---|
| Qwen3-VL-8B | 0.195 [0.176, 0.215] | 0.153 | 0.60 |
| Qwen3-VL-4B | 0.194 [0.175, 0.213] | 0.144 | 0.60 |
| Qwen3-VL-2B | 0.101 | 0.074 | 0.69 |
| Qwen2.5-VL-7B | 0.091 | 0.067 | 0.75 |
| InternVL3-8B | 0.084 | 0.059 | 0.72 |
| Pixtral-12B | 0.078 | 0.053 | 0.70 |
| LLaVA-OneVision-7B | 0.028 | 0.018 | 0.80 |
| SmolVLM2-2.2B† | 0.016 | 0.010 | 0.80 |
| Idefics3-8B† | 0.013 | 0.011 | 0.75 |
| *ray tracer (oracle masks)* | *0.734* | *0.645* | *0.15* |
| *fine-tuned Qwen3-VL-4B (§5)* | *0.924* | *0.909* | *0.03* |

- † These two models give a legal candidate in only about 35% of answers, so they are reported separately.
- The pre-registered H_general holds for 9 of 9 models.
- **Prompt robustness:** step-by-step reasoning leaves the top three models at 0.10–0.19, against the tracer's 0.69 on the
  same leader-leader pairs.
- **[PENDING] Frontier API models**, and models larger than 12B.

### 4.2 Why they fail (Qwen3-VL-4B; `research/gate2_final_report.md`, `research/exp1_report.md`)

- **Perception is not the bottleneck.**
  - Annotation text is read correctly 95.4% of the time [93.4, 97.2] (crop probe).
  - Relational accuracy given that both labels were read correctly is 0.539.
- **Crossings drive the errors.** Pair accuracy falls from 0.77 with no crossing to 0.30 with an entity-line crossing, and to
  0.15 when the two leaders cross.
- **The image hurts once the graph is known.**
  - Given the oracle relation graph as text, a2e pair accuracy is 0.996 with a blank image.
  - It drops to 0.820 with the diagram: −17.6 pp, p = 1.7e−24.
- **Path following is weak even without binding.**
  - With annotation text removed and the start of the line marked, pair accuracy rises only from 0.162 to 0.253
    (+9.2 pp [5.3, 13.0]).
  - A linear probe on frozen visual-token features recovers the link at 0.44 in-distribution and 0.31 on unseen topology.

## 5. Learnability (`research/transfer_report.md`, `research/datascale_report.md`)

### 5.1 Fine-tuning

- **Recipe:** answer-only supervised fine-tuning of Qwen3-VL-4B; LoRA r16 plus the vision-language merger; 600 groups; 1 epoch.
- **Results, 3 seeds:**
  - val 0.977 ± 0.011;
  - unseen topology 0.845 ± 0.026;
  - unseen template 0.833 ± 0.032;
  - unseen family 0.895 ± 0.015;
  - pooled 0.858 ± 0.012.
- The untrained model scores 0.05–0.25 on the same splits.
- Results hold on leader-leader crossings: 0.82 on unseen templates and 0.88 on circuits.
- **[PLANNED] Second base model family** (InternVL3-8B).

### 5.2 Ablations: what drives the fix

- **Counterfactual pairing.** At matched data (9,000 examples), training with the answer-flipping twin beats training with a
  style-only twin by **+3.9 pp** (mean of 3 seeds: +3.6, +1.8, +6.2; all > 0; pre-registered rule passed). The gain is
  mostly on unseen topology.
- **Data scale.** Pooled unseen accuracy is 0.49 at 50 groups, 0.72 at 150 and 0.86 at 600, and still rising.
  Scale matters more than pairing.

## 6. What does not help (pre-registered negative result; `research/pilot_report.md`, `research/transfer_report.md`)

| split | answer-only SFT (M1) | relation-SFT + contrastive relational head (M3) | rationale-first coordinates (M2R) |
|---|---|---|---|
| val | 0.983 | 0.897 (−8.6) | 0.811 (−17.2) |
| unseen topology | 0.815 | 0.598 (−21.7) | 0.472 (−34.3) |
| unseen template | 0.857 | 0.575 (−28.2) | 0.345 (−51.2) |
| unseen family | 0.900 | 0.767 (−13.3) | 0.617 (−28.3) |

All differences are p < 1e−7. The pre-registered transfer rule returned "drop the method claim".

## 7. Real technical drawings [PENDING]

- **Planned set:** about 100 US patent figures (public domain).
- **Real-image strict twins:** the two labels' pixel patches are swapped, lines and markers are untouched, and the answer flips.
- Protocol and analyses are pre-registered in `research/real_diagram_protocol.md`.

## 8. Limitations

- One generator and style system.
- Rejection sampling introduces a selection bias toward layouts where strict swaps are feasible.
- Fine-tuning has been tested on one base model so far.
- Open models only, up to 12B.
- On real drawings, the markers are an added visual element.
- Transparency: the protocol log records every deviation, including two aborted runs caused by our own logging bug,
  which was later retracted.

## Reproducibility statement

- Pre-registration and the change log: `research/experimental_protocol.md`.
- Build hashes are in the manifests, and adapter sha256 values are in the logs.
- Versions are pinned, and model revisions are recorded.
- Total compute: about $121 of H100 time.
