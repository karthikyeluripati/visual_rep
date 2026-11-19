# Novelty Audit — RelBind (Phase 0)

Date of audit: 2026-11-19. Auditor: research assistant (automated literature read, see "Evidence quality").

## Evidence quality

Papers were read through arXiv abstract/HTML pages, proceedings pages, and (for CofactGVR) a text
extraction of the PMLR PDF. Several HTML pages were long; the later sections were read through a
model-generated summary rather than verbatim. Claims marked **[partial]** rest partly on that summary
and must be re-checked against the PDF before being cited in a paper. Nothing here was taken from
memory alone; papers dated after mid-2026 were all looked up.

## 1. Paper-by-paper record

### 1.1 ViGoRL — Grounded RL for Visual Reasoning (Sarch et al., arXiv 2505.23678, NeurIPS 2025)
- **Task/contribution:** RL-trained VLM that ties each reasoning step to an image coordinate; multi-turn variant zooms into predicted coordinates.
- **Data/supervision:** MCTS-generated grounded reasoning traces for SFT warm start, then GRPO.
- **Method/reward:** SFT → GRPO with answer reward (+ format). Multi-turn zoom environment.
- **Metrics:** accuracy on SAT-2, BLINK, V*Bench, RoboSpatial, ScreenSpot, VisualWebArena; human rating of grounding.
- **Code:** project page visually-grounded-rl.github.io.
- **Overlap with us:** Establishes that coordinate-grounded RL with crop/zoom actions helps spatial/fine-grained tasks. Our §9 agent (inspect/crop/compare/submit) is essentially a ViGoRL-style environment on a new task. **An RL contribution from us would be incremental unless the reward is the thing being studied.**

### 1.2 Faithful GRPO (Kancheti et al., arXiv 2604.08476, COLM 2026)
- **Task:** visual spatial reasoning with faithful CoT.
- **Method/reward:** GRPO with logical-consistency and visual-grounding *constraints* enforced via Lagrangian dual ascent (not plain reward terms).
- **Models/benchmarks:** Qwen2.5-VL 3B/7B, seven spatial benchmarks.
- **Results:** inconsistency rate 24.5% (or 26.1%, versions differ) → 1.7%; +13 pp semantic grounding; +2 pp accuracy over GRPO.
- **Overlap:** Grounding-aware RL with verifiable grounding signals is established. A "relation-grounding reward" (our R2) is a domain instance of this idea.

### 1.3 DRAGON (Iyengar et al., arXiv 2604.25231, NeurIPS 2026)
- **Task:** given diagram + question + answer, predict evidence boxes that justify the answer.
- **Data:** 6 sources (AI2D, ChartQA, Circuit-VQA, InfographicsVQA, MapIQ, MapWise); 11,664 instances, 2,445 human-verified test instances released; MLLM-proposed evidence verified by humans.
- **Metrics:** MPIoU, grounding IoU, box P/R/F1.
- **Results:** grounding weak everywhere; Claude Opus 4.6 GIoU 23.8% on AI2D. No training.
- **Overlap:** Establishes that VLMs' evidence grounding on diagrams is poor. Does **not** isolate annotation→entity association, leader lines, or connectivity interventions. Potential real-world evaluation source (AI2D/Circuit-VQA subsets), but its labels are evidence *sets*, not binding edges.

### 1.4 CofactGVR (Zhang et al., ICML 2026, PMLR 306)
- **Task:** grounded visual reasoning (evidence box → answer).
- **Method:** Causal Grounding Graph with evidence as mediator; counterfactual intervention **perturbs the predicted region** with image–question context fixed; factual–counterfactual reward gap as intermediate bonus; Q-PAS (quantile-filtered prioritized advantage sampling).
- **Benchmarks:** VisCoT test split (doc/text, chart, general VQA, relation reasoning, fine-grained VQA), general LVLM benchmarks, referring-expression grounding. Baseline table includes Qwen2.5-VL-7B and Ground-R1 **[partial: base model inferred from tables]**.
- **Code:** promised at github.com/zhangy0822/CofactGVR (not verified to exist).
- **Overlap:** "Counterfactual" there means *removing/perturbing evidence*, used as a reward signal. Not diagrams, not topology. Our counterfactual is a *semantic rewiring* of the scene where the correct answer changes; that's a different intervention type.

### 1.5 Structure-aware Contrastive Learning for Diagram Understanding (SaCLIP; Sasaki, arXiv 2509.01959, ICCV Workshops 2025)
- **Task:** image–text matching and VQA on flowcharts with CLIP-style models.
- **Method:** TripletCLIP-style training with synthetic hard negatives + a Structure-aware Contrastive loss (plus a second loss).
- **Results:** gains over CLIP and hard-negative CLIP on a flowchart benchmark.
- **Overlap:** **Directly overlaps our H2 idea** (contrastive learning with structural hard negatives on synthetic diagrams). Difference: dual-encoder CLIP, whole-image–caption matching, not annotation-entity scoring inside a generative VLM, and no counterfactual minimal pairs under held-out topology.

### 1.6 Pseudo Contrastive Learning for Diagram Comprehension (Sasaki, arXiv 2602.23589, WACV Workshops 2026 — venue not independently confirmed)
- **Method:** editable pseudo-diagram synthesis from random text elements, rule-based hard contrastive data, structure-aware contrastive learning; CLIP-style.
- **Overlap:** same as 1.5; shows rule-based structural edits generate useful hard negatives. Our "connectivity-swap" negatives are one such rule.

### 1.7 AI2D-RST (Hiippala et al., arXiv 1912.03879; github.com/thiippal/AI2D-RST; CC-BY-4.0)
- **Content:** 1,000 AI2D primary-school science diagrams with expert grouping, **connectivity** (arrows/lines between elements/groups; directed/undirected) and RST discourse layers.
- **Overlap/use:** The only real-diagram source found with explicit, human-annotated *connectivity* links. Candidate for a real-world adapter, **but** connectivity links are between groups (often text-to-image-region via lines), and arrowheads are taken from original AI2D. Compatibility with our annotation→entity task must be validated per link type before use (see §5 of protocol). Field-level semantics were **not yet verified** from the JSON.

### 1.8 RefChartQA (Vogel et al., arXiv 2503.23131)
- **Task:** chart QA with bounding-box grounding of answer evidence; 73,702 samples; instruction-tuned 5 VLMs; grounding improves accuracy >15%.
- **Overlap:** Grounding-supervised SFT helps charts. Charts have legend↔series binding, which is a *color/marker-matching* relation, not a connector relation. Weak fit as real-world test; possible secondary "legend binding" transfer probe.

### 1.9 RD-Bench / RD-MCTS (Hong et al., IJCAI 2026)
- **Task:** relational-diagram reasoning (flowcharts, hierarchies, fishbone); 17K synthetic images from draw.io templates, 390K QA, four levels.
- **Method:** inference-time MCTS over a parsed graph; LoRA SFT baseline on training QA.
- **Results:** GCSR (strict global-constraint success) ~0.46–0.56 for best models; MCTS adds ~10 pp to Qwen2.5-VL-7B.
- **Overlap:** Flowchart relational reasoning benchmark + SFT baseline exists. Doesn't isolate label-to-target binding or use matched counterfactual pairs.

### 1.10 VC-OPD (Feng, arXiv 2610.14939)
- **Method:** on-policy distillation where teacher scores student tokens under the real image and a corrupted image (default patch dropout 60% **[partial]**); token weight from visual-dependence gap.
- **Benchmarks:** math/visual reasoning (Geo3K, MathVista, MathVerse, ...). Code "coming soon".
- **Overlap:** Counterfactual = degraded image; no diagrams-as-topology, no binding.

## 2. Additional related work found (through Nov 2026)

| Work | What it establishes | Overlap |
|---|---|---|
| **Knossos & Ariadne** (Guo et al., arXiv 2610.04721, Oct 2026; code + data released) | Synthetic diagram→graph topology extraction (19.2K diagrams, 6 domains). Qwen3-VL-8B direct Edge F1 46.1 → 91.6 after staged QLoRA SFT; GPT-5.6 81.1, Opus 5 70.4. External TopoBench-180 transfer partial. | **Strongest competitor for H2.** Shows relation-structured SFT on synthetic diagrams works in-distribution. No annotation/leader-line binding, **no counterfactual minimal pairs**, no test of whether gains follow connectivity rather than layout priors. |
| **VLMs Trace Without Tracking** (Hong et al., arXiv 2605.15672, ICML 2026) | Controlled path-following: VLMs switch to locally similar distractor paths; scaling, reasoning, and tracing instructions don't fix it. | **Strongest competitor for H1 mechanism.** A leader line *is* a path. Our H1 must be distinguished as *binding* (assigning text to a target through a path amid proximity/text priors), not tracing per se. |
| **TraversalBench** (arXiv 2604.10999) | Path-following accuracy collapses at crossings. | Predicts our crossing stratum; we should replicate, not claim. |
| **Enginuity** (Kumar et al., arXiv 2606.03410) | Real engineering parts diagrams; names "callout–part misalignment" failure (right part number on wrong item), worst on dense overlapping leaders; **not quantified**, OCR vs association not separated. Zero-shot only. CC-BY-4.0. | Independent real-world evidence that our failure mode exists. Candidate real-world test source (license permits). |
| **DeFacto** (Xu et al., arXiv 2509.20912) | Counterfactual (masked-evidence → "unknown") + random-mask GRPO training; R = answer + format + IoU selection reward. | Abstention-on-removed-evidence and IoU-reward designs exist; we should reuse rather than reinvent. |
| **SADU** (arXiv 2604.04009) | Software-architecture diagram benchmark; VLMs struggle with relational grounding and complex arrow layouts. | More evidence for H1-type failures; no training. |
| **VLM Reality Check** (CVPR-W 2026) | Minimal-edit counterfactual pairs on natural images, incl. compositional binding; 12–38% drops. | The *counterfactual-pair evaluation instrument* exists in natural images; novelty for us is only in the diagram-connectivity instantiation. |
| **BlindTest** ("VLMs are blind", 2024) | VLMs fail at counting line intersections etc. | Background for low-level line perception. |

## 3. What is already established (do not claim)

1. VLMs fail at diagram relational structure and evidence grounding (DRAGON, SADU, RD-Bench, Knossos).
2. VLMs fail at following lines through local distractors and crossings (Trace Without Tracking, TraversalBench, BlindTest).
3. Relation-structured SFT on synthetic diagrams strongly improves in-distribution relation extraction (Knossos/Ariadne, RD-Bench SFT).
4. Structural hard-negative contrastive learning helps CLIP-style models on synthetic flowcharts (SaCLIP, Pseudo-CL).
5. Grounding-aware RL with verifiable grounding rewards / counterfactual rewards improves grounded reasoning (ViGoRL, FGRPO, CofactGVR, DeFacto, VC-OPD).
6. Counterfactual minimal pairs as an evaluation instrument (VLM Reality Check, CCTVBench etc.).

## 4. The gap that remains (narrow, testable)

No paper we found answers:

> **When only the connectivity between a fixed annotation and fixed candidate entities changes, does a VLM's assignment follow the connector — and does training make it follow connectivity *causally*, or merely fit the layout/proximity priors of the training generator?**

Concretely, none of the above:
- (a) measures annotation→entity binding with **matched connectivity-swap pairs** where text, entities, annotation positions and terminator positions are pixel-identical and only connector paths differ (a design under which *every* position/text-only strategy scores **0% paired accuracy** by construction);
- (b) separates recognition (OCR, entity localization) from association with oracle conditions on the same items;
- (c) tests whether gains from relational SFT (Knossos-style) survive **counterfactual pairs at held-out topology/templates**, i.e., distinguishes "learned to trace" from "learned priors";
- (d) separates the effect of **counterfactual paired data** from the effect of a **pairwise/contrastive objective** at matched data and compute.

(d) is the most interesting scientific question and the one where an ML contribution could live:
SaCLIP shows hard negatives help *dual encoders*; Knossos shows plain SFT helps *generative VLMs*. Whether an explicit
contrastive/pairwise objective adds anything beyond SFT on the *same* counterfactual-paired data for a generative
VLM is open — and plausibly the answer is "no", which would itself be a useful negative result.

## 5. What our paper would establish that the strongest work has not

If (and only if) experiments support it:
1. A quantified **connectivity-following** deficit for annotation binding (paired CF accuracy, "stuck" rate) that persists after controlling for OCR/crop/oracle boxes — beyond the generic path-following deficit (we must show binding errors exceed what tracing errors on the same connector predict, or reframe as a domain instance of it).
2. A **data-vs-objective decomposition** for fixing it: answer-only SFT vs relation SFT vs relation SFT on CF-paired batches vs + pairwise contrastive scoring, at matched examples/compute, evaluated on held-out topology/templates/domain with paired tests.

## 6. Novelty verdict

- **C1 (benchmark):** partially novel. Counterfactual connectivity pairs for annotation binding are new as far as we found, but adjacent to Knossos, path-following benchmarks and Enginuity. Alone, it is a workshop/datasets-track-level contribution.
- **C2 (learned relational representation):** novelty is **at risk**. Contrastive structural hard negatives (SaCLIP) and relation SFT (Knossos) exist. Defensible only as the controlled data-vs-objective study in §4(d), not as "a new method".
- **C3 (grounding-aware RL):** **low novelty.** Five recent papers cover grounding/counterfactual RL. Keep only as a conditional experiment; drop from the paper unless it beats SFT + inference-time verification at matched compute.
- **C4 (failure decomposition):** novel in this specific form, modest in size.

**Decision: REVISE (not STOP).** Reframe the paper as:

> *"Do VLMs bind annotations by connectivity or by prior? A counterfactual study of relational grounding and how to train for it."*

Primary contribution: counterfactual diagnostic + data-vs-objective training study. RL demoted to a gated secondary question. We must cite and compare against Knossos/Ariadne (released code) as the structured-SFT baseline, and against "Trace Without Tracking" for the mechanism.

## 7. Open verification items

- Confirm AI2D-RST connectivity JSON semantics and which link types are annotation→target.
- Confirm Enginuity data release (HF DOI 10.57967/hf/8702) and whether item-number callouts have coordinates (needed for a binding adapter).
- Read Knossos/Ariadne appendices (matching rules, training config) verbatim before re-implementing it as a baseline.
- Confirm WACV-W venue for Pseudo-CL, and CofactGVR base model.
- Re-run a targeted search before submission (field moves monthly).
