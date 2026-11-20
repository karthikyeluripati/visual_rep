# Paper outline: target NeurIPS 2027 Evaluations & Datasets track (revised 2026-11-20)

**Working title:** *Following the Line: Strict Counterfactual Evaluation of Connector Binding in Vision-Language Models*

The method track (relational head, contrastive grounding, RL) was **stopped** by the pre-registered transfer rule
(`research/transfer_report.md`). The previous outline is in git history (before commit 2c498f5).
Every claim is tagged with its evidence: ✅ done, ⏳ running, ❌ pending a PI action.

## Contributions

- **C1, benchmark.** RelBind-Bench uses strict counterfactual twins: connector paths are swapped while arrowheads stay
  fixed, and the images are pixel-identical outside a 3-px mask. It adds reroute and style controls, crossing-balanced
  cells, held-out topology/template/family splits, a leakage audit and a reproducible build. ✅
  The human audit is ❌.
- **C2, diagnosis.**
  - Zero-shot VLMs fail: Qwen3-VL-4B gets 0.17 pair accuracy against the tracer's 0.66, with a stuck rate of 0.51. ✅
  - Recognition is not the bottleneck: annotations are read 95% of the time, yet relational accuracy given recognition is 0.54. ✅
  - The failure is concentrated on crossings: 0.77 → 0.15. ✅
  - The oracle graph closes most of the gap. ✅
  - The failure holds across 8 open VLMs. ⏳ (sweep)
- **C3, learnability.**
  - 600 groups of answer-only SFT reach 0.97 in-distribution, 0.82 on unseen topology, 0.86 on unseen templates and
    0.90 on an unseen family. ✅
  - The counterfactual-pair contribution and the data-efficiency curve. ⏳
  - Seed variance. ⏳
- **C4, negative result.** Explicit relational supervision hurts: relation-SFT, rationale-first coordinates and a
  contrastive relational head are all worse than answer-only SFT on every split, by 9–51 pp. ✅
- **C5, external validity.** A real patent-drawing set with real-image strict counterfactuals (label-patch swaps). ❌ (labels)

## Sections

1. **Introduction.** A model reads the label correctly but attaches it to the wrong part. Why strict counterfactuals
   are needed: they defeat arrowhead and layout priors.
2. **Related work.** Diagram topology and path following; counterfactual and minimal-pair VLM benchmarks; chart and
   diagram QA grounding; data-centric fixes versus objective-centric fixes.
3. **RelBind-Bench.** Generation; verifier; strict swap construction and pixel check; splits; leakage; human audit;
   datasheet.
4. **Zero-shot diagnosis.** The multi-model sweep; conditions A–F and P_read; error taxonomy; crossing strata;
   tracer baselines; frozen-feature probe (Exp. 1).
5. **Learnability.** Answer-only SFT and transfer; counterfactual-pair ablation; data-efficiency curve; seeds.
6. **What does not help.** Relation-SFT, rationale-first and the relational head (a pre-registered negative result).
7. **Real drawings.** Patent set, labeller agreement, zero-shot versus fine-tuned results.
8. **Limitations.** Single generator and style system; rejection-sampling selection bias; fine-tuning on one base
   model; real-set size; markers as an added visual element.

## Reproducibility appendix

Pre-registration and the full protocol change log, including the errors and retractions. Also: build hashes, adapter
sha256 values, pinned versions and the cost ledger.
