# Paper outline (conditional on evidence)

Working title: **Do Vision-Language Models Bind Annotations by Connectivity or by Prior?
Counterfactual Relational Grounding in Diagrams.**
(Original working title "RelBind: Learning Counterfactual Relational Grounding in VLMs" is kept
only if Gate 3 passes.)

Every claim below is tagged with the gate that must pass before it can appear in the paper.

## Contributions

- **C1 (benchmark, Gate 1 ✓ partial, audit pending).** RelBind-Bench: procedurally generated
  annotation→entity binding with *connectivity-swap* pairs that are pixel-identical outside the swapped
  connectors, plus style and reroute negative controls, across 3 diagram families and held-out
  template, topology, font and domain splits. PCA is immune to position/text shortcuts by construction.
- **C4 (diagnosis, Gate 2).** Decomposition of failures into perception, proximity capture,
  stuck/partner capture, crossing switches and other association errors, under oracle crop, OCR,
  box and graph conditions.
- **C2 (training study, Gate 3).** Data vs. objective: unpaired relation SFT vs. counterfactually paired
  SFT vs. paired SFT + pairwise counterfactual-negative margin, at matched data and compute.
  Framed as *discriminative grounding post-training*, not a new representation architecture.
- **C3 (RL, Gate 5).** Only if it beats the best SFT and inference-time crop+verify at matched compute.
  Otherwise it is removed.

## Sections

1. Introduction: the motivating failure (text read correctly, attached to the wrong entity);
   what is new relative to Knossos/Ariadne, Trace-Without-Tracking, Enginuity and SaCLIP.
2. Related work: diagram topology extraction; path following; grounded/counterfactual RL
   (ViGoRL, FGRPO, CofactGVR, DeFacto, VC-OPD); diagram evidence grounding (DRAGON, RefChartQA, RD-Bench).
3. RelBind-Bench: generation, verification rules, counterfactual construction, pixel checks,
   splits and leakage audit, human audit, real-world set.
4. Diagnostic study (Phase 1): conditions A–F, P_read, heuristics; error taxonomy; strata
   (crossings, length, proximity rank).
5. Training study (Phase 2, if Gate 2 passes).
6. Limitations: synthetic generator, selection bias of rejection sampling, single base model family,
   real-world set size.

## Figures and tables (planned)

- Fig 1: an orig/cf pair with the model's answer stuck on the old target.
- Table 1: diagnostic conditions × splits (PCA, acc_orig, stuck, invariance) with 95% CIs.
- Fig 2: PCA vs. number of crossings and connector length.
- Table 2: error taxonomy.
- Table 3: training arms on held-out splits (paired differences, McNemar, 3 seeds).
