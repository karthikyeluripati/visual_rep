# Datasheet: RelBind-Bench strict_v3 (draft, 2026-11-20)

Format: Gebru et al., "Datasheets for Datasets". Items marked **[PI]** need a decision before release.

## Motivation

- **Purpose.** To test whether vision-language models bind a diagram annotation to the entity it designates by
  following the drawn connector (leader line), rather than by layout or text priors.
  Each item has a *strict counterfactual twin*: two connectors' paths are swapped while arrowhead positions stay
  pixel-identical, so the answer flips while everything outside the swapped connectors is pixel-identical
  (zero tolerance outside a 3-px dilated mask).
- **Creators and funding.** **[PI]**

## Composition

- **Instances.** Groups of 4 rendered diagrams (orig, cf, reroute control, style control) with queries:
  - annotation→entity (a2e): "which entity does this annotation point to?";
  - entity→annotation (e2a).
  Every example has the image, question, gold answer, candidate list, and family/template/topology/font/crossing-cell
  metadata. It also has the full scene graph (the graph is built *before* rendering).
- **Size.** 1,440 groups, 28,800 examples, 5,760 images (strict_v3).

| split | groups | families / templates | connector routes |
|---|---|---|---|
| train | 600 | schematic/elevation, flowchart/tb | straight, elbow |
| val | 90 | same | straight, elbow |
| test_id | 300 | same | straight, elbow |
| test_unseen_topology | 150 | same | zigzag |
| test_unseen_template | 150 | schematic/plan, flowchart/lr | straight, elbow |
| test_unseen_family | 150 | circuit/ladder, circuit/grid | straight, elbow |

- **Crossing type** of the swapped pair (assigned per group before generation, over the cells feasible under
  strict swaps): leader_leader 1,025, entity_only 304, none 111.
- **Labels.** Ground truth comes from the generator's scene graph. It is verified by a geometric verifier and a pixel check.
  There is no human labelling.
- **Human validation.** A blind 2-reviewer audit kit (172 items, diag_v1) is prepared. **It is not yet completed, so the
  benchmark is not claimed to be human-validated.**
- **Leakage.** Groups never straddle splits. There are no identical images across splits and no shared relation signatures.
  7 cross-split base-image pairs have dHash distance ≤ 4 (reported in the leakage report; too few to affect results).
- **Personal or sensitive data.** None. All content is synthetic: procedural shapes, labels and values.

## Collection process

- **Procedural generation.** Seeded per split from the config (`configs/strict_v3.json`, seed 20261121).
- **Rejection sampling** with a fixed per-group config draw, so rejection cannot shift the config mix.
  This adds a selection bias toward layouts where strict swaps are feasible (documented in the paper's limitations).
- **Reproducibility.** The exact build is reproducible from git commit and config. The manifest records per-split
  example sha256 values, config sha256 and library versions. A rebuild on a fresh Linux x86-64 machine was verified byte-identical (2026-11-19, 2026-11-20). A Windows build differs at the sub-pixel level (text layout engine), so the released images are the Linux build.

## Preprocessing and uses

- **Images:** 1024×700 PNG, rendered with vendored DejaVu fonts (Bitstream Vera license, permissive).
- **Intended uses:**
  - zero-shot evaluation (pair accuracy, stuck rate, invariance);
  - controlled fine-tuning studies with the train split.
- **Not intended for** claims about real-world diagram understanding without the real-diagram set
  (`research/real_diagram_protocol.md`, labels pending).
- **Known shortcut-relevant facts.** The swaps are strict, so arrowhead and position priors cannot solve a pair.
  Answer-only fine-tuning on 600 groups reaches 0.86–0.90 pair accuracy on held-out templates and family.
  So the benchmark measures a learnable skill, and it should be reported alongside a fine-tuned reference.

## Distribution and maintenance

- **License:** **[PI]** Suggested: code MIT/Apache-2.0, data CC BY 4.0.
- **Hosting:** **[PI]** Suggested: Hugging Face Datasets, with the generator in the GitHub repository.
- **Versioning.** Releases are identified by config sha256 plus git commit. Errata go in the repository changelog
  (`research/experimental_protocol.md` already records every protocol change).
- **Maintainer contact:** **[PI]**
