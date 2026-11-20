# External validity on real diagrams: protocol (pre-registered 2026-11-20; labels pending)

## Purpose

Every result so far comes from one procedural generator. Reviewers at the target venue (NeurIPS 2027 E&D) will ask
whether two findings hold on real drawings:

1. Zero-shot VLMs fail strict counterfactual binding.
2. Answer-only fine-tuning on synthetic strict pairs fixes the failure.

## Data

- **Source:** about 100 US patent figures (public domain) with ≥ 4 numeric reference labels and leader lines.
- **Labelling:** two labellers following `data/real/README_LABELLERS.md`. 20 figures are labelled by both.
- **Labeller-agreement gate:** the median `end`-point disagreement must be ≤ 2% of the image diagonal before any model is scored.
  Callouts where the labellers disagree by more than 5% are dropped.
- **Labels are not fabricated.** No model output is used in labelling. I cannot label these, and the PI decides who does.

## Items

`src/relbind/real.py`, tested in `tests/test_real.py`. Per figure, up to 3 label pairs with similar text-box sizes.

- **Original image:** markers A–D at the endpoints of the pair and 2 distractor callouts. Letters are shuffled with a fixed seed.
- **Counterfactual image:** the two labels' pixel patches are swapped. Lines and markers are untouched, so the answers flip.
  The test verifies that orig and cf differ only inside the two text boxes.
- **Metric:** pair accuracy (both images correct). Chance level is 1/16 for a pair; 1/4 per image.

## Planned analyses (fixed now)

- **R1:** zero-shot pair accuracy for every sweep model, plus fine-tuned M1 (adapter rebuilt from code, seed and data;
  sha256 recorded), with cluster bootstrap over figures.
- **R2:** "fine-tuning transfers to real drawings" iff M1 − (its own base model) ≥ +10 pp, with CI lower bound > 0.
- **R3:** "synthetic failure is predictive" iff the Spearman correlation between sweep models' synthetic pooled pair accuracy
  and their real pair accuracy is ≥ 0.5.

## Known limitations

- Markers are an added visual element, not present in the original drawing.
- Patch swaps with slightly different box sizes are resized, which can leave small artefacts. The identical-outside-boxes
  property still holds.
