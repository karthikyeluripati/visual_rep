# Validation: blind human audit

**Status: NOT PERFORMED. Benchmark validity is PENDING.** No item in this repository has been
human-verified. The author's visual spot checks during development do not count as an audit.

## Current package: `audit_diag_v1/` (for the evaluated diagnostic set diag_v1)

Built with `python -m relbind.viz audit2 --data <diag_v1> --pairs 70 --extra 30 --out data/validation/audit_diag_v1`.

- 172 items: 71 queries shown on **both** the original and the counterfactual image (the primary
  metric needs both members to be answerable), plus 30 style/reroute/unanswerable items. Split-stratified,
  test_id-heavy.
- `sheet_reviewer_A.csv` and `sheet_reviewer_B.csv`: the same items in independently shuffled order.
- `README_REVIEWERS.md`: instructions (work alone, about 45–75 min).
- Give each reviewer **only** this folder, as a zip of `audit_diag_v1/`. Do not share the repository: the
  answer key is deliberately **not** in git. It is kept at
  `pod_results/20261119_082056/visual_rep/data/validation/audit_diag_v1_ANSWER_KEY_DO_NOT_SHARE/answer_key.json`
  (gitignored; back it up privately). SHA-256: `6e4d7f505b2f5b17fa8e5c41d9d55aca57f9e67d2a81062fc959404d604b2233`.

Scoring after both sheets come back:

```bash
python -m relbind.validation score2 --key <path>/answer_key.json --sheet-a filled_A.csv --sheet-b filled_B.csv
```

It reports per-reviewer accuracy and ambiguity flags, Cohen's kappa (correctness, ambiguity), pair-level
answerability, and the list of failed items. Pre-registered pass criterion: each reviewer ≥ 98% accuracy and
≤ 2% ambiguous, and ≥ 95% of orig/cf pairs answered correctly on both members by both reviewers.

## Superseded: `audit_v0/` (prototype build)

This single-sheet package was built from the prototype build and **contains its answer key inside the
folder**. Do not use it for blind review. It is kept for the record.

Real-world validation sets (AI2D-RST connectivity, Enginuity callouts) are not yet built.
