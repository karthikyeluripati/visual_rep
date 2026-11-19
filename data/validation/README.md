# Validation: blind human audit

`audit_v0/` was produced by

```bash
python -m relbind.viz audit --data data/prototype --n 70 --out data/validation/audit_v0
```

It holds 70 items: 10 per split, one random variant each. Give reviewers **only**
`audit_sheet.csv` and `images/`. Each reviewer fills in `reviewer_answer` and
`unambiguous(y/n)` without seeing `answer_key.json`. Then:

```bash
python -m relbind.validation score --audit data/validation/audit_v0 --sheet <reviewer>.csv
```

Pass criterion (pre-registered): reviewer accuracy ≥ 98% and ≤ 2% items flagged ambiguous.
**Status: not yet performed.** Two independent human reviewers are needed. The author's visual
spot-checks during development do not count as the audit. Gate 2 cannot be passed until it is done.

Real-world validation sets (AI2D-RST connectivity, Enginuity callouts) are not yet built. See
`research/experimental_protocol.md` §1.
