# Data generation

Data is not committed; it is regenerated from a frozen config:

```bash
python -m relbind.build --config configs/prototype.json --out data/prototype [--svg] [--limit N]
```

Output layout:

```
<out>/manifest.json            config + sha256, git commit, package versions, image hashes, stats
<out>/leakage_report.json      split-rule conformity, straddling groups, duplicates, seed overlap, near-dups
<out>/<split>/examples.jsonl   one line per (image variant, query); see fields below
<out>/<split>/images/*.png     rendered model inputs (canonical raster)
<out>/<split>/scenes/*.json    ground-truth scene graph + render style + scale (source of truth)
<out>/<split>/svg/*.svg        optional archival vector copy
```

Example fields: `example_id, query_id, group_id, scene_id, variant (orig|cf|reroute|style), split,
image, family, template, topology, font, style, attach_fixed, query_type (a2e|e2a), query_ref,
query_text, question, answer, answerable, flips_under_cf, candidate_entities, candidate_values,
gt_annotation_bbox, gt_entity_bbox, features{connector_length, n_bends, cross_connectors,
cross_entities, cross_distractors, target_proximity_rank, nearest_is_target, ann_target_distance,
n_entities, n_annotations, n_texts, text_density, n_distractor_strokes}`.

Rejection-sampling yields per (family, template, topology) were measured on 40 seeds each
(Phase 0). Low-yield configurations (flowchart/lr/straight ≈ 15%, circuit/grid ≈ 20–45%) are
over-filtered toward geometrically easy layouts. That is a known selection bias; the group's
configuration is fixed before seeds are retried, so the family/template/topology mix is unbiased.

Builds so far:

| build | config | groups | examples | leakage | notes |
|---|---|---|---|---|---|
| prototype_v0 | configs/prototype.json @ b33ebce | 330 | 6,600 | ok (0 near-dups; min cross-split dHash 7) | 249/330 strict swaps (arrowhead positions fixed) |
| diag_v1 | configs/diag_v1.json | 430 | 8,600 | see manifest | Phase-1 diagnostic set |
