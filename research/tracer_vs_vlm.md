# Deterministic Tracer vs VLM

All systems were scored on the **same diag_v1 examples, ground truth and metric** (a2e pair accuracy =
both the original and the counterfactual answer correct; cluster bootstrap over groups;
paired exact McNemar). Source: `results/phase1/final_analysis.json`.

## 1. What each baseline is allowed to see

| system | input at inference | privileged information | class |
|---|---|---|---|
| proximity | GT annotation and entity boxes/geometry | **yes** (layout) | oracle-layout heuristic |
| text prior | question text; answer statistics from the prototype train split | no image | text-only |
| connected-component tracer | raster image + GT text boxes + GT entity shapes as masks | **yes** (masks) | oracle-assisted raster |
| ray tracer (`src/relbind/baselines/tracer.py`) | raster image + GT text boxes + GT node shapes (entities, decision diamonds) as masks | **yes** (masks); no connector geometry, no relations | oracle-assisted raster |
| vector-path tracer | SVG connector polylines | **yes** (connectivity itself) | trivial oracle; 100% by construction, not run |
| VLM A / B | raster image + question | none | image-only |
| VLM E | image + GT boxes as 0–1000 text | boxes | oracle-assisted |
| VLM F | image + GT relations as text | relations | oracle |

No raster baseline here is a fair image-only competitor. An image-only tracer would also need text and
node detection, which would add errors. The tracer results are an *upper-bound-style reference* for
"geometry suffices", not a claim about deployable CV.

## 2. Results (a2e pair accuracy)

| subset | proximity | CC tracer | ray tracer | VLM A | VLM B | VLM E | VLM F |
|---|---|---|---|---|---|---|---|
| test_id (in-distribution), n=500 | 0.000 | 0.084 | **0.706** [0.652, 0.758] | 0.268 [0.228, 0.310] | 0.346 | 0.276 | 0.820 |
| test_id strict pairs, n=334 | 0.000 | 0.045 | 0.575 [0.509, 0.641] | 0.153 [0.117, 0.189] | 0.192 | 0.159 | 0.820 |
| unseen template, n=120 | 0.000 | — | 0.758 | 0.233 | n/a | n/a | n/a |
| unseen topology (zigzag), n=120 | 0.000 | — | 0.583 [0.500, 0.667] | 0.083 [0.033, 0.142] | n/a | n/a | n/a |
| cross-domain circuit, n=120 | 0.000 | — | 0.975 [0.942, 1.000] | 0.192 [0.125, 0.267] | n/a | n/a | n/a |
| cluttered: > 5 entities (test_id), n=116 | 0.000 | — | 0.500 | 0.147 | 0.207 | 0.181 | — |
| real diagrams | unavailable (no labelled real set exists yet) | | | | | | |

n/a = condition not run on that split (B, D, E, F were test_id only by design).

Paired differences (VLM A − ray tracer): test_id −43.8 pp [−49.6, −38.0], discordant 26 / 245,
p = 7.7e-46. Strict −42.2 pp. Unseen topology −50.0 pp. Cross-domain −78.3 pp.
VLM F − ray tracer: +11.4 pp [4.4, 18.2] on test_id. With stated relations the VLM beats the tracer.

## 3. Where each fails (test_id)

| crossing type | n | ray tracer | VLM A |
|---|---|---|---|
| none | 43 | 0.977 | 0.767 |
| entity lines only | 210 | 0.933 | 0.305 |
| leaders cross each other (no entity line) | 211 | 0.393 | 0.152 |
| leaders cross each other + entity line | 36 | 0.889 | 0.139 |

Ray-tracer outcomes (a2e, orig+cf): test_id 764 correct, 190 "no entity reached", 46 wrong entity.
Unseen topology: 185 / 33 / 22.
**The tracer's remaining errors are visual parsing failures.** It dead-ends at arrowheads and at
leader–leader crossings of orthogonal flowchart arrows. There is no association step to get wrong.
Its style invariance is poor (0.710 answers unchanged; −19.3 pp under style changes, mainly dashed
leaders). The VLM is more style-robust (0.886 unchanged, −3.6 pp).

The VLM's errors persist where the tracer succeeds: entity-line crossings, where the VLM gets 0.305 and the
tracer 0.933. Given correct recognition (§2 of the Gate-2 report), these are association/path errors of the
VLM, not text parsing.

## 4. Does the tracer challenge the premise?

Partly. On this synthetic distribution, connectivity is recoverable by simple geometry (given text/node
masks) far better than the VLM recovers it. So the paper cannot claim the task is intrinsically hard, nor
present a learned method as needed for *this* synthetic data. The defensible framing is that a 4B VLM that
reads every label correctly still fails to follow lines a 150-line procedure follows, and that zoom, OCR,
boxes and reasoning do not help. Any trained model must be compared against this tracer and should beat it
on the held-out regimes where geometry is brittle: styles, dashed leaders, zigzags. Otherwise the learning
contribution is not motivated.
