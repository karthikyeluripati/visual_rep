# visual_rep — RelBind

**Do vision-language models bind annotations by connectivity or by prior?**
A counterfactual benchmark and study of annotation→entity relational grounding in diagrams.

Status (2026-11-19): Phase 0 and the Phase-1 diagnostic (inference only) are complete.
**Gate 2 = REVISE** (`research/gate2_final_report.md`). The human audit is pending, so benchmark validity is
pending. No training has been run, and Phase 2 is not authorised.

Start here: `research/gate2_final_report.md`, `research/error_taxonomy_final.md`, `research/tracer_vs_vlm.md`,
`research/phase2_go_no_go.md`, `research/phase2_experimental_design.md`, `research/phase2_compute_budget.md`.
Reproduce the final numbers from preserved artifacts with
`python -m relbind.final_analysis --bundle <pulled bundle> --out results/phase1/final_analysis.json`.

## Core idea

Every benchmark item comes as a group of four rendered images of one diagram:

| variant | what changes | gold answer |
|---|---|---|
| `orig` | — | a1→e1, a2→e2 |
| `cf` | only the two connector paths are swapped (text, entities, annotation and arrowhead positions are identical; checked at the pixel level) | a1→e2, a2→e1 |
| `reroute` | every connector gets a different path shape to the same endpoint | unchanged |
| `style` | ink, stroke widths, terminator glyph, dash | unchanged |

The primary metric is **paired counterfactual accuracy (PCA)**: a query counts only if it is answered
correctly on both `orig` and `cf`. Any strategy that ignores the connector pixels (proximity, text
priors, layout priors) scores exactly 0 on PCA by construction.

## Layout

```
research/          novelty_audit.md, research_questions.md, experimental_protocol.md, failure_analysis.md
configs/           frozen build configs (split rules, seeds): prototype.json, diag_v1.json
src/relbind/       schema, geometry, rendering, verification
  generation/      schematic / flowchart / circuit generators (relation graph first, then render)
  counterfactual.py  swap / reroute / style variants
  build.py         dataset builder (+ pixel checks, leakage audit, manifest with hashes)
  baselines/       B0 heuristics (random, text prior, proximity, connected-component tracer)
  metrics.py, eval.py   PCA, stuck rate, invariance, strata, cluster bootstrap, McNemar
  vlm/             diagnostic conditions A–F + P_read, parser, Qwen3-VL backend, mock backend
  analysis.py      error taxonomy + Gate-2 decision
  viz.py, validation.py  figure panels, blind human-audit sheets and scoring
data/generation/   how to regenerate data (data itself is not committed)
data/validation/   blind audit sheets (answer keys kept separate)
experiments/       run scripts and logs
results/           metrics JSON and figures
paper/             outline (contributions conditional on evidence)
tests/             unit + end-to-end smoke tests
```

## Reproduce the Phase-0 prototype (CPU)

```bash
python -m venv --system-site-packages venv && . venv/bin/activate
pip install -e .[dev]
pytest -q                                                    # 55 tests
python -m relbind.build --config configs/prototype.json --out data/prototype --svg
for m in random text_prior proximity cc_tracer; do
  python -m relbind.baselines.heuristics --data data/prototype --method $m --out results/b0/$m.jsonl
  python -m relbind.eval --data data/prototype --pred results/b0/$m.jsonl --out results/b0/${m}_metrics.json
done
python -m relbind.viz audit  --data data/prototype --n 70 --out data/validation/audit_v0
python -m relbind.viz panels --data data/prototype --split test_id --n 3 --out results/figures/panels_test_id.png
```

The build is deterministic given the config, the commit and the Pillow/FreeType versions (fonts are vendored).
`manifest.json` records the SHA-256 of every image and examples file, package versions and the commit.
The reference build used Python 3.12.3, numpy 2.5.4, Pillow 13.0.0 on Linux.

## Phase-1 diagnostic (GPU, inference only)

```bash
pip install -e .[vlm]
python -m relbind.build --config configs/diag_v1.json --out data/diag_v1
bash experiments/phase1_diag/run_all.sh data/diag_v1 experiments/phase1_diag
python -m relbind.analysis --data data/diag_v1 --diag experiments/phase1_diag --heuristics results/b0_diag --out results/phase1
```

Model: `Qwen/Qwen3-VL-4B-Instruct` @ `ebb281ec70b05090aa6165b016eac8ec08e71b17`, bf16, greedy,
native input resolution (1024×700 → 704 visual tokens), transformers 5.23.0, torch 2.8.0+cu129, 1× H100 80GB.

## Licensing

Code: see repository license (to be chosen). Fonts: DejaVu (Bitstream Vera derivative), see
`src/relbind/assets/fonts/LICENSE_DEJAVU`. All benchmark images are procedurally generated.
