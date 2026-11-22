# Status report — 2026-11-22

Target: NeurIPS 2027 **Evaluations & Datasets track** (deadline ~mid-May 2027, projected). A main-track
"technical contribution" route is under discussion (see §4). Every number below traces to a committed report.

## 1. Work done

| Stage | What | Result | Report |
|---|---|---|---|
| Benchmark | RelBind-Bench strict_v3: strict counterfactual twins (connector paths swapped, arrowheads fixed, pixel-identical outside a 3px mask), reroute and style controls, 1,440 groups / 28,800 examples, held-out topology / template / family splits | Leakage audit 0 errors; byte-reproducible on Linux | `datasheet.md` |
| Diagnosis (Qwen3-VL-4B) | Crop probes, oracle conditions, crossing breakdown, path-following control, frozen-feature linear probe | Reads labels (95.4%) but misbinds; crossings drive errors (0.77 → 0.15); image hurts given oracle graph (−17.6 pp); path control +9.2 pp ("intermediate"); probe peaks mid-depth, 0.44 ID / 0.31 unseen topology (inconclusive) | `gate2_final_report.md`, `exp1_report.md` |
| Zero-shot sweep | 9 open models, 6 families, 2B–12B; B_reasoning prompt check | Best 0.195 pair acc. vs ray tracer 0.734; stuck 60–80%; H_general holds 9/9; CoT does not help | `sweep_report.md` |
| Method arms | M1 answer-only SFT vs M2R rationale-first vs M3 relational head | M1 best everywhere; M2R/M3 hurt by 9–51 pp; pre-registered rule → drop method claim | `pilot_report.md`, `transfer_report.md` |
| Learnability | M1, 3 seeds, data-scale curve, CF-pairing ablation | 0.858 ± 0.012 pooled held-out; 0.49 / 0.72 / 0.86 at 50 / 150 / 600 groups; CF pairing +3.9 pp (replicated) | `datascale_report.md` |
| Cross-family | Same recipe on InternVL3-8B | 0.099 → 0.616 (+51.7 pp [47.9, 55.4]), rule passed; weaker template transfer (0.47 vs 0.83) | `xfam_report.md` |
| Human audit kit | 180-item blind 2-reviewer kit (strict_v3), images hash-verified vs release; answer key out of git, sha256 af1f35e4… logged | Kit ready; reviewer zips prepared (local, gitignored) | `experimental_protocol.md` |
| Real-drawing tooling | Patent-figure item builder with label-patch swap twins, tests | Code ready; no labels yet | `real_diagram_protocol.md` |
| Paper | Outline + draft v0 | Results sections mostly drafted; related work, LaTeX, figures not started | `paper/draft.md` |

Compute to date: about $130 of H100 time. The pod (uux57hw32tdm1j) is confirmed EXITED via the provider API.

## 2. Hypothesis scorecard

- **H1 (binding deficit):** supported. Binding vs pure path-following is "intermediate" (+9.2 pp).
- **H2 / H2b (explicit relational supervision beats answer-only SFT):** **negative** (pre-registered).
- **H2a (counterfactual pairing helps):** modest positive, +3.9 pp, replicated over 3 seeds.
- **H3 (RL):** not run; gate not met.

## 3. Blockers before submission

| # | Blocker | Status | Waiting on |
|---|---|---|---|
| 1 | Human audit (2 blind reviewers, ~75 min each) | Kit ready | PI: recruit reviewers, return CSVs |
| 2 | Real patent drawings (~100 figures) | Code + protocol ready | Labelling |
| 3 | Frontier API models | Not started | PI: decision, keys, spending cap |
| 4 | Release (license, HF hosting, contact, anonymized mirror) | Suggestions in datasheet | PI decisions |
| 5 | >12B models, Gemma | Blocked | Larger pod disk (~150GB); Gemma license approval |
| 6 | Paper (related work with verified citations, LaTeX, figures) | Draft v0 | — |
| 7 (new) | Knossos/Ariadne structured-SFT baseline | Not started | Plan to be pre-registered |

## 4. Open question: main track vs E&D

- **Current assessment:** benchmark + diagnosis + learnability is a solid E&D paper. It is not main-track-level:
  there is no winning method (ours lost to plain SFT), and no causal mechanism (one correlational probe, one model, inconclusive).
- **Closest prior work:**
  - "VLMs Trace Without Tracking" (Hong et al., ICML 2026) for the mechanism;
  - Knossos & Ariadne (Guo et al., 2026) for the fix.
  Citations come from an automated audit (`novelty_audit.md`) and must be verified against the PDFs.
- **Candidate main-track route:** a causal account of *how* SFT fixes binding (activation patching between twins,
  base vs fine-tuned, Qwen + InternVL) plus the Knossos baseline. About 2–4 weeks and $30–80; outcome uncertain.
- **Next:** the PI is discussing approach options externally. No compute is committed until a plan is pre-registered.

## 5. Housekeeping done today

- Datasheet and draft updated: audit is now 180 items (strict_v3); limitations cover two base models and the missing
  Knossos baseline; compute total is ~$130.
- Reminders for the PI:
  - back up both answer keys off-machine;
  - rotate the HF token (it was pasted in chat).
