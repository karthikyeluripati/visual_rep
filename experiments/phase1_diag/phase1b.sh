#!/usr/bin/env bash
# Phase 1b (inference + CPU analysis only; no training). Runs after the Phase-1 diagnostic.
set -u
cd /workspace/visual_rep
D=/workspace/data/diag_v1
PY=/workspace/venv/bin/python
export HF_HOME=/workspace/hf
REV=ebb281ec70b05090aa6165b016eac8ec08e71b17
OUT=experiments/phase1_diag
log() { echo "$(date -Is) $*"; }

# F audit control: identical oracle-graph prompt with a blank image
if [ ! -d $OUT/F_text_only ]; then
  $PY -m relbind.vlm.run --data $D --splits test_id --condition F_text_only --backend qwen --revision $REV \
      --batch-size 32 --out $OUT/F_text_only > $OUT/logs/F_text_only.log 2>&1
  $PY -m relbind.eval --data $D --splits test_id --pred $OUT/F_text_only/predictions.jsonl \
      --out $OUT/F_text_only/metrics.json --B 2000 >> $OUT/logs/F_text_only.log 2>&1
  log "done F_text_only"
fi

# deterministic baselines on the diagnostic set (text prior fitted on prototype train split)
$PY -m relbind.baselines.heuristics --data $D --method ray_tracer --out results/b0_diag/ray_tracer.jsonl
$PY -m relbind.eval --data $D --pred results/b0_diag/ray_tracer.jsonl --out results/b0_diag/ray_tracer_metrics.json --B 2000
mv results/b0_diag/text_prior.jsonl results/b0_diag/text_prior_NO_TRAIN_SPLIT.jsonl 2>/dev/null
mv results/b0_diag/text_prior_metrics.json results/b0_diag/text_prior_NO_TRAIN_SPLIT_metrics.json 2>/dev/null
$PY -m relbind.baselines.heuristics --data $D --method text_prior --train-data /workspace/data/prototype_v0 --out results/b0_diag/text_prior.jsonl
$PY -m relbind.eval --data $D --pred results/b0_diag/text_prior.jsonl --out results/b0_diag/text_prior_metrics.json --B 1000
log "done heuristics"

# two-reviewer blind audit package from the evaluated set
[ -d data/validation/audit_diag_v1 ] || $PY -m relbind.viz audit2 --data $D --pairs 70 --extra 30 --out data/validation/audit_diag_v1

# figures with model predictions
for s in test_id test_unseen_topology; do
  $PY -m relbind.viz panels --data $D --split $s --n 4 --pred $OUT/A_direct/predictions.jsonl --out results/figures/diag_A_${s}.png
done

# only a successful analysis releases the pod for auto-stop; on failure the pod stays up for a fix
if $PY -m relbind.gate2 --data $D --diag $OUT --heur results/b0_diag --out results/phase1/gate2_analysis.json \
     && [ -f data/validation/audit_diag_v1/sheet_reviewer_A.csv ] && [ -f $OUT/P_read/predictions.jsonl ]; then
  log "PHASE1B_DONE"
else
  log "PHASE1B_FAILED (pod kept running)"
fi
