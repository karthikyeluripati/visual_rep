#!/usr/bin/env bash
# Decisive Experiment 1 (inference + frozen-feature linear probe; the VLM is never trained).
# Runs on the pod from /workspace/visual_rep. Emits EXP1_DONE only if every step succeeded.
set -euo pipefail
cd /workspace/visual_rep
PY=/workspace/venv/bin/python
export HF_HOME=/workspace/hf
REV=ebb281ec70b05090aa6165b016eac8ec08e71b17
D=/workspace/data/strict_v2
O=experiments/exp1
R=results/exp1
log() { echo "$(date -Is) $*"; }
mkdir -p $O/logs $R

log "env"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > $O/logs/gpu.txt
$PY -m pip freeze > $O/logs/pip_freeze.txt
$PY -m pytest -q > $O/logs/pytest.txt 2>&1 && log "tests ok"

log "build strict_v2"
[ -f $D/manifest.json ] || RELBIND_GIT_COMMIT=$(cat COMMIT) $PY -m relbind.build --config configs/strict_v2.json --out $D --workers 24 > $O/logs/build.log 2>&1
tail -12 $O/logs/build.log

log "path-control images"
$PY -m relbind.pathctl --data $D --splits test_id test_unseen_topology > $O/logs/pathctl.log 2>&1

log "VLM A (binding) and A (path control)"
for name in A_direct A_path; do
  [ -d $O/$name ] && continue
  EX=examples.jsonl; [ $name = A_path ] && EX=path_examples.jsonl
  $PY -m relbind.vlm.run --data $D --splits test_id test_unseen_topology --condition A_direct --backend qwen \
      --revision $REV --batch-size 32 --examples-name $EX --out $O/$name > $O/logs/$name.log 2>&1
  $PY -m relbind.eval --data $D --splits test_id test_unseen_topology --pred $O/$name/predictions.jsonl \
      --examples-name $EX --out $O/$name/metrics.json --B 2000 --no-strata >> $O/logs/$name.log 2>&1
done

log "tracer"
$PY -m relbind.baselines.heuristics --data $D --splits test_id test_unseen_topology --method ray_tracer --out $R/ray_tracer.jsonl

log "probe extract"
[ -d /workspace/feats_v2 ] || $PY -m relbind.probe extract --data $D --splits train val test_id test_unseen_topology \
    --batch-size 8 --out /workspace/feats_v2 > $O/logs/probe_extract.log 2>&1
log "probe fit"
$PY -m relbind.probe fit --data $D --feats /workspace/feats_v2 --out $R/probe.json > $O/logs/probe_fit.log 2>&1

log "analysis"
$PY -m relbind.exp1_analysis --data $D --binding $O/A_direct/predictions.jsonl --path $O/A_path/predictions.jsonl \
    --tracer $R/ray_tracer.jsonl --probe $R/probe.json --out $R/exp1_analysis.json > $O/logs/analysis.log 2>&1
cat $O/logs/analysis.log
log "EXP1_DONE"
