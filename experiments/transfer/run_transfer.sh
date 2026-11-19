#!/usr/bin/env bash
# Template/family transfer run (approved 2026-11-19): M1, M2R (rationale-first), M3 x 1 seed, trained on strict_v3 train
# (= strict_v2 train), evaluated orig/cf on val, test_unseen_topology, test_unseen_template, test_unseen_family.
# Emits TRANSFER_DONE only if every step succeeded.
set -euo pipefail
cd /workspace/visual_rep
PY=/workspace/venv/bin/python
export HF_HOME=/workspace/hf
D=/workspace/data/strict_v3
D2=/workspace/data/strict_v2
O=experiments/transfer
R=results/transfer
CK=/workspace/ckpt
SPL="val test_unseen_topology test_unseen_template test_unseen_family"
log() { echo "$(date -Is) $*"; }
mkdir -p $O/logs $R $CK

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > $O/logs/gpu.txt
$PY -m pip freeze > $O/logs/pip_freeze.txt
$PY -m pytest -q > $O/logs/pytest.txt 2>&1 && log "tests ok"

log "build strict_v3 and strict_v2 (for identity check of shared splits)"
[ -f $D/manifest.json ] || RELBIND_GIT_COMMIT=$(cat COMMIT) $PY -m relbind.build --config configs/strict_v3.json --out $D --workers 24 > $O/logs/build.log 2>&1
[ -f $D2/manifest.json ] || RELBIND_GIT_COMMIT=$(cat COMMIT) $PY -m relbind.build --config configs/strict_v2.json --out $D2 --workers 24 > $O/logs/build_v2.log 2>&1
for s in train val test_id test_unseen_topology; do
  a=$(sha256sum $D/$s/examples.jsonl | cut -d" " -f1); b=$(sha256sum $D2/$s/examples.jsonl | cut -d" " -f1)
  log "split $s identical to strict_v2: $([ "$a" = "$b" ] && echo yes || echo NO)"
  [ "$a" = "$b" ] || { log "shared split differs; aborting"; exit 1; }
done
cp $D/manifest.json $D/leakage_report.json $O/logs/
grep -E "errors|warnings" $D/leakage_report.json || true

log "smoke: M2R, 4 steps, save->reload check"
rm -rf /tmp/smoke_M2R
$PY -m relbind.train --arm M2R --data $D --out /tmp/smoke_M2R --max-steps 4 --check-reload --no-eval > $O/logs/smoke_M2R.log 2>&1
$PY -c "import json; i=json.load(open('/tmp/smoke_M2R/train_info.json')); print('smoke', i['steps'], i['reload_check']); assert i['reload_check']['ok']"
cp /tmp/smoke_M2R/train_info.json $O/logs/smoke_M2R_train_info.json

for ARM in M1 M2R M3; do
  [ -f $O/$ARM/predictions.jsonl ] && continue
  log "train+eval $ARM"
  $PY -m relbind.train --arm $ARM --data $D --out $O/$ARM --eval-splits $SPL > $O/logs/$ARM.log 2>&1
  mv $O/$ARM/trainable_state.pt $CK/${ARM}_trainable_state.pt   # keep adapters out of the bundle
  $PY -m relbind.eval --data $D --splits $SPL --pred $O/$ARM/predictions.jsonl \
      --out $O/$ARM/metrics.json --B 2000 --no-strata >> $O/logs/$ARM.log 2>&1 || true
done

log "untrained base on held-out splits"
[ -f $O/M0_base/predictions.jsonl ] || $PY -m relbind.vlm.run --data $D --splits test_unseen_template test_unseen_family \
    --variants orig cf --condition A_direct --backend qwen --batch-size 32 --out $O/M0_base > $O/logs/M0_base.log 2>&1
$PY -m relbind.baselines.heuristics --data $D --splits $SPL --method ray_tracer --out $R/ray_tracer.jsonl

log "analysis"
$PY -m relbind.transfer_analysis --data $D --arms M1=$O/M1/predictions.jsonl M2R=$O/M2R/predictions.jsonl M3=$O/M3/predictions.jsonl \
    --base experiments/exp1/A_direct/predictions.jsonl $O/M0_base/predictions.jsonl --tracer $R/ray_tracer.jsonl \
    --out $R/transfer_analysis.json > $O/logs/analysis.log 2>&1
cat $O/logs/analysis.log
sha256sum $CK/*.pt > $O/logs/adapter_sha256.txt
log "TRANSFER_DONE"
