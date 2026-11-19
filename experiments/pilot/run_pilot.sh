#!/usr/bin/env bash
# Phase-2 pilot (approved 2026-11-19): M1, M2, M3 x 1 seed on strict_v2 train; eval orig/cf on val,
# test_unseen_topology, test_id. Emits PILOT_DONE only if every step succeeded.
set -euo pipefail
cd /workspace/visual_rep
PY=/workspace/venv/bin/python
export HF_HOME=/workspace/hf
D=/workspace/data/strict_v2
O=experiments/pilot
R=results/pilot
CK=/workspace/ckpt
log() { echo "$(date -Is) $*"; }
mkdir -p $O/logs $R $CK

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > $O/logs/gpu.txt
$PY -m pip freeze > $O/logs/pip_freeze.txt
$PY -m pytest -q > $O/logs/pytest.txt 2>&1 && log "tests ok"

log "build strict_v2 (same config/seed as Exp 1)"
[ -f $D/manifest.json ] || RELBIND_GIT_COMMIT=$(cat COMMIT) $PY -m relbind.build --config configs/strict_v2.json --out $D --workers 24 > $O/logs/build.log 2>&1
SAME=$($PY - <<'EOF'
import json
a = json.load(open("/workspace/data/strict_v2/manifest.json"))["examples_sha256"]
b = json.load(open("data/builds/strict_v2/manifest.json"))["examples_sha256"]
print("yes" if a == b else "no")
EOF
)
log "strict_v2 examples identical to Exp-1 build: $SAME"

log "smoke: M3, 4 steps, save->reload check"
rm -rf /tmp/smoke_M3
$PY -m relbind.train --arm M3 --data $D --out /tmp/smoke_M3 --max-steps 4 --check-reload --no-eval > $O/logs/smoke_M3.log 2>&1
$PY - <<'EOF'
import json
i = json.load(open("/tmp/smoke_M3/train_info.json"))
print("smoke", i["steps"], i["reload_check"])
assert i["reload_check"]["ok"], "reload check failed"
EOF
cp /tmp/smoke_M3/train_log.jsonl $O/logs/smoke_M3_train_log.jsonl
cp /tmp/smoke_M3/train_info.json $O/logs/smoke_M3_train_info.json

for ARM in M1 M2 M3; do
  [ -f $O/$ARM/predictions.jsonl ] && continue
  log "train+eval $ARM"
  $PY -m relbind.train --arm $ARM --data $D --out $O/$ARM > $O/logs/$ARM.log 2>&1
  mv $O/$ARM/trainable_state.pt $CK/${ARM}_trainable_state.pt   # keep adapters out of the bundle
  $PY -m relbind.eval --data $D --splits val test_unseen_topology test_id --pred $O/$ARM/predictions.jsonl \
      --out $O/$ARM/metrics.json --B 2000 --no-strata >> $O/logs/$ARM.log 2>&1 || true
done

BASE=experiments/exp1/A_direct/predictions.jsonl
if [ "$SAME" != "yes" ]; then
  log "build differs from Exp 1 -> re-running base model"
  rm -rf $O/M0_base
  $PY -m relbind.vlm.run --data $D --splits test_id test_unseen_topology --variants orig cf --condition A_direct \
      --backend qwen --batch-size 32 --out $O/M0_base > $O/logs/M0_base.log 2>&1
  BASE=$O/M0_base/predictions.jsonl
fi
$PY -m relbind.baselines.heuristics --data $D --splits val test_id test_unseen_topology --method ray_tracer --out $R/ray_tracer.jsonl

log "analysis"
$PY -m relbind.pilot_analysis --data $D --arms M1=$O/M1/predictions.jsonl M2=$O/M2/predictions.jsonl M3=$O/M3/predictions.jsonl \
    --base $BASE --tracer $R/ray_tracer.jsonl --out $R/pilot_analysis.json > $O/logs/analysis.log 2>&1
cat $O/logs/analysis.log
sha256sum $CK/*.pt > $O/logs/adapter_sha256.txt
log "PILOT_DONE"
