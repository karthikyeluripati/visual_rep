#!/usr/bin/env bash
# Data-efficiency / cf-pair ablation / seed robustness for M1 (approved 2026-11-20). Pilot recipe, strict_v3.
# M1_n600_s0 is the transfer-run M1 (reference predictions in experiments/datascale/ref/). Emits DATASCALE_DONE.
set -euo pipefail
cd /workspace/visual_rep
PY=/workspace/venv/bin/python
export HF_HOME=/workspace/hf
D=/workspace/data/strict_v3
O=experiments/datascale
R=results/datascale
CK=/workspace/ckpt
SPL="val test_unseen_topology test_unseen_template test_unseen_family"
log() { echo "$(date -Is) $*"; }
mkdir -p $O/logs $R $CK
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > $O/logs/gpu.txt
$PY -m pip freeze > $O/logs/pip_freeze.txt
$PY -m pytest -q > $O/logs/pytest.txt 2>&1 && log "tests ok"
[ -f $D/manifest.json ] || RELBIND_GIT_COMMIT=$(cat COMMIT) $PY -m relbind.build --config configs/strict_v3.json --out $D --workers 24 > $O/logs/build.log 2>&1
$PY - <<'P'
import json
a = json.load(open("/workspace/data/strict_v3/manifest.json"))["examples_sha256"]
b = json.load(open("experiments/datascale/ref/strict_v3_manifest.json"))["examples_sha256"]
print("strict_v3 identical to transfer-run build:", a == b); assert a == b
P
log "strict_v3 identical to transfer-run build"

run() {  # name, extra args
  local N=$1; shift
  [ -f $O/$N/predictions.jsonl ] && return 0
  log "train+eval $N ($*)"
  $PY -m relbind.train --arm M1 --data $D --out $O/$N --eval-splits $SPL "$@" > $O/logs/$N.log 2>&1
  mv $O/$N/trainable_state.pt $CK/${N}_trainable_state.pt
}
run M1_noCF_s0    --drop-variants cf
run M1_noStyle_s0 --drop-variants style
run M1_n150_s0    --n-groups 150
run M1_n50_s0     --n-groups 50
run M1_n600_s1    --seed 1
run M1_n600_s2    --seed 2

log "analysis"
A="M1_n600_s0=$O/ref/M1_n600_s0_predictions.jsonl"
for N in M1_n600_s1 M1_n600_s2 M1_n150_s0 M1_n50_s0 M1_noCF_s0 M1_noStyle_s0; do A="$A $N=$O/$N/predictions.jsonl"; done
$PY -m relbind.datascale_analysis --data $D --arms $A --out $R/datascale_analysis.json > $O/logs/analysis.log 2>&1
cat $O/logs/analysis.log | tail -12
sha256sum $CK/*.pt > $O/logs/adapter_sha256.txt
log "DATASCALE_DONE"
