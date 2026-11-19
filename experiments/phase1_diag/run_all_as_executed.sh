#!/usr/bin/env bash
# Phase-1 diagnostic (inference only). Qwen3-VL-4B-Instruct, pinned revision, greedy decoding.
# Usage: bash experiments/phase1_diag/run_all.sh /workspace/data/diag_v1 experiments/phase1_diag
set -euo pipefail
DATA=${1:?data dir}
OUT=${2:?output dir}
REV=ebb281ec70b05090aa6165b016eac8ec08e71b17
PY=${PY:-/workspace/venv/bin/python}
export HF_HOME=${HF_HOME:-/workspace/hf}
run () {  # cond splits... ; batch from $BS
  local cond=$1; shift
  local tag=$cond
  [ -d "$OUT/$tag" ] && { echo "skip $tag (exists)"; return; }
  $PY -m relbind.vlm.run --data "$DATA" --splits "$@" --condition "$cond" --backend qwen \
      --revision $REV --batch-size ${BS:-32} --out "$OUT/$tag" > "$OUT/logs/$tag.log" 2>&1
  $PY -m relbind.eval --data "$DATA" --pred "$OUT/$tag/predictions.jsonl" --splits "$@" \
      --out "$OUT/$tag/metrics.json" --B 2000 >> "$OUT/logs/$tag.log" 2>&1 || true
  echo "$(date -Is) done $tag"
}
mkdir -p "$OUT/logs"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$OUT/logs/gpu.txt"
$PY -m pip freeze > "$OUT/logs/pip_freeze.txt"
ALL="test_id test_unseen_template test_unseen_topology test_cross_domain"
BS=32 run A_direct $ALL
BS=32 run P_read $ALL
BS=32 run C_oracle_crop $ALL
BS=32 run D_oracle_ocr test_id
BS=32 run E_oracle_box test_id
BS=32 run F_oracle_graph test_id
BS=48 run B_reasoning test_id
echo "$(date -Is) ALL DONE"
