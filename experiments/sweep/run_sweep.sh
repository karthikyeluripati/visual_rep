#!/usr/bin/env bash
# Zero-shot multi-model sweep on strict_v3 (A_direct, orig+cf). Approved 2026-11-20 ("meet the targets").
# Per model: download -> 2-group smoke -> full run -> delete weights. Failures are logged, not fatal. Emits SWEEP_DONE.
set -uo pipefail
cd /workspace/visual_rep
PY=/workspace/venv/bin/python
export HF_HOME=/workspace/hf HF_HUB_ENABLE_HF_TRANSFER=1
D=/workspace/data/strict_v3
O=experiments/sweep
R=results/sweep
log() { echo "$(date -Is) $*"; }
mkdir -p $O/logs $R
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > $O/logs/gpu.txt
df -h /workspace > $O/logs/disk.txt
[ -f $D/manifest.json ] || RELBIND_GIT_COMMIT=$(cat COMMIT) $PY -m relbind.build --config configs/strict_v3.json --out $D --workers 24 > $O/logs/build.log 2>&1
$PY -c "import json;a=json.load(open('$D/manifest.json'))['examples_sha256'];b=json.load(open('experiments/datascale/ref/strict_v3_manifest.json'))['examples_sha256'];assert a==b;print('strict_v3 identical')" || { log "build differs"; echo "$(date -Is) SWEEP_FAILED"; exit 1; }
SPL=$($PY -c "import json;print(' '.join(json.load(open('configs/sweep_models.json'))['splits']))")
$PY -c "import json;[print(m['id'],m['bs'],m.get('revision','')) for m in json.load(open('configs/sweep_models.json'))['models']]" | while read MID BS REV; do
  TAG=$(echo $MID | tr '/' '_')
  [ -f $O/$TAG/predictions.jsonl ] && [ -f $O/$TAG/run_info.json ] && continue
  log "model $MID"
  RA=""; [ -n "$REV" ] && RA="--revision $REV"
  rm -rf /tmp/smoke_$TAG
  if ! $PY -m relbind.vlm.run --data $D --splits test_id --variants orig cf --condition A_direct --backend qwen \
        --model $MID $RA --batch-size 4 --max-groups 2 --out /tmp/smoke_$TAG > $O/logs/${TAG}_smoke.log 2>&1 < /dev/null; then
    log "SKIP $MID (smoke failed: $(tail -1 $O/logs/${TAG}_smoke.log | cut -c1-200))"
  else
    head -c 600 /tmp/smoke_$TAG/predictions.jsonl > $O/logs/${TAG}_smoke_sample.txt
    rm -rf $O/$TAG
    $PY -m relbind.vlm.run --data $D --splits $SPL --variants orig cf --condition A_direct --backend qwen \
        --model $MID $RA --batch-size $BS --out $O/$TAG > $O/logs/$TAG.log 2>&1 < /dev/null && log "done $MID" || log "FAILED $MID"
  fi
  rm -rf $HF_HOME/hub/models--$(echo $MID | sed 's#/#--#')   # free container disk
done
$PY -m relbind.baselines.heuristics --data $D --splits $SPL --method ray_tracer --out $R/ray_tracer.jsonl
log "analysis"
$PY -m relbind.sweep_analysis --data $D --sweep $O --tracer $R/ray_tracer.jsonl \
    --ref M1_finetuned=experiments/datascale/ref/M1_n600_s0_predictions.jsonl M1_finetuned=experiments/sweep/ref/M1_pilot_predictions.jsonl --out $R/sweep_analysis.json > $O/logs/analysis.log 2>&1
tail -25 $O/logs/analysis.log
log "SWEEP_DONE"
