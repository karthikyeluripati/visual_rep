#!/usr/bin/env bash
# Cross-family learnability (2026-11-20): M1 answer-only SFT on InternVL3-8B (pilot recipe), + Gemma-3-12B zero-shot
# if the PI's HF token has access. Emits XFAM_DONE.
set -euo pipefail
cd /workspace/visual_rep
PY=/workspace/venv/bin/python
export HF_HOME=/workspace/hf HF_HUB_DISABLE_XET=1
[ -f /workspace/.hf_token ] && export HF_TOKEN=$(cat /workspace/.hf_token)   # never echoed
D=/workspace/data/strict_v3
O=experiments/xfam
R=results/xfam
CK=/workspace/ckpt
MID=OpenGVLab/InternVL3-8B-hf
REVI=259a3b64a14623c0ec91a045cb43f7c5af5fa6af   # revision used in the sweep
log() { echo "$(date -Is) $*"; }
mkdir -p $O/logs $R $CK
$PY -m pytest -q > $O/logs/pytest.txt 2>&1 && log "tests ok"
[ -f $D/manifest.json ] || RELBIND_GIT_COMMIT=$(cat COMMIT) $PY -m relbind.build --config configs/strict_v3.json --out $D --workers 24 > $O/logs/build.log 2>&1
$PY -c "import json;a=json.load(open('$D/manifest.json'))['examples_sha256'];b=json.load(open('$O/ref/strict_v3_manifest.json'))['examples_sha256'];assert a==b;print('strict_v3 identical')"

log "smoke: InternVL3-8B M1, 4 steps, reload check"
rm -rf /tmp/smoke_xf
$PY -m relbind.train --arm M1 --model $MID --revision $REVI --data $D --out /tmp/smoke_xf --max-steps 4 --check-reload --no-eval > $O/logs/smoke.log 2>&1
$PY -c "import json;i=json.load(open('/tmp/smoke_xf/train_info.json'));print('smoke',i['steps'],i['n_trainable_params'],i['reload_check']);assert i['reload_check']['ok']"
cp /tmp/smoke_xf/train_info.json $O/logs/smoke_train_info.json; cp /tmp/smoke_xf/train_log.jsonl $O/logs/smoke_train_log.jsonl
log "train+eval InternVL3-8B M1 seed 0"
[ -f $O/InternVL3-8B_M1/predictions.jsonl ] || {
  $PY -m relbind.train --arm M1 --model $MID --revision $REVI --data $D --out $O/InternVL3-8B_M1 \
      --eval-splits val test_unseen_topology test_unseen_template test_unseen_family > $O/logs/InternVL3-8B_M1.log 2>&1
  mv $O/InternVL3-8B_M1/trainable_state.pt $CK/InternVL3-8B_M1.pt; }
$PY -m relbind.xfam_analysis --data $D --ft $O/InternVL3-8B_M1/predictions.jsonl --base $O/ref/InternVL3-8B_zeroshot.jsonl \
    --out $R/xfam_analysis.json > $O/logs/analysis.log 2>&1
tail -25 $O/logs/analysis.log
rm -rf $HF_HOME/hub/models--OpenGVLab--InternVL3-8B-hf $HF_HOME/xet

log "Gemma-3-12B zero-shot (only if the token has access)"
G=google/gemma-3-12b-it
if $PY -m relbind.vlm.run --data $D --splits test_id --variants orig cf --condition A_direct --backend qwen --model $G \
     --batch-size 4 --max-groups 2 --out /tmp/smoke_gemma > $O/logs/gemma_smoke.log 2>&1 < /dev/null; then
  $PY -m relbind.vlm.run --data $D --splits test_id test_unseen_template test_unseen_family --variants orig cf --condition A_direct \
      --backend qwen --model $G --batch-size 12 --out $O/google_gemma-3-12b-it > $O/logs/gemma.log 2>&1 < /dev/null && log "gemma done" || log "gemma FAILED"
else
  log "SKIP gemma: $(tail -1 $O/logs/gemma_smoke.log | cut -c1-160)"
fi
sed -i "s/hf_[A-Za-z0-9]\{20,\}/hf_REDACTED/g" $O/logs/*.log 2>/dev/null || true
sha256sum $CK/*.pt > $O/logs/adapter_sha256.txt
log "XFAM_DONE"
