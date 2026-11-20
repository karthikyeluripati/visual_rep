#!/usr/bin/env bash
# Robustness checks (2026-11-20, under the "meet the targets" approval): R1 cf-pair replication (noCF/noStyle seeds 1,2),
# R2 B_reasoning prompt for the 3 best zero-shot models (100 groups per sweep split). Emits ROBUST_DONE.
set -euo pipefail
cd /workspace/visual_rep
PY=/workspace/venv/bin/python
export HF_HOME=/workspace/hf HF_HUB_DISABLE_XET=1
D=/workspace/data/strict_v3
O=experiments/robust
R=results/robust
CK=/workspace/ckpt
log() { echo "$(date -Is) $*"; }
mkdir -p $O/logs $R $CK
$PY -m pytest -q > $O/logs/pytest.txt 2>&1 && log "tests ok"
[ -f $D/manifest.json ] || RELBIND_GIT_COMMIT=$(cat COMMIT) $PY -m relbind.build --config configs/strict_v3.json --out $D --workers 24 > $O/logs/build.log 2>&1
$PY -c "import json;a=json.load(open('$D/manifest.json'))['examples_sha256'];b=json.load(open('$O/ref/strict_v3_manifest.json'))['examples_sha256'];assert a==b;print('strict_v3 identical')"
SPL="val test_unseen_topology test_unseen_template test_unseen_family"
for S in 1 2; do for V in cf style; do
  N=M1_no$( [ $V = cf ] && echo CF || echo Style )_s$S
  [ -f $O/$N/predictions.jsonl ] && continue
  log "train+eval $N"
  $PY -m relbind.train --arm M1 --data $D --out $O/$N --eval-splits $SPL --drop-variants $V --seed $S > $O/logs/$N.log 2>&1
  mv $O/$N/trainable_state.pt $CK/${N}.pt
done; done
for MID in Qwen/Qwen3-VL-8B-Instruct Qwen/Qwen3-VL-4B-Instruct OpenGVLab/InternVL3-8B-hf; do
  T=B_$(echo $MID | tr '/' '_')
  [ -f $O/$T/predictions.jsonl ] && continue
  log "B_reasoning $MID"
  $PY -m relbind.vlm.run --data $D --splits test_id test_unseen_template test_unseen_family --variants orig cf \
      --condition B_reasoning --backend qwen --model $MID --batch-size 32 --max-groups 100 --out $O/$T > $O/logs/$T.log 2>&1
  rm -rf $HF_HOME/hub/models--$(echo $MID | sed 's#/#--#') $HF_HOME/xet
done
$PY -m relbind.baselines.heuristics --data $D --splits test_id test_unseen_template test_unseen_family --method ray_tracer --out $R/ray_tracer.jsonl
log "analysis"
CFA=""; for S in 0 1 2; do for A in noCF noStyle; do
  F=$O/M1_${A}_s$S/predictions.jsonl; [ $S = 0 ] && F=$O/ref/M1_${A}_s0.jsonl; CFA="$CFA M1_${A}_s$S=$F"; done; done
BA=""; for T in $O/B_*; do BA="$BA $(basename $T)=$T/predictions.jsonl"; done
$PY -m relbind.robustness_analysis --data $D --cf $CFA --breason $BA --tracer $R/ray_tracer.jsonl --out $R/robustness.json > $O/logs/analysis.log 2>&1
cat $O/logs/analysis.log | head -60
sha256sum $CK/*.pt > $O/logs/adapter_sha256.txt
log "ROBUST_DONE"
