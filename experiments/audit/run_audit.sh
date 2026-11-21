#!/usr/bin/env bash
# Build the strict_v3 blind two-reviewer audit kit on Linux (the released build's platform). CPU only. Emits AUDIT_DONE.
set -euo pipefail
cd /workspace/visual_rep
PY=/workspace/venv/bin/python
D=/workspace/data/strict_v3
log() { echo "$(date -Is) $*"; }
mkdir -p experiments/audit/logs
[ -f $D/manifest.json ] || RELBIND_GIT_COMMIT=$(cat COMMIT) $PY -m relbind.build --config configs/strict_v3.json --out $D --workers 24 > experiments/audit/logs/build.log 2>&1
$PY -c "import json;a=json.load(open('$D/manifest.json'))['examples_sha256'];b=json.load(open('experiments/audit/strict_v3_manifest.json'))['examples_sha256'];assert a==b;print('strict_v3 identical')"
$PY -m relbind.viz audit3 --data $D --splits test_id test_unseen_topology test_unseen_template test_unseen_family \
    --pairs-per-split 20 --extra 20 --out data/validation/audit_strict_v3
ls data/validation/audit_strict_v3 | head; ls data/validation/audit_strict_v3/images | wc -l
log "AUDIT_DONE"
