#!/usr/bin/env bash
# Runs ON the pod. Waits for a pipeline marker, bundles results, waits for the client to pull the
# bundle (it touches $W/PULLED), then stops the pod. This RunPod pod has no persistent volume, so the
# container disk is wiped on stop. Hence: pull first. If nobody pulls within GRACE_MIN minutes, the
# pod stops anyway: idling longer would cost about as much as re-running the diagnostic.
#   usage: nohup bash auto_stop.sh <marker-log> <marker-string> [GRACE_MIN] > /workspace/logs/auto_stop.log 2>&1 &
set -u
MARK_LOG=${1:?log file to watch}
MARK=${2:?marker string}
GRACE_MIN=${3:-90}
W=/workspace
R=$W/visual_rep
log() { echo "$(date -Is) $*"; }

log "waiting for '$MARK' in $MARK_LOG"
until grep -q "$MARK" "$MARK_LOG" 2>/dev/null; do sleep 30; done
log "pipeline done; bundling"
rm -f $W/BUNDLE_READY $W/PULLED
cd $W
# small, irreplaceable outputs only; images/scenes are regenerable from config + commit
tar -czf $W/bundle.tgz visual_rep/experiments visual_rep/results visual_rep/data/validation visual_rep/COMMIT \
  logs $(ls -d data/*/manifest.json data/*/leakage_report.json data/*/*/examples.jsonl data/*/*/path_examples.jsonl 2>/dev/null)
sha256sum $W/bundle.tgz | cut -d" " -f1 > $W/bundle.sha256
touch $W/BUNDLE_READY
log "bundle ready ($(du -h $W/bundle.tgz | cut -f1)); waiting up to ${GRACE_MIN} min for client pull"

deadline=$(( $(date +%s) + GRACE_MIN * 60 ))
while [ ! -f $W/PULLED ] && [ "$(date +%s)" -lt "$deadline" ]; do sleep 20; done
if [ -f $W/PULLED ]; then log "client confirmed pull"; else log "WARNING: grace period expired without pull"; fi

# never stop while GPU work is still running
while pgrep -f "relbind.vlm.run" > /dev/null; do log "GPU job still running; waiting"; sleep 60; done

export RUNPOD_API_KEY=$(tr "\0" "\n" < /proc/1/environ | sed -n "s/^RUNPOD_API_KEY=//p")
POD=$(tr "\0" "\n" < /proc/1/environ | sed -n "s/^RUNPOD_POD_ID=//p")
log "stopping pod $POD"
runpodctl stop pod "$POD" 2>&1 | sed "s/^/runpodctl: /"
log "stop requested (exit ${PIPESTATUS[0]})"
