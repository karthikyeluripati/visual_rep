#!/usr/bin/env bash
# Generic CLIENT-side orchestration: resume pod (retry until a GPU is free) -> sync -> setup -> run script
# -> auto_stop (stops on <TAG>_DONE or <TAG>_FAILED after the client pulls the bundle) -> pull+verify -> confirm EXITED.
# usage: bash experiments/pod/orchestrate.sh <pod_id> <run_script relative to repo> <TAG> [MAX_WAIT_MIN] [GRACE_MIN]
set -u
POD=${1:?pod}; RUN=${2:?run script}; TAG=${3:?tag}; MAX_WAIT_MIN=${4:-60}; GRACE=${5:-60}
INCLUDE=${6:-}  # optional bundle include list (paths relative to /workspace)
CTL="python experiments/pod/runpod_ctl.py"
log() { echo "$(date -u +%FT%TZ) $*"; }
deadline=$(( $(date +%s) + MAX_WAIT_MIN * 60 ))
until $CTL resume "$POD" > /dev/null 2>&1; do
  [ "$(date +%s)" -ge "$deadline" ] && { log "GAVE_UP: no free GPU within ${MAX_WAIT_MIN} min"; exit 2; }
  log "no free GPU yet; retrying in 120 s"; sleep 120
done
ST=$($CTL wait "$POD" RUNNING 900) || { log "not RUNNING; stopping"; $CTL stop "$POD"; exit 3; }
HP=$(echo "$ST" | python -c "import sys,json; print(json.load(sys.stdin)['ssh'])"); HOST=${HP%:*}; PORT=${HP#*:}
log "running at $HOST:$PORT"
SSH="ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -o ServerAliveInterval=30 -p $PORT root@$HOST"
for i in $(seq 1 20); do $SSH true 2>/dev/null && break; sleep 15; done
tar --force-local --exclude=__pycache__ -czf orch_src.tgz src tests configs pyproject.toml experiments/pod \
  "$(dirname "$RUN")" experiments/exp1 results/exp1 data/builds/strict_v2
scp -q -o StrictHostKeyChecking=accept-new -P "$PORT" orch_src.tgz root@"$HOST":/workspace/ && rm orch_src.tgz
COMMIT=$(git rev-parse HEAD)
$SSH "mkdir -p /workspace/visual_rep /workspace/logs && cd /workspace/visual_rep && tar --no-same-owner -xzf /workspace/orch_src.tgz && echo $COMMIT > COMMIT && \
  bash experiments/pod/setup_pod.sh > /workspace/logs/setup.log 2>&1; tail -3 /workspace/logs/setup.log; \
  cp experiments/pod/auto_stop.sh /workspace/auto_stop.sh; \
  (nohup bash -c 'bash $RUN || echo \"\$(date -Is) ${TAG}_FAILED\"' > /workspace/logs/run.log 2>&1 &); \
  (nohup bash /workspace/auto_stop.sh /workspace/logs/run.log '${TAG}_(DONE|FAILED)' $GRACE '$INCLUDE' > /workspace/logs/auto_stop.log 2>&1 &); echo launched"
log "launched $RUN; waiting for bundle"
bash experiments/pod/pull_bundle.sh "$HOST" "$PORT"
log "pull finished; waiting for pod to stop"
$CTL wait "$POD" EXITED 3600 && log "CONFIRMED_EXITED" || { log "requesting stop"; $CTL stop "$POD"; $CTL wait "$POD" EXITED 600 && log "CONFIRMED_EXITED"; }
