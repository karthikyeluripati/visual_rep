#!/usr/bin/env bash
# CLIENT-side orchestration of Decisive Experiment 1 on an existing (stopped) RunPod pod.
#   1. retry podResume until a GPU is free (no cost while EXITED), up to MAX_WAIT_MIN
#   2. sync repo, recreate pinned env, launch run_exp1.sh + auto_stop (stops on DONE *or* FAILED)
#   3. pull + checksum-verify the bundle, then confirm EXITED via the API
# usage: bash experiments/pod/orchestrate_exp1.sh <pod_id> [MAX_WAIT_MIN]
set -u
POD=${1:?pod id}
MAX_WAIT_MIN=${2:-180}
CTL="python experiments/pod/runpod_ctl.py"
log() { echo "$(date -u +%FT%TZ) $*"; }

deadline=$(( $(date +%s) + MAX_WAIT_MIN * 60 ))
until $CTL resume "$POD" > /dev/null 2>&1; do
  [ "$(date +%s)" -ge "$deadline" ] && { log "GAVE_UP: no free GPU on host within ${MAX_WAIT_MIN} min"; exit 2; }
  log "no free GPU yet; retrying in 120 s"; sleep 120
done
log "resume accepted"
ST=$($CTL wait "$POD" RUNNING 900) || { log "pod did not reach RUNNING; requesting stop"; $CTL stop "$POD"; exit 3; }
HP=$(echo "$ST" | python -c "import sys,json; print(json.load(sys.stdin)['ssh'])")
HOST=${HP%:*}; PORT=${HP#*:}
log "running at $HOST:$PORT"
SSH="ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -o ServerAliveInterval=30 -p $PORT root@$HOST"
for i in $(seq 1 20); do $SSH true 2>/dev/null && break; sleep 15; done

tar --force-local --exclude=.git --exclude=__pycache__ --exclude=pod_results --exclude=.env -czf exp1_src.tgz \
  src tests configs pyproject.toml experiments/pod experiments/exp1
scp -q -o StrictHostKeyChecking=accept-new -P "$PORT" exp1_src.tgz root@"$HOST":/workspace/ && rm exp1_src.tgz
COMMIT=$(git rev-parse HEAD)
$SSH "mkdir -p /workspace/visual_rep /workspace/logs && cd /workspace/visual_rep && tar --no-same-owner -xzf /workspace/exp1_src.tgz && echo $COMMIT > COMMIT && \
  bash experiments/pod/setup_pod.sh > /workspace/logs/setup.log 2>&1; tail -3 /workspace/logs/setup.log; \
  cp experiments/pod/auto_stop.sh /workspace/auto_stop.sh; \
  (nohup bash -c 'bash experiments/exp1/run_exp1.sh || echo \"\$(date -Is) EXP1_FAILED\"' > /workspace/logs/exp1.log 2>&1 &); \
  (nohup bash /workspace/auto_stop.sh /workspace/logs/exp1.log 'EXP1_(DONE|FAILED)' 60 > /workspace/logs/auto_stop.log 2>&1 &); echo launched"
log "experiment launched; waiting for bundle"
bash experiments/pod/pull_bundle.sh "$HOST" "$PORT"
log "pull finished; waiting for pod to stop"
$CTL wait "$POD" EXITED 1800 && log "CONFIRMED_EXITED" || { log "pod not stopped by itself; requesting stop"; $CTL stop "$POD"; $CTL wait "$POD" EXITED 600 && log "CONFIRMED_EXITED"; }
