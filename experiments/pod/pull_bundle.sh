#!/usr/bin/env bash
# Runs on the CLIENT. Polls the pod for BUNDLE_READY, pulls bundle.tgz, verifies its sha256 and
# archive integrity, unpacks it into ./pod_results/<stamp>/, then touches PULLED on the pod so
# auto_stop.sh may stop it.
#   usage: bash experiments/pod/pull_bundle.sh <host> <port>
set -u
HOST=${1:?host}
PORT=${2:?port}
SSH="ssh -o ConnectTimeout=20 -o ServerAliveInterval=30 -p $PORT root@$HOST"
until $SSH "test -f /workspace/BUNDLE_READY" 2>/dev/null; do sleep 60; done
STAMP=$(date +%Y%m%d_%H%M%S)
DEST=pod_results/$STAMP
mkdir -p "$DEST"
for attempt in 1 2 3; do
  scp -q -P "$PORT" root@"$HOST":/workspace/bundle.tgz "$DEST/bundle.tgz" || { sleep 30; continue; }
  want=$($SSH "cat /workspace/bundle.sha256")
  got=$(sha256sum "$DEST/bundle.tgz" | cut -d" " -f1)
  if [ "$want" = "$got" ] && tar --force-local -tzf "$DEST/bundle.tgz" > /dev/null; then
    tar --force-local -xzf "$DEST/bundle.tgz" -C "$DEST" && rm "$DEST/bundle.tgz"
    $SSH "touch /workspace/PULLED"
    echo "PULLED_OK $DEST"
    exit 0
  fi
  echo "checksum/integrity mismatch (attempt $attempt)"; sleep 30
done
echo "PULL_FAILED"; exit 1
