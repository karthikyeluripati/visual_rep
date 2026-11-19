"""Minimal RunPod GraphQL control (status / resume / stop). Key from .env (RUNPOD_API_KEY); never printed.

    python experiments/pod/runpod_ctl.py status <pod_id>
    python experiments/pod/runpod_ctl.py resume <pod_id>
    python experiments/pod/runpod_ctl.py stop <pod_id>
    python experiments/pod/runpod_ctl.py wait <pod_id> RUNNING|EXITED [timeout_s]
"""
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

URL = "https://api.runpod.io/graphql"


def key() -> str:
    for line in Path(".env").read_text(encoding="utf-8").splitlines():
        if "RUNPOD" in line.upper() and "=" in line:
            return line.split("=")[-1].strip().strip('"').strip("'")
    sys.exit("no RunPod key in .env")


def gql(q: str) -> dict:
    req = urllib.request.Request(URL, data=json.dumps({"query": q}).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {key()}",
                                          "User-Agent": "relbind-ctl"})
    with urllib.request.urlopen(req, timeout=30) as r:
        d = json.loads(r.read())
    if "errors" in d or "error" in d:
        raise RuntimeError(re.sub(r"rpa_[A-Za-z0-9]+", "<redacted>", json.dumps(d)))
    return d["data"]


def status(pod: str) -> dict:
    d = gql('query { pod(input: {podId: "%s"}) { id desiredStatus costPerHr '
            'runtime { uptimeInSeconds ports { ip isIpPublic privatePort publicPort type } } } }' % pod)["pod"]
    ssh = None
    for p in ((d.get("runtime") or {}).get("ports") or []):
        if p["privatePort"] == 22 and p["isIpPublic"]:
            ssh = f"{p['ip']}:{p['publicPort']}"
    return {"status": d["desiredStatus"], "costPerHr": d["costPerHr"], "ssh": ssh,
            "uptime_s": (d.get("runtime") or {}).get("uptimeInSeconds")}


def main():
    cmd, pod = sys.argv[1], sys.argv[2]
    if cmd == "status":
        print(json.dumps(status(pod)))
    elif cmd == "resume":
        print(json.dumps(gql('mutation { podResume(input: {podId: "%s", gpuCount: 1}) { id desiredStatus } }' % pod)))
    elif cmd == "stop":
        print(json.dumps(gql('mutation { podStop(input: {podId: "%s"}) { id desiredStatus } }' % pod)))
    elif cmd == "wait":
        want, timeout = sys.argv[3], float(sys.argv[4]) if len(sys.argv) > 4 else 600
        t0 = time.time()
        while time.time() - t0 < timeout:
            s = status(pod)
            if s["status"] == want and (want != "RUNNING" or s["ssh"]):
                print(json.dumps(s))
                return
            time.sleep(15)
        sys.exit(f"timeout waiting for {want}: {status(pod)}")


if __name__ == "__main__":
    main()
