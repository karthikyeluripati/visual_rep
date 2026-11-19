#!/usr/bin/env bash
# Recreate the Phase-1 environment on a fresh pod (container disk is wiped on stop).
# Pinned to the versions recorded in experiments/phase1_diag/logs/pip_freeze.txt.
set -euo pipefail
export HF_HOME=/workspace/hf
python3 -m venv --system-site-packages /workspace/venv
/workspace/venv/bin/pip install -q "transformers==5.23.0" "accelerate==1.16.0" "qwen-vl-utils==0.0.14" \
    "huggingface_hub" "hf_transfer" pytest
/workspace/venv/bin/pip install -q -e /workspace/visual_rep
HF_HUB_ENABLE_HF_TRANSFER=1 /workspace/venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download("Qwen/Qwen3-VL-4B-Instruct", revision="ebb281ec70b05090aa6165b016eac8ec08e71b17")
print("MODEL_OK", p)
PY
/workspace/venv/bin/pip freeze | grep -iE "^(torch|transformers|accelerate|qwen|tokenizers|pillow|numpy)=="
echo SETUP_DONE
