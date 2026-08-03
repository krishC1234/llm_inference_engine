#!/usr/bin/env bash
#
# Bootstrap the LLM inference engine on a fresh Amazon Linux 2023 GPU box (T4).
# Assumes the NVIDIA driver + CUDA are already installed (`nvidia-smi` works).
#
# Usage (from the repo root):
#     bash setup.sh
#     source .venv/bin/activate
#     python main.py
#
set -euo pipefail

# Must run from the repo root (where requirements.txt lives).
if [[ ! -f requirements.txt ]]; then
  echo "ERROR: run this from the repo root (requirements.txt not found)." >&2
  exit 1
fi

echo "==> Checking GPU is visible"
nvidia-smi -L || { echo "ERROR: nvidia-smi failed — is this the GPU box?" >&2; exit 1; }

echo "==> Checking disk space (need ~15 GB for torch + weights)"
df -h --output=avail / | tail -1

echo "==> Installing system packages (dnf)"
sudo dnf install -y git python3 python3-pip

echo "==> Creating virtualenv (.venv)"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

echo "==> Installing PyTorch (CUDA build; driver supports up to CUDA 13.3)"
pip install torch          # default Linux wheel is a CUDA build

echo "==> Installing project requirements"
pip install -r requirements.txt

echo "==> Verifying PyTorch sees the GPU"
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
else:
    raise SystemExit("CUDA NOT available to torch — reinstall from "
                     "https://download.pytorch.org/whl/cu124 (or matching cuXXX).")
PY

echo ""
echo "==> Done. Next:"
echo "    source .venv/bin/activate"
echo "    python main.py        # first run downloads TinyLlama (~2.2 GB)"
