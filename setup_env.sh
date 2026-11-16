#!/usr/bin/env bash
# Set up the training environment next to this repo:
#   ../venv                    Python 3.12 virtualenv (made with uv)
#   ../piper-sample-generator  rhasspy/piper-sample-generator + LibriTTS-R generator
#
# Usage: bash setup_env.sh [cu126|cpu]     (default: cu126 if nvidia-smi works)
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$HERE")"
VENV="$ROOT/venv"
PSG="$ROOT/piper-sample-generator"

PSG_COMMIT=2971426a55072f7d22fec416ca7800df8bd23207          # v3.2.0
OWW_COMMIT=368c03716d1e92591906a84949bc477f3a834455          # openWakeWord main, 2025-12-30
GENERATOR=https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/en_US-libritts_r-medium.pt

TORCH_FLAVOUR="${1:-}"
if [ -z "$TORCH_FLAVOUR" ]; then
  if nvidia-smi >/dev/null 2>&1; then TORCH_FLAVOUR=cu126; else TORCH_FLAVOUR=cpu; fi
fi
echo "== torch: $TORCH_FLAVOUR, python env: $VENV"

command -v uv >/dev/null || pip install -q uv
[ -x "$VENV/bin/python" ] || uv venv -q -p 3.12 "$VENV"
export VIRTUAL_ENV="$VENV"

if [ ! -d "$PSG/.git" ]; then
  git clone -q https://github.com/rhasspy/piper-sample-generator "$PSG"
fi
git -C "$PSG" fetch -q --depth 1 origin "$PSG_COMMIT" 2>/dev/null || true
git -C "$PSG" checkout -q "$PSG_COMMIT"
[ -s "$PSG/models/en_US-libritts_r-medium.pt" ] || \
  curl -fsSL -o "$PSG/models/en_US-libritts_r-medium.pt" "$GENERATOR"

uv pip install -q torch==2.8.0 torchaudio==2.8.0 \
  --index-url "https://download.pytorch.org/whl/$TORCH_FLAVOUR"
uv pip install -q --no-deps "openwakeword @ git+https://github.com/dscripka/openWakeWord@$OWW_COMMIT"
uv pip install -q -r "$HERE/requirements.txt"

SITE="$("$VENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
# train.py's --convert_to_tflite defaults to the string "False", which is
# true, so it always tries the TensorFlow export: only convert when asked
sed -i 's/if args.convert_to_tflite:/if args.convert_to_tflite is True:/' "$SITE/openwakeword/train.py"
# openWakeWord's feature models (melspectrogram, embedding)
"$VENV/bin/python" -c "import openwakeword.utils as u; u.download_models(model_names=['-'])" >/dev/null 2>&1

# check that everything imports
PYTHONWARNINGS=ignore HERE="$HERE" "$VENV/bin/python" - <<'PY'
import os, sys
sys.path.insert(0, os.environ["HERE"])
import torch, torchaudio, yaml, onnxruntime, openwakeword, openwakeword.data
from generate_samples import generate_samples   # this repo + piper-sample-generator
print("torch", torch.__version__, "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
PY
echo "== ready: $VENV/bin/python, train.py at $SITE/openwakeword/train.py"
