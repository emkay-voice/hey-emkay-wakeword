"""Write train_hey_emkay.ipynb (the Colab notebook) from the cells below.

Usage: python tools/make_notebook.py
"""
import json
from pathlib import Path

REPO = "https://github.com/emkay-voice/hey-emkay-wakeword"

CELLS = [
    ("markdown", """# Train "hey M K", Emkay's wake word

This notebook trains an [openWakeWord](https://github.com/dscripka/openWakeWord) model for
**"hey M K"** (said as the letters, in English) for [Emkay](https://github.com/emkay-voice/emkay),
with openWakeWord's own training script and synthetic speech from
[piper-sample-generator](https://github.com/rhasspy/piper-sample-generator) (904 voices).

**Before you start:** *Runtime → Change runtime type → T4 GPU*. Then *Runtime → Run all*.
Each step ends with a green ✅ line; if one fails it stops there with a red ❌ error.
It takes about 1½–2 hours on a free T4. Keep the tab open; at the end the browser downloads
`hey_emkay.onnx`.

Everything is in [{repo}]({repo}): the training config (`hey_emkay.yml`: phrases, negatives,
sizes), the data download and the environment."""),
    ("code", """#@title Settings
N_SAMPLES = 50000      #@param {type:"integer"}
N_SAMPLES_VAL = 5000   #@param {type:"integer"}
STEPS = 50000          #@param {type:"integer"}
AUDIOSET_SHARDS = 4    #@param {type:"integer"}
TTS_BATCH_SIZE = 50    #@param {type:"integer"}

import subprocess
gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                     capture_output=True, text=True)
print("GPU:", gpu.stdout.strip() or "none - choose Runtime → Change runtime type → T4 GPU, "
      "or this runs on CPU (many hours)")
PY = "/content/venv/bin/python"
WORK = "/content/work"

def sh(cmd, done="done"):
    \"\"\"Run a shell command, show its output as it comes, and stop here (red) if it fails.\"\"\"
    import subprocess, sys
    p = subprocess.Popen(cmd, shell=True, cwd="/content", stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    for chunk in iter(lambda: p.stdout.read(256), ""):
        sys.stdout.write(chunk)
        sys.stdout.flush()
    if p.wait() != 0:
        raise RuntimeError(f"❌ FAILED (exit {p.returncode}) - send the output above to get help")
    print(f"\\n✅ {done}")"""),
    ("code", f"""#@title Install (~3 min)
sh("(test -d hey-emkay-wakeword || git clone -q {REPO}) && git -C hey-emkay-wakeword pull -q"
   " && bash hey-emkay-wakeword/setup_env.sh", "installed")"""),
    ("code", """#@title Download data (~20 GB, ~10 min)
# room impulse responses, AudioSet background audio, openWakeWord's negative
# and validation features (Hugging Face)
sh(f"{PY} hey-emkay-wakeword/prepare_data.py {WORK}/data --audioset-shards {AUDIOSET_SHARDS}"
   " && df -h /content | tail -1", "data downloaded")"""),
    ("code", """#@title 1. Generate synthetic clips (~20–30 min)
# positive "hey M K" clips and look-alike negatives; rerun to go on after a break
sh(f"{PY} hey-emkay-wakeword/run.py {WORK} --stage generate --n-samples {N_SAMPLES} "
   f"--n-samples-val {N_SAMPLES_VAL} --steps {STEPS} --tts-batch-size {TTS_BATCH_SIZE}", "clips generated")"""),
    ("code", """#@title Listen to a few clips
import glob, random
from IPython.display import Audio, display
for kind in ("positive_train", "negative_train"):
    print(kind)
    clips = glob.glob(f"{WORK}/output/hey_emkay/{kind}/*.wav")
    for f in random.sample(clips, min(4, len(clips))):
        display(Audio(f))"""),
    ("code", """#@title 2. Augment and compute features (~30–40 min)
sh(f"{PY} hey-emkay-wakeword/run.py {WORK} --stage augment --n-samples {N_SAMPLES} "
   f"--n-samples-val {N_SAMPLES_VAL} --steps {STEPS} --tts-batch-size {TTS_BATCH_SIZE}", "features computed")"""),
    ("code", """#@title 3. Train (~20–40 min)
sh(f"{PY} hey-emkay-wakeword/run.py {WORK} --stage train --n-samples {N_SAMPLES} "
   f"--n-samples-val {N_SAMPLES_VAL} --steps {STEPS} --tts-batch-size {TTS_BATCH_SIZE}", "model trained")"""),
    ("code", """#@title Check the model and download it
sh(f"{PY} hey-emkay-wakeword/evaluate.py {WORK}", "evaluated")
import shutil
from google.colab import files
shutil.copy(f"{WORK}/hey_emkay_run.yml", f"{WORK}/output/hey_emkay_run.yml")
files.download(f"{WORK}/output/hey_emkay.onnx")
files.download(f"{WORK}/output/hey_emkay_run.yml")"""),
    ("markdown", """## Next

Emkay will take `hey_emkay.onnx` as a wake word model with this `mycroft.conf` (the support for
it lands in Emkay with the first model):

```json
{
  "listener": {"wake_word": "hey_emkay"},
  "hotwords": {
    "hey_emkay": {"module": "ovos-ww-plugin-openwakeword", "model": "hey_emkay.onnx",
                  "threshold": 0.5, "listen": true}
  }
}
```

Try it in a real room and tune `threshold`: higher means fewer false wake-ups,
lower means it hears you more easily."""),
]


def cell(kind, src):
    lines = src.splitlines(keepends=True)
    c = {"cell_type": kind, "metadata": {}, "source": lines}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c


nb = {
    "cells": [cell(k, s.replace("{repo}", REPO)) for k, s in CELLS],
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}
out = Path(__file__).resolve().parent.parent / "train_hey_emkay.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
print(out)
