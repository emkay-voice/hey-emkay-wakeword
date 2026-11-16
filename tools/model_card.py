"""Write the Hugging Face model card (README.md) for a release's model.

Usage: python tools/model_card.py TAG OUT_DIR [NOTES_FILE]

OUT_DIR holds the release's files (hey_emkay_<tag>.onnx, .yml); NOTES_FILE
is the release's notes (gh release view TAG --json body), put in as they are.
"""
import sys
from pathlib import Path

REPO = "https://github.com/emkay-voice/hey-emkay-wakeword"

CARD = """---
license: cc-by-nc-sa-4.0
language:
- en
library_name: onnx
pipeline_tag: audio-classification
tags:
- openwakeword
- wake-word
- hotword
- openvoiceos
- ovos
- emkay
---

# hey_emkay {tag}: "hey M K", Emkay's wake word

An [openWakeWord](https://github.com/dscripka/openWakeWord) model for **"hey M K"**
(said as the letters, in English: /heɪ ɛm keɪ/), the wake word of
[Emkay](https://github.com/emkay-voice/emkay), OpenVoiceOS in a browser.

It is trained with openWakeWord's own training script on synthetic speech
([piper-sample-generator](https://github.com/rhasspy/piper-sample-generator), 904
voices), with look-alike phrases ("okay", "hey Mike", "hey Kay", ...) as negatives.
The training setup, a Colab notebook and results per threshold are in
[{repo_name}]({repo}); this model's config is `{yml}`.

## Files

| File | What it is |
|---|---|
| `{onnx}` | the model: input `[1, 16, 96]` (16 openWakeWord embeddings, 1.28 s), output a score 0–1 |
| `{yml}` | the training config it was made with |

It needs openWakeWord's shared front end (`melspectrogram.onnx`, `embedding_model.onnx`),
from openWakeWord's releases.

## Use

With openWakeWord in Python:

```python
from openwakeword.model import Model
model = Model(wakeword_models=["{onnx}"], inference_framework="onnx")
scores = model.predict(chunk)   # 1280 samples of 16 kHz, 16-bit audio
```

In OpenVoiceOS (`mycroft.conf`), with `ovos-ww-plugin-openwakeword`:

```json
{{"listener": {{"wake_word": "hey_emkay"}},
 "hotwords": {{"hey_emkay": {{"module": "ovos-ww-plugin-openwakeword",
                            "model": "{onnx}", "threshold": 0.8, "listen": true}}}}}}
```

Emkay ships it and offers it under *Wake word* in the control panel's Voice settings.

## This release

{notes}

## Licence

CC BY-NC-SA 4.0 (non-commercial), like openWakeWord's own pre-trained models: it is
trained with openWakeWord's negative features (ACAV100M) and AudioSet audio, whose
terms do not allow more. The training code is Apache-2.0.
"""

if __name__ == "__main__":
    tag, out = sys.argv[1], Path(sys.argv[2])
    notes = Path(sys.argv[3]).read_text().strip() if len(sys.argv) > 3 else ""
    onnx = next(out.glob("*.onnx")).name
    ymls = list(out.glob("*.yml"))
    card = CARD.format(tag=tag, onnx=onnx, yml=ymls[0].name if ymls else "the repo's hey_emkay.yml",
                       repo=REPO, repo_name="emkay-voice/hey-emkay-wakeword",
                       notes=notes or f"See {REPO}/releases/tag/{tag}.")
    (out / "README.md").write_text(card, encoding="utf-8")
    print(out / "README.md")
