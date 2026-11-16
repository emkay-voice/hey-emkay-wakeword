# hey-emkay-wakeword

Training for **"hey M K"**, the wake word of [Emkay](https://github.com/emkay-voice/emkay):
an [openWakeWord](https://github.com/dscripka/openWakeWord) model, trained with
openWakeWord's own `train.py` on synthetic speech, which Emkay runs in the browser
(`ovos-ww-plugin-openwakeword`, see Emkay's `web/wakeword.js`).

It is said as the letters, in English: /heɪ ɛm keɪ/.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/emkay-voice/hey-emkay-wakeword/blob/main/train_hey_emkay.ipynb)

## Train it

**On Google Colab (free T4 GPU, about 1½–2 hours):** open the notebook with the
badge above, choose *Runtime → Change runtime type → T4 GPU*, then *Runtime → Run all*.
At the end the browser downloads `hey_emkay.onnx` and the config it was trained with.

**On your own Linux machine** (an NVIDIA GPU helps a lot; on CPU it takes many hours):

```sh
git clone https://github.com/emkay-voice/hey-emkay-wakeword
bash hey-emkay-wakeword/setup_env.sh                       # ./venv, ./piper-sample-generator
venv/bin/python hey-emkay-wakeword/prepare_data.py work/data   # ~20 GB
venv/bin/python hey-emkay-wakeword/run.py work               # generate, augment, train
venv/bin/python hey-emkay-wakeword/evaluate.py work
```

Training needs about 12 GB of RAM (openWakeWord loads its validation set whole).

## What it does

1. **Synthetic clips.** [piper-sample-generator](https://github.com/rhasspy/piper-sample-generator)'s
   LibriTTS-R generator says "hey M K" in mixes of its 904 voices, at three speeds
   (50,000 clips by default). It also says look-alike phrases: those openWakeWord's
   `train.py` finds by phoneme overlap, and the ones in `hey_emkay.yml`
   (`custom_negative_phrases`: "okay" in many forms, "hey Mike", "hey Kay", ...).
2. **Augmentation.** Each clip gets room echo (MIT impulse responses), background
   audio (AudioSet), noise, EQ and gain, and becomes openWakeWord features.
3. **Training.** openWakeWord's automatic training: a small DNN over 1.28 s of
   features, against ~2,000 hours of general speech (openWakeWord's ACAV100M
   features) and the look-alikes, with early stopping on false activations per
   hour in ~11 hours of validation audio.

`evaluate.py` then prints, per threshold, the recall on held-out clips, how many
look-alikes it hears, and the false wake-ups per hour.

## Files

| File | What it is |
|---|---|
| `train_hey_emkay.ipynb` | the Colab notebook (written by `tools/make_notebook.py`) |
| `hey_emkay.yml` | the training config: phrases, negatives, sizes |
| `setup_env.sh` | the Python 3.12 environment, pinned (`requirements.txt`) |
| `prepare_data.py` | room impulse responses, background audio and features from Hugging Face |
| `run.py` | runs openWakeWord's `train.py` with the config |
| `generate_samples.py` | the TTS step for `train.py` (see below) |
| `evaluate.py` | recall and false wake-ups per threshold |

### Why `generate_samples.py`

openWakeWord's `train.py` imports `generate_samples` from a 2023 fork of
piper-sample-generator that needs an old phonemizer stack. This file gives it the same
function on today's piper-sample-generator (v3.2.0), gives every clip a random pair of
voices (the upstream loop starts with one voice paired with all others) and writes
16 kHz clips, which openWakeWord trains on.

Two more changes to the stock setup: openWakeWord is installed without its TensorFlow
extras (only needed for `.tflite`; Emkay uses `.onnx`), and `setup_env.sh` fixes
`train.py`'s `--convert_to_tflite`, whose default (the string `"False"`) made it always
try the TensorFlow export.

## Using the model in Emkay

```json
{
  "listener": {"wake_word": "hey_emkay"},
  "hotwords": {
    "hey_emkay": {"module": "ovos-ww-plugin-openwakeword", "model": "hey_emkay.onnx",
                  "threshold": 0.5, "listen": true}
  }
}
```

A higher `threshold` means fewer false wake-ups, a lower one that it hears you more easily.

Emkay does not read `model` for openWakeWord yet (it ships openWakeWord's own
models by name); that change comes with the first model.

## Models

Trained models are attached to this repo's
[releases](https://github.com/emkay-voice/hey-emkay-wakeword/releases); the config each
was trained with is in [`models/`](models).

### v0.1 (2026-11-16)

[`hey_emkay_v0.1.onnx`](https://github.com/emkay-voice/hey-emkay-wakeword/releases/download/v0.1/hey_emkay_v0.1.onnx)
(205 kB), trained on Colab's free T4 with the defaults here: 50,000 clips, 50,000 steps,
4 AudioSet shards ([`models/hey_emkay_v0.1.yml`](models/hey_emkay_v0.1.yml)).

`evaluate.py` on the held-out clips (with noise and room echo added) and 10.7 h of
validation audio:

| threshold | recall | look-alikes heard | false wake-ups/h |
|---:|---:|---:|---:|
| 0.3 | 30.8% | 0.1% | 1.40 |
| 0.5 | 28.9% | 0.1% | 0.93 |
| 0.7 | 26.7% | 0.1% | 0.75 |
| 0.8 | 25.1% | 0.1% | 0.28 |
| 0.9 | 22.3% | 0.1% | 0.09 |

On clean clips (new synthetic voices, a little room noise, streamed 80 ms at a time as
Emkay does) it heard "hey M K" in all 60 and none of "okay", "hey Mike", "hey Mickey",
"hey Mycroft" or "what time is it" (one "hey Kay" in 20). Those voices are from the
generator it was trained on, so real voices are the real test; threshold 0.8 is the
place to start.

## Licence

The code here is Apache-2.0.

The trained models are **CC BY-NC-SA 4.0** (non-commercial), like openWakeWord's own
pre-trained models: they are trained with openWakeWord's negative features (ACAV100M)
and AudioSet audio, whose terms do not allow more.
