"""generate_samples() as openWakeWord's train.py expects it, on today's
rhasspy/piper-sample-generator.

openWakeWord's train.py does `from generate_samples import generate_samples`
from `piper_sample_generator_path`, the layout of dscripka's 2023 fork of
piper-sample-generator. That fork needs an old espeak-phonemizer stack; this
file gives train.py the same function on top of rhasspy/piper-sample-generator
(v3, the LibriTTS-R generator, piper-tts' own espeak), with two changes:

* every clip gets a random pair of the generator's 904 speakers and a random
  mix between them (the upstream loop walks speaker 0 with every other speaker
  first, so a few thousand clips would mostly be speaker 0's voice)
* clips are written as 16 kHz, 16-bit mono WAV, which openWakeWord trains on
  (the generator speaks at 22.05 kHz)

Put this file's folder in the training config as `piper_sample_generator_path`.
rhasspy/piper-sample-generator is imported from its repo (its wheel lacks
`piper_train`), by default the clone setup_env.sh makes next to this repo
(PIPER_SAMPLE_GENERATOR overrides it; PIPER_GENERATOR_MODEL the model).
"""
import logging
import os
import random
import sys
import uuid
import wave
from pathlib import Path
from typing import Iterable, List, Optional, Union

import torch
import torchaudio

# setup_env.sh puts piper-sample-generator next to this repo
_PSG = Path(os.environ.get("PIPER_SAMPLE_GENERATOR",
                           Path(__file__).resolve().parent.parent / "piper-sample-generator"))
if _PSG.is_dir() and str(_PSG) not in sys.path:
    sys.path.insert(0, str(_PSG))

from piper_sample_generator.__main__ import (  # noqa: E402
    audio_float_to_int16,
    generate_audio,
    get_phonemes,
)

_LOGGER = logging.getLogger(__name__)

#: The generator, by default next to piper-sample-generator's own models
DEFAULT_MODEL = os.environ.get(
    "PIPER_GENERATOR_MODEL", str(_PSG / "models" / "en_US-libritts_r-medium.pt"))

_cache = {}


def _load(model_path: str):
    if model_path not in _cache:
        import json
        model = torch.load(model_path, weights_only=False)
        model.eval()
        if torch.cuda.is_available():
            model.cuda()
        with open(f"{model_path}.json", encoding="utf-8") as f:
            config = json.load(f)
        resampler = torchaudio.transforms.Resample(
            config["audio"]["sample_rate"], 16000,
            lowpass_filter_width=64, rolloff=0.9475937167399596,
            resampling_method="sinc_interp_kaiser", beta=14.769656459379492)
        _cache[model_path] = (model, config, resampler)
    return _cache[model_path]


def generate_samples(
    text: Union[List[str], str],
    output_dir: Union[str, Path],
    max_samples: Optional[int] = None,
    file_names: Optional[Iterable[str]] = None,
    batch_size: int = 1,
    slerp_weights=(0.2, 0.35, 0.5, 0.65, 0.8),
    length_scales=(0.75, 1.0, 1.25),
    noise_scales=(0.667,),
    noise_scale_ws=(0.8,),
    max_speakers: Optional[int] = None,
    auto_reduce_batch_size: bool = False,
    model: str = DEFAULT_MODEL,
    verbose: bool = False,
    **kwargs,
) -> None:
    """Write `max_samples` clips of `text` (a phrase, a list of phrases used in
    turn, or a file with one per line) to `output_dir`."""
    torch_model, config, resampler = _load(model)
    voice = config["espeak"]["voice"]
    n_speakers = config["num_speakers"]
    if max_speakers is not None:
        n_speakers = min(n_speakers, max_speakers)

    if isinstance(text, str) and os.path.isfile(text):
        with open(text, encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
    elif isinstance(text, str):
        texts = [text]
    else:
        texts = list(text)
    if max_samples is None:
        max_samples = len(texts)
    names = iter(file_names) if file_names is not None else None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    done, t_idx, size = 0, 0, max(1, batch_size)
    while done < max_samples:
        n = min(size, max_samples - done)
        phrases = [texts[(t_idx + i) % len(texts)] for i in range(n)]
        ids = [get_phonemes(voice, config, p, verbose) for p in phrases]
        longest = max(len(i) for i in ids)
        ids = [i + [1] * (longest - len(i)) for i in ids]   # pad with '^', as upstream
        s1 = torch.LongTensor([random.randrange(n_speakers) for _ in range(n)])
        s2 = torch.LongTensor([random.randrange(n_speakers) for _ in range(n)])
        try:
            with torch.no_grad():
                audio, phoneme_samples = generate_audio(
                    torch_model, s1, s2, ids, random.choice(slerp_weights),
                    random.choice(noise_scales), random.choice(noise_scale_ws),
                    random.choice(length_scales), None)
        except torch.OutOfMemoryError:
            if not auto_reduce_batch_size or size == 1:
                raise
            torch.cuda.empty_cache()
            size = max(1, size // 2)
            _LOGGER.warning("Out of GPU memory: batch size now %s", size)
            continue
        audio = audio.cpu()
        for i in range(n):
            # the clip up to its last phoneme (the batch is padded), at 16 kHz,
            # normalised on its own
            end = int(phoneme_samples[i].flatten().sum().item())
            clip = resampler(audio[i, 0, :end + 1][None, :]).numpy()
            clip = audio_float_to_int16(clip)[0]
            name = next(names) if names is not None else f"{uuid.uuid4().hex}.wav"
            with wave.open(str(output_dir / name), "wb") as w:
                w.setframerate(16000)
                w.setsampwidth(2)
                w.setnchannels(1)
                w.writeframes(clip.tobytes())
        done += n
        t_idx += n
        if verbose or done % 1000 < n:
            _LOGGER.info("%s/%s clips", done, max_samples)
