"""Fetch what openWakeWord's automatic training needs besides the TTS clips.

Usage: python prepare_data.py DATA_DIR [--audioset-shards N] [--no-acav]

Into DATA_DIR:
  mit_rirs/          room impulse responses (MIT IR survey, 270 files, 16 kHz)
  background/        background audio from AudioSet's balanced train set,
                     10 s clips as 16 kHz mono WAV (~500 per shard, ~1.4 h)
  validation_set_features.npy
                     ~11 h of speech, noise and music as openWakeWord
                     features: false activations per hour during training
  openwakeword_features_ACAV100M_2000_hrs_16bit.npy
                     ~2,000 h of general negative speech as openWakeWord
                     features (17 GB); --no-acav skips it (only for a quick
                     test of the pipeline: the model needs it)

All from Hugging Face; everything already there is kept, so a rerun goes on
where a broken one stopped.
"""
import argparse
import io
import sys
from pathlib import Path

import numpy as np
import scipy.io.wavfile
import soundfile
from huggingface_hub import hf_hub_download, snapshot_download

FEATURES = "davidscripka/openwakeword_features"
RIRS = "davidscripka/MIT_environmental_impulse_responses"
AUDIOSET = "agkphysics/AudioSet"


def to_16k_mono(data: np.ndarray, sr: int) -> np.ndarray:
    import torch
    import torchaudio
    if data.ndim == 2:
        data = data.mean(axis=1)
    if sr != 16000:
        data = torchaudio.functional.resample(
            torch.from_numpy(data.astype(np.float32)), sr, 16000).numpy()
    return (np.clip(data, -1, 1) * 32767).astype(np.int16)


def rirs(out: Path):
    target = out / "mit_rirs"
    if target.exists() and len(list(target.glob("*.wav"))) >= 270:
        return print(f"{target}: there")
    tmp = Path(snapshot_download(RIRS, repo_type="dataset", allow_patterns="16khz/*",
                                 local_dir=out / ".rirs"))
    target.mkdir(exist_ok=True)
    for f in (tmp / "16khz").glob("*.wav"):
        sr, d = scipy.io.wavfile.read(f)
        if d.dtype != np.int16:
            d = to_16k_mono(d.astype(np.float32), sr)
        scipy.io.wavfile.write(target / f.name, 16000, d)
    print(f"{target}: {len(list(target.glob('*.wav')))} files")


def background(out: Path, shards: int):
    import pyarrow.parquet as pq
    target = out / "background"
    target.mkdir(exist_ok=True)
    for i in range(shards):
        done = target / f".shard{i:02d}"
        if done.exists():
            continue
        path = hf_hub_download(AUDIOSET, f"data/bal_train/{i:02d}.parquet",
                               repo_type="dataset", local_dir=out / ".audioset")
        table = pq.read_table(path, columns=["video_id", "audio"])
        for vid, audio in zip(table["video_id"].to_pylist(), table["audio"].to_pylist()):
            try:
                d, sr = soundfile.read(io.BytesIO(audio["bytes"]))
            except Exception as e:     # a few clips are broken
                print(f"skipped {vid}: {e}", file=sys.stderr)
                continue
            scipy.io.wavfile.write(target / f"{vid}.wav", 16000, to_16k_mono(d, sr))
        Path(path).unlink()            # 0.7 GB each
        done.touch()
        print(f"{target}: shard {i} done, {len(list(target.glob('*.wav')))} clips")


def features(out: Path, acav: bool):
    names = ["validation_set_features.npy"]
    if acav:
        names.append("openwakeword_features_ACAV100M_2000_hrs_16bit.npy")
    for name in names:
        p = hf_hub_download(FEATURES, name, repo_type="dataset", local_dir=out)
        print(f"{p}: {Path(p).stat().st_size / 1e9:.2f} GB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--audioset-shards", type=int, default=4)
    ap.add_argument("--no-acav", action="store_true")
    a = ap.parse_args()
    out = Path(a.data_dir)
    out.mkdir(parents=True, exist_ok=True)
    rirs(out)
    background(out, a.audioset_shards)
    features(out, not a.no_acav)
