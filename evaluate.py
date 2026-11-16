"""How a trained hey_emkay.onnx does, per threshold, as Emkay would use it.

Usage: python evaluate.py WORK_DIR [--model PATH]

* recall: share of the held-out "hey M K" clips (augmented, with noise and
  room echo) that score at or above the threshold
* look-alikes heard: share of the held-out look-alike phrases ("okay",
  "hey Mike", ...) that do
* false wake-ups per hour: on openWakeWord's validation set (~11 h of speech,
  noise and music), scored every 80 ms as Emkay does, one wake-up per 2 s
"""
import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort

THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
REST_FRAMES = 25          # 2 s of 80 ms frames, as Emkay's REST_MS


def scores(session, x: np.ndarray, batch: int = 4096) -> np.ndarray:
    name = session.get_inputs()[0].name
    fixed = session.get_inputs()[0].shape[0] == 1
    out = []
    for i in range(0, len(x), 1 if fixed else batch):
        chunk = x[i:i + (1 if fixed else batch)].astype(np.float32)
        out.append(session.run(None, {name: chunk})[0].reshape(-1))
    return np.concatenate(out)


def stream_scores(session, feats: np.ndarray, window: int) -> np.ndarray:
    windows = np.lib.stride_tricks.sliding_window_view(feats, (window, feats.shape[1]))[:, 0]
    return scores(session, windows)


def wakeups(s: np.ndarray, threshold: float) -> int:
    n, rest = 0, 0
    for v in s:
        if rest:
            rest -= 1
        elif v >= threshold:
            n, rest = n + 1, REST_FRAMES
    return n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("work_dir")
    ap.add_argument("--model")
    a = ap.parse_args()
    work = Path(a.work_dir)
    model = Path(a.model) if a.model else work / "output" / "hey_emkay.onnx"
    session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    window = session.get_inputs()[0].shape[1]

    feats = work / "output" / "hey_emkay"
    pos = scores(session, np.load(feats / "positive_features_test.npy"))
    neg = scores(session, np.load(feats / "negative_features_test.npy"))
    val = np.load(work / "data" / "validation_set_features.npy", mmap_mode="r")
    hours = len(val) * 0.08 / 3600
    stream = stream_scores(session, np.asarray(val), window)

    print(f"{model}: {len(pos)} positive and {len(neg)} look-alike test clips, "
          f"{hours:.1f} h of validation audio\n")
    print(f"{'threshold':>9}  {'recall':>7}  {'look-alikes heard':>17}  {'false wake-ups/h':>16}")
    for t in THRESHOLDS:
        print(f"{t:>9.1f}  {(pos >= t).mean():>7.1%}  {(neg >= t).mean():>17.1%}  "
              f"{wakeups(stream, t) / hours:>16.2f}")
