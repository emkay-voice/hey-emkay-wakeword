"""Train hey_emkay with openWakeWord's own train.py.

Usage (with the venv setup_env.sh makes):
  python run.py WORK_DIR [--stage generate|augment|train|all]
                [--n-samples N] [--n-samples-val N] [--steps N] [--tts-batch-size N]

WORK_DIR holds data/ (prepare_data.py) and gets output/ and the run's config
(hey_emkay_run.yml: hey_emkay.yml with the overrides, paths made absolute).
Each stage can be rerun; generation goes on from the clips already there.
"""
import argparse
import subprocess
import sys
from pathlib import Path

import yaml

import openwakeword

HERE = Path(__file__).resolve().parent
TRAIN = Path(openwakeword.__file__).parent / "train.py"
STAGES = {"generate": "--generate_clips", "augment": "--augment_clips", "train": "--train_model"}


def config(work: Path, a) -> Path:
    c = yaml.safe_load((HERE / "hey_emkay.yml").read_text())
    for key in ("n_samples", "n_samples_val", "steps", "tts_batch_size"):
        if getattr(a, key) is not None:
            c[key] = getattr(a, key)
    c["piper_sample_generator_path"] = str(HERE)
    c["output_dir"] = str(work / "output")
    c["rir_paths"] = [str(work / p) for p in c["rir_paths"]]
    c["background_paths"] = [str(work / p) for p in c["background_paths"]]
    c["false_positive_validation_data_path"] = str(work / c["false_positive_validation_data_path"])
    c["feature_data_files"] = {k: str(work / v) for k, v in c["feature_data_files"].items()}
    path = work / "hey_emkay_run.yml"
    path.write_text(yaml.safe_dump(c, sort_keys=False))
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("work_dir")
    ap.add_argument("--stage", choices=[*STAGES, "all"], default="all")
    ap.add_argument("--n-samples", dest="n_samples", type=int)
    ap.add_argument("--n-samples-val", dest="n_samples_val", type=int)
    ap.add_argument("--steps", type=int)
    ap.add_argument("--tts-batch-size", dest="tts_batch_size", type=int)
    a = ap.parse_args()
    work = Path(a.work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    cfg = config(work, a)
    for stage in (STAGES if a.stage == "all" else [a.stage]):
        print(f"== {stage}", flush=True)
        subprocess.run([sys.executable, str(TRAIN), "--training_config", str(cfg), STAGES[stage]],
                       cwd=work, check=True)
    model = work / "output" / "hey_emkay.onnx"
    if model.exists():
        print(f"== model: {model} ({model.stat().st_size / 1e3:.0f} kB)")
