"""Fine-tunes YOLO11n (COCO-pretrained) on the cleaned RDD2022 subset.

Run training/prepare_dataset.py first. Saves the fine-tuned checkpoint to
weights/yolo11n/yolo11n_rdd2022.pt, which detector_interface/plugins/yolo11n.py loads for
real inference. Prints final val-set accuracy metrics (mAP50, mAP50-95, precision, recall)
straight from ultralytics' own evaluation.

Trains against data_fast.yaml (val_small — a small val slice, prepare_dataset.py writes it
alongside the full data.yaml) rather than data.yaml, so per-epoch validation for the
patience/early-stop check doesn't re-score the entire ~5.8k-image val set every epoch. The final
accuracy numbers below still come from the full test split either way — data_fast.yaml's "test"
key points at the same place as data.yaml's.

CPU-only defaults below (imgsz=256, epochs=15, batch=8) — see ASBcode.md for real measured
per-epoch timing from an actual run, not a guess. Raise epochs/imgsz/batch/MAX_TRAIN_IMAGES
yourself if you have more time and want a better model — this is a real training run, not a demo.

Pause/resume (added 2026-08-27, see training/pause_control.py for the full mechanism and why):
    python training/pause_control.py pause   # from another terminal, while this is running
    python training/train_yolo11n.py --resume   # continues from the checkpoint afterward
Stops cleanly at the next epoch boundary and exits the process (frees its RAM), not an
in-process sleep — see training/pause_control.py's docstring for why it's epoch-boundary, not
batch-boundary (a first version tried the finer-grained approach and it silently broke resume;
caught and fixed by actually testing resume end-to-end, not just the pause half).
"""

import shutil
from pathlib import Path

import pause_control

ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = ROOT / "data" / "RDD2022" / "data.yaml"
FAST_DATA_YAML = ROOT / "data" / "RDD2022" / "data_fast.yaml"
BASE_WEIGHTS = ROOT / "weights" / "yolo11n" / "yolo11n.pt"
FINETUNED_WEIGHTS = ROOT / "weights" / "yolo11n" / "yolo11n_rdd2022.pt"
RUNS_DIR = ROOT / "training" / "runs"
RUN_NAME = "yolo11n_rdd2022"


def main() -> None:
    if not FAST_DATA_YAML.exists():
        raise RuntimeError(f"{FAST_DATA_YAML} not found — run training/prepare_dataset.py first.")

    model, results, paused = pause_control.run_training(
        source_weights=BASE_WEIGHTS,
        run_name=RUN_NAME,
        runs_dir=RUNS_DIR,
        train_kwargs=dict(
            data=str(FAST_DATA_YAML),
            epochs=15,
            imgsz=256,
            batch=8,
            device="cpu",
            project=str(RUNS_DIR),
            name=RUN_NAME,
            exist_ok=True,
            patience=5,
            # Pinned instead of optimizer="auto" (added 2026-08-27): auto's choice depends on
            # train-set size (it switches SGD vs AdamW and scales lr0 by iteration count), which
            # would make this drift out of sync with train_yolo12s.py/train_nsr51324_yolo11n.py's
            # pinned lr0=0.001 any time MAX_TRAIN_IMAGES in prepare_dataset.py changes — confirmed
            # this actually happened: with the 2000-image train set currently on disk, auto
            # resolves to AdamW/lr0≈0.0011, nearly identical to those two scripts' "10x lower" LR,
            # which would have silently erased the intended contrast. SGD/lr0=0.01 matches
            # Ultralytics' own stock SGD defaults (see cfg/default.yaml) — this model is
            # COCO-pretrained only, not road-damage-tuned by anyone yet, so it gets the higher
            # "adapt to a new domain" LR; see ASBcode.md's 2026-08-27 LR entry for the full split.
            optimizer="SGD",
            lr0=0.01,
            momentum=0.9,
        ),
    )
    if paused:
        return

    best = Path(results.save_dir) / "weights" / "best.pt"
    shutil.copy2(best, FINETUNED_WEIGHTS)
    print(f"Fine-tuned weights saved to {FINETUNED_WEIGHTS}")

    metrics = model.val(data=str(DATA_YAML), split="test", imgsz=256, device="cpu")
    print("\nTest-set accuracy:")
    print(f"  mAP50:    {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print(f"  precision: {metrics.box.mp:.4f}")
    print(f"  recall:    {metrics.box.mr:.4f}")


if __name__ == "__main__":
    main()
