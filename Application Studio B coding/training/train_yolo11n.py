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
"""

import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = ROOT / "data" / "RDD2022" / "data.yaml"
FAST_DATA_YAML = ROOT / "data" / "RDD2022" / "data_fast.yaml"
BASE_WEIGHTS = ROOT / "weights" / "yolo11n" / "yolo11n.pt"
FINETUNED_WEIGHTS = ROOT / "weights" / "yolo11n" / "yolo11n_rdd2022.pt"
RUNS_DIR = ROOT / "training" / "runs"


def main() -> None:
    if not FAST_DATA_YAML.exists():
        raise RuntimeError(f"{FAST_DATA_YAML} not found — run training/prepare_dataset.py first.")

    model = YOLO(str(BASE_WEIGHTS))
    results = model.train(
        data=str(FAST_DATA_YAML),
        epochs=15,
        imgsz=256,
        batch=8,
        device="cpu",
        project=str(RUNS_DIR),
        name="yolo11n_rdd2022",
        exist_ok=True,
        patience=5,
    )

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
