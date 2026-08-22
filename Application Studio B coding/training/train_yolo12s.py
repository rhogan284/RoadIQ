"""Continues fine-tuning YOLO12s from the rezzzq/yolo12s-road-damage-rdd2022 checkpoint
(already RDD2022-fine-tuned by a third party) on our own cleaned RDD2022 subset.

Run training/prepare_dataset.py first. Saves the continued checkpoint to
weights/yolo12s/yolo12s_rdd2022_continued.pt, which detector_interface/plugins/yolo12s.py
loads for real inference once present (falls back to the original rezzzq checkpoint
otherwise) — the original weights/yolo12s/yolo12s_rdd2022.pt file is left untouched.

Class-mapping caveat: the source checkpoint's class order is
{0: D00 longitudinal crack, 1: D10 transverse crack, 2: D20 alligator crack, 3: D40 pothole,
4: Repair}, but data.yaml (from prepare_dataset.py) uses our RDD2022_CLASS_NAMES order
(..., 3: other corruption, 4: pothole). Classes 0-2 line up already; continuing training here
will retrain output slots 3 and 4 to mean our "other corruption" / "pothole" instead of the
checkpoint's original "pothole" / "Repair" — the backbone/feature layers transfer as useful
initialization, the head just needs epochs to relearn what slots 3-4 mean under our labels.

Same CPU-only defaults as train_yolo11n.py (imgsz=256, epochs=15, batch=8, patience=5) so the
two runs are on equal footing for the benchmark comparison — not because this starting point
needs the same epoch budget (it likely needs fewer, since it's not starting from COCO).

Trains against data_fast.yaml (val_small — a small val slice prepare_dataset.py writes
alongside the full data.yaml), not data.yaml, so per-epoch validation for the patience/
early-stop check doesn't re-score the entire ~5.8k-image val set every epoch — that was most of
why the first attempt at this run was taking ~29 min/epoch. Final test-set accuracy below still
comes from the full test split regardless (data_fast.yaml's "test" key points at the same place
as data.yaml's).
"""

import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = ROOT / "data" / "RDD2022" / "data.yaml"
FAST_DATA_YAML = ROOT / "data" / "RDD2022" / "data_fast.yaml"
SOURCE_WEIGHTS = ROOT / "weights" / "yolo12s" / "yolo12s_rdd2022.pt"
CONTINUED_WEIGHTS = ROOT / "weights" / "yolo12s" / "yolo12s_rdd2022_continued.pt"
RUNS_DIR = ROOT / "training" / "runs"


def main() -> None:
    if not FAST_DATA_YAML.exists():
        raise RuntimeError(f"{FAST_DATA_YAML} not found — run training/prepare_dataset.py first.")
    if not SOURCE_WEIGHTS.exists():
        raise RuntimeError(f"{SOURCE_WEIGHTS} not found.")

    model = YOLO(str(SOURCE_WEIGHTS))
    results = model.train(
        data=str(FAST_DATA_YAML),
        epochs=15,
        imgsz=256,
        batch=8,
        device="cpu",
        project=str(RUNS_DIR),
        name="yolo12s_rdd2022_continued",
        exist_ok=True,
        patience=5,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    shutil.copy2(best, CONTINUED_WEIGHTS)
    print(f"Continued fine-tuned weights saved to {CONTINUED_WEIGHTS}")

    metrics = model.val(data=str(DATA_YAML), split="test", imgsz=256, device="cpu")
    print("\nTest-set accuracy:")
    print(f"  mAP50:    {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print(f"  precision: {metrics.box.mp:.4f}")
    print(f"  recall:    {metrics.box.mr:.4f}")


if __name__ == "__main__":
    main()
