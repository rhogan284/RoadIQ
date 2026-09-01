"""One-off: run the final full-test-set eval for the completed YOLO12s v4 run.

train_yolo12s.py --resume crashed on this run because pause happened to land right after the
very last scheduled epoch (25/25) — Ultralytics' resume=True refuses to resume a run with
nothing left to train (AssertionError: "training to 25 epochs is finished, nothing to resume"),
which isn't a KeyboardInterrupt so pause_control.py's except clause doesn't catch it. Training
itself is genuinely complete and best.pt is valid; this just does the two post-training steps
train_yolo12s.py's main() would have done itself, manually.
"""

import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = ROOT / "data" / "RDD2022" / "data.yaml"
BEST = ROOT / "training" / "runs" / "yolo12s_rdd2022_continued" / "weights" / "best.pt"
CONTINUED_WEIGHTS = ROOT / "weights" / "yolo12s" / "yolo12s_rdd2022_continued.pt"

model = YOLO(str(BEST))
shutil.copy2(BEST, CONTINUED_WEIGHTS)
print(f"Continued fine-tuned weights saved to {CONTINUED_WEIGHTS}")

metrics = model.val(data=str(DATA_YAML), split="test", imgsz=416, device="cpu")
print("\nTest-set accuracy:")
print(f"  mAP50:    {metrics.box.map50:.4f}")
print(f"  mAP50-95: {metrics.box.map:.4f}")
print(f"  precision: {metrics.box.mp:.4f}")
print(f"  recall:    {metrics.box.mr:.4f}")
