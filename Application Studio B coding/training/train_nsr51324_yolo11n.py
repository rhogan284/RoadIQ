"""Continues fine-tuning YOLO11n from the nsr51324/Road_Damage_Object_Detection checkpoint
(already road-damage-fine-tuned by a third party on a Roboflow dataset) on our own cleaned
RDD2022 subset — same pattern as train_yolo12s.py's continued-fine-tune approach.

Run training/prepare_dataset.py first. Saves the continued checkpoint to
weights/nsr51324_yolo11n/nsr51324_yolo11n_continued.pt. There is no detector_interface plugin
for this checkpoint yet (only weights + this training script have been set up so far) — add one
under detector_interface/plugins/ modeled on yolo12s.py before this can be benchmarked or served.

Source checkpoint: weights/nsr51324_yolo11n/nsr51324_yolo11n_road.pt, downloaded 2026-08-27 from
https://huggingface.co/nsr51324/Road_Damage_Object_Detection
(runs/detect/yolov11_road-2/weights/best.pt — the repo bundles three separately-trained
architectures (YOLOv8n/YOLOv10n/YOLO11n) benchmarked against each other; this is the YOLO11n one,
chosen to match this project's existing yolo11n architecture family). Sibling args.yaml
(weights/nsr51324_yolo11n/args.yaml) confirms it started from stock yolo11n.pt, imgsz=640,
batch=16, up to 50 epochs with patience=3, on a Roboflow "road-damage-1" dataset export — the HF
repo card doesn't say how many classes or what order, and args.yaml doesn't carry class names
either (that lives in the checkpoint's own model.names, not the training args).

Class-mapping caveat (checked 2026-08-27 by loading the checkpoint —
`YOLO(str(SOURCE_WEIGHTS)).names`): this checkpoint has **7** classes, not our 5, and they don't
line up the way yolo12s.py's checkpoint does:
    {0: 'alligator', 1: 'block', 2: 'crack', 3: 'edge', 4: 'longitudinal', 5: 'pothole', 6: 'transverse'}
vs. our RDD2022_CLASS_NAMES (detector_interface/plugins/yolo11n.py):
    ['longitudinal crack', 'transverse crack', 'alligator crack', 'other corruption', 'pothole']
There's a generic standalone 'crack' class plus separate 'alligator'/'longitudinal'/'transverse'
labels (unclear from the checkpoint alone whether their source Roboflow dataset annotates
crack-type and 'crack' as two boxes per defect, or something else), and 'block'/'edge' have no
equivalent in our labels at all. Unlike yolo12s.py's D00-D40 remap, there's no safe mechanical
1:1 slot mapping here — don't write an inference-plugin remap assuming one. Continuing training
here just lets the 5-slot head relearn our order over epochs from whatever the pretrained head
currently outputs (which is closer to "reinitializing the head" than "adjusting a few slots",
practically) — the backbone/FPN features (crack/pothole visual patterns) are still useful
transfer regardless of the head's original class count.

Same CPU-only imgsz/epochs/batch/patience as train_yolo11n.py/train_yolo12s.py (256/15/8/5) for
comparability. Trains against data_fast.yaml, evaluates final accuracy against the full
data.yaml test split — see train_yolo12s.py's docstring for why (per-epoch val on the full
~5.8k-image val set is what made an earlier run take ~29 min/epoch).

Learning rate (added 2026-08-27): pinned to optimizer="AdamW", lr0=0.001, momentum=0.9 — same
reasoning and same value as train_yolo12s.py's continued fine-tune, 10x lower than
train_yolo11n.py/train_cvtechniques_yolo11s.py's pinned optimizer="SGD"/lr0=0.01, since this
checkpoint is already road-damage-fine-tuned by nsr51324. Pinned rather than left on
optimizer="auto" because auto's choice is train-set-size-dependent — see train_yolo11n.py's
docstring for why that would've silently collapsed this exact contrast.

Pause/resume (added 2026-08-27, see training/pause_control.py for the full mechanism and why):
    python training/pause_control.py pause   # from another terminal, while this is running
    python training/train_nsr51324_yolo11n.py --resume   # continues from the checkpoint after
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
SOURCE_WEIGHTS = ROOT / "weights" / "nsr51324_yolo11n" / "nsr51324_yolo11n_road.pt"
CONTINUED_WEIGHTS = ROOT / "weights" / "nsr51324_yolo11n" / "nsr51324_yolo11n_continued.pt"
RUNS_DIR = ROOT / "training" / "runs"
RUN_NAME = "nsr51324_yolo11n_continued"


def main() -> None:
    if not FAST_DATA_YAML.exists():
        raise RuntimeError(f"{FAST_DATA_YAML} not found — run training/prepare_dataset.py first.")
    if not SOURCE_WEIGHTS.exists():
        raise RuntimeError(f"{SOURCE_WEIGHTS} not found.")

    model, results, paused = pause_control.run_training(
        source_weights=SOURCE_WEIGHTS,
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
            # Same reasoning as train_yolo12s.py: this checkpoint is already road-damage-tuned
            # (by nsr51324), so use a 10x-lower LR than the from-scratch scripts' pinned lr0=0.01
            # instead of retraining it as hard as a fresh COCO checkpoint — see ASBcode.md's
            # 2026-08-27 LR entry.
            optimizer="AdamW",
            lr0=0.001,
            momentum=0.9,
        ),
    )
    if paused:
        return

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
