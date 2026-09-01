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

Learning rate pinned to optimizer="AdamW", lr0=0.001, momentum=0.9 (added 2026-08-27) — 10x lower
than train_yolo11n.py's pinned optimizer="SGD"/lr0=0.01, since this checkpoint is already
road-damage-fine-tuned and a from-scratch-strength LR risks catastrophic forgetting of what
rezzzq's run already learned. Pinned rather than using optimizer="auto" — auto's choice is
train-set-size-dependent (see train_yolo11n.py's docstring for why that broke a benchmark
comparison once already).

**2026-08-28 update — no longer matching train_yolo11n.py's imgsz/epochs/batch:** the first real
15-epoch/2000-image/batch=8 run (mAP50=0.3256, best of 5 models attempted, see
`modelresults.md`) checked out clearly not plateaued — `results.csv` showed train loss still
dropping and val mAP50 still climbing through epoch 15, with the LR schedule only just finishing
its decay. Bumped to epochs=25 (patience=5 still applies, so this is a ceiling, not a forced full
run), batch=16, and MAX_TRAIN_IMAGES 2000->4000 in prepare_dataset.py (imgsz left at 256
deliberately — not part of that change). This v2 run got mAP50=0.3724, see `modelresults.md`.
This is a YOLO12s-specific improvement run, decoupled from train_yolo11n.py's settings (that
script is unchanged).

**2026-08-30 update — imgsz 256->416, everything else unchanged from v2:** next lever on the
same "raise imgsz/epochs/MAX_TRAIN_IMAGES" list from `modelresults.md` — cracks/potholes are
small objects, so more pixels to detect them in should help disproportionately. epochs=25,
batch=16, MAX_TRAIN_IMAGES=4000, optimizer/lr0/momentum all left as-is; only this script's train
imgsz and the final test-set eval's imgsz changed (kept matched to each other, since evaluating
at a different resolution than trained would confound the comparison). Only YOLO12s — the other
three scripts are untouched.

Trains against data_fast.yaml (val_small — a small val slice prepare_dataset.py writes
alongside the full data.yaml), not data.yaml, so per-epoch validation for the patience/
early-stop check doesn't re-score the entire ~5.8k-image val set every epoch — that was most of
why the first attempt at this run was taking ~29 min/epoch. Final test-set accuracy below still
comes from the full test split regardless (data_fast.yaml's "test" key points at the same place
as data.yaml's).

Pause/resume (added 2026-08-27, see training/pause_control.py for the full mechanism and why):
    python training/pause_control.py pause   # from another terminal, while this is running
    python training/train_yolo12s.py --resume   # continues from the checkpoint afterward
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
SOURCE_WEIGHTS = ROOT / "weights" / "yolo12s" / "yolo12s_rdd2022.pt"
CONTINUED_WEIGHTS = ROOT / "weights" / "yolo12s" / "yolo12s_rdd2022_continued.pt"
RUNS_DIR = ROOT / "training" / "runs"
RUN_NAME = "yolo12s_rdd2022_continued"


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
            epochs=25,
            imgsz=416,
            batch=16,
            device="cpu",
            project=str(RUNS_DIR),
            name=RUN_NAME,
            exist_ok=True,
            patience=5,
            # Pinned instead of optimizer="auto": auto would pick MuSGD/lr0=0.01 here (same as a
            # from-scratch COCO fine-tune, see train_yolo11n.py) — too high for a checkpoint
            # that's already road-damage-tuned, risks catastrophic forgetting of what rezzzq's
            # run already learned. 10x lower than the from-scratch scripts' effective lr0=0.01,
            # AdamW because a low constant-ish LR fine-tune is its usual use case (vs. SGD's need
            # for a LR schedule to converge well). See ASBcode.md's 2026-08-27 LR entry.
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

    metrics = model.val(data=str(DATA_YAML), split="test", imgsz=416, device="cpu")
    print("\nTest-set accuracy:")
    print(f"  mAP50:    {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print(f"  precision: {metrics.box.mp:.4f}")
    print(f"  recall:    {metrics.box.mr:.4f}")


if __name__ == "__main__":
    main()
