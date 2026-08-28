"""Fine-tunes YOLO11s on our own cleaned RDD2022 subset, replicating the recipe documented by
cvtechniques/road-damage-detection-yolov11 on Hugging Face — NOT continuing that repo's actual
checkpoint, because it doesn't have one.

**Important — read before assuming this is a continued fine-tune like train_yolo12s.py or
train_nsr51324_yolo11n.py:** checked the HF repo's file listing directly (API call, 2026-08-27)
— https://huggingface.co/cvtechniques/road-damage-detection-yolov11 only contains README.md,
Picture1.jpg, results.png, and val_batch0_pred.jpg. No .pt/.pth/.onnx weight file was ever
uploaded. The README is a full model card (training recipe, RDD2022-Japan/Roboflow subset,
per-class metrics: mAP50 ~0.47, mAP50-95 ~0.19, precision 0.63, recall 0.40 on their own held-out
test set) describing a real YOLOv11s fine-tune that was trained (per the card: Colab, T4 GPU, 15
epochs) — the trained weights themselves just aren't published anywhere this project can fetch
them from. So there is nothing to download or continue-fine-tune from; this script instead
starts from stock COCO-pretrained yolo11s.pt (ultralytics auto-downloads it) and fine-tunes
directly on our own data, which is the closest equivalent this project can produce. If real
cvtechniques weights ever surface (e.g. a future HF commit, or a linked Roboflow/Colab export),
switch this to the SOURCE_WEIGHTS continued-fine-tune pattern used by train_yolo12s.py instead.

Run training/prepare_dataset.py first. Saves the fine-tuned checkpoint to
weights/cvtechniques_yolo11s/cvtechniques_yolo11s_rdd2022.pt. There is no detector_interface
plugin for this model yet — add one under detector_interface/plugins/ modeled on yolo11n.py
before this can be benchmarked or served.

Same CPU-only defaults as the other training scripts here (imgsz=256, epochs=15, batch=8,
patience=5) for comparability — the cvtechniques card itself used imgsz=640/batch=16/epochs=15
on a GPU, which isn't CPU-feasible for this project's laptop-CPU constraint (see
train_yolo11n.py's docstring). Trains against data_fast.yaml, evaluates final accuracy against
the full data.yaml test split.

Learning rate (added 2026-08-27): pinned to optimizer="SGD", lr0=0.01, momentum=0.9 — same as
train_yolo11n.py, since this model (like that one) starts from a COCO-pretrained checkpoint with
no prior road-damage fine-tuning to protect, unlike train_yolo12s.py/train_nsr51324_yolo11n.py's
lower pinned lr0=0.001. See train_yolo11n.py's docstring for why this is pinned rather than left
on optimizer="auto".

Pause/resume (added 2026-08-27, see training/pause_control.py for the full mechanism and why):
    python training/pause_control.py pause   # from another terminal, while this is running
    python training/train_cvtechniques_yolo11s.py --resume   # continues after
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
BASE_WEIGHTS = "yolo11s.pt"  # no local copy — ultralytics auto-downloads on first use
FINETUNED_WEIGHTS = ROOT / "weights" / "cvtechniques_yolo11s" / "cvtechniques_yolo11s_rdd2022.pt"
RUNS_DIR = ROOT / "training" / "runs"
RUN_NAME = "cvtechniques_yolo11s_rdd2022"


def main() -> None:
    if not FAST_DATA_YAML.exists():
        raise RuntimeError(f"{FAST_DATA_YAML} not found — run training/prepare_dataset.py first.")

    FINETUNED_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)

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
            # Pinned instead of optimizer="auto" — same reasoning as train_yolo11n.py's
            # 2026-08-27 change: auto's LR depends on train-set size and would drift out of sync
            # with the continued-fine-tune scripts' pinned lr0=0.001 as MAX_TRAIN_IMAGES changes.
            # This model has no real pretrained road-damage weights to protect (see this file's
            # module docstring), so it gets the same from-scratch SGD/lr0=0.01 as train_yolo11n.py.
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
