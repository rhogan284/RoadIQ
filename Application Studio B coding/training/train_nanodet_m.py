"""Fine-tunes NanoDet-m (COCO-pretrained) on the cleaned RDD2022 subset.

Unlike YOLOX (whose pip-installed package has no usable CPU training path — see the abandoned
attempt in git history / chat log), NanoDet's own tools/train.py genuinely supports CPU training
out of the box (`cfg.device.gpu_ids: -1` picks accelerator="cpu" in nanodet/util's Lightning
Trainer setup, no hardcoded CUDA calls in the training path), and it ships a built-in
`YoloDataset` loader (nanodet/data/dataset/yolo.py) that reads YOLO-format `.txt` labels
directly — no COCO-format conversion needed. This script drives that vendored tools/train.py
programmatically (loaded via importlib so third_party/nanodet stays an untouched clone) instead
of reimplementing a training loop by hand.

One dataset-layout wrinkle: YoloDataset's image lookup (`_find_image` in yolo.py) only looks for
an image next to its .txt label, in the *same* directory passed as `ann_path` — but
prepare_dataset.py's output keeps images/ and labels/ as sibling folders, not merged. So this
script first copies (not moves — additive, non-destructive, cheap since labels are tiny .txt
files) each split's label files into its images/ folder, idempotently, before training starts.

Same CPU-time-budget convention as train_yolo11n.py / train_yolo12s.py: imgsz=256 (not NanoDet's
usual 320), batch=8, 15 epochs, trains against the train split and validates every epoch (via
NanoDet's own CocoDetectionEvaluator, computed automatically — no custom eval code needed here)
against val_small. Only the classification head is affected by the 80->5 class change;
nanodet's own load_model_weight() utility already auto-skips shape-mismatched layers when
loading the pretrained checkpoint, so no manual state_dict surgery is needed here (unlike the
abandoned YOLOX attempt).

Not included yet (kept out to keep this script's first run low-risk): a final full-test-set eval
matching the "Test-set accuracy" printout train_yolo11n.py/train_yolo12s.py produce. Once a
training run here is confirmed working, that can be added as a follow-up in the same style.
"""

import importlib.util
import shutil
import sys
import types
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
NANODET_DIR = ROOT / "third_party" / "nanodet"
DATA_ROOT = ROOT / "data" / "RDD2022" / "clean"
TRAIN_IMAGES = DATA_ROOT / "train" / "images"
TRAIN_LABELS = DATA_ROOT / "train" / "labels"
VAL_IMAGES = DATA_ROOT / "val_small" / "images"
VAL_LABELS = DATA_ROOT / "val_small" / "labels"
BASE_WEIGHTS = ROOT / "weights" / "nanodet_m" / "nanodet_m.ckpt"
FINETUNED_WEIGHTS = ROOT / "weights" / "nanodet_m" / "nanodet_m_rdd2022.ckpt"
SAVE_DIR = ROOT / "training" / "runs" / "nanodet_m_rdd2022"
GENERATED_CONFIG = ROOT / "training" / "nanodet_m_rdd2022.generated.yml"

INPUT_SIZE = [256, 256]  # [w, h]
BATCH_SIZE = 8
EPOCHS = 15
CLASS_NAMES = [
    "longitudinal crack",
    "transverse crack",
    "alligator crack",
    "other corruption",
    "pothole",
]


def _merge_labels_into_images(images_dir: Path, labels_dir: Path) -> None:
    for label_file in labels_dir.glob("*.txt"):
        dest = images_dir / label_file.name
        if not dest.exists():
            shutil.copy2(label_file, dest)


def _ensure_importable() -> None:
    if str(NANODET_DIR) not in sys.path:
        sys.path.insert(0, str(NANODET_DIR))
    if "torch._six" not in sys.modules:
        shim = types.ModuleType("torch._six")
        shim.string_classes = str
        sys.modules["torch._six"] = shim

    # tools/train.py's `ckpt = torch.load(cfg.schedule.load_model)` (no map_location) crashes
    # on this CPU-only machine when loading a checkpoint saved from a CUDA device. Patch
    # torch.load process-wide to default to CPU — safe here since this whole project is
    # CPU-only anyway. Idempotent so re-running main() in the same process doesn't double-wrap.
    if not getattr(torch.load, "_cpu_patched", False):
        _original_torch_load = torch.load

        def _cpu_torch_load(*args, **kwargs):
            kwargs.setdefault("map_location", "cpu")
            return _original_torch_load(*args, **kwargs)

        _cpu_torch_load._cpu_patched = True
        torch.load = _cpu_torch_load

    # tools/train.py passes `strategy=None` for single-device (non-DDP) runs, which was valid
    # on the pytorch_lightning version it was written against but raises ValueError on the
    # version installed here (must be "auto" or a Strategy instance). Patch Trainer.__init__ to
    # normalize that one value rather than editing the vendored file.
    import pytorch_lightning as pl

    if not getattr(pl.Trainer.__init__, "_strategy_patched", False):
        _original_trainer_init = pl.Trainer.__init__

        def _patched_trainer_init(self, *args, **kwargs):
            if kwargs.get("strategy", "auto") is None:
                kwargs["strategy"] = "auto"
            # Same story for `devices=None` on the CPU accelerator — valid on the
            # pytorch_lightning version tools/train.py targeted, rejected on this one
            # ("`devices` selected with `CPUAccelerator` should be an int > 0").
            if kwargs.get("accelerator") == "cpu" and kwargs.get("devices") is None:
                kwargs["devices"] = 1
            return _original_trainer_init(self, *args, **kwargs)

        _patched_trainer_init._strategy_patched = True
        pl.Trainer.__init__ = _patched_trainer_init


def _build_config() -> dict:
    normalize = [[103.53, 116.28, 123.675], [57.375, 57.12, 58.395]]
    return {
        "save_dir": str(SAVE_DIR),
        "model": {
            "arch": {
                "name": "OneStageDetector",
                "backbone": {
                    "name": "ShuffleNetV2",
                    "model_size": "1.0x",
                    "out_stages": [2, 3, 4],
                    "activation": "LeakyReLU",
                },
                "fpn": {
                    "name": "PAN",
                    "in_channels": [116, 232, 464],
                    "out_channels": 96,
                    "start_level": 0,
                    "num_outs": 3,
                },
                "head": {
                    "name": "NanoDetHead",
                    "num_classes": len(CLASS_NAMES),
                    "input_channel": 96,
                    "feat_channels": 96,
                    "stacked_convs": 2,
                    "share_cls_reg": True,
                    "octave_base_scale": 5,
                    "scales_per_octave": 1,
                    "strides": [8, 16, 32],
                    "reg_max": 7,
                    "norm_cfg": {"type": "BN"},
                    "loss": {
                        "loss_qfl": {
                            "name": "QualityFocalLoss",
                            "use_sigmoid": True,
                            "beta": 2.0,
                            "loss_weight": 1.0,
                        },
                        "loss_dfl": {"name": "DistributionFocalLoss", "loss_weight": 0.25},
                        "loss_bbox": {"name": "GIoULoss", "loss_weight": 2.0},
                    },
                },
            }
        },
        "data": {
            "train": {
                "name": "YoloDataset",
                "class_names": CLASS_NAMES,
                "img_path": str(TRAIN_IMAGES),
                "ann_path": str(TRAIN_IMAGES),
                "input_size": INPUT_SIZE,
                "keep_ratio": True,
                "pipeline": {
                    "perspective": 0.0,
                    "scale": [0.6, 1.4],
                    "stretch": [[1, 1], [1, 1]],
                    "rotation": 0,
                    "shear": 0,
                    "translate": 0.2,
                    "flip": 0.5,
                    "brightness": 0.2,
                    "contrast": [0.6, 1.4],
                    "saturation": [0.5, 1.2],
                    "normalize": normalize,
                },
            },
            "val": {
                "name": "YoloDataset",
                "class_names": CLASS_NAMES,
                "img_path": str(VAL_IMAGES),
                "ann_path": str(VAL_IMAGES),
                "input_size": INPUT_SIZE,
                "keep_ratio": True,
                "pipeline": {"normalize": normalize},
            },
        },
        "device": {
            "gpu_ids": -1,
            "workers_per_gpu": 0,
            "batchsize_per_gpu": BATCH_SIZE,
        },
        "schedule": {
            "load_model": str(BASE_WEIGHTS),
            "optimizer": {"name": "SGD", "lr": 0.001, "momentum": 0.9, "weight_decay": 0.0001},
            "warmup": {"name": "linear", "steps": 100, "ratio": 0.1},
            "total_epochs": EPOCHS,
            "lr_schedule": {"name": "MultiStepLR", "milestones": [10, 13], "gamma": 0.1},
            "val_intervals": 1,
        },
        "evaluator": {"name": "CocoDetectionEvaluator", "save_key": "mAP"},
        "log": {"interval": 20},
        "class_names": CLASS_NAMES,
    }


def main() -> None:
    if not BASE_WEIGHTS.exists():
        raise RuntimeError(f"{BASE_WEIGHTS} not found.")
    if not TRAIN_IMAGES.exists() or not VAL_IMAGES.exists():
        raise RuntimeError(f"{TRAIN_IMAGES} / {VAL_IMAGES} not found — run training/prepare_dataset.py first.")

    _merge_labels_into_images(TRAIN_IMAGES, TRAIN_LABELS)
    _merge_labels_into_images(VAL_IMAGES, VAL_LABELS)

    config = _build_config()
    GENERATED_CONFIG.write_text(yaml.safe_dump(config, sort_keys=False))
    print(f"Wrote generated config to {GENERATED_CONFIG}")

    _ensure_importable()

    spec = importlib.util.spec_from_file_location("nanodet_train_tool", NANODET_DIR / "tools" / "train.py")
    nanodet_train_tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nanodet_train_tool)

    sys.argv = ["train.py", str(GENERATED_CONFIG)]
    args = nanodet_train_tool.parse_args()
    nanodet_train_tool.main(args)

    best_ckpt = SAVE_DIR / "model_best" / "nanodet_model_best.pth"
    fallback_ckpt = SAVE_DIR / "model_last.ckpt"
    source_ckpt = best_ckpt if best_ckpt.exists() else fallback_ckpt
    if source_ckpt.exists():
        FINETUNED_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_ckpt, FINETUNED_WEIGHTS)
        print(f"Fine-tuned weights saved to {FINETUNED_WEIGHTS} (from {source_ckpt.name})")
        eval_results = SAVE_DIR / "model_best" / "eval_results.txt"
        if eval_results.exists():
            print(f"\nBest-epoch validation metrics ({eval_results}):")
            print(eval_results.read_text())
    else:
        print(f"Warning: no checkpoint found under {SAVE_DIR} — check what was actually saved there.")


if __name__ == "__main__":
    main()
