"""Fine-tunes NanoDet-m on our own cleaned RDD2022 subset, starting from the COCO-pretrained
weights/nanodet_m/nanodet_m.ckpt. Unlike the Ultralytics-based training scripts in this folder,
NanoDet doesn't expose a simple Python training API — its own tools/train.py is a CLI entry
point driven entirely by a config YAML, so this script is a thin subprocess wrapper around it
rather than a direct model.train() call. Saves checkpoints under
training/runs/nanodet_m_rdd2022/ (see nanodet_rdd2022.yml's save_dir) — NOT a single output .pt
matching the other scripts' pattern; NanoDet's own checkpoint naming is used as-is.

Run training/prepare_dataset.py first.

**Dataset layout fix (resolved 2026-08-27, was previously an open TODO):** confirmed by reading
third_party/nanodet/nanodet/data/dataset/yolo.py + coco.py that NanoDet's YoloDataset needs each
label .txt's matching image *findable in the same directory as ann_path* (only used to read
width/height via imagesize.get() during dataset indexing — the actual per-sample pixel load
later uses img_path + file_name via a plain os.path.join, so img_path can point anywhere as long
as it's consistent). prepare_dataset.py's separate images/ + labels/ subdirectories don't satisfy
that indexing step on their own — every annotation would silently fail the `_find_image` lookup
and get skipped, producing an empty training set with no error, not a crash. Fixed here by
hardlinking each split's images + labels into one combined flat directory
(data/RDD2022/nanodet_flat/{split}/) before invoking nanodet's trainer, and pointing both
img_path and ann_path in nanodet_rdd2022.yml at that combined directory. Hardlinks (not copies)
so this doesn't duplicate ~10GB of image data on disk — falls back to a real copy only if
hardlinking fails (e.g. across filesystems).

Class-count mismatch on load (not something to "fix" — just how transfer learning works here):
nanodet_m.ckpt's head was trained for COCO's 80 classes; nanodet_rdd2022.yml sets num_classes=5.
nanodet/util/check_point.py's load_model_weight handles this gracefully (confirmed by reading
it) — shape-mismatched head params are skipped and left at their random init, everything else
(backbone, FPN) transfers — so this isn't a crash risk, just means the detection head starts
from scratch, same spirit as yolo12s.py's class-remap caveat but a full head reinit rather than
a partial remap.

Also needs `pip install pytorch_lightning termcolor` (same as the nanodet_m.py inference
plugin — see its docstring) and the same torch._six shim it installs at import time, which this
script does NOT install itself since it shells out to nanodet's own tools/train.py rather than
importing nanodet directly — if that script errors on `from torch._six import string_classes`,
patch nanodet/data/collate.py or run once through a Python entry point that installs the shim
first (see detector_interface/plugins/nanodet_m.py's _ensure_importable()).

CPU-only (device.gpu_ids: -1 in the config), 15 epochs, batch=8 — matching the epoch/batch
budget of this project's other training scripts, not NanoDet's original 280-epoch/batch=192
defaults (also lowered in the config, along with the learning rate, to suit batch=8). NanoDet-m
is COCO-pretrained only (no third party has road-damage-fine-tuned it, unlike yolo12s/
nsr51324_yolo11n), so — same bucket as train_yolo11n.py/train_cvtechniques_yolo11s.py — this
does NOT get the lower "already road-damage-tuned" learning rate those two scripts use; see
ASBcode.md's 2026-08-27 LR entry.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "third_party" / "nanodet" / "tools" / "train.py"
CONFIG_PATH = ROOT / "training" / "nanodet_rdd2022.yml"
BASE_WEIGHTS = ROOT / "weights" / "nanodet_m" / "nanodet_m.ckpt"
CLEAN_DIR = ROOT / "data" / "RDD2022" / "clean"
FLAT_DIR = ROOT / "data" / "RDD2022" / "nanodet_flat"
IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def _link_or_copy(src: Path, dest: Path) -> None:
    if dest.exists():
        return
    try:
        os.link(src, dest)
    except OSError:
        shutil.copy2(src, dest)


def _flatten_split(split: str) -> Path:
    """NanoDet's YoloDataset needs each split's images and labels in one directory (see this
    file's docstring) — builds data/RDD2022/nanodet_flat/{split}/ as a hardlinked flattening of
    prepare_dataset.py's separate clean/{split}/images + clean/{split}/labels."""
    images_dir = CLEAN_DIR / split / "images"
    labels_dir = CLEAN_DIR / split / "labels"
    if not images_dir.exists() or not labels_dir.exists():
        raise RuntimeError(f"{images_dir} or {labels_dir} not found — run training/prepare_dataset.py first.")

    out_dir = FLAT_DIR / split
    out_dir.mkdir(parents=True, exist_ok=True)

    linked = 0
    for label_path in sorted(labels_dir.glob("*.txt")):
        image_path = next(
            (images_dir / f"{label_path.stem}{ext}" for ext in IMAGE_EXTS if (images_dir / f"{label_path.stem}{ext}").exists()),
            None,
        )
        if image_path is None:
            continue
        _link_or_copy(label_path, out_dir / label_path.name)
        _link_or_copy(image_path, out_dir / image_path.name)
        linked += 1

    print(f"{split}: {linked} image+label pairs flattened into {out_dir}")
    return out_dir


def main() -> None:
    if not CLEAN_DIR.exists():
        raise RuntimeError(f"{CLEAN_DIR} not found — run training/prepare_dataset.py first.")
    if not BASE_WEIGHTS.exists():
        raise RuntimeError(f"{BASE_WEIGHTS} not found.")
    if not TRAIN_SCRIPT.exists():
        raise RuntimeError(
            f"{TRAIN_SCRIPT} not found. Run: "
            f"git clone --depth 1 https://github.com/RangiLyu/nanodet.git {ROOT / 'third_party' / 'nanodet'}"
        )

    _flatten_split("train")
    _flatten_split("val_small")

    subprocess.run(
        [sys.executable, str(TRAIN_SCRIPT), str(CONFIG_PATH)],
        cwd=str(ROOT),
        check=True,
    )


if __name__ == "__main__":
    main()
