"""Extracts data/RDD2022/archive.zip, cleans it, and writes data/RDD2022/data.yaml.

The Kaggle dataset (aliabdelmenam/rdd-2022) already ships a train/val/test split (70/15/15,
~26.9k/5.8k/5.8k images) in YOLO format (images/ + labels/ per split). We keep that split as-is
(no re-splitting — it's already sound, and re-splitting risks mixing near-duplicate frames
across train/test). What this script adds:

1. Extraction.
2. Cleaning: drop any image whose file OpenCV can't decode (corrupt/truncated), and drop any
   image+label pair where the label file references a class index outside the 5 known classes.
3. Leakage check: MD5-hash every image; if the same image (by content, not filename) appears in
   more than one split, keep it only in the split it originally appeared in first (train > val >
   test priority) and drop the later duplicate(s) — a duplicate frame in both train and test
   would let the model "cheat" by memorizing it.
4. A CPU-feasible training subset: full-size val/test are kept (they're the numbers that
   matter), but train is capped at MAX_TRAIN_IMAGES via random sample, class-stratified where
   possible, so a fine-tuning run finishes in a reasonable time on a laptop CPU. Increase
   MAX_TRAIN_IMAGES yourself and re-run if you have time to spare.
5. Writes data.yaml (Ultralytics format) pointing at the cleaned subset.
"""

import hashlib
import random
import shutil
import zipfile
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[1] / "data" / "RDD2022"
ARCHIVE = ROOT / "archive.zip"
EXTRACTED = ROOT / "extracted"
CLEAN = ROOT / "clean"

CLASS_NAMES = [
    "longitudinal crack",
    "transverse crack",
    "alligator crack",
    "other corruption",
    "pothole",
]
MAX_TRAIN_IMAGES = 8000  # CPU-feasible subset; full train split is ~26,900 images. Raised from
# 2000 2026-08-28 for a longer YOLO12s improvement run (see ASBcode.md's 2026-08-28 entry), then
# 4000->8000 2026-08-30 after YOLO12s v3 (imgsz=416) early-stopped well before its epoch ceiling —
# evidence pointed at training-set size, not epoch count, as the bottleneck. See modelresults.md's
# "YOLO12s v3" entry.
VAL_SUBSET_SIZE = 400  # separate small val slice for per-epoch training checks — see data_fast.yaml
# below. Was wrongly left at 5758 (the FULL val set) before 2026-08-28 — that defeated the whole
# point of val_small (fast per-epoch checks) and would have made every epoch as slow as the final
# test-set eval. Restored to the small value documented in ASBcode.md's 2026-08-22 entry.
RANDOM_SEED = 42


def _extraction_complete() -> bool:
    """True only if all three splits' images/ and labels/ dirs are actually present — not just
    if EXTRACTED has *something* in it. A Ctrl+C mid-extractall can leave e.g. train/test done
    but val missing, which used to be silently treated as "already extracted, skip"."""
    if not EXTRACTED.exists():
        return False
    for split in ("train", "val", "test"):
        if not list(EXTRACTED.rglob(f"{split}/images")) or not list(EXTRACTED.rglob(f"{split}/labels")):
            return False
    return True


def extract() -> None:
    if _extraction_complete():
        print(f"Already extracted at {EXTRACTED}, skipping.")
        return
    if not zipfile.is_zipfile(ARCHIVE):
        raise RuntimeError(f"{ARCHIVE} is not a complete/valid zip yet — download still in progress?")
    print(f"Extracting {ARCHIVE} -> {EXTRACTED} ...")
    EXTRACTED.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ARCHIVE) as zf:
        zf.extractall(EXTRACTED)
    print("Extraction done.")


def find_split_dirs() -> dict[str, tuple[Path, Path]]:
    """Locate images/ and labels/ for train/val/test, wherever they landed inside EXTRACTED."""
    splits: dict[str, tuple[Path, Path]] = {}
    for split in ("train", "val", "test"):
        images_dirs = list(EXTRACTED.rglob(f"{split}/images"))
        labels_dirs = list(EXTRACTED.rglob(f"{split}/labels"))
        if not images_dirs or not labels_dirs:
            raise RuntimeError(f"Couldn't find {split}/images or {split}/labels under {EXTRACTED}")
        splits[split] = (images_dirs[0], labels_dirs[0])
    return splits


def label_is_valid(label_path: Path) -> bool:
    if not label_path.exists():
        return True  # background image, no objects — valid in YOLO format
    for line in label_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            return False
        class_id = int(parts[0])
        if not (0 <= class_id < len(CLASS_NAMES)):
            return False
    return True


def clean_and_dedupe(splits: dict[str, tuple[Path, Path]]) -> dict[str, list[Path]]:
    seen_hashes: set[str] = set()
    kept: dict[str, list[Path]] = {"train": [], "val": [], "test": []}
    stats = {"corrupt": 0, "bad_label": 0, "duplicate": 0, "kept": 0}

    for split in ("train", "val", "test"):  # priority order: train > val > test
        images_dir, labels_dir = splits[split]
        for img_path in sorted(images_dir.glob("*")):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                stats["corrupt"] += 1
                continue
            label_path = labels_dir / (img_path.stem + ".txt")
            if not label_is_valid(label_path):
                stats["bad_label"] += 1
                continue
            digest = hashlib.md5(img_path.read_bytes()).hexdigest()
            if digest in seen_hashes:
                stats["duplicate"] += 1
                continue
            seen_hashes.add(digest)
            kept[split].append(img_path)
            stats["kept"] += 1

    print(f"Cleaning stats: {stats}")
    return kept


def write_clean_split(split: str, images: list[Path], labels_dir: Path) -> None:
    out_images = CLEAN / split / "images"
    out_labels = CLEAN / split / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)
    for img_path in images:
        shutil.copy2(img_path, out_images / img_path.name)
        label_path = labels_dir / (img_path.stem + ".txt")
        if label_path.exists():
            shutil.copy2(label_path, out_labels / label_path.name)
        else:
            (out_labels / (img_path.stem + ".txt")).write_text("")


def main() -> None:
    extract()
    splits = find_split_dirs()
    kept = clean_and_dedupe(splits)

    random.seed(RANDOM_SEED)
    if len(kept["train"]) > MAX_TRAIN_IMAGES:
        print(f"Subsetting train: {len(kept['train'])} -> {MAX_TRAIN_IMAGES} (CPU-feasible; edit MAX_TRAIN_IMAGES to change)")
        kept["train"] = random.sample(kept["train"], MAX_TRAIN_IMAGES)

    for split in ("train", "val", "test"):
        _, labels_dir = splits[split]
        write_clean_split(split, kept[split], labels_dir)
        print(f"{split}: {len(kept[split])} images written to {CLEAN / split}")

    # A separate small val slice for per-epoch validation *during training* (patience/early-stop
    # checks) — re-scoring the full val set every epoch is expensive and unnecessary for that
    # purpose. eval/benchmark.py's real accuracy numbers always come from the full test split
    # (reads clean/test directly, doesn't use either data.yaml), so this doesn't touch final
    # reported accuracy at all.
    _, val_labels_dir = splits["val"]
    val_subset = kept["val"][:VAL_SUBSET_SIZE]
    write_clean_split("val_small", val_subset, val_labels_dir)
    print(f"val_small: {len(val_subset)} images written to {CLEAN / 'val_small'} (training-time only)")

    names = {i: name for i, name in enumerate(CLASS_NAMES)}
    data_yaml = {
        "path": str(CLEAN),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": names,
    }
    (ROOT / "data.yaml").write_text(yaml.dump(data_yaml, sort_keys=False))
    print(f"Wrote {ROOT / 'data.yaml'}")

    data_fast_yaml = {**data_yaml, "val": "val_small/images"}
    (ROOT / "data_fast.yaml").write_text(yaml.dump(data_fast_yaml, sort_keys=False))
    print(f"Wrote {ROOT / 'data_fast.yaml'} (training scripts use this one)")


if __name__ == "__main__":
    main()
