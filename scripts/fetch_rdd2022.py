"""Download the RDD2022 test split and turn it into a feed-sim manifest + ground truth.

    uv run python -m scripts.fetch_rdd2022

Source: Hugging Face `dronefreak/RDD2022` (CC-BY-SA-4.0), a 4-class YOLO export whose
26,869 / 5,758 / 5,758 split matches the Kaggle `aliabdelmenam/rdd-2022` set Shervin trained
from. Per-file download, so no Kaggle token and no 10.6 GB zip on a nearly full disk.

Boxes come from each shard's `metadata.jsonl` (pixel x, y, w, h — checked against the YOLO
label file for China_Drone_000008), so the 5,758 label files are never fetched. Images are
fetched in parallel over plain HTTPS; the hub client's snapshot download managed ~3 files/s
unauthenticated, which put the split at an hour.

Outputs, all under `data/` (gitignored):
  data/rdd2022/data/images/test/shard_*/X.jpg   images
  data/rdd2022/manifest.json                    feed-sim pools, same shape as the fixtures'
  data/rdd2022/ground_truth.json                pixel boxes keyed by the same source_ref

Paths are relative to the repo root, which is also /app in the containers, so the
`source_ref` feed-sim sends, the one the API reads the frame back from, and the one ground
truth joins on are one string.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

REPO = "dronefreak/RDD2022"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main"
ROOT = Path("data/rdd2022")
SHARDS = {"test": ["shard_000", "shard_001"]}
#: The export's class ids → RDD damage codes used everywhere else.
#:
#: ⚠ The mirror's data.yaml calls id 3 `pothole`. It is not. The mirror was cut from the
#: Kaggle RDD_SPLIT export, whose order is 0 longitudinal, 1 transverse, 2 alligator,
#: 3 OTHER CORRUPTION, 4 pothole (Shervin's prepare_dataset.py uses the same list), and the
#: mirror "dropped class 4 as other" — i.e. it dropped the potholes and kept the other
#: corruption under the pothole name. Found 2026-09-27 on the first full e2e run: the model's
#: D40 predictions matched 15 of 1,587 "pothole" boxes, and a visual check of those boxes
#: showed manhole covers, drain grates and faded lane lines (RDD D43/D44/D50). So id 3 is
#: loaded as `other` (stored, never scored) and pothole accuracy is NOT measurable on this
#: mirror. Shervin's own test split still has potholes (his pothole mAP50: 0.433).
CLASS_CODES = {0: "D00", 1: "D10", 2: "D20", 3: "other"}


def _get(rel: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(8):
        try:
            with urllib.request.urlopen(f"{BASE}/{rel}", timeout=60) as resp:
                tmp.write_bytes(resp.read())
            tmp.rename(dest)
            return
        except OSError:
            # Unauthenticated hub requests hit HTTP 429 about 3,000 files in; back off
            # rather than hammer it. Re-running resumes: finished files are skipped.
            if attempt == 7:
                raise
            time.sleep(2 ** attempt)


def _records(split: str) -> list[tuple[str, dict]]:
    out = []
    for shard in SHARDS[split]:
        rel = f"data/images/{split}/{shard}/metadata.jsonl"
        _get(rel, ROOT / rel)
        for line in (ROOT / rel).read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                out.append((f"data/images/{split}/{shard}/{rec['file_name']}", rec["objects"]))
    return out


def download(records: list[tuple[str, dict]], workers: int) -> None:
    with ThreadPoolExecutor(workers) as pool:
        for i, _ in enumerate(pool.map(lambda r: _get(r[0], ROOT / r[0]), records), 1):
            if i % 500 == 0:
                print(f"{i}/{len(records)}", flush=True)


def build(split: str, records: list[tuple[str, dict]]) -> dict:
    clean, defect, truth = [], [], []
    for rel, objects in records:
        ref = str(ROOT / rel)
        if not objects["bbox"]:
            clean.append({"path": ref})
            continue
        with Image.open(ref) as im:
            width, height = im.size
        boxes = [{"defect_class": CLASS_CODES[c], "x": max(0, x), "y": max(0, y),
                  "w": max(1, min(width - max(0, x), w)), "h": max(1, min(height - max(0, y), h))}
                 for (x, y, w, h), c in zip(objects["bbox"], objects["categories"])]
        defect.append({"path": ref})
        truth.append({"source_ref": ref, "width": width, "height": height, "boxes": boxes})

    (ROOT / "manifest.json").write_text(json.dumps(
        {"source": f"hf:{REPO}", "split": split, "clean": clean, "defect": defect}))
    (ROOT / "ground_truth.json").write_text(json.dumps(truth))
    return {"images": len(records), "clean": len(clean), "defect": len(defect),
            "boxes": sum(len(t["boxes"]) for t in truth)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=sorted(SHARDS))
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    records = _records(args.split)
    download(records, args.workers)
    print(build(args.split, records))


if __name__ == "__main__":
    main()
