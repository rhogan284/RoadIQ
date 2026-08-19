"""Generate synthetic road-surface images with exact ground truth.

Clean: uniform grey + gaussian noise.
Defect: same, plus one dark rectangle whose bbox is recorded in the manifest.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import cv2
import numpy as np

BASE_GREY = 160
NOISE_SIGMA = 6
DEFECT_GREY = 60


def _canvas(size: int, rng: np.random.Generator) -> np.ndarray:
    noise = rng.normal(0, NOISE_SIGMA, (size, size))
    return np.clip(BASE_GREY + noise, 0, 255).astype(np.uint8)


def generate(out_dir: Path, *, n_clean: int, n_defect: int, size: int = 256,
             seed: int = 0) -> dict:
    out_dir = Path(out_dir)
    (out_dir / "clean").mkdir(parents=True, exist_ok=True)
    (out_dir / "defect").mkdir(parents=True, exist_ok=True)

    np_rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)
    manifest: dict[str, list[dict]] = {"clean": [], "defect": []}

    for i in range(n_clean):
        img = _canvas(size, np_rng)
        path = out_dir / "clean" / f"clean_{i:04d}.png"
        cv2.imwrite(str(path), img)
        manifest["clean"].append({"path": str(path)})

    for i in range(n_defect):
        img = _canvas(size, np_rng)
        w = py_rng.randint(12, 30)
        h = py_rng.randint(12, 30)
        x = py_rng.randint(8, size - w - 8)
        y = py_rng.randint(8, size - h - 8)
        img[y:y + h, x:x + w] = DEFECT_GREY
        path = out_dir / "defect" / f"defect_{i:04d}.png"
        cv2.imwrite(str(path), img)
        manifest["defect"].append(
            {"path": str(path), "x": x, "y": y, "w": w, "h": h,
             "defect_class": "pothole"}
        )

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("tests/fixtures/generated"))
    ap.add_argument("--clean", type=int, default=50)
    ap.add_argument("--defect", type=int, default=20)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    m = generate(args.out, n_clean=args.clean, n_defect=args.defect,
                 size=args.size, seed=args.seed)
    print(f"wrote {len(m['clean'])} clean, {len(m['defect'])} defect to {args.out}")
