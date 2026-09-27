"""Download the YOLOv12s road-damage weights into `weights/` (gitignored).

    uv run --with huggingface_hub python -m scripts.fetch_weights

The public `rezzzq/yolo12s-road-damage-rdd2022` checkpoint (MIT) is the one Shervin's v4
continued from. When his `yolo12s_rdd2022_continued.pt` arrives, drop it in `weights/` and
set `YOLO_WEIGHTS=weights/yolo12s_rdd2022_continued.pt` — nothing else changes.
"""
from __future__ import annotations

from pathlib import Path

REPO = "rezzzq/yolo12s-road-damage-rdd2022"
FILENAME = "yolo12s_RDD2022_best.pt"


def main() -> None:
    from huggingface_hub import hf_hub_download

    out = Path("weights")
    out.mkdir(exist_ok=True)
    path = hf_hub_download(repo_id=REPO, filename=FILENAME, local_dir=out)
    print(f"weights at {path} ({Path(path).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
