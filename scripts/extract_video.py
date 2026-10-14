"""Turn a dashcam video into a feed-sim manifest, so `make video` can replay it.

    uv run python -m scripts.extract_video VIDEO [--fps 8] [--out data/video/<stem>]

Samples the video at --fps (the feed rate, so a replay at the same fps runs in real
time), writes each sample as a JPEG under <out>/frames/, and writes <out>/manifest.json
in the RDD2022 manifest shape. A video has no labels, so every frame goes in one pool
and the manifest is marked `ordered`: feed-sim must replay it in capture order, not
shuffle it like the dataset. There is no ground truth, so the bench step has nothing to
score against.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def extract(video: Path, out: Path, fps: float, quality: int = 90) -> int:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = src_fps / fps
    frames_dir = out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    index, next_keep = 0, 0.0
    while True:
        ok, image = cap.read()
        if not ok:
            break
        if index >= next_keep:
            path = frames_dir / f"{len(entries):06d}.jpg"
            cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, quality])
            entries.append({"path": path.as_posix(), "video_frame": index,
                            "t_s": round(index / src_fps, 3)})
            next_keep += step
        index += 1
    cap.release()

    manifest = {"source": f"video:{video.name}", "split": "video", "ordered": True,
                "source_fps": src_fps, "sample_fps": fps,
                "clean": [], "defect": entries}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return len(entries)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("video", type=Path)
    ap.add_argument("--fps", type=float, default=8.0)
    ap.add_argument("--out", type=Path, default=None,
                    help="must be under data/, which the containers mount")
    args = ap.parse_args()
    out = args.out or Path("data/video") / args.video.stem
    n = extract(args.video, out, args.fps)
    print(f"{n} frames at {args.fps} fps → {out}/manifest.json")


if __name__ == "__main__":
    main()
