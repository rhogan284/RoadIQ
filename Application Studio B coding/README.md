# Application Studio B coding

Code home for **Contract 2 — the detector plugin interface**, from
`University/Application Studio B/Research Plan — Shervin.md`: `detect(image) → list[Detection]`
+ name/version/params, frozen so Ryan can build the ingestion→inference→DB pipeline against a
stub instead of it becoming a Week 6 integration weekend.

Team project: **Automated Edge-CV Defect Detection & Pipeline Data Logging** (Portfolio
Component 3/4), domain locked to Option A — edge-first road assessment on the RDD2022 dataset.
Team: Ryan, Shervin, Ilana, Dexter, Joseph. Shervin owns model selection, training, and the
accuracy/latency trade-off (ILC lane T3).

## Docker setup (recommended for a fresh clone)
Containerises the detector environment (Python, torch, opencv, YOLOX, NanoDet and all their
install-time quirks below) so a fresh clone doesn't need to repeat the bare-metal setup at all.
Files live under `docker_images/`; nothing here touches the root RoadIQ `docker-compose.yml` or
`Dockerfile` — this is a separate, standalone Compose project.

**Prerequisites:** Docker + Docker Compose (v2, i.e. `docker compose`, not the old `docker-compose`
binary). No local Python/uv install needed for anything that runs inside the container.

**What you must obtain separately before first run — this repo does not (and per `.gitignore`,
should not) contain any of it:**
- **Dataset:** [Kaggle — aliabdelmenam/rdd-2022](https://www.kaggle.com/datasets/aliabdelmenam/rdd-2022)
  (CC BY-SA 4.0). Download `archive.zip` and place it at `data/RDD2022/archive.zip` (create the
  `data/RDD2022/` folder if it doesn't exist yet). Nothing else under `data/` needs to exist yet —
  `training/prepare_dataset.py` (run inside the container, see below) extracts/cleans/splits it.
- **Model weights:** downloaded automatically the first time each plugin runs, from the links in
  that plugin's own docstring under `detector_interface/plugins/` — e.g. `yolo11n.py` for
  `weights/yolo11n/yolo11n.pt`. The one exception is `weights/yolo12s/yolo12s_rdd2022.pt` (the
  [rezzzq/yolo12s-road-damage-rdd2022](https://huggingface.co/rezzzq/yolo12s-road-damage-rdd2022)
  Hugging Face checkpoint) — fetch that one by hand and place it at that exact path; there's no
  auto-download for it today.
- **NanoDet source** (`third_party/nanodet`): nothing to do — the Docker image fetches the exact
  pinned commit itself at build time (see below). This is only a manual `git clone` step in the
  bare-metal path further down this file.

**Build and start:**
```bash
cd docker_images
docker compose build      # ~a few minutes first time — torch alone is several hundred MB
docker compose up -d
```
This builds one image containing all six detector plugins' dependencies (Python 3.13, CPU-only
PyTorch, opencv, ultralytics, YOLOX from source, NanoDet at a pinned commit) and starts one
persistent `detector` container in the background.

**How the mounts work:** the container's `/app/data`, `/app/weights`, `/app/training/runs` and
`/app/eval/results` are bind-mounted straight to this folder's `data/`, `weights/`,
`training/runs/` and `eval/results/` — nothing written by a script running in the container is
container-local; it lands directly in your working tree, and anything you place in those folders
on the host is immediately visible inside the container too. Everything else (the actual code —
`detector_interface/`, `eval/benchmark.py`, `training/*.py`, `examples/`) is baked into the image
at build time, not mounted — see "Rebuild after changes" below for what that means in practice.
`third_party/nanodet` is also baked in (a pinned commit, not your local clone — see the Dockerfile
comment for why), so it does not need to exist on the host at all for the Docker path.

**Run something:**
```bash
docker compose exec detector python examples/list_detectors.py
docker compose exec detector python training/prepare_dataset.py
docker compose exec detector python training/train_yolo11n.py
docker compose exec detector python eval/benchmark.py --models yolo11n --limit 50
docker compose exec detector bash          # interactive shell, for anything else
```
(No `PYTHONPATH=.` needed inside the container — the image sets `PYTHONPATH=/app` already.)

**Stop the container(s)** — this leaves `data/`, `weights/`, `training/runs/` and `eval/results/`
on your host filesystem exactly as they were, since those are bind mounts, not container storage:
```bash
docker compose stop     # or: docker compose down (without -v)
```
**Never run `docker compose down -v`** for this project — it would only remove container-local
state (there isn't any persistent named volume here, so `-v` has nothing to do today, but don't
add one later without checking this note first).

**Rebuild after source or dependency changes:** editing `detector_interface/`, `eval/`,
`training/`, `examples/`, or `requirements.txt` on the host does **not** change the running
container — that code is `COPY`'d into the image at build time, not mounted. After such a change:
```bash
docker compose build
docker compose up -d      # recreates the container from the new image
```
Weights, dataset and run outputs are untouched by a rebuild (they're bind mounts, not part of the
image), so a rebuild never loses training progress.

## Setup (bare metal / without Docker)
The Docker path above is recommended for a fresh clone. This section is the original manual
install, still the reference for exactly which pip commands the Docker image itself runs — and a
fallback if you'd rather not use Docker.

No dedicated venv for this project — a `.venv` created inside this folder fails on Windows
(`WinError 206: filename too long`), because torch ships deeply nested license files and this
project's own path (`...\Mountain of Flowers and Fruits\Application Studio B coding\.venv\...`)
is already long enough to blow past the 260-char limit once torch's paths stack on top. So:
everything is installed straight into the base Python instead.

**Important — if your prompt shows `(canvas-mcp)`, deactivate it first.** That's a *different*
project's venv (the Canvas MCP server) and doesn't have these packages:
```powershell
deactivate
pip install -r requirements.txt
```
Confirm you're on the right interpreter before running anything:
```powershell
python -c "import sys; print(sys.executable)"
```
It should print `...\AppData\Local\Programs\Python\Python313\python.exe` — not a path
containing `canvas-mcp\.venv`.

**Extra one-time setup for the YOLOX and NanoDet plugins** (not in `requirements.txt` because
both need special install steps, not a plain `pip install`):
```powershell
# YOLOX-Tiny / YOLOX-Nano — the yolox PyPI package is broken (its sdist is missing
# requirements.txt), and one of its deps (onnx-simplifier) fails to build on Windows due to
# MAX_PATH, so install runtime deps directly and skip yolox's own dependency resolution:
pip install cmake
pip install loguru tabulate thop ninja tqdm tensorboard pycocotools
pip install --no-build-isolation --no-deps git+https://github.com/Megvii-BaseDetection/YOLOX.git

# NanoDet-m — no usable pip package at all (upstream repo bug drops nanodet/model and
# nanodet/data from any built wheel), so it's vendored as a plain git clone instead:
git clone --depth 1 https://github.com/RangiLyu/nanodet.git third_party/nanodet
pip install "pytorch_lightning<2.0.0" termcolor imagesize
```
`pytorch_lightning` **must** stay pinned below 2.0.0 — `third_party/nanodet/requirements.txt`
itself pins `pytorch-lightning>=1.9.0,<2.0.0`, but a plain unpinned `pip install pytorch_lightning`
resolves the latest release regardless (2.6.5 as of 2026-08-23). PL 2.0 removed the
`training_epoch_end` hook NanoDet's `TrainingTask` relies on, so training crashes immediately
with a `NotImplementedError` on an unpinned install — training/train_nanodet_m.py won't run
without this pin. `imagesize` is needed by NanoDet's built-in `YoloDataset` loader
(`nanodet/data/dataset/yolo.py`), used only for training, not inference — easy to miss since it's
never imported by the detection plugin itself. The Docker image (see above) already has both of
these pinned correctly, so this only matters for the bare-metal path.

See the docstrings in `detector_interface/plugins/yolox_tiny.py` and `plugins/nanodet_m.py`
for exactly why each of these steps is needed, and `training/train_nanodet_m.py`'s module
docstring for two further CPU/library-version bugs in NanoDet's vendored `tools/train.py`
(a missing `map_location="cpu"` on checkpoint load, and two `pytorch_lightning` Trainer kwargs
that changed meaning between versions) — that script patches around both from the outside rather
than editing the vendored file, so nothing further is needed to run it, just worth knowing about
if `tools/train.py` is ever invoked directly instead.

CPU-only is assumed throughout (no CUDA on this machine) — training scripts default to
`device="cpu"` and small image sizes/epoch counts accordingly.

## Layout
- `detector_interface/` — the frozen interface (`base.py`, `detection.py`,
  `detection.schema.json`, `registry.py`) plus one adapter per surveyed model under `plugins/`:
  YOLOX-Tiny, YOLOX-Nano, YOLOv4-tiny, YOLO11n, YOLO12s, NanoDet-m.
- `weights/` — pretrained weights per model (downloaded from each model's official source —
  see the docstring at the top of each file in `plugins/` for the exact link and license).
- `third_party/nanodet` — vendored git clone of RangiLyu/nanodet (source code, not a weight —
  see `plugins/nanodet_m.py` for why it's cloned rather than pip-installed).
- `data/RDD2022/` — the dataset (see below).
- `training/` — dataset prep + training scripts.
- `eval/benchmark.py` — the shared accuracy/speed harness, works against any wired-up plugin.

## Dataset
Source: [Kaggle — aliabdelmenam/rdd-2022](https://www.kaggle.com/datasets/aliabdelmenam/rdd-2022)
(CC BY-SA 4.0). Already pre-split 70/15/15 train/val/test, YOLO-format labels, 5 classes:
longitudinal crack, transverse crack, alligator crack, other corruption, pothole.

`training/prepare_dataset.py` extracts `data/RDD2022/archive.zip`, then:
- **Cleans**: drops any image OpenCV can't decode, and any label file with an out-of-range
  class index.
- **De-duplicates**: MD5-hashes every image; if the same image appears in more than one split
  (frame reuse), keeps it only in the split it first appears in (train > val > test priority) —
  otherwise a duplicate frame in both train and test would let a model "cheat" by memorizing it.
- **Subsets train** to `MAX_TRAIN_IMAGES` (4,000 by default) via random sample — val/test are
  kept full-size, since those are the numbers that matter. This is purely to keep CPU training
  time reasonable; raise it yourself and re-run if you have time to spare.
- Writes `data/RDD2022/data.yaml` (Ultralytics format) pointing at the cleaned/split result in
  `data/RDD2022/clean/`.

The original train/val/test split from the dataset author is kept as-is rather than
re-splitting from scratch — it's already sound, and re-splitting risks mixing near-duplicate
frames across train and test.

## Status per model
| Model | Status |
|---|---|
| **YOLO11n** | Real inference wired up; still COCO-pretrained — `training/train_yolo11n.py` fine-tunes it on RDD2022 but hasn't been run yet, so it reads ~0% mAP against RDD2022 until then |
| **YOLO12s** | Real inference, already fine-tuned on RDD2022 by a third party ([rezzzq/yolo12s-road-damage-rdd2022](https://huggingface.co/rezzzq/yolo12s-road-damage-rdd2022) on Hugging Face) — no training run of our own needed, just benchmark it. Class order differs from ours; see the remap caveat in `plugins/yolo12s.py`. |
| **YOLOv4-tiny** | Real inference via `cv2.dnn.readNetFromDarknet`, but COCO-pretrained only — ~0% mAP against RDD2022 until fine-tuned (needs the darknet C/CUDA build toolchain, not set up here) |
| **YOLOX-Tiny / YOLOX-Nano** | Real inference via the official `yolox` GitHub source (see Setup) — COCO-pretrained only, same ~0% mAP caveat until fine-tuned |
| **NanoDet-m** | Real inference via a vendored git clone of the official repo (see Setup) — COCO-pretrained only, same ~0% mAP caveat until fine-tuned |

Only YOLO12s currently has real (non-zero) accuracy against RDD2022 — it's the only one fine-tuned
on this dataset so far. Everything else needs either our own fine-tuning run (YOLO11n has a
script for this; the rest don't yet) or should be read purely as a speed/latency data point
until they get one.

## How to run it
**PowerShell** (set `PYTHONPATH` once per session, then run scripts normally — PowerShell
doesn't support bash's inline `VAR=value command` syntax):
```powershell
$env:PYTHONPATH = "."
python training\prepare_dataset.py    # extract, clean, dedupe, split
python training\train_yolo11n.py      # trains live, prints per-epoch progress
python eval\benchmark.py               # accuracy + speed for every wired-up model
```
**Bash / Git Bash:**
```
PYTHONPATH=. python training/prepare_dataset.py
PYTHONPATH=. python training/train_yolo11n.py
PYTHONPATH=. python eval/benchmark.py
```
- `train_yolo11n.py` saves the fine-tuned checkpoint to `weights/yolo11n/yolo11n_rdd2022.pt` —
  the plugin picks it up automatically once present (falls back to COCO-pretrained otherwise).
- `eval/benchmark.py` prints mAP@0.5 / precision / recall / ms-per-image / img-per-sec per
  model and writes full results to `eval/results/benchmark_results.json`. Flags:
  `--models yolo11n yolov4-tiny` to pick specific models, `--limit 0` for the full test split
  instead of the default 500-image sample.
- `examples/list_detectors.py` — quick registry demo, no weights/dataset needed.
