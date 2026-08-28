# Application Studio B coding

Code home for **Contract 2 — the detector plugin interface**, from
`University/Application Studio B/Research Plan — Shervin.md`: `detect(image) → list[Detection]`
+ name/version/params, frozen so Ryan can build the ingestion→inference→DB pipeline against a
stub instead of it becoming a Week 6 integration weekend.

Team project: **Automated Edge-CV Defect Detection & Pipeline Data Logging** (Portfolio
Component 3/4), domain locked to Option A — edge-first road assessment on the RDD2022 dataset.
Team: Ryan, Shervin, Ilana, Dexter, Joseph. Shervin owns model selection, training, and the
accuracy/latency trade-off (ILC lane T3).

## Setup
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
pip install pytorch_lightning termcolor
```
See the docstrings in `detector_interface/plugins/yolox_tiny.py` and `plugins/nanodet_m.py`
for exactly why each of these steps is needed.

CPU-only is assumed throughout (no CUDA on this machine) — training scripts default to
`device="cpu"` and small image sizes/epoch counts accordingly.

## Layout
- `detector_interface/` — the frozen interface (`base.py`, `detection.py`,
  `detection.schema.json`, `registry.py`) plus one adapter per surveyed model under `plugins/`:
  YOLOX-Tiny, YOLOX-Nano, YOLOv4-tiny, YOLO11n, YOLO12s, NanoDet-m. (nsr51324's YOLO11n and
  cvtechniques' YOLO11s, added 2026-08-27, have weights/training scripts but no plugin yet.)
- `weights/` — pretrained weights per model (downloaded from each model's official source —
  see the docstring at the top of each file in `plugins/` for the exact link and license).
  `weights/nsr51324_yolo11n/` and `weights/cvtechniques_yolo11s/` follow the same pattern but
  aren't wired into `plugins/` yet — see the model table below and each `training/train_*.py`
  script's docstring.
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
| **YOLO12s** | Real inference, already fine-tuned on RDD2022 by a third party ([rezzzq/yolo12s-road-damage-rdd2022](https://huggingface.co/rezzzq/yolo12s-road-damage-rdd2022) on Hugging Face) — no training run of our own needed, just benchmark it. Class order differs from ours; see the remap caveat in `plugins/yolo12s.py`. Weights re-downloaded 2026-08-27 (local run outputs had been deleted). |
| **YOLOv4-tiny** | Real inference via `cv2.dnn.readNetFromDarknet`, but COCO-pretrained only — ~0% mAP against RDD2022 until fine-tuned (needs the darknet C/CUDA build toolchain, not set up here) |
| **YOLOX-Tiny / YOLOX-Nano** | Real inference via the official `yolox` GitHub source (see Setup) — COCO-pretrained only, same ~0% mAP caveat until fine-tuned |
| **NanoDet-m** | Real inference via a vendored git clone of the official repo (see Setup) — COCO-pretrained only. `training/train_nanodet_m.py` + `training/nanodet_rdd2022.yml` (added 2026-08-27) aren't runnable yet — see that script's docstring for the dataset-layout gap that needs fixing first |
| **nsr51324 YOLO11n** *(no plugin yet)* | Weights only (`weights/nsr51324_yolo11n/`), downloaded 2026-08-27 from [nsr51324/Road_Damage_Object_Detection](https://huggingface.co/nsr51324/Road_Damage_Object_Detection) — already road-damage-fine-tuned by a third party, but on an unverified class taxonomy. `training/train_nsr51324_yolo11n.py` continues fine-tuning it; not run yet, no `plugins/` adapter written |
| **cvtechniques YOLO11s** *(no plugin yet, no real weights)* | [cvtechniques/road-damage-detection-yolov11](https://huggingface.co/cvtechniques/road-damage-detection-yolov11) documents a fine-tune (mAP50 ~0.47) but never published the weights — checked its file listing directly, only README + images exist there. `training/train_cvtechniques_yolo11s.py` fine-tunes stock COCO yolo11s.pt instead; not run yet |

Only YOLO12s currently has real (non-zero) accuracy against RDD2022 — it's the only one fine-tuned
on this dataset so far. Everything else needs either our own fine-tuning run (YOLO11n, nsr51324
YOLO11n, and cvtechniques YOLO11s all have scripts for this now; YOLOv4-tiny/YOLOX/NanoDet-m
don't yet, or aren't runnable yet) or should be read purely as a speed/latency data point until
they get one.

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
