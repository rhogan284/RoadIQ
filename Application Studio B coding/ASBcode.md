# ASBcode — project memory

Context for `Application Studio B coding/`: Contract 2 (the detector plugin interface) for the
team's **Automated Edge-CV Defect Detection & Pipeline Data Logging** project (Portfolio
Component 3/4, Application Studio B), domain locked to Option A — edge-first road assessment on
RDD2022. Team: Ryan, Shervin, Ilana, Dexter, Joseph. Shervin owns model selection, training, and
the accuracy/latency trade-off (ILC lane T3). Full team/course context lives in
`University/Application Studio B/Research Plan — Shervin.md` (outside this folder) — this file
only covers what's specific to the code here, so it still makes sense once this folder becomes
its own repo and that University vault isn't alongside it anymore.

This file is meant to be read at the start of any future session touching this code — update it
as things change rather than letting it go stale. Not auto-loaded like `CLAUDE.md` would be
(deliberately named `ASBcode.md` instead), so point a fresh session at it explicitly.

## What this repo is
`detect(image) -> list[Detection]` + name/version/params — a frozen interface
(`detector_interface/base.py`, `detection.py`) so Ryan's ingestion→inference→DB pipeline can
swap detectors without changing its own code, plus one adapter per model under `plugins/` for
benchmarking the accuracy/latency trade-off (`eval/benchmark.py`).

## Model status (as of 2026-08-22)
| Model | Real inference? | Fine-tuned on RDD2022? |
|---|---|---|
| YOLO11n | Yes | Not yet — `training/train_yolo11n.py` exists but hasn't been run to completion |
| YOLO12s | Yes | Yes — base checkpoint from [rezzzq/yolo12s-road-damage-rdd2022](https://huggingface.co/rezzzq/yolo12s-road-damage-rdd2022) (HF, MIT weights), already fine-tuned by a third party. `training/train_yolo12s.py` continues fine-tuning that checkpoint on our own subset (in progress as of this writing) — output lands at `weights/yolo12s/yolo12s_rdd2022_continued.pt`, which the plugin prefers automatically once it exists |
| YOLOv4-tiny | Yes (`cv2.dnn`) | No — blocked, needs the darknet C/CUDA toolchain, not installed |
| YOLOX-Tiny / YOLOX-Nano | Yes | No — no training script written yet |
| NanoDet-m | Yes | No — no training script written yet |

Only YOLO12s currently has non-zero measured accuracy against RDD2022; everything else is either
mid-training or still COCO-pretrained-only (reads ~0% mAP on RDD2022 until fine-tuned).

## Setup gotchas (all Windows-specific — see "Moving to Docker" below, most of this may not
apply on Linux)
- **No local `.venv`** — one inside this folder hits `WinError 206` (filename too long) because
  torch's nested license files stack on top of an already-long project path. Everything installs
  straight into base Python instead. Watch for a stray `(canvas-mcp)` venv prompt (a different
  project) — `deactivate` it first.
- **YOLOX-Tiny/Nano**: the `yolox` PyPI package is broken (sdist missing `requirements.txt`).
  Installed from GitHub instead, and with `--no-deps` because one of its deps
  (`onnx-simplifier`) vendors a test-data tree deep enough to blow past Windows' MAX_PATH during
  its own build:
  ```
  pip install cmake
  pip install loguru tabulate thop ninja tqdm tensorboard pycocotools
  pip install --no-build-isolation --no-deps git+https://github.com/Megvii-BaseDetection/YOLOX.git
  ```
  (`cmake` itself is only needed because no prebuilt `onnx` wheel exists for this Python version —
  see "Moving to Docker.")
- **NanoDet-m**: no usable pip package at all — upstream repo's `nanodet/model/` and
  `nanodet/data/` are missing `__init__.py`, so `setuptools.find_packages()` silently drops them
  from any built wheel (confirmed empirically). Vendored as a plain git clone instead:
  `git clone --depth 1 https://github.com/RangiLyu/nanodet.git third_party/nanodet`, imported via
  `sys.path` insertion in `plugins/nanodet_m.py` (works because Python 3 namespace packages don't
  need `__init__.py`). Also needs `pip install pytorch_lightning termcolor`, and the plugin
  monkey-patches a `torch._six` shim at runtime (removed from modern torch; nanodet's
  `collate.py` still imports from it at module level).
- Full detail/reasoning for both of the above is in the docstrings at the top of
  `detector_interface/plugins/yolox_tiny.py` and `plugins/nanodet_m.py`.

## Data
`data/RDD2022/archive.zip` (10.6GB, Kaggle `aliabdelmenam/rdd-2022`, CC BY-SA 4.0) —
**don't commit this or the extracted/cleaned splits to git**, it's a documented download step
(see README). `training/prepare_dataset.py` extracts/cleans/dedupes it into
`data/RDD2022/clean/{train,val,test}` and writes `data/RDD2022/data.yaml`.
`MAX_TRAIN_IMAGES` (in `prepare_dataset.py`) is currently **2000** (down from the original 4000
default) and `imgsz` in both training scripts is **256** (down from 416) — both lowered
2026-08-22 to keep CPU training time down.

## ⚠️ There were briefly two copies of this project on disk (resolved 2026-08-22)
`Desktop\...\Application Studio B coding` (this one — where all real work happens) and a second,
older `OneDrive\Desktop\...\Application Studio B coding` that nothing today touched. **A stale
`data.yaml` already sitting in this (Desktop) copy had its internal `path:` field hardcoded to
the OneDrive copy** — so the first YOLO12s training attempt silently trained against the
OneDrive copy's *old* dataset (4,000 train images — the pre-today `MAX_TRAIN_IMAGES` default,
not the 2,000 set today — and the full 5,758-image val set), not anything `prepare_dataset.py`
produced today, even though the script ran without any error. If dataset numbers ever look
inconsistent with what `prepare_dataset.py`'s constants say they should be, **check `data.yaml`'s
`path:` field is actually this folder**, and check `data/RDD2022/clean/{train,val}/images` isn't
suspiciously empty despite training "working." The OneDrive copy hasn't been deleted or
investigated further (why it exists, whether it's an intentional backup) — flag it to a human
before doing anything destructive to it.

## Training timing — the first measurement was against the wrong dataset (see above), don't trust it
The initial ~28-30 min/epoch reading for YOLO12s (epoch training ≈ 20-21 min, per-epoch
validation ≈ 7-9 min) was measured against the stale OneDrive-referenced data (4,000 train images,
full 5,758-image val) — not a real reading of this project's actual settings, so don't extrapolate
from it. Two real fixes went in before the restart, both in `training/prepare_dataset.py` /
`train_yolo11n.py` / `train_yolo12s.py`:
1. The stale `data.yaml` was deleted so a fresh one (pointing at *this* folder) gets written.
2. `prepare_dataset.py` now also writes `val_small` (`VAL_SUBSET_SIZE = 400` images sampled from
   val) and a second `data_fast.yaml` pointing training at that small val slice instead of the
   full ~5.8k one — training scripts use `data_fast.yaml` for `model.train()`, but still use the
   full `data.yaml` for the final test-set accuracy printout, so reported accuracy is unaffected.
**Real reading from the corrected setup (2026-08-22, YOLO12s continued fine-tune, 2000 train /
400 val, imgsz=256, batch=8): ~10-13 min/epoch, holding steady across all 6 epochs it ran.**
Total wall time ≈ 71 min training + final test-set eval — well under the ~2.5-3hr worst case.

## YOLO12s continued fine-tune — completed, results (2026-08-22)
Stopped at **epoch 6** (`patience=5` triggered — see below). Weights at
`weights/yolo12s/yolo12s_rdd2022_continued.pt`, plugin picks it up automatically
(`version: rdd2022-continued-ours`). Test-set (5,758 images): **mAP50 = 0.263, mAP50-95 = 0.122,
precision = 0.369, recall = 0.315**. Per-class mAP50: alligator crack 0.448, longitudinal crack
0.310, transverse crack 0.281, pothole 0.166, other corruption 0.110.

Two findings worth keeping for the report/R-S2b writeup:
1. **The class-remapping caveat in `plugins/yolo12s.py` played out exactly as predicted** — the
   two classes whose output slot got relabeled during continued training (pothole: was "Repair"
   in the base checkpoint; other corruption: was "pothole") scored roughly 2-4x worse than the
   three classes that kept their original slot. 6 epochs wasn't enough for the head to fully
   relearn what those two slots mean now.
2. **Per `training/runs/yolo12s_rdd2022_continued/results.csv`, epoch 1 had the best validation
   mAP50 (0.278) of the entire run** — epochs 2-6 never beat it, which is why `patience=5` cut
   it off at epoch 6. Validation accuracy got *worse* after epoch 1 and never recovered within
   the budget — plausibly the class-remapping disruption outweighing any relearning benefit that
   early on. `best.pt` (what actually got copied to the `_continued.pt` file) is epoch 1's
   checkpoint, not epoch 6's.

## Moving to Docker / a new GitHub repo (planned, not yet done)
Plan: move this folder into its own repo, containerize it, teammates use it via Docker instead of
each repeating today's manual install saga. Notes for whoever does this:
- Most of the Windows-specific pain above (MAX_PATH, the no-venv workaround) doesn't apply on
  Linux — a normal `venv` inside a Docker image should just work.
- Consider pinning an **older Python** (e.g. 3.11/3.12) in the Docker image instead of 3.13 —
  the `cmake`-to-build-`onnx`-from-source step exists specifically because no prebuilt `onnx`
  wheel exists yet for 3.13. An older Python likely has one and skips that step entirely.
- Don't bake `archive.zip` / the extracted dataset into the image — mount as a volume or keep it
  a documented download step.
- Weights (~100MB total) are small enough to commit or fetch via a small download script —
  prefer a script (not yet written) over shipping raw binaries, keeps the repo light.
- `third_party/nanodet` should be a Dockerfile `git clone` step, not a copied folder (avoids
  vendoring its `.git` history into your own repo).
- **Open design question, not yet decided**: is this meant to be imported as a Python package
  directly inside Ryan's pipeline code (matches the original Contract 2 framing — "so Ryan can
  code the pipeline against a stub"), or served as a standalone model API other team members'
  code calls over the network? Changes the Docker architecture significantly — settle this with
  the team before building the Dockerfile.

## Open TODOs
- [ ] Finish YOLO12s continued fine-tune (in progress).
- [ ] Run `training/train_yolo11n.py` (dataset now prepped as of whenever `prepare_dataset.py`
      last completed — check `data/RDD2022/clean/train/images` isn't empty before assuming so).
- [ ] No training script exists yet for YOLOv4-tiny (blocked on darknet toolchain), YOLOX-Tiny/
      Nano, or NanoDet-m — would need to be written if the team wants fine-tuned numbers for
      those too, not just COCO-pretrained speed data points.
- [ ] Docker/repo move — see above, architecture question unresolved.
