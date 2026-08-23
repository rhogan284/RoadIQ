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
| YOLO11n | Yes | Yes — trained from base COCO weights, 15 epochs, output at `weights/yolo11n/yolo11n_rdd2022.pt` |
| YOLO12s | Yes | Yes — base checkpoint from [rezzzq/yolo12s-road-damage-rdd2022](https://huggingface.co/rezzzq/yolo12s-road-damage-rdd2022) (HF, MIT weights), already fine-tuned by a third party. `training/train_yolo12s.py` continued fine-tuning that checkpoint on our own subset — output at `weights/yolo12s/yolo12s_rdd2022_continued.pt`, which the plugin prefers automatically |
| YOLOv4-tiny | Yes (`cv2.dnn`) | No — blocked, needs the darknet C/CUDA toolchain, not installed |
| YOLOX-Tiny / YOLOX-Nano | Yes | No — a fine-tuning attempt was abandoned: the pip-installed `yolox` package's own trainer hardcodes CUDA (`torch.cuda.set_device(...)` unconditionally), and `tools/train.py` isn't even included in the pip package, only the full GitHub repo. Would need a from-scratch CPU training loop, same category of work NanoDet-m's `train_nanodet_m.py` ended up needing but deeper — not attempted. |
| NanoDet-m | Yes | Yes — trained from base COCO weights, 15 epochs, output at `weights/nanodet_m/nanodet_m_rdd2022.ckpt` |

YOLO12s, YOLO11n, and NanoDet-m all have real measured accuracy against RDD2022 now (see results
sections below) — YOLO12s is clearly ahead on every metric, NanoDet-m is behind both on every
metric. YOLOv4-tiny is still COCO-pretrained-only (reads ~0% mAP on RDD2022 until fine-tuned).

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
  need `__init__.py`). Also needs `pip install "pytorch_lightning<2.0.0" termcolor imagesize`
  (**must** stay pinned below 2.0.0 — `third_party/nanodet/requirements.txt` itself pins
  `<2.0.0`, but an unpinned install resolves the latest release and its removed
  `training_epoch_end` hook crashes NanoDet's trainer immediately; `imagesize` is needed by the
  training-only `YoloDataset` loader, easy to miss since inference never imports it), and the
  plugin monkey-patches a `torch._six` shim at runtime (removed from modern torch; nanodet's
  `collate.py` still imports from it at module level). `training/train_nanodet_m.py` and
  `test_nanodet_m.py` patch three more CPU/library-version bugs in the vendored `tools/train.py`
  from the outside (missing `map_location="cpu"` on checkpoint load, two `pytorch_lightning`
  `Trainer` kwargs that changed meaning between versions, and a validation-only log line that
  assumes an optimizer exists) — see those scripts' module docstrings, nothing further needed to
  run them.
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

## YOLO11n fine-tune — completed, results (2026-08-23)
Trained from base COCO weights (not a pre-fine-tuned checkpoint like YOLO12s), 2000 train / 400
val_small, imgsz=256, batch=8, `patience=5`. Ran the **full 15 epochs** without early-stopping —
`metrics/mAP50(B)` climbed steadily the whole way (0.011 → 0.133 on val_small), unlike YOLO12s
which peaked at epoch 1. Per-epoch time ≈ 5:38 train + 1:12 val ≈ 6:50, matching the ~40min
(early-stop) to ~1hr45 (full run) prediction — landed at the full-run end. Final test-set eval
(5,758 images) added ~15-20min more (linear-scaled from the 400-image val time), CPU inference
speed **33.9ms/image (~29.5 img/sec)**. Weights at `weights/yolo11n/yolo11n_rdd2022.pt`.

**Test-set (5,758 images): mAP50 = 0.1264, mAP50-95 = 0.0504, precision = 0.2219, recall =
0.2023.** Per-class mAP50: other corruption 0.248, alligator crack 0.165, longitudinal crack
0.096, transverse crack 0.090, pothole 0.033.

**YOLO11n vs YOLO12s comparison** — YOLO12s wins on every metric (~2x mAP50, ~2.4x mAP50-95,
~1.7x precision, ~1.6x recall), expected given it started from a checkpoint already fine-tuned
on RDD2022 by a third party and has ~3.5x the params (9M vs 2.6M). More interesting: **the
per-class ranking flips**. YOLO12s's best class is alligator crack (0.448) and worst is other
corruption (0.110); YOLO11n's best is other corruption (0.248) and worst is pothole (0.033).
This lines up with the class-remapping caveat already documented in `plugins/yolo12s.py` — the
"other corruption" and "pothole" output slots got relabeled during YOLO12s's continued
fine-tune and never fully recovered in only 6 epochs, specifically hurting those two classes for
that model. YOLO11n trained fresh has no such handicap, so "other corruption" tops its table
instead. Pothole is the weakest class for *both* models regardless — likely a genuinely harder/
rarer class, not a training artifact.

Note: `eval/results/benchmark_results.json` is stale (pre-fine-tuning, imgsz=640 not 256, and
its `weights_path` still points at the old OneDrive copy from the two-copies incident above)
— don't read speed/accuracy numbers from it, it predates both models' real fine-tunes.

## NanoDet-m fine-tune — completed, results (2026-08-23)
Trained from base COCO weights via `training/train_nanodet_m.py`, which drives NanoDet's own
vendored `tools/train.py` programmatically (`pytorch_lightning` Trainer under the hood) rather
than a hand-written loop — same 2000 train / 400 val_small, imgsz=256, batch=8 convention as the
other two, but **no early-stop/patience mechanism** (would have needed patching NanoDet's
vendored trainer, skipped to keep the first run lower-risk), so it always runs the full 15
epochs. An earlier fine-tuning attempt on YOLOX-Tiny was abandoned first — see the model status
table above — before landing on NanoDet-m as the next model to train.

Per-epoch time was inconsistent, tracked live via the timestamp prefix on each log line:
7, 11, 10, 10, 10, 6, 9, 7, 11, 9, 7, [13 combined over 2 epochs], 5 min — averaging **~9
min/epoch**, total wall time **~2hrs**. The variance (unlike YOLO11n/YOLO12s's steadier timing)
is most likely Docker Desktop running in the background competing for CPU, not a NanoDet-specific
issue — it was started partway through this session to build the `docker_images/` image.

Best checkpoint per NanoDet's own tracking (only logs a new line when validation improves) was
**epoch 13**, not epoch 15 — validation mAP on val_small got *worse* over the last 2 epochs
(0.0277 at epoch 13 → 0.0239 at epoch 15). `nanodet_model_best.pth` (epoch 13's weights) is what
`train_nanodet_m.py` copies to `weights/nanodet_m/nanodet_m_rdd2022.ckpt`, not the last epoch —
same "best checkpoint isn't the last one" pattern YOLO12s hit.

`train_nanodet_m.py` only ever validates against val_small (400 images) — getting a number
comparable to YOLO11n/YOLO12s needed a separate script, `training/test_nanodet_m.py`, written
after training finished. It reuses NanoDet's own `TrainingTask` + `CocoDetectionEvaluator` via
`pytorch_lightning`'s `Trainer.validate()` rather than reimplementing scoring by hand, pointed at
the full test split with the fine-tuned checkpoint loaded instead of the base COCO one. Needed
two more from-outside patches beyond `train_nanodet_m.py`'s three (wrong Lightning logger class
for `Trainer.validate()`; a per-batch log line that assumes `trainer.optimizers` is non-empty,
which is only true when training via `.fit()`) — all in that script's own docstring/comments.

**Test-set (5,758 images), from `test_nanodet_m.py`: mAP50 = 0.1005, mAP50-95 = 0.0374, AP75 =
0.0218.** Speed: 720 batches (8/batch) in 5:13 → **54.4ms/image (~18.4 img/sec)**. Per-class
mAP50: other corruption 0.183, alligator crack 0.172, longitudinal crack 0.078, transverse crack
0.060, pothole 0.011 (weakest, consistent with every other model so far). NanoDet doesn't report
single-number precision/recall the way Ultralytics does — it gives COCO-style Average Recall at
several maxDets instead (AR@100 = 0.209 is the loosest analog), so that row isn't directly
comparable across all three models; mAP50/mAP50-95 are, since both tools compute those the same
way.

**Three-way comparison (all on the same full 5,758-image test set now):**
| Model | mAP50 | mAP50-95 | Speed |
|---|---|---|---|
| YOLO12s | 0.263 | 0.122 | not measured |
| YOLO11n | 0.126 | 0.050 | 33.9ms/img (~29.5 img/s) |
| NanoDet-m | 0.101 | 0.037 | 54.4ms/img (~18.4 img/s) |

**NanoDet-m is last on both accuracy and speed** — worth a sentence in the report specifically
because it's counterintuitive: it's by far the smallest model (0.95M params vs YOLO11n's 2.6M,
YOLO12s's 9M) but the *slowest* measured here, not the fastest. Not about raw FLOPs — framework
overhead: NanoDet's eval runs through `pytorch_lightning`'s loop plus its own Python-level
post-processing, versus Ultralytics' more heavily optimized inference path. Smaller param count
didn't translate to either better accuracy or faster wall-clock time in this setup. Accuracy-wise,
plausible given it started from a plain COCO checkpoint (same as YOLO11n) but has ~2.7x fewer
params to work with.

## Docker (done — this section used to say "planned, not yet done", it's stale no longer)
`docker_images/` (Dockerfile + docker-compose.yml, standalone from the root RoadIQ compose
stack — see this folder's own `README.md`) now containerizes the whole detector environment;
`docker compose build` gets a teammate a working setup without repeating the manual install saga
above. Built and confirmed working 2026-08-23 — the Dockerfile's own `pip install
pytorch_lightning termcolor` line had the exact same unpinned-version bug flagged in the NanoDet-m
setup gotcha above (would've hit every teammate who built it fresh), fixed there too so it's not
just documented here but actually prevented at build time.
- Still true: a normal Linux `venv` inside the image sidesteps the Windows-specific MAX_PATH/
  no-venv pain above entirely.
- **Still an open design question, not yet decided**: is this meant to be imported as a Python
  package directly inside Ryan's pipeline code (matches the original Contract 2 framing — "so
  Ryan can code the pipeline against a stub"), or served as a standalone model API other team
  members' code calls over the network? The current `docker_images/docker-compose.yml` doesn't
  answer this either way (just a persistent `sleep infinity` container you `exec` into) — settle
  with the team before assuming either direction for a real deployment.

## Open TODOs
- [x] Finish YOLO12s continued fine-tune — done 2026-08-22, see results above.
- [x] Run `training/train_yolo11n.py` — done 2026-08-23, see results above.
- [x] Train NanoDet-m — done 2026-08-23 (`training/train_nanodet_m.py` +
      `training/test_nanodet_m.py` for the full-test-set number), see results above.
- [ ] No training script exists yet for YOLOv4-tiny (blocked on darknet toolchain) or YOLOX-Tiny/
      Nano (attempted for YOLOX, abandoned — see model status table) — would need real work if
      the team wants fine-tuned numbers for those too, not just COCO-pretrained speed data points.
- [ ] `eval/results/benchmark_results.json` is stale (pre-fine-tuning) — re-run
      `eval/benchmark.py` against all three fine-tuned checkpoints for a real accuracy+speed
      comparison on equal footing (same imgsz, same image set); would also finally give YOLO12s
      a measured speed number, the one still missing from the three-way comparison above.
- [ ] Docker repo move (splitting this folder into its own repo) — separate from the
      containerization itself, which is done (see Docker section above).
