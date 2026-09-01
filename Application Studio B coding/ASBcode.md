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

## Model status (as of 2026-09-01 — YOLO12s v4 completed, see `modelresults.md`)
| Model | Real inference? | Fine-tuned on RDD2022? |
|---|---|---|
| YOLO11n | Yes | Yes — full 15-epoch run completed 2026-08-28. Test-set (5758 images): mAP50=0.1012, mAP50-95=0.0408, precision=0.1867, recall=0.1847 |
| YOLO12s | Yes | Yes — continued fine-tune from [rezzzq/yolo12s-road-damage-rdd2022](https://huggingface.co/rezzzq/yolo12s-road-damage-rdd2022), latest run (v4, imgsz=416, MAX_TRAIN_IMAGES=8000) completed 2026-09-01, AdamW lr0=0.001. **Best of all real runs, wins on every metric**: mAP50=0.5362, mAP50-95=0.2701, precision=0.6160, recall=0.5244. Output at `weights/yolo12s/yolo12s_rdd2022_continued.pt` — see `modelresults.md`'s "YOLO12s v4" entry |
| YOLOv4-tiny | Yes (`cv2.dnn`) | No — blocked, needs the darknet C/CUDA toolchain, not installed |
| YOLOX-Tiny / YOLOX-Nano | Yes | No — no training script written yet |
| NanoDet-m | Yes | **Attempted 2026-08-28, failed**: `ModuleNotFoundError: No module named 'nanodet'` — `training/train_nanodet_m.py`'s subprocess call never adds `third_party/nanodet` to the child process's `sys.path`/`PYTHONPATH` (the inference plugin does this manually; the training wrapper doesn't). Dataset-layout flattening step itself worked fine (2000 train + 400 val pairs). Fix: pass `env={**os.environ, "PYTHONPATH": str(REPO_DIR)}` to the `subprocess.run()` call. Not fixed yet — skipped per instruction to move on from failures rather than debug them mid-run |
| nsr51324 YOLO11n (no plugin yet) | No plugin | Yes — continued fine-tune completed 2026-08-28, AdamW lr0=0.001. Weakest of the continued-fine-tune pair: mAP50=0.1563, mAP50-95=0.0638, precision=0.2618, recall=0.2234 — expected, given its source checkpoint's 7-class taxonomy (`alligator, block, crack, edge, longitudinal, pothole, transverse`) has no clean 1:1 mapping to our 5 classes, unlike YOLO12s's D00-D40 remap. Output at `weights/nsr51324_yolo11n/nsr51324_yolo11n_continued.pt` |
| cvtechniques YOLO11s (no plugin yet) | No plugin | Yes — COCO-base fine-tune completed 2026-08-28 (no real cvtechniques weights exist to continue from, see `weights/cvtechniques_yolo11s/NOTE.md`), SGD lr0=0.01, paused mid-epoch-3 and resumed cleanly (confirmed via the run's own "Resuming training ... from epoch 3" log line, not just assumed). mAP50=0.1787, mAP50-95=0.0752, precision=0.2936, recall=0.2481. Output at `weights/cvtechniques_yolo11s/cvtechniques_yolo11s_rdd2022.pt` |

**All numbers above are real, measured against the full 5,758-image test split** (not the small
val_small slice used for per-epoch early-stop checks during training). YOLO12s wins clearly —
~2x the next-best on every metric — because it's the only one continuing a checkpoint whose
class order already lines up with ours. Overall accuracy across all four is low in absolute
terms (0.10-0.33 mAP50) by design, not by accident: `MAX_TRAIN_IMAGES=2000` (of ~26,900
available), `imgsz=256` (vs. YOLO's typical 640), only 15 epochs, and batch=8 were all chosen
specifically to keep CPU training feasible in hours rather than days — see `modelresults.md` for
the full reasoning. Raising those three constants is the lever if better numbers are wanted,
at the direct cost of much longer training time.

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
`MAX_TRAIN_IMAGES` (in `prepare_dataset.py`) started at **2000** (down from the original 4000
default, lowered 2026-08-22 to keep CPU training time down), then raised back to **4000**
2026-08-28 for YOLO12s's v2 improvement run — applies to whichever model is trained next, since
it's shared by all four training scripts. `imgsz` was similarly lowered to 256 in all four
scripts 2026-08-22, but as of 2026-08-30 **only `train_yolo12s.py` has been raised back to 416**
— YOLO11n/nsr51324/cvtechniques are still at 256 (untouched, not part of that change).

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

## YOLO12s continued fine-tune — results
**Current (2026-09-01) — v4, the numbers that matter now:** direct follow-through on v3's own
conclusion below — raised `MAX_TRAIN_IMAGES` 4000→8000 in `prepare_dataset.py` (re-ran it first;
the full re-scan/re-hash of ~38,400 source images took ~30+ min on its own), everything else
identical to v3 (`imgsz=416`, `epochs=25` ceiling, `batch=16`, AdamW lr0=0.001/momentum=0.9,
`patience=5`). **Ran the full 25 epochs, no early stop** — unlike v3, which plateaued and
early-stopped at epoch 15; with double the data the model kept improving epoch-to-epoch instead.
Test-set (5,758 images): **mAP50 = 0.5362, mAP50-95 = 0.2701, precision = 0.6160, recall =
0.5244** — a clear gain over v3 on every metric (+29% mAP50, +38% mAP50-95, +20% precision,
+20% recall), confirming the "training-set size was the bottleneck" read from v3. Weights at
`weights/yolo12s/yolo12s_rdd2022_continued.pt` (overwrote v3's, which no longer exist on disk).
See `modelresults.md`'s "YOLO12s v4" entry for the per-class breakdown and full reasoning.

**Two real things worth knowing before touching this again:**
1. **Per-epoch timing was highly inconsistent this run** (~1h10min to ~3h45min for nominally
   identical epochs), traced to system-wide RAM pressure (free RAM seen as low as ~1.2-1.7GB of
   15.7GB total during the slow stretches, Windows actively using Memory Compression) — not a
   training bug. `AcerSense` (Acer's bundled system monitor) consistently showed unusually high
   cumulative CPU in every check during this run; never confirmed as the direct cause of a
   specific stall, but worth investigating/disabling if this recurs.
2. **Pausing on the exact last scheduled epoch breaks `--resume`** — paused right after epoch 25
   (the final epoch of the ceiling) finished, and `--resume` then crashed with
   `AssertionError: ...training to 25 epochs is finished, nothing to resume` (Ultralytics'
   `resume_training()` asserts `0 < start_epoch < self.epochs`, never true once training already
   hit its full epoch count). Not a `KeyboardInterrupt`, so `pause_control.py`'s except clause
   doesn't catch it — training itself was genuinely fine, only the script's post-training steps
   (copy `best.pt`, final test-set eval) never ran. **Worked around, not fixed in
   `pause_control.py`**: wrote `training/_finalize_v4.py`, a one-off that manually redoes those
   two steps against the already-valid `best.pt`. If this happens again on a future run, avoid
   pausing right as the epoch ceiling is about to be hit, or reuse/adapt that script.

**v3 (2026-08-30):** raised `imgsz` 256→416 only, everything else identical to v2 below (4000
img / batch=16 / epochs=25 ceiling / AdamW lr0=0.001). Ran as a detached background process,
unattended overnight, uninterrupted. Early-stopped at epoch 15 (`patience=5`, best weights from
epoch 10) — didn't reach the 25-epoch ceiling, unlike v2 or v4. Test-set: mAP50 = 0.4172,
mAP50-95 = 0.1953, precision = 0.5132, recall = 0.4366 (+12% mAP50 over v2). Plateauing well
before the epoch ceiling is what motivated v4's `MAX_TRAIN_IMAGES` increase above — also worth
remembering: because early-stop cut this run short, its cosine LR schedule (sized for the full
25 epochs) never finished decaying as intended. Superseded by v4 above — weights no longer exist
on disk. Full account in `modelresults.md`'s "YOLO12s v3" entry.

**v2 (2026-08-28/29):** `MAX_TRAIN_IMAGES` 2000→4000, batch 8→16, epochs 15→25 (ceiling,
`patience=5` still applies), `imgsz` unchanged at 256. Ran the full 25 epochs with no early stop,
paused once mid-run (epoch 5) and resumed cleanly. Test-set: mAP50 = 0.3724, mAP50-95 = 0.1750,
precision = 0.5072, recall = 0.3818 (+14% mAP50 over v1). Superseded by v3, then v4, above —
weights no longer exist on disk. Full account in `modelresults.md`'s "YOLO12s v2" entry.

**v1 (2026-08-28), the original 15-epoch/2000-image run — this paragraph was left mislabeled as
"v2" in this file until 2026-08-30's correction; the numbers were always v1's:** ran the full 15
epochs (4.49 hours, `patience=5` never triggered this time — plausibly because this run used the
lower pinned `lr0=0.001` added 2026-08-27, vs. whatever `optimizer="auto"` picked for the deleted
2026-08-22 run below; a slower-improving loss curve is less likely to look "plateaued" to the
early-stop check). Test-set (5,758 images): **mAP50 = 0.3256, mAP50-95 = 0.1467, precision =
0.4468, recall = 0.3570** — best of all 5 models attempted that session, see `modelresults.md`
for the full comparison table. Superseded by v2 and now v3.

**Historical (2026-08-22, run itself deleted 2026-08-27, numbers kept for context only — do not
treat as current):** stopped at epoch 6 (`patience=5` triggered), mAP50 = 0.263, mAP50-95 =
0.122, precision = 0.369, recall = 0.315. Per-class mAP50: alligator crack 0.448, longitudinal
crack 0.310, transverse crack 0.281, pothole 0.166, other corruption 0.110. Two findings from
that run, still conceptually relevant even though the weights are gone:
1. The class-remapping caveat in `plugins/yolo12s.py` played out as predicted — the two classes
   whose output slot got relabeled during continued training (pothole: was "Repair" in the base
   checkpoint; other corruption: was "pothole") scored roughly 2-4x worse than the three classes
   that kept their original slot.
2. Per that run's `results.csv`, epoch 1 had the best validation mAP50 of the entire 6-epoch run
   — validation got worse after epoch 1 and never recovered, which is why `patience=5` cut it off
   so early. Today's lower-LR, full-15-epoch run doesn't show this pattern (no early stop at
   all), consistent with the LR-was-too-high theory above.

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

## Learning rates — pretrained-on-road-damage vs. from-scratch (2026-08-27)
Split the four Ultralytics training scripts into two buckets and pinned explicit
optimizer/lr0/momentum for all of them — **not** left on `optimizer="auto"` for any of them
(first pass of this change only pinned the "already road-damage-tuned" pair and left the other
two on auto; caught and corrected same day, see "caught by checking the real data" below):
- **Already road-damage-fine-tuned by a third party** (`train_yolo12s.py`, continuing rezzzq's
  checkpoint; `train_nsr51324_yolo11n.py`, continuing nsr51324's) — pinned to
  `optimizer="AdamW", lr0=0.001, momentum=0.9`. Retraining an already-specialized checkpoint at
  the same LR as a from-scratch COCO fine-tune risks catastrophic forgetting of what the source
  run already learned, so these two use a low LR.
- **COCO-pretrained only, not road-damage-tuned by anyone yet** (`train_yolo11n.py`,
  `train_cvtechniques_yolo11s.py`) — pinned to `optimizer="SGD", lr0=0.01, momentum=0.9`
  (Ultralytics' own stock SGD defaults). NanoDet-m is in this bucket too by the same logic
  (COCO-only, no third-party road-damage tune exists), but its LR was already set separately in
  `training/nanodet_rdd2022.yml` (0.01, scaled down from the stock config's 0.14 for our
  batch=8 vs. its batch=192 default — not an Ultralytics mechanism, NanoDet has none).
- Net effect: the road-damage-pretrained pair trains at **10x lower LR** (0.001 vs 0.01) than the
  from-scratch pair, as a fixed ratio regardless of dataset size.
- **Momentum**: the fraction of the previous update direction carried into the next step, so the
  optimizer keeps moving through small/noisy gradient fluctuations instead of reacting to every
  single batch — dampens oscillation and speeds convergence versus plain SGD/Adam with no
  momentum term. Same 0.9 in both buckets — only LR (and optimizer family) differs between them,
  to keep the comparison to one deliberate variable.
- **Caught by checking the real data, not by re-deriving the theory:** the first version of this
  change computed `optimizer="auto"`'s effective LR using `prepare_dataset.py`'s *current*
  `MAX_TRAIN_IMAGES` constant (8000) → 15,000 iterations → auto picks MuSGD/lr0≈0.01, so the
  "from-scratch" bucket was left on `auto` assuming it'd land there. Actually checking
  `data/RDD2022/clean/train/images` on disk found only **2000** images (from an earlier
  `prepare_dataset.py` run, before that constant was raised) → 3,750 iterations → auto actually
  resolves to AdamW/lr0≈0.0011 for the *current* data — almost identical to the "road-damage-
  pretrained" bucket's pinned 0.001, which would have silently erased the entire intended
  contrast. Fixed by pinning explicit values for all four scripts instead of relying on
  `auto`, whose choice depends on train-set size and would drift every time `MAX_TRAIN_IMAGES`
  changes or `prepare_dataset.py` gets re-run.
- **Same check also found `data/RDD2022/{data.yaml,data_fast.yaml}`'s internal `path:` field was
  stale** — hardcoded to `C:\Users\sherv\Desktop\Mountain of Flowers and Fruits\Application
  Studio B coding\data\RDD2022\clean` (this project's location before it became part of the
  RoadIQ git repo at `D:\Projects\RoadIQ\Application Studio B coding`), not this repo's current
  location. That old Desktop path still exists on disk with what looks like the same 2000-image
  data (not verified byte-for-byte, not deleted — same "don't touch a second copy without
  flagging it" caution as the earlier OneDrive incident above), so training wouldn't have
  crashed, just silently trained against whatever's at that stale path instead of this repo's
  own `data/`. **Fixed 2026-08-27** by editing both yaml files' `path:` to the current
  `D:\Projects\RoadIQ\Application Studio B coding\data\RDD2022\clean` location. If dataset
  numbers ever look inconsistent again, check this field first — same advice as the original
  incident note above, this is now the second time it's happened after a project move.

## Open TODOs
- [x] ~~Finish YOLO12s continued fine-tune~~ — **done 2026-08-28**, full 15 epochs, mAP50=0.3256.
      Best of all 5 models attempted. See `modelresults.md`.
- [x] ~~YOLO12s v2 — raise MAX_TRAIN_IMAGES/epochs/batch~~ — **done 2026-08-28/29**, mAP50=0.3724
      (+14% over v1). Superseded by v3.
- [x] ~~YOLO12s v3 — raise imgsz 256→416~~ — **done 2026-08-30**, mAP50=0.4172 (+12% over v2,
      +14% recall). Early-stopped at epoch 15/25 (best=epoch 10) — training-set size, not epoch
      count, was the bottleneck, motivating v4 below.
- [x] ~~YOLO12s v4 — raise MAX_TRAIN_IMAGES 4000→8000~~ — **done 2026-09-01**, mAP50=0.5362
      (+29% over v3, +20% precision/recall). Ran the full 25 epochs, no early stop this time.
      Current best model — see `modelresults.md`'s "YOLO12s v4" entry, including the real
      `--resume`-crashes-on-the-last-epoch bug found and worked around this run.
- [ ] `training/pause_control.py` doesn't handle pausing on the last scheduled epoch — `--resume`
      crashes with an uncaught `AssertionError` rather than gracefully proceeding to
      post-training steps (see v4 notes above). Not fixed, only worked around with a one-off
      script (`training/_finalize_v4.py`) for that specific run. Worth a real fix if this project
      keeps doing multi-day paused runs.
- [x] ~~Run `training/train_yolo11n.py`~~ — **done 2026-08-28**, full 15 epochs, mAP50=0.1012.
- [x] ~~Run `training/train_nsr51324_yolo11n.py`~~ — **done 2026-08-28**, mAP50=0.1563 (weakest
      of the two continued-fine-tunes, expected given its unmapped 7-class source taxonomy).
- [x] ~~Run `training/train_cvtechniques_yolo11s.py`~~ — **done 2026-08-28**, mAP50=0.1787.
- [ ] `training/train_nanodet_m.py` — **attempted 2026-08-28, failed**:
      `ModuleNotFoundError: No module named 'nanodet'`. The dataset-flattening fix from
      2026-08-27 worked correctly, but the subprocess call itself never puts
      `third_party/nanodet` on the child process's import path (`sys.path`/`PYTHONPATH`) the way
      the inference plugin does. Fix: pass `env={**os.environ, "PYTHONPATH": str(REPO_DIR)}` to
      the `subprocess.run()` call in that script. Not fixed yet — skipped per instruction to move
      on rather than debug mid-run.
- [ ] No `detector_interface/plugins/` adapter exists yet for nsr51324 YOLO11n or cvtechniques
      YOLO11s — needed before either can be benchmarked via `eval/benchmark.py` or served, model
      on `plugins/yolo12s.py`/`plugins/yolo11n.py`.
- [ ] No training script exists yet for YOLOv4-tiny (blocked on darknet toolchain) or YOLOX-Tiny/
      Nano — would need to be written if the team wants fine-tuned numbers for those too, not
      just COCO-pretrained speed data points.
- [ ] All 5 real results so far used `MAX_TRAIN_IMAGES=2000`/`imgsz=256`/`epochs=15`/`batch=8` —
      deliberately small for CPU-feasible runtime, at the cost of low absolute accuracy
      (0.10-0.33 mAP50 across the board). Raising those is the lever for meaningfully better
      numbers, at the cost of much longer training time — not done yet, no decision made on
      whether it's worth it.
- [ ] Docker/repo move — see above, architecture question unresolved.

## Pause/resume for training runs (added 2026-08-27)
All four Ultralytics training scripts support pausing (e.g. to free up RAM for something else
running alongside training) and resuming:
```
python training/pause_control.py pause      # from another terminal, while a script is running
python training/train_yolo12s.py --resume    # continues from the checkpoint afterward
```
**Real Ultralytics gotcha this ran into, worth knowing before touching this again:** the first
implementation used `trainer.stop = True` (an Ultralytics API meant for exactly this) and it
silently broke resume — confirmed by reading `ultralytics/utils/torch_utils.py`'s
`strip_optimizer()`: it's called unconditionally at the end of every `train()` call (normal
finish, `patience` early-stop, or a `stop` flag) and unconditionally sets `epoch = -1` and
`optimizer = None` on the saved checkpoint. `resume=True` checks exactly those two fields, finds
them stripped, and silently starts a brand-new default run instead (confirmed with a real
smoke test: it landed in `runs/detect/train` with Ultralytics' default 100 epochs, ignoring the
original run's project/name/epoch settings entirely — this also means any of these scripts'
*existing* early-stop-via-`patience` behavior would have had the exact same problem, not just
the new pause feature). Fixed by having the pause callback raise `KeyboardInterrupt` instead —
deliberately mimicking a real Ctrl+C, which never reaches `strip_optimizer()` — registered on
`on_fit_epoch_end` (fires right after that epoch's own checkpoint save), so pause granularity is
"next epoch boundary," not "next batch." Re-verified end-to-end after the fix: a resumed run
correctly continues the original run's project/name/epoch count. Full account in
`training/pause_control.py`'s own docstring.
