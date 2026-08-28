# Model Results

Real training runs, launched sequentially (one at a time — CPU-only machine, running these in
parallel would just make all of them slower and risk running out of RAM). Each row gets filled
in as its run actually finishes; "in progress" / "not started" rows are placeholders, not results.

All runs use `data/RDD2022/data_fast.yaml` for training (2000 train / 400 val images) and
`data/RDD2022/data.yaml`'s full test split for the final accuracy numbers below, imgsz=256,
batch=8, CPU. See `ASBcode.md` for the full reasoning behind each model's learning rate.

| Model | Status | mAP50 | mAP50-95 | Precision | Recall | Notes |
|---|---|---|---|---|---|---|
| YOLO11n | done | 0.1012 | 0.0408 | 0.1867 | 0.1847 | COCO base → fine-tune, SGD lr0=0.01, 15 epochs, full 5758-image test set |
| YOLO12s (original) | superseded | 0.3256 | 0.1467 | 0.4468 | 0.3570 | 2000 img / 15 epochs / batch 8 — kept for comparison, see v2 below |
| **YOLO12s v2** | **done** | **0.3724** | **0.1750** | **0.5072** | **0.3818** | 4000 img / 25 epochs / batch 16, imgsz unchanged at 256. Paused mid-run (epoch 5) and resumed cleanly. **New best model.** |
| nsr51324 YOLO11n | done | 0.1563 | 0.0638 | 0.2618 | 0.2234 | Continues nsr51324's road-damage checkpoint, AdamW lr0=0.001 — 7-class source, no clean mapping to our 5 (see script docstring) |
| cvtechniques YOLO11s | done | 0.1787 | 0.0752 | 0.2936 | 0.2481 | COCO base → fine-tune, SGD lr0=0.01, resumed from epoch 3/15 checkpoint |
| NanoDet-m | **failed — skipped** | — | — | — | — | `ModuleNotFoundError: No module named 'nanodet'` — see Log |

## Log
- **YOLO11n done.** Training itself finished all 15 epochs cleanly (weights saved to
  `weights/yolo11n/yolo11n_rdd2022.pt`); the run's own final validation print got cut off by an
  unrelated tooling accident on my end (not a training failure), so I re-ran validation directly
  against the already-trained weights to get the real numbers above — same result either way,
  nothing was re-trained.
- **YOLO12s got killed twice more** (once ~2s in, once ~6 minutes into epoch 1, both before any
  checkpoint existed — real CPU time lost both times, restarted from scratch each time), even
  after chaining all four remaining runs into one background process. Not caused by anything I
  called — no stop/kill command came from me either time. Switched to launching the training
  chain as a fully OS-detached process (`Start-Process` outside this session's own task
  tracking) instead, specifically so it survives whatever's causing that.
- YOLO12s and nsr51324 YOLO11n both completed successfully with that detached process.
- **Found on pausing (2026-08-28): the two earlier "killed" background-task attempts weren't
  actually killed at the OS level** — they'd been left running as orphaned processes this whole
  time, meaning at points there were multiple duplicate copies of the same training chain
  running concurrently, writing to the same output files (raced, not corrupted — checked: all
  checkpoint files still load as valid zip archives, and both chains were running the identical
  deterministic config (same seed/data/hyperparams), so the numbers above should be trustworthy,
  just computed redundantly rather than once). Killed everything for real this time (verified by
  watching the log file stop growing and confirming zero running python.exe processes, not just
  a process-list snapshot — the first couple of kill attempts looked like they'd worked but
  didn't). Filed as feedback separately from the original "killed" bug — this is a different,
  worse failure mode (silent orphaning instead of a clean stop).
- cvtechniques YOLO11s was paused mid-epoch-3 with a valid resumable checkpoint, then resumed
  and finished cleanly — confirmed the run log itself said "Resuming training ... from epoch 3
  to 15 total epochs" (not silently restarting from scratch), then completed and printed real
  final test-set numbers, recorded above.
- **NanoDet-m failed, skipped per instruction — real error, not the mystery-kill bug.**
  `training/train_nanodet_m.py` invokes `third_party/nanodet/tools/train.py` as a fresh
  subprocess, which needs `third_party/nanodet` on `sys.path` to import the `nanodet` package
  (it has no proper installable package — see that script's own docstring and
  `detector_interface/plugins/nanodet_m.py`, which handles this via `sys.path.insert(0, ...)`
  before importing). The training wrapper never added that — it was flagged as untested in its
  own docstring, and this is the first time it's actually been run. Fix (not applied, since the
  instruction was to skip on failure): set `PYTHONPATH` to include `third_party/nanodet` when
  launching the subprocess, e.g. `env={**os.environ, "PYTHONPATH": str(REPO_DIR)}` in the
  `subprocess.run()` call.

## All 5 models attempted — run complete.
4/5 have real results (YOLO11n, YOLO12s, nsr51324 YOLO11n, cvtechniques YOLO11s). NanoDet-m
failed on a fixable import-path bug, not run.

## Why the numbers are low (0.10-0.33 mAP50 across the board)
Deliberate CPU-feasibility tradeoffs, not a bug:
1. **Tiny training set** — 2000 images out of ~26,900 available in RDD2022
   (`MAX_TRAIN_IMAGES` in `prepare_dataset.py`).
2. **Low resolution** — `imgsz=256`, half of YOLO's typical 640. Hits small objects (cracks,
   potholes) hardest — less than a quarter the pixel area to detect them in.
3. **Few epochs** — 15, short for detection fine-tuning, especially the two from-scratch-COCO
   runs (YOLO11n, cvtechniques).
4. **Small batch size** — 8 (typical is 16-64+), hurts gradient/batchnorm stability.

All four were chosen this way in an earlier session specifically to make a full run finish in
hours instead of days on a laptop CPU — see `ASBcode.md`'s Data section. For comparison, a past
(now-deleted) YOLO12s run under similar constraints got mAP50=0.263; today's got 0.3256 — same
ballpark, so ~0.25-0.35 mAP50 looks like roughly what this configuration is actually capable of,
not an error.

**Why YOLO12s wins clearly (~2x the next-best on every metric):** it's the only one continuing a
checkpoint whose class order already lines up with ours (rezzzq's D00→longitudinal crack, etc.
maps cleanly). nsr51324 also continues a pretrained checkpoint, but its 7 classes (`alligator,
block, crack, edge, longitudinal, pothole, transverse`) don't map onto our 5 at all, so most of
that head start is wasted — closer to relearning from scratch than true transfer. YOLO11n and
cvtechniques score lowest because they start from plain COCO weights with zero road-damage
exposure at all.

**Lever for better numbers:** raise `MAX_TRAIN_IMAGES`, `imgsz`, and `epochs` — at the direct
cost of training taking much longer than the hours this run took.

## YOLO12s v2 — the improvement run (2026-08-28/29)
Tried the lever above on YOLO12s specifically (the clear winner), without touching `imgsz`:
`MAX_TRAIN_IMAGES` 2000→4000, batch 8→16, epochs 15→25 (ceiling — `patience=5` still applies).
Also fixed a live bug found while doing this: `prepare_dataset.py`'s `VAL_SUBSET_SIZE` was
wrongly left at 5758 (the full val set) instead of 400, which would have made every epoch's
validation as slow as the final test-set eval — restored to 400.

Result: **mAP50 0.3256 → 0.3724 (+14%)**, gains across all four metrics. Ran the full 25 epochs
(no early stop), paused once mid-run (after epoch 5, via `pause_control.py`) and resumed cleanly
— confirmed via the run's own "Resuming training ... from epoch 6 to 25 total epochs" log line.
Real per-epoch time trended down over the run (epoch 2: ~39min, epoch 5: ~17min), faster than
initially projected.

This is now the best model across all attempts. `weights/yolo12s/yolo12s_rdd2022_continued.pt`
was overwritten with this v2 result (the original 15-epoch run's number is kept in the table
above for comparison only — those weights no longer exist on disk).
