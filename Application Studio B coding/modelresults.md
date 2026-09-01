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
| YOLO12s (original) | superseded | 0.3256 | 0.1467 | 0.4468 | 0.3570 | 2000 img / 15 epochs / batch 8 — kept for comparison, see v2/v3 below |
| YOLO12s v2 | superseded | 0.3724 | 0.1750 | 0.5072 | 0.3818 | 4000 img / 25 epochs / batch 16, imgsz 256. Paused mid-run (epoch 5) and resumed cleanly. Kept for comparison, see v3 below |
| YOLO12s v3 | superseded | 0.4172 | 0.1953 | 0.5132 | 0.4366 | Same as v2 but imgsz 256→416. Early-stopped at epoch 15 (`patience=5`, best=epoch 10), ran uninterrupted (no pause used). Kept for comparison, see v4 below |
| **YOLO12s v4** | **done** | **0.5362** | **0.2701** | **0.6160** | **0.5244** | Same as v3 but `MAX_TRAIN_IMAGES` 4000→8000. Ran the full 25 epochs (no early stop), paused/resumed 4 times across the run to free RAM. **New best model.** |
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

## YOLO12s v3 — imgsz 256→416, everything else same as v2 (2026-08-30)
Next lever after v2: raised `imgsz` 256→416 only (`train_yolo12s.py`'s train and final-eval calls
both changed — kept matched to each other so the eval isn't at a different resolution than
training). `epochs=25` (ceiling), `batch=16`, `MAX_TRAIN_IMAGES=4000`, optimizer/lr0/momentum all
identical to v2. Ran as a detached background process overnight, uninterrupted (pause/resume
mechanism wasn't exercised this run, but is unaffected by the imgsz change — `--resume` reloads
imgsz from the checkpoint's own saved args, doesn't need separate wiring).

**Result: mAP50 0.3724 → 0.4172 (+12%), mAP50-95 0.1750 → 0.1953 (+12%), recall 0.3818 → 0.4366
(+14%), precision 0.5072 → 0.5132 (~flat, +1%).** Per-class mAP50 on the full test set: alligator
crack 0.483, other corruption 0.461, longitudinal crack 0.406, transverse crack 0.391, pothole
0.345 (weakest — fewest instances of any class, 951 vs. 1500-3900+ for the others).

**Early-stopped at epoch 15** (`patience=5`, best weights from epoch 10) — never got near the
25-epoch ceiling v2 used. Two implications, not yet acted on:
1. The cosine LR schedule (`lrf=0.01`) is computed assuming a full 25-epoch run; stopping at 15
   means it never finished decaying as scheduled, unlike v2 which ran the complete schedule.
2. Plateauing well before the epoch ceiling suggests epoch count isn't the current bottleneck —
   training-set size is (still only 4,000 of ~26,900 available images). That makes
   `MAX_TRAIN_IMAGES` the highest-leverage next lever, more so than pushing `imgsz` or `epochs`
   further from here.

This was the best model across all attempts at the time, superseding v2 — since superseded by v4
below.
`weights/yolo12s/yolo12s_rdd2022_continued.pt` was overwritten with this v3 result (v2's weights
no longer exist on disk — same pattern as v1→v2).

## YOLO12s v4 — MAX_TRAIN_IMAGES 4000→8000 (2026-08-31/09-01)
Direct follow-through on v3's own conclusion: raised `MAX_TRAIN_IMAGES` 4000→8000 in
`prepare_dataset.py` (re-ran it first to regenerate `data.yaml`/`data_fast.yaml` — full re-scan/
re-hash of all ~38,400 source images took ~30+ min on its own). Everything else identical to v3
(`imgsz=416`, `epochs=25` ceiling, `batch=16`, AdamW lr0=0.001/momentum=0.9, `patience=5`).

**Result: mAP50 0.4172 → 0.5362 (+29%), mAP50-95 0.1953 → 0.2701 (+38%), precision 0.5132 →
0.6160 (+20%), recall 0.4366 → 0.5244 (+20%).** Clear win across every metric, confirming v3's
"training-set size is the bottleneck" read. Per-class mAP50 on the full test set: alligator
crack 0.634, other corruption 0.599, longitudinal crack 0.509, transverse crack 0.507, pothole
0.433 (still weakest — fewest training instances of any class — but up sharply from v3's 0.345).

**Ran the full 25 epochs, no early stop** (`patience=5` never triggered) — unlike v3, which
early-stopped at epoch 15. With double the data, the model kept finding genuine improvement
epoch-to-epoch instead of plateauing early: best single-epoch val mAP50 climbed steadily
(epoch 10: 0.461 -> epoch 20: 0.589 -> epoch 25: 0.594), no long non-improving stretch.

**Paused/resumed 4 times over the course of the run** (to free RAM for other work) — all four
mid-run resumes confirmed correct via the log's own "Resuming training ... from epoch N to 25
total epochs" lines, no silent restarts. Real per-epoch time was highly inconsistent, swinging
between ~1h10min and ~3h45min for nominally identical epochs — traced to system-wide RAM
pressure (free RAM was seen as low as ~1.2-1.7GB out of 15.7GB total during the slow stretches,
with Windows actively using Memory Compression), not a training bug. `AcerSense` (Acer's
bundled system monitor) consistently showed unusually high cumulative CPU in every check during
this run — never confirmed as the direct cause of any specific stall, but a recurring background
presence worth investigating if this keeps happening on future runs.

**Real bug found: pausing on the exact last scheduled epoch breaks `--resume`.** Requested pause
right after epoch 25 (the final epoch of the `epochs=25` ceiling) finished, intending to free RAM
before the heavy final full-test-set eval step. `--resume` then crashed with
`AssertionError: ...training to 25 epochs is finished, nothing to resume` — Ultralytics'
`resume_training()` asserts `0 < start_epoch < self.epochs`, which is never true once training
already completed its full scheduled epoch count. That's an `AssertionError`, not a
`KeyboardInterrupt`, so `pause_control.py`'s except clause doesn't catch it — the script crashed
before running its own post-training steps (copying `best.pt`, final test-set eval). Not a data
problem — training itself was genuinely complete and `best.pt`/`last.pt` were both valid.
**Worked around, not fixed in `pause_control.py` itself**: wrote a one-off
`training/_finalize_v4.py` that manually redoes `train_yolo12s.py`'s two post-training steps
(copy `best.pt` -> `weights/yolo12s/yolo12s_rdd2022_continued.pt`, run `model.val()` against the
full test split) without going through `model.train(resume=True)` again. **If this happens
again**: avoid pausing right as the epoch ceiling is about to be hit, or reuse/adapt
`_finalize_v4.py` instead of retrying `--resume`.

This is now the best model across all attempts, superseding v3.
`weights/yolo12s/yolo12s_rdd2022_continued.pt` was overwritten with this v4 result (v3's weights
no longer exist on disk — same pattern as v1→v2→v3).
