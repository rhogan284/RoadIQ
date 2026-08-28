"""Shared pause-request helper for the Ultralytics training scripts (train_yolo11n.py,
train_yolo12s.py, train_nsr51324_yolo11n.py, train_cvtechniques_yolo11s.py).

"Pausing" here means a real process stop, not an in-process sleep loop — a sleeping process
still holds all its model/optimizer/dataloader memory, so it wouldn't actually free any RAM for
something like Roblox running alongside it (see ASBcode.md's 2026-08-27 RAM-contention note,
which is the whole reason this exists). Instead, requesting a pause makes the training script
stop at the next epoch boundary and exit the process entirely, in a way that's actually resumable.

**Design note — a first version of this used `trainer.stop = True` and was wrong, caught by
testing it end-to-end, not by re-reading the docs harder:** setting `trainer.stop` lets
Ultralytics' training loop exit through its normal "run is complete" path, which calls
`final_eval()` -> `strip_optimizer()` on last.pt. Confirmed by reading
ultralytics/utils/torch_utils.py's strip_optimizer(): it unconditionally sets `epoch = -1` and
`optimizer = None` in the saved checkpoint, every single time, whether training finished
normally, hit `patience`, or was stopped via that flag. `resume=True` checks exactly those two
fields to decide whether a checkpoint is resumable — finding them stripped, it silently gives up
and starts a brand-new default run instead (confirmed this happened: a real smoke test resumed
into `runs/detect/train` with Ultralytics' default 100 epochs, silently ignoring the original
run's project/name/epoch settings entirely). A genuine Ctrl+C doesn't have this problem, because
it interrupts the process *before* reaching that finalization code, leaving the last per-epoch
checkpoint (saved by `save_model()`, before `final_eval()` runs) with full optimizer/epoch state
intact. So instead of a graceful stop flag, the pause callback below raises KeyboardInterrupt —
deliberately mimicking a real Ctrl+C rather than inventing a new code path — registered on
`on_fit_epoch_end`, which (confirmed by reading trainer.py) fires right after that epoch's
`save_model()` call, so the checkpoint being interrupted *into* is always a fully valid, freshly
saved, resumable one. Re-verified end-to-end after this fix: a resumed run now correctly
continues the original run's project/name/epoch count instead of falling back to defaults.

Net effect: pause granularity is "next epoch boundary" (up to ~10-13 min in this project's real
runs, not "next batch" as an earlier version of this docstring claimed before the fix above), in
exchange for actually working. Resume: rerun the same script with --resume — Ultralytics reloads
last.pt plus that run's saved training args and continues from there.

Usage (from a second terminal, while a training script is running in another one):
    python training/pause_control.py pause   # request a stop at the next epoch boundary
    python training/pause_control.py clear    # manually clear a stuck flag (scripts already
                                                # clear it themselves on every fresh start/resume)
"""

import argparse
import sys
from pathlib import Path
from typing import Any

PAUSE_FLAG = Path(__file__).resolve().parent / ".pause_requested"


def pause_requested() -> bool:
    return PAUSE_FLAG.exists()


def request_pause() -> None:
    PAUSE_FLAG.touch()


def clear_pause() -> None:
    PAUSE_FLAG.unlink(missing_ok=True)


def make_pause_callback():
    """Ultralytics callback — register with model.add_callback("on_fit_epoch_end", ...).
    Raises KeyboardInterrupt (deliberately, see this module's docstring) once a pause has been
    requested via `python training/pause_control.py pause`, right after the just-finished
    epoch's checkpoint has already been saved with full resumable state."""
    announced = False

    def _callback(trainer) -> None:
        nonlocal announced
        if pause_requested():
            if not announced:
                print(
                    "\nPause requested — stopping now that this epoch's checkpoint is saved. "
                    "Rerun this script with --resume to continue."
                )
                announced = True
            raise KeyboardInterrupt("Pause requested via training/pause_control.py")

    return _callback


def run_training(source_weights: Path, run_name: str, runs_dir: Path, train_kwargs: dict[str, Any]):
    """Shared fresh-vs-resume + pause-callback wiring for the four Ultralytics training scripts.

    Handles: parsing --resume, clearing any stale pause flag at the start of every run (fresh or
    resumed), loading either source_weights (fresh) or that run's weights/last.pt (--resume,
    via Ultralytics' own `YOLO(last.pt).train(resume=True)` idiom — reloads that run's saved
    training args automatically, no need to pass train_kwargs again), and registering the pause
    callback in both cases so a paused-then-resumed run can be paused again. Catches the
    KeyboardInterrupt the pause callback raises so callers don't see a scary traceback for a
    deliberate pause (a genuine accidental Ctrl+C from the user is caught the same way and
    treated the same way — exit cleanly, point at --resume — which is the right outcome either way).

    Returns (model, results, paused). Callers should skip their normal "training genuinely
    finished" steps (copying best.pt out, final full-test-set val, etc.) when paused is True —
    those only make sense once training has actually run to completion or triggered its own
    early-stop via `patience`, not on an interrupted mid-run pause. `results` is None when paused.
    """
    from ultralytics import YOLO

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the last checkpoint of a paused/interrupted run instead of starting fresh.",
    )
    args, _ = parser.parse_known_args()

    clear_pause()
    last_pt = runs_dir / run_name / "weights" / "last.pt"

    if args.resume:
        if not last_pt.exists():
            raise RuntimeError(f"{last_pt} not found — nothing to resume. Run without --resume to start fresh.")
        model = YOLO(str(last_pt))
        model.add_callback("on_fit_epoch_end", make_pause_callback())
        try:
            results = model.train(resume=True)
            paused = False
        except KeyboardInterrupt:
            results = None
            paused = True
    else:
        model = YOLO(str(source_weights))
        model.add_callback("on_fit_epoch_end", make_pause_callback())
        try:
            results = model.train(**train_kwargs)
            paused = False
        except KeyboardInterrupt:
            results = None
            paused = True

    if paused:
        print(f"\nPaused. Checkpoint saved at {last_pt} — resume with: python <this script> --resume")

    return model, results, paused


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("pause", "clear"):
        print("Usage: python training/pause_control.py [pause|clear]")
        sys.exit(1)
    if sys.argv[1] == "pause":
        request_pause()
        print(f"Pause requested ({PAUSE_FLAG}). The running training script will stop at the next epoch boundary.")
    else:
        clear_pause()
        print("Pause flag cleared.")
