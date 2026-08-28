# Must Remember — Pause/Resume Training

## The commands

```
python training/pause_control.py pause          # run this in a SEPARATE terminal, while training is running
python training/train_yolo12s.py --resume         # run this later to pick back up
```

Works for all four Ultralytics scripts:
- `train_yolo11n.py`
- `train_yolo12s.py`
- `train_nsr51324_yolo11n.py`
- `train_cvtechniques_yolo11s.py`

`NanoDet-m` (`train_nanodet_m.py`) does **not** have pause/resume — different training framework, and it's never even been run once yet.

## What actually happens when you pause

- It does **not** just freeze the process in place — it fully **stops and exits**, which frees up the RAM it was using. That's the whole point: so Roblox/Discord/whatever actually gets that memory back, not just idle CPU.
- It saves a checkpoint before exiting, so `--resume` picks up where it left off (not from zero).
- Timing: it stops at the next **epoch boundary**, not instantly. On this project's real settings that's up to ~10-13 minutes — so if you hit pause, it might take a few minutes to actually quit, not stop immediately.

## Why (in case future-me wonders why it's not instant)

First version tried to pause instantly (mid-epoch). It looked like it worked, but it secretly broke `--resume` — Ultralytics wipes some required data from the checkpoint the moment training "finishes" (even an early stop counts as finishing), so resuming would silently ignore everything and start a brand new run from scratch instead of continuing. Fixed by making pause behave exactly like hitting Ctrl+C for real, which avoids that wipe — but that only works cleanly between epochs, not mid-epoch. Confirmed this fix actually works by running it start-to-finish twice, not just trusting it.

## Reminder on why this exists at all

Your PC only has 16GB RAM, and Roblox alone eats multiple GB. Training + Roblox + Discord open all at once can push free RAM down to ~1GB, causing stutter in Roblox and general slowdowns. Pausing training when you want to play, resuming when you're done, is the fix for that.
