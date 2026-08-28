# No weight file here — by design, not an oversight

`https://huggingface.co/cvtechniques/road-damage-detection-yolov11` does not host a trained
checkpoint. Checked its file listing directly via the HF API on 2026-08-27 — the repo contains
only `README.md`, `Picture1.jpg`, `results.png`, `val_batch0_pred.jpg`. It's a model card
describing a real YOLOv11s fine-tune (Colab/T4, 15 epochs, RDD2022-Japan via Roboflow, mAP50
~0.47) but the trained weights themselves were never uploaded.

`training/train_cvtechniques_yolo11s.py` is set up to fine-tune stock COCO yolo11s.pt on our own
RDD2022 subset instead, as the closest available equivalent — see that script's docstring. If a
real cvtechniques checkpoint ever gets published, swap it in there and continue-fine-tune it
instead (same pattern as train_yolo12s.py).
