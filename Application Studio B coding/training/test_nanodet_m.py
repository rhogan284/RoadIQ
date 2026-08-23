"""Runs the fine-tuned NanoDet-m checkpoint (weights/nanodet_m/nanodet_m_rdd2022.ckpt, from
training/train_nanodet_m.py) against the full RDD2022 test split (5,758 images) — the same split
train_yolo11n.py / train_yolo12s.py print "Test-set accuracy" against. train_nanodet_m.py itself
only ever validated against val_small (400 images) during training, so this fills that gap and
makes NanoDet-m's numbers directly comparable to the other two.

Reuses NanoDet's own validation machinery (TrainingTask + CocoDetectionEvaluator) via
pytorch_lightning's `Trainer.validate()`, rather than reimplementing inference/scoring by hand —
this is the exact same code path train_nanodet_m.py's per-epoch validation already exercised
(and that was smoke-tested before handing that script over), just pointed at the test split with
the fine-tuned checkpoint loaded instead of val_small with the base COCO checkpoint. Writes to a
separate save_dir so its own "best model" bookkeeping doesn't touch
training/runs/nanodet_m_rdd2022/ (the actual training run's output).
"""

from pathlib import Path

import torch
import yaml

import training.train_nanodet_m as train_mod

ROOT = train_mod.ROOT
DATA_ROOT = train_mod.DATA_ROOT
TEST_IMAGES = DATA_ROOT / "test" / "images"
TEST_LABELS = DATA_ROOT / "test" / "labels"
FINETUNED_WEIGHTS = ROOT / "weights" / "nanodet_m" / "nanodet_m_rdd2022.ckpt"
SAVE_DIR = ROOT / "training" / "runs" / "nanodet_m_rdd2022_test_eval"
GENERATED_CONFIG = ROOT / "training" / "nanodet_m_rdd2022_test.generated.yml"


def main() -> None:
    if not FINETUNED_WEIGHTS.exists():
        raise RuntimeError(f"{FINETUNED_WEIGHTS} not found — run training/train_nanodet_m.py first.")
    if not TEST_IMAGES.exists():
        raise RuntimeError(f"{TEST_IMAGES} not found.")

    train_mod._merge_labels_into_images(TEST_IMAGES, TEST_LABELS)

    config = train_mod._build_config()
    config["data"]["val"]["img_path"] = str(TEST_IMAGES)
    config["data"]["val"]["ann_path"] = str(TEST_IMAGES)
    config["schedule"]["load_model"] = str(FINETUNED_WEIGHTS)
    config["save_dir"] = str(SAVE_DIR)
    GENERATED_CONFIG.write_text(yaml.safe_dump(config, sort_keys=False))
    print(f"Wrote generated config to {GENERATED_CONFIG}")

    train_mod._ensure_importable()

    import pytorch_lightning as pl
    from nanodet.data.collate import naive_collate
    from nanodet.data.dataset import build_dataset
    from nanodet.evaluator import build_evaluator
    from nanodet.trainer.task import TrainingTask
    from nanodet.util import NanoDetLightningLogger, cfg, load_config, load_model_weight, mkdir

    load_config(cfg, str(GENERATED_CONFIG))
    mkdir(-1, cfg.save_dir)
    # NanoDetLightningLogger (not the plain `Logger` the inference-only plugin code uses) —
    # pl.Trainer's `logger=` kwarg needs a real pytorch_lightning Logger subclass (`save_dir`,
    # `finalize()`, etc.), which only this one provides. Mixing them up crashes with
    # `AttributeError: 'Logger' object has no attribute 'save_dir'` inside Trainer.validate().
    logger = NanoDetLightningLogger(cfg.save_dir)

    test_dataset = build_dataset(cfg.data.val, "test")
    print(f"Test set size: {len(test_dataset)} images")
    evaluator = build_evaluator(cfg.evaluator, test_dataset)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.device.batchsize_per_gpu,
        shuffle=False,
        num_workers=cfg.device.workers_per_gpu,
        collate_fn=naive_collate,
        drop_last=False,
    )

    task = TrainingTask(cfg, evaluator)
    ckpt = torch.load(str(FINETUNED_WEIGHTS), map_location="cpu")
    load_model_weight(task.model, ckpt, logger)
    print(f"Loaded fine-tuned weights from {FINETUNED_WEIGHTS}")

    trainer = pl.Trainer(
        default_root_dir=str(SAVE_DIR),
        accelerator="cpu",
        devices=1,
        logger=logger,
        num_sanity_val_steps=0,
    )
    # TrainingTask.validation_step's per-batch log line reads
    # self.trainer.optimizers[0].param_groups[0]["lr"] unconditionally. That's only ever empty
    # when validating standalone via Trainer.validate() (as here) instead of through Trainer.fit()
    # (which sets it up via configure_optimizers()) — crashes with IndexError otherwise. This
    # optimizer is never stepped; it exists only to satisfy that log line.
    trainer.optimizers = [torch.optim.SGD(task.model.parameters(), lr=0.0)]
    trainer.validate(task, dataloaders=test_dataloader)


if __name__ == "__main__":
    main()
