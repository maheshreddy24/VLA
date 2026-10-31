""" Train the VLA on LIBERO demonstrations.

    python train.py --config configs/config.yaml [--resume ckpt.pth] [key.subkey=value ...]
    python train.py data.suites=[libero_spatial] trainer.num_epochs=10
"""
import argparse
import os

import torch
from torch.utils.data import DataLoader

from src.backbone import load_processor
from src.config import load_config
from src.dataset import LiberoDataset
from src.models import VLA
from src.policy import build_collator
from src.processing import Normalizer
from src.training import Trainer


def build_datasets(config):
    d, m = config["data"], config["model"]
    kwargs = dict(data_root=d["root"], suites=d["suites"], horizon=m["horizon"], cameras=d["cameras"],
                  val_demos_per_task=d["val_demos_per_task"], filter_noops=d["filter_noops"])
    train_set = LiberoDataset(split="train", **kwargs)
    val_set = LiberoDataset(split="val", **kwargs) if d["val_demos_per_task"] > 0 else None
    return train_set, val_set


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--resume", default=None, help="checkpoint to resume (weights, optimizer, step)")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    config = load_config(args.config, args.overrides)
    torch.backends.cuda.matmul.allow_tf32 = True

    train_set, val_set = build_datasets(config)
    print(f"train samples: {len(train_set)} ({len(train_set.episodes)} demos), "
          f"val samples: {len(val_set) if val_set else 0}")

    if args.resume:  # keep the normalization the checkpoint was trained with
        stats = torch.load(args.resume, map_location="cpu", weights_only=False)["config"]["normalization_stats"]
        normalizer = Normalizer(stats)
    else:
        normalizer = Normalizer(Normalizer.compute_stats(*train_set.compute_stats()))
    config["normalization_stats"] = {k: {s: v.tolist() for s, v in d.items()} for k, d in normalizer.stats.items()}

    processor = load_processor(config["model"]["vlm_name"])
    d = config["data"]
    loader_kwargs = dict(batch_size=d["batch_size"], num_workers=d["num_workers"], pin_memory=True,
                         persistent_workers=d["num_workers"] > 0)
    train_loader = DataLoader(train_set, shuffle=True, drop_last=True,
                              collate_fn=build_collator(config, processor, normalizer, train=True), **loader_kwargs)
    val_loader = None
    if val_set is not None:
        val_loader = DataLoader(val_set, shuffle=True, collate_fn=build_collator(config, processor, normalizer, train=False),
                                **loader_kwargs)

    model = VLA(config)
    trainer = Trainer(model, train_loader, val_loader, config)
    normalizer.save(os.path.join(trainer.checkpoint_dir, "stats.json"))
    if args.resume:
        trainer.load_checkpoint(args.resume)
    trainer.logger.info(f"checkpoints: {trainer.checkpoint_dir}")
    path = trainer.train()
    trainer.logger.info(f"done, final checkpoint: {path}")


if __name__ == "__main__":
    main()
