import logging
import math
import os
import time
from typing import Optional

import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from .processing import move_to_device


def get_logger(log_path: str) -> logging.Logger:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger(f"trainer.{log_path}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler = logging.FileHandler(log_path, mode="a")
        file_handler.setFormatter(fmt)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    return logger


def _warmup_cosine_factor(step: int, warmup_steps: int, total_steps: int, min_lr_ratio: float) -> float:
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        eval_loader: Optional[DataLoader],
        config: dict,
    ):
        self.full_config = config
        self.config = config["trainer"]
        self.device = self.config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.eval_loader = eval_loader

        self.checkpoint_dir = os.path.join(self.config["checkpoint_dir"], f"exp_{time.time()}")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.logger = get_logger(os.path.join(self.checkpoint_dir, "train.log"))

        self.optimizer = None
        self.scheduler = None
        self.amp = bool(self.config["amp"]) and self.device == "cuda"
        self.epoch = 0
        self.global_step = 0
        self.local_step = 0
        self.init_optim()

    def init_optim(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.logger.info(f"trainable params: {sum(p.numel() for p in params) / 1e6:.1f}M")
        self.optimizer = torch.optim.AdamW(
            params,
            lr=float(self.config["lr"]),
            weight_decay=float(self.config["weight_decay"]),
            betas=tuple(float(b) for b in self.config["betas"]),
            eps=float(self.config["eps"]),
        )

        total_steps = self.total_steps = len(self.train_loader) * int(self.config["num_epochs"])
        warmup_steps = int(round(total_steps * float(self.config["warmup_ratio"])))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: _warmup_cosine_factor(
                step, warmup_steps, total_steps, float(self.config["min_lr_ratio"])
            ),
        )

        wandb.init(
            project=self.config["wandb_project"],
            name=self.config["wandb_name"],
            config=self.full_config,
            mode="online" if self.config.get("wandb", True) else "disabled",
        )

    def _model_state_dict(self):
        if hasattr(self.model, "trainable_state_dict"):
            return self.model.trainable_state_dict()  # skip frozen weights (the VLM is reloaded from the hub)
        return self.model.state_dict()

    def save_checkpoint(self, name=None):
        name = name or f"model_epoch{self.epoch}_step{self.global_step}.pth"
        path = os.path.join(self.checkpoint_dir, name)
        torch.save(
            {
                "model": self._model_state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict() if self.scheduler else None,
                "epoch": self.epoch,
                "global_step": self.global_step,
                "local_step": self.local_step,
                "config": self.full_config,
            },
            path,
        )
        self.logger.info(f"saved checkpoint: {path}")
        return path

    def load_checkpoint(self, path: str, load_optim: bool = True):
        ckpt = torch.load(path, map_location=self.device)
        if hasattr(self.model, "load_trainable_state_dict"):
            self.model.load_trainable_state_dict(ckpt["model"])
        else:
            self.model.load_state_dict(ckpt["model"])
        if load_optim and ckpt.get("optimizer") is not None:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        if self.scheduler is not None and ckpt.get("scheduler") is not None:
            self.scheduler.load_state_dict(ckpt["scheduler"])
        self.epoch = ckpt.get("epoch", 0)
        self.global_step = ckpt.get("global_step", 0)
        self.local_step = ckpt.get("local_step", 0)
        return ckpt

    def train_step(self, batch):
        batch = move_to_device(batch, self.device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp):
            out = self.model(batch)
        loss = out["loss"]
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad], float(self.config["grad_clip"]))
        self.optimizer.step()
        self.scheduler.step()
        return {"train/loss": loss.item(), "train/grad_norm": grad_norm.item(), "train/lr": self.scheduler.get_last_lr()[0]}

    def train(self, num_epochs: Optional[int] = None):
        num_epochs = num_epochs or int(self.config["num_epochs"])
        log_every = int(self.config.get("log_every", 50))
        eval_every = int(self.config.get("eval_every", 2000))
        save_every = int(self.config.get("save_every", 5000))
        path = None

        for _ in tqdm(range(self.epoch, num_epochs), total=num_epochs - self.epoch, leave=False):
            self.model.train()
            t0 = time.time()
            for step, batch in tqdm(enumerate(self.train_loader), total=len(self.train_loader), leave=True):
                if step < self.local_step:  # fast-forward a resumed epoch
                    continue
                metrics = self.train_step(batch)
                self.global_step += 1
                self.local_step = step + 1

                if self.global_step % log_every == 0:
                    metrics["train/steps_per_sec"] = log_every / (time.time() - t0)
                    t0 = time.time()
                    wandb.log(metrics, step=self.global_step)
                    self.logger.info(f"epoch {self.epoch} step {self.global_step}/{self.total_steps} "
                                     + " ".join(f"{k}={v:.4g}" for k, v in metrics.items()))
                if self.eval_loader is not None and self.global_step % eval_every == 0:
                    wandb.log(self.eval(), step=self.global_step)
                    self.model.train()
                if self.global_step % save_every == 0:
                    self.save_checkpoint()

            self.epoch += 1
            self.local_step = 0
            path = self.save_checkpoint(name="last.pth")
        if self.eval_loader is not None:
            wandb.log(self.eval(), step=self.global_step)
        return path

    @torch.no_grad()
    def eval(self):
        """ Held-out flow-matching loss (the success rate is measured in simulation by eval_libero.py). """
        self.model.eval()
        max_batches = int(self.config.get("eval_max_batches", 50))
        losses = []
        for step, batch in tqdm(enumerate(self.eval_loader), total=min(len(self.eval_loader), max_batches), leave=False):
            if step >= max_batches:
                break
            batch = move_to_device(batch, self.device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp):
                losses.append(self.model(batch)["loss"].item())
        metrics = {"eval/loss": sum(losses) / max(1, len(losses))}
        self.logger.info(f"step {self.global_step} " + " ".join(f"{k}={v:.4g}" for k, v in metrics.items()))
        return metrics
