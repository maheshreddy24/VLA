from collections import deque

import numpy as np
import torch

from .backbone import load_processor
from .models import VLA
from .processing import Normalizer, VLACollator, move_to_device


def build_collator(config, processor, normalizer, train):
    d = config["data"]
    return VLACollator(processor, normalizer, image_size=d["vlm_image_size"], crop_scale=d.get("crop_scale", 1.0),
                       train=train, prompt=d["prompt"])


class VLAPolicy:
    """ Closed-loop wrapper: predicts an action chunk, executes its first `execute_steps` actions open loop, then
    replans. Returns unnormalized LIBERO actions (delta eef pose + gripper in {-1: open, 1: close}). """

    def __init__(self, model, collator, normalizer, execute_steps, num_denoise_steps=None, device="cuda"):
        self.model = model.eval()
        self.collator = collator
        self.normalizer = normalizer
        self.execute_steps = execute_steps
        self.num_denoise_steps = num_denoise_steps
        self.device = device
        self.queue = deque()

    def reset(self):
        self.queue.clear()

    @torch.no_grad()
    def predict_chunk(self, images, instruction, state):
        batch = move_to_device(self.collator([{"images": images, "instruction": instruction, "state": state}]),
                               self.device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device == "cuda"):
            chunk = self.model.predict_action(batch, self.num_denoise_steps)
        chunk = self.normalizer.unnormalize_action(chunk[0].float().cpu().numpy())
        chunk[:, -1] = np.where(chunk[:, -1] > 0, 1.0, -1.0)
        return chunk

    def act(self, images, instruction, state):
        if not self.queue:
            self.queue.extend(self.predict_chunk(images, instruction, state)[:self.execute_steps])
        return self.queue.popleft()


def load_policy(checkpoint_path, device="cuda", execute_steps=None, num_denoise_steps=None):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    model = VLA(config)
    model.load_trainable_state_dict(ckpt["model"])
    model = model.to(device)
    normalizer = Normalizer(config["normalization_stats"])
    collator = build_collator(config, load_processor(config["model"]["vlm_name"]), normalizer, train=False)
    execute_steps = execute_steps or config["eval"]["execute_steps"]
    return VLAPolicy(model, collator, normalizer, execute_steps, num_denoise_steps, device), config
