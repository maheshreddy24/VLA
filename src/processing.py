""" Preprocessing shared by training and simulator evaluation, so both see identical inputs. """
import json

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def move_to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    return batch


def flip_image(img):
    """ LIBERO renders upside down (OpenGL convention); rotate 180 degrees. Applied to demos and sim obs alike. """
    return np.ascontiguousarray(img[::-1, ::-1])


class Normalizer:
    """ Per-dimension normalization for states (mean / std) and actions (q01 / q99 -> [-1, 1]).
    Dimensions listed in `action_skip_dims` (the binary gripper) are left untouched. """

    def __init__(self, stats, action_skip_dims=(-1,)):
        self.stats = {k: {s: np.asarray(v, dtype=np.float32) for s, v in d.items()} for k, d in stats.items()}
        self.action_skip_dims = list(action_skip_dims)

    @staticmethod
    def compute_stats(states, actions):
        """ states: N, S; actions: N, A """
        return {
            "state": {"mean": states.mean(0), "std": states.std(0)},
            "action": {"q01": np.quantile(actions, 0.01, axis=0), "q99": np.quantile(actions, 0.99, axis=0),
                       "mean": actions.mean(0), "std": actions.std(0)},
        }

    def normalize_state(self, state):
        s = self.stats["state"]
        return (state - s["mean"]) / (s["std"] + 1e-6)

    def _action_range(self):
        a = self.stats["action"]
        lo, hi = a["q01"].copy(), a["q99"].copy()
        lo[self.action_skip_dims], hi[self.action_skip_dims] = -1.0, 1.0  # identity on skipped dims
        return lo, hi

    def normalize_action(self, action):
        lo, hi = self._action_range()
        return np.clip(2 * (action - lo) / (hi - lo + 1e-6) - 1, -1, 1)

    def unnormalize_action(self, action):
        lo, hi = self._action_range()
        return (np.clip(action, -1, 1) + 1) / 2 * (hi - lo + 1e-6) + lo

    def save(self, path):
        with open(path, "w") as f:
            json.dump({k: {s: v.tolist() for s, v in d.items()} for k, d in self.stats.items()}, f, indent=2)

    @classmethod
    def load(cls, path, **kwargs):
        with open(path) as f:
            return cls(json.load(f), **kwargs)


def preprocess_image(img, size, crop_scale=1.0, train=False):
    """ img: H, W, 3 uint8 numpy -> size x size PIL image.
    crop_scale < 1 crops a square covering that fraction of the area: random position at train time, centered at
    eval time (the standard LIBERO augmentation). """
    x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float()[None]  # 1, 3, H, W
    if crop_scale < 1.0:
        H, W = x.shape[-2:]
        ch, cw = int(round(H * crop_scale ** 0.5)), int(round(W * crop_scale ** 0.5))
        if train:
            top, left = np.random.randint(0, H - ch + 1), np.random.randint(0, W - cw + 1)
        else:
            top, left = (H - ch) // 2, (W - cw) // 2
        x = x[..., top:top + ch, left:left + cw]
    x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False, antialias=True)
    return Image.fromarray(x[0].clamp(0, 255).round().byte().permute(1, 2, 0).numpy())


class VLACollator:
    """ Turns a list of samples {images: [H, W, 3 uint8, ...], instruction, state, [actions, action_mask]} into a
    batch with the VLM inputs built by the Qwen3-VL processor. Runs in the DataLoader workers. """

    def __init__(self, processor, normalizer, image_size=256, crop_scale=1.0, train=False,
                 prompt="What action should the robot take to {instruction}?"):
        self.processor = processor
        self.normalizer = normalizer
        self.image_size = image_size
        self.crop_scale = crop_scale
        self.train = train
        self.prompt = prompt

    def build_conversation(self, images, instruction):
        content = [{"type": "image"} for _ in images]
        content.append({"type": "text", "text": self.prompt.format(instruction=instruction.strip().rstrip("."))})
        return [{"role": "user", "content": content}]

    def __call__(self, samples):
        texts, images = [], []
        for s in samples:
            texts.append(self.processor.apply_chat_template(
                self.build_conversation(s["images"], s["instruction"]), tokenize=False, add_generation_prompt=True))
            images.extend(preprocess_image(im, self.image_size, self.crop_scale, self.train) for im in s["images"])
        vlm_inputs = self.processor(text=texts, images=images, padding=True, return_tensors="pt")

        batch = {
            "vlm_inputs": dict(vlm_inputs),
            "state": torch.from_numpy(np.stack([self.normalizer.normalize_state(s["state"]) for s in samples])).float(),
        }
        if "actions" in samples[0]:
            batch["actions"] = torch.from_numpy(
                np.stack([self.normalizer.normalize_action(s["actions"]) for s in samples])).float()
            batch["action_mask"] = torch.from_numpy(np.stack([s["action_mask"] for s in samples])).bool()
        return batch
