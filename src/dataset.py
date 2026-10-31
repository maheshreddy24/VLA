""" LIBERO demonstrations (the official HDF5 release, https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets).

Layout: <data_root>/<suite>/<task>_demo.hdf5, each with data/demo_<i>/{actions, obs/{agentview_rgb, eye_in_hand_rgb,
ee_states, gripper_states, ...}}. State = ee_states (eef pos + axis-angle, 6) + gripper_states (2) = 8 dims, which is
exactly what `libero_utils.get_state` rebuilds from simulator observations. """
import glob
import json
import os
import re

import h5py
import numpy as np
from torch.utils.data import Dataset

from .processing import flip_image

CAMERA_KEYS = {"agentview": "agentview_rgb", "wrist": "eye_in_hand_rgb"}


def instruction_from_file(f, path):
    try:
        return json.loads(f["data"].attrs["problem_info"])["language_instruction"]
    except (KeyError, TypeError, ValueError):
        name = os.path.basename(path).replace("_demo.hdf5", "")
        name = re.sub(r"^([A-Z0-9_]+_SCENE\d+_)", "", name)  # libero_10 / 90 prefix, e.g. KITCHEN_SCENE3_
        return name.replace("_", " ")


def non_noop_indices(actions, threshold=1e-4):
    """ Drop idle steps (no arm motion and no gripper change), as in OpenVLA's LIBERO regeneration: the policy then
    never learns to stand still at the start of an episode. """
    keep = []
    for i, a in enumerate(actions):
        moving = np.linalg.norm(a[:-1]) > threshold
        gripper_changed = i > 0 and a[-1] != actions[i - 1][-1]
        if moving or gripper_changed:
            keep.append(i)
    return np.asarray(keep, dtype=np.int64)



# 	"pick up the ketchup and place it in the basket"
class LiberoDataset(Dataset):
    def __init__(self, data_root, suites, horizon, split="train", val_demos_per_task=2, cameras=("agentview", "wrist"),
                 filter_noops=True):
        self.horizon = horizon
        self.cameras = list(cameras)
        self.files, self.instructions = [], []
        self.episodes = []  # (file_idx, demo_key, kept frame indices)
        for suite in suites:
            paths = sorted(glob.glob(os.path.join(data_root, suite, "*.hdf5")))
            if not paths:
                raise FileNotFoundError(f"no hdf5 files in {os.path.join(data_root, suite)}; run scripts/download_libero.sh")
            for path in paths:
                with h5py.File(path, "r") as f:
                    demos = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[-1]))
                    n_val = min(val_demos_per_task, len(demos) - 1)
                    demos = demos[:len(demos) - n_val] if split == "train" else demos[len(demos) - n_val:]
                    file_idx = len(self.files)
                    self.files.append(path)
                    self.instructions.append(instruction_from_file(f, path))
                    for demo in demos:
                        actions = f["data"][demo]["actions"][()]
                        kept = non_noop_indices(actions) if filter_noops else np.arange(len(actions))
                        if len(kept) > 0:
                            self.episodes.append((file_idx, demo, kept))

        self.index = [(e, p) for e, (_, _, kept) in enumerate(self.episodes) for p in range(len(kept))]
        self._handles = {}  # opened lazily, one set per DataLoader worker

    def __len__(self):
        return len(self.index)

    def _file(self, file_idx):
        if file_idx not in self._handles:
            self._handles[file_idx] = h5py.File(self.files[file_idx], "r")
        return self._handles[file_idx]

    def close(self):
        for f in self._handles.values():
            f.close()
        self._handles = {}

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handles"] = {}
        return state

    def __getitem__(self, i):
        ep, pos = self.index[i]
        file_idx, demo, kept = self.episodes[ep]
        g = self._file(file_idx)["data"][demo]
        frame = kept[pos]
        chunk = kept[pos:pos + self.horizon]
        action_mask = np.zeros(self.horizon, dtype=bool)
        action_mask[:len(chunk)] = True
        chunk = np.concatenate([chunk, np.full(self.horizon - len(chunk), chunk[-1])])  # pad with the last action
        # h5py fancy indexing needs increasing unique indices
        uniq, inv = np.unique(chunk, return_inverse=True)
        actions = g["actions"][uniq][inv].astype(np.float32)
        obs = g["obs"]
        state = np.concatenate([obs["ee_states"][frame], obs["gripper_states"][frame]]).astype(np.float32)
        images = [flip_image(obs[CAMERA_KEYS[c]][frame]) for c in self.cameras]
        return {"images": images, "instruction": self.instructions[file_idx], "state": state,
                "actions": actions, "action_mask": action_mask}

    def compute_stats(self):
        """ State / action statistics over the (non-noop) frames of this split. """
        states, actions = [], []
        for file_idx, demo, kept in self.episodes:
            g = self._file(file_idx)["data"][demo]
            states.append(np.concatenate([g["obs"]["ee_states"][()], g["obs"]["gripper_states"][()]], -1)[kept])
            actions.append(g["actions"][()][kept])
        self.close()  # don't leak open h5py handles into forked DataLoader workers
        return np.concatenate(states).astype(np.float32), np.concatenate(actions).astype(np.float32)
