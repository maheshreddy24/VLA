""" LIBERO simulator helpers. Import this before anything that imports robosuite / mujoco. """
import importlib.util
import os

os.environ.setdefault("MUJOCO_GL", "egl")  # headless GPU rendering
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import yaml

from .processing import flip_image

# Standard per-suite episode lengths (longest training demo + margin), as in OpenVLA / pi0 evaluations.
SUITE_MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520, "libero_90": 400}
NUM_WAIT_STEPS = 10  # objects fall into place during the first steps after reset
DUMMY_ACTION = [0.0] * 6 + [-1.0]  # no motion, gripper open


def ensure_libero_config():
    """ `import libero.libero` blocks on input() when ~/.libero/config.yaml is missing; write the default one. """
    config_dir = os.environ.get("LIBERO_CONFIG_PATH", os.path.expanduser("~/.libero"))
    config_file = os.path.join(config_dir, "config.yaml")
    if os.path.exists(config_file):
        return
    spec = importlib.util.find_spec("libero")
    if spec is None:
        raise ImportError("LIBERO is not installed; run scripts/setup_libero.sh")
    root = os.path.join(list(spec.submodule_search_locations)[0], "libero")
    os.makedirs(config_dir, exist_ok=True)
    with open(config_file, "w") as f:
        yaml.dump({"benchmark_root": root, "bddl_files": os.path.join(root, "bddl_files"),
                   "init_states": os.path.join(root, "init_files"), "datasets": os.path.join(root, "../datasets"),
                   "assets": os.path.join(root, "assets")}, f)


ensure_libero_config()
import torch  # noqa: E402
from libero.libero import benchmark, get_libero_path  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402
from robosuite.utils.transform_utils import quat2axisangle  # noqa: E402


def get_suite(suite_name):
    return benchmark.get_benchmark_dict()[suite_name]()


def load_init_states(suite, task_id):
    """ Same as suite.get_task_init_states, which breaks on torch >= 2.6 (weights_only defaults to True). """
    task = suite.tasks[task_id]
    path = os.path.join(get_libero_path("init_states"), task.problem_folder, task.init_states_file)
    return torch.load(path, weights_only=False)


def make_env(task, resolution=128, seed=0):
    bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl_file, camera_heights=resolution, camera_widths=resolution)
    env.seed(seed)
    return env


def get_state(obs):
    """ 8-dim state matching the dataset's ee_states (pos + axis-angle) + gripper_states. """
    return np.concatenate([obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]),
                           obs["robot0_gripper_qpos"]]).astype(np.float32)


def get_images(obs, cameras=("agentview", "wrist")):
    keys = {"agentview": "agentview_image", "wrist": "robot0_eye_in_hand_image"}
    return [flip_image(obs[keys[c]]) for c in cameras]
