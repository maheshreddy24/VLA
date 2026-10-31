#!/bin/bash
# Installs the training deps + LIBERO simulator into the active env (run after setup.sh: conda activate temporal).
set -e
ROOT="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"

pip install -r "${ROOT}/requirements.txt"

if [ ! -d "${ROOT}/third_party/LIBERO" ]; then
    git clone https://github.com/Lifelong-Robot-Learning/LIBERO "${ROOT}/third_party/LIBERO"
fi
# --no-deps: LIBERO's requirements.txt pins 2022 versions (torch, transformers, numpy 1.22) that would break this env
# The top-level libero/ has no __init__.py, so with modern setuptools find_packages() finds nothing and the editable
# install maps no packages; add it and use compat mode (plain .pth path entry).
touch "${ROOT}/third_party/LIBERO/libero/__init__.py"
pip install -e "${ROOT}/third_party/LIBERO" --no-deps --config-settings editable_mode=compat

# headless rendering (EGL) for MuJoCo
sudo apt install -y libegl1 libgl1 libglew-dev || true

# Write ~/.libero/config.yaml (otherwise `import libero` blocks on an interactive prompt) and smoke test the sim
cd "${ROOT}" && MUJOCO_GL=egl python -c "
from src import libero_utils as lu
suite = lu.get_suite('libero_spatial')
task = suite.get_task(0)
env = lu.make_env(task)
env.reset()
obs = env.set_init_state(lu.load_init_states(suite, 0)[0])
obs, *_ = env.step(lu.DUMMY_ACTION)
print('LIBERO OK:', task.language, 'state', lu.get_state(obs).shape, 'images', [i.shape for i in lu.get_images(obs)])
env.close()
"
