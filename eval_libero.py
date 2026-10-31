""" Closed-loop evaluation in the LIBERO simulator (standard protocol: 50 fixed initial states per task).

    python eval_libero.py --checkpoint checkpoints/exp_.../last.pth --suites libero_spatial libero_object
    python eval_libero.py --checkpoint ... --suites libero_10 --task_ids 0 1 --num_trials 5   # quick check
"""
import argparse
import json
import os
import time

from src import libero_utils as lu  # sets MUJOCO_GL before robosuite is imported

import textwrap

import imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from src.policy import load_policy


def annotate_video(frames, instruction, success, scale=2):
    """ Upscales the frames and adds a banner with the instruction, the step and the episode outcome. """
    font = ImageFont.load_default(size=14)
    width = frames[0].shape[1] * scale
    lines = textwrap.wrap(instruction, width=width // 8)
    banner_h = 18 * (len(lines) + 1) + 6
    out = []
    for t, frame in enumerate(frames):
        img = Image.fromarray(frame).resize((width, frame.shape[0] * scale), Image.NEAREST)
        canvas = Image.new("RGB", (width, img.height + banner_h), "black")
        canvas.paste(img, (0, banner_h))
        draw = ImageDraw.Draw(canvas)
        for i, line in enumerate(lines):
            draw.text((4, 3 + 18 * i), line, fill="white", font=font)
        status = ("SUCCESS" if success else "FAIL") if t == len(frames) - 1 else ""
        draw.text((4, 3 + 18 * len(lines)), f"step {t}  {status}", fill="lime" if success else "orange", font=font)
        out.append(np.asarray(canvas))
    return out


def run_episode(env, policy, instruction, init_state, max_steps, cameras, record=False):
    policy.reset()
    env.reset()
    obs = env.set_init_state(init_state)
    frames = []
    for t in range(max_steps + lu.NUM_WAIT_STEPS):
        if t < lu.NUM_WAIT_STEPS:
            obs, _, done, _ = env.step(lu.DUMMY_ACTION)
            continue
        images = lu.get_images(obs, cameras)
        if record:
            frames.append(np.concatenate(images, axis=1))
        action = policy.act(images, instruction, lu.get_state(obs))
        obs, _, done, _ = env.step(action.tolist())
        if done:
            return True, frames
    return False, frames


def evaluate_suite(policy, config, suite_name, num_trials, task_ids, out_dir, num_videos):
    suite = lu.get_suite(suite_name)
    max_steps = lu.SUITE_MAX_STEPS[suite_name]
    cameras = config["data"]["cameras"]
    task_ids = task_ids if task_ids is not None else range(suite.n_tasks)
    results = {}
    for task_id in task_ids:
        task = suite.get_task(task_id)
        init_states = lu.load_init_states(suite, task_id)
        env = lu.make_env(task, resolution=config["eval"]["resolution"], seed=config["eval"]["seed"])
        successes = []
        for trial in tqdm(range(num_trials), desc=f"{suite_name} task {task_id}", leave=False):
            record = trial < num_videos
            success, frames = run_episode(env, policy, task.language, init_states[trial % len(init_states)],
                                          max_steps, cameras, record)
            successes.append(success)
            if record and frames:
                name = f"{suite_name}_task{task_id}_trial{trial}_{'success' if success else 'fail'}.mp4"
                imageio.mimwrite(os.path.join(out_dir, "videos", name), annotate_video(frames, task.language, success),
                                 fps=20)
        env.close()
        results[task.language] = float(np.mean(successes))
        print(f"[{suite_name}] task {task_id} '{task.language}': {results[task.language]:.2%}", flush=True)
    return {"success_rate": float(np.mean(list(results.values()))), "tasks": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--suites", nargs="+", default=None, help="default: the suites the model was trained on")
    parser.add_argument("--task_ids", nargs="+", type=int, default=None)
    parser.add_argument("--num_trials", type=int, default=None)
    parser.add_argument("--execute_steps", type=int, default=None)
    parser.add_argument("--num_denoise_steps", type=int, default=None)
    parser.add_argument("--num_videos", type=int, default=None, help="videos saved per task")
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    policy, config = load_policy(args.checkpoint, execute_steps=args.execute_steps,
                                 num_denoise_steps=args.num_denoise_steps)
    e = config["eval"]
    np.random.seed(e["seed"])
    torch.manual_seed(e["seed"])
    suites = args.suites or config["data"]["suites"]
    num_trials = args.num_trials or e["num_trials"]
    num_videos = e["num_videos_per_task"] if args.num_videos is None else args.num_videos
    out_dir = args.out_dir or os.path.join(os.path.dirname(args.checkpoint), f"eval_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(os.path.join(out_dir, "videos"), exist_ok=True)

    all_results = {}
    for suite_name in suites:
        all_results[suite_name] = evaluate_suite(policy, config, suite_name, num_trials, args.task_ids, out_dir,
                                                 num_videos)
        print(f"==> {suite_name}: {all_results[suite_name]['success_rate']:.2%}", flush=True)
        with open(os.path.join(out_dir, "results.json"), "w") as f:
            json.dump(all_results, f, indent=2)
    if len(suites) > 1:
        print("==> average:", f"{np.mean([r['success_rate'] for r in all_results.values()]):.2%}")
    print(f"results: {os.path.join(out_dir, 'results.json')}")


if __name__ == "__main__":
    main()
