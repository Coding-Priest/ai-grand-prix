"""Simulate the competition as in the IROS 2022 Safe Robot Learning competition.

Run as:

    $ python scripts/sim.py --config level0.toml

Look for instructions in `README.md` and in the official documentation.
"""

from __future__ import annotations
import os
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import fire
from collections import deque
from tqdm import tqdm
import gymnasium
import jax.numpy as jp
import numpy as np
from gymnasium.wrappers.vector import JaxToNumpy as JaxToNumpyVec
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy as JaxToNumpySingle


import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import time
from lsy_drone_racing.utils import load_config, load_controller

# from .visualize import VisualizeSim
from .rollout_buffer import RolloutBuffer

# from models.waypointac import WaypointActorCritic
from models.waypointac2 import WaypointActorCritic2


if TYPE_CHECKING:
    from ml_collections import ConfigDict

    from lsy_drone_racing.control.controller import Controller
    from lsy_drone_racing.envs.drone_race import DroneRaceEnv


logger = logging.getLogger(__name__)

# sim_visualizer = VisualizeSim()

ACTION_LOW = np.array([-1.5707964, -1.5707964, -1.5707964, 0.08545052])
ACTION_HIGH = np.array([1.5707964, 1.5707964, 1.5707964, 0.8])


def scale_action(model_action):
    """Maps the [-1, 1] network output to the physical drone bounds."""
    normalized = (model_action + 1.0) / 2.0
    return ACTION_LOW + normalized * (ACTION_HIGH - ACTION_LOW)


def create_env(config, num_envs, render=False):
    """Instantiate multiple environments."""
    sim_config = config.sim
    sim_config.render = render

    if num_envs > 1:
        # Batched environments for training
        env = gymnasium.make_vec(
            config.env.id,
            num_envs=num_envs,
            freq=config.env.freq,
            sim_config=sim_config,
            sensor_range=config.env.sensor_range,
            control_mode=config.env.control_mode,
            track=config.env.track,
            disturbances=config.env.get("disturbances"),
            randomizations=config.env.get("randomizations"),
            max_episode_steps=1350,
            seed=config.env.seed,
            disable_termination=False,
            disable_collisions=False,
            device="gpu",  # Match your sim script
        )
        return JaxToNumpyVec(env)
    else:
        # Single environment for validation
        env = gymnasium.make(
            config.env.id,
            freq=config.env.freq,
            sim_config=sim_config,
            sensor_range=config.env.sensor_range,
            control_mode=config.env.control_mode,
            track=config.env.track,
            disturbances=config.env.get("disturbances"),
            randomizations=config.env.get("randomizations"),
            max_episode_steps=850,
            seed=config.env.seed,
            disable_termination=False,
            disable_collisions=False,
            device="gpu",
        )

        return JaxToNumpySingle(env)


def train(
    config_file: str = "level0_baby_steps2.toml",
    iterations: int = 250,
    val_freq: int = 15,  # Run validation every 10 iterations
    checkpoint_dir: str = "checkpoints",  # <--- NEW: Where to save
    resume_path: str = "/home/homefree/Development/anduril-drone-race/ai-grand-prix/checkpoints/routine/best_model_50.pth",
    routine_checkpoint_dir: str = "checkpoints/routine",
):
    """Train the Drone Actor-Critic using PPO with Validation."""

    config = load_config(Path(__file__).parents[1] / "config" / config_file)

    train_device = "cpu" if torch.cuda.is_available() else "cpu"
    # 1. Setup Separate Environments
    # Training env runs as fast as possible (no rendering)
    # Validation env is used to watch the drone (rendering enabled)

    # 2. PPO Hyperparameters
    num_steps = 32
    batch_size = 512
    ppo_epochs = 10
    learning_rate = 1e-4
    clip_coef = 0.2
    ent_coef = 0.01

    # 3. Initialize Models and Buffer
    num_gates = len(config.env.track.gates)
    model = WaypointActorCritic2()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # buffer = RolloutBuffer(
    #     num_steps=num_steps,
    #     num_envs=num_envs,
    #     obs_dim=model.state_dim,
    #     action_dim=model.action_dim,
    #     device=train_device,
    # )

    model.to(train_device)
    # buffer.to(train_device)

    # ==========================================
    # CHECKPOINT LOADING & SETUP
    # ==========================================
    start_iteration = 1
    best_avg_reward = -float("inf")

    # Ensure our save folder exists
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(routine_checkpoint_dir).mkdir(parents=True, exist_ok=True)

    if resume_path is not None:
        if os.path.isfile(resume_path):
            print(f"=> Loading checkpoint from '{resume_path}'")
            # map_location ensures safe loading even if moving between CPU/GPU
            checkpoint = torch.load(
                resume_path, map_location=train_device, weights_only=False
            )
            # print(checkpoint.keys())
            # exit()
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_iteration = checkpoint["iteration"] + 1
            # best_avg_reward = checkpoint.get("best_avg_reward", -float("inf"))

            print(f"=> Successfully resumed from iteration {checkpoint['iteration']}")
        else:
            print(
                f"=> WARNING: No checkpoint found at '{resume_path}'. Starting from scratch."
            )
    val_env = create_env(config, num_envs=1, render=True)
    for iteration in range(1, iterations + 1):
        print(f"\n>>> Running Validation Episode (Iteration {iteration}) <<<")

        model.eval()

        # 1. Initialize the environment fresh

        val_obs, _ = val_env.reset()

        val_done = False
        total_val_reward = 0.0
        steps_survived = 0

        # Run one full episode until crash or completion
        while not val_done:

            val_state = model.format_state(
                val_obs["pos"],
                val_obs["quat"],
                val_obs["vel"],
                val_obs["ang_vel"],
                val_obs["target_gate"],
                val_obs["gates_pos"],
                val_obs["gates_quat"],
                use_pass_through=True,
            ).to(train_device)

            with torch.no_grad():
                action_dist, _ = model(val_state)
                val_action = action_dist.mean

            numpy_action = val_action.cpu().numpy()[0]
            env_action = scale_action(numpy_action)

            val_obs, val_reward, terminated, truncated, _ = val_env.step(env_action)
            val_done = terminated or truncated
            total_val_reward += val_reward
            steps_survived += 1

            # Render the validation environment
            val_env.render()

        gates_passed = val_obs["target_gate"]
        if gates_passed == -1:
            gates_passed = num_gates

        print(
            f"Validation Results | Reward: {total_val_reward:.2f} | Gates Passed: {gates_passed}/{num_gates} | Steps Survived: {steps_survived}"
        )
        # 2. Close the environment to free up the rendering window and memory
    val_env.close()


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger("lsy_drone_racing").setLevel(logging.INFO)
    logger.setLevel(logging.INFO)

    fire.Fire(train, serialize=lambda _: None)
