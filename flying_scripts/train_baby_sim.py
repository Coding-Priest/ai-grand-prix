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

from lsy_drone_racing.utils import load_config, load_controller

# from .visualize import VisualizeSim
from .rollout_buffer import RolloutBuffer

# from models.waypointac import WaypointActorCritic
from models.waypointac3 import WaypointActorCritic3


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
            max_episode_steps=500,
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
            max_episode_steps=500,
            seed=config.env.seed,
            disable_termination=False,
            disable_collisions=False,
            device="cpu",
        )

        return JaxToNumpySingle(env)


def run_validation(model, config, num_episodes=10, device="cuda"):
    """Runs a set of evaluation episodes and returns averaged statistics."""
    model.eval()
    # Use a single env for validation; set render=False if you want it to run fast
    val_env = create_env(config, num_envs=1, render=False)

    stats = {"rewards": [], "gates_passed": [], "steps_survived": [], "success_rate": 0}

    for ep in range(num_episodes):
        obs, _ = val_env.reset()
        done = False
        ep_reward = 0
        ep_steps = 0

        while not done:
            state_tensor = model.format_state(
                obs["pos"],
                obs["quat"],
                obs["vel"],
                obs["ang_vel"],
                obs["target_gate"],
                obs["gates_pos"],
                obs["gates_quat"],
                use_pass_through=True,
            ).to(device)

            with torch.no_grad():
                # Use mean for deterministic validation
                action_dist, _ = model(state_tensor)
                action = action_dist.mean.cpu().numpy()[0]

            env_action = scale_action(np.clip(action, -1.0, 1.0))
            obs, reward, terminated, truncated, _ = val_env.step(env_action)

            ep_reward += reward
            ep_steps += 1
            done = terminated or truncated

        # Extract gate info (Logic depends on your env's target_gate behavior)
        # If target_gate is the index of the next gate, passed = target_gate
        gates = obs["target_gate"]
        if gates == -1:  # Usually indicates track completion in some envs
            gates = len(config.env.track.gates)
            stats["success_rate"] += 1

        stats["rewards"].append(ep_reward)
        stats["gates_passed"].append(gates)
        stats["steps_survived"].append(ep_steps)

    val_env.close()

    # Calculate Averages
    results = {
        "avg_reward": np.mean(stats["rewards"]),
        "avg_gates": np.mean(stats["gates_passed"]),
        "avg_steps": np.mean(stats["steps_survived"]),
        "success_rate": (stats["success_rate"] / num_episodes) * 100,
    }
    return results


def train(
    config_file: str = "level0_baby_steps2.toml",  # "level0_boot_strap.toml",  #
    iterations: int = 250,
    val_freq: int = 5,  # Run validation every 10 iterations
    render_sim_freq: int = 5,
    checkpoint_dir: str = "checkpoints",  # <--- NEW: Where to save
    resume_path: str = "/home/homefree/Development/anduril-drone-race/ai-grand-prix/model_weights/3_waypoints_ac/best_model_check_2.pth",
    routine_checkpoint_dir: str = "checkpoints/routine",
):
    """Train the Drone Actor-Critic using PPO with Validation."""

    config = load_config(Path(__file__).parents[1] / "config" / config_file)

    train_device = "cuda" if torch.cuda.is_available() else "cpu"
    # 1. Setup Separate Environments
    # Training env runs as fast as possible (no rendering)
    num_envs = int(512 * 2)
    train_env = create_env(config, num_envs=num_envs, render=False)
    # Validation env is used to watch the drone (rendering enabled)

    # 2. PPO Hyperparameters
    num_steps = 128
    batch_size = 512
    ppo_epochs = 5
    learning_rate = 1e-4
    clip_coef = 0.2
    ent_coef = 0.005

    # 3. Initialize Models and Buffer
    num_gates = len(config.env.track.gates)
    model = WaypointActorCritic3()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, eps=1e-5)
    buffer = RolloutBuffer(
        num_steps=num_steps,
        num_envs=num_envs,
        obs_dim=model.state_dim,
        action_dim=model.action_dim,
        device=train_device,
    )

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
                resume_path, map_location=buffer.device, weights_only=False
            )
            # print(checkpoint.keys())
            # exit()
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_iteration = checkpoint["iteration"] + 1
            # best_avg_reward = checkpoint.get("best_avg_reward", -float("inf"))
            print("Loaded model weights successfully.")
            # print(f"=> Successfully resumed from iteration {checkpoint['iteration']}")
        else:
            print(
                f"=> WARNING: No checkpoint found at '{resume_path}'. Starting from scratch."
            )
    # exit()

    train_obs, _ = train_env.reset()

    # ==========================================
    # MAIN TRAINING LOOP
    # ==========================================
    model.train()
    for iteration in range(1, iterations + 1):
        print(f"\n--- Iteration {iteration}/{iterations} ---")

        # --- Tracking Variables for this Iteration ---
        rollout_reward_sum = 0.0
        episodes_completed = 0
        frac = 1.0  # - (iteration - 1.0) / iterations
        current_ent_coef = ent_coef * frac

        # ------------------------------------------
        # PHASE 1: Data Collection
        # ------------------------------------------
        rollout_bar = tqdm(range(num_steps), desc="Rollout Phase", leave=False)
        for step in rollout_bar:
            # model.format_state must be able to handle batched dict inputs
            state_tensor = model.format_state(
                train_obs["pos"],
                train_obs["quat"],
                train_obs["vel"],
                train_obs["ang_vel"],
                train_obs["target_gate"],
                train_obs["gates_pos"],
                train_obs["gates_quat"],
                use_pass_through=True,
            ).to(train_device)

            with torch.no_grad():
                action_dist, value = model(state_tensor)
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action).sum(axis=-1)

            # 1. Extract the BATCHED actions (Remove the [0] index)
            numpy_action = action.cpu().numpy()  # Shape: (num_envs, action_dim)
            clipped_numpy_action = np.clip(numpy_action, -1.0, 1.0)
            env_action = scale_action(clipped_numpy_action)

            # Match your simulation script requirement for the step function
            env_action = np.expand_dims(env_action, axis=1)

            # 2. Step the batched environments
            next_obs, reward, terminated, truncated, _ = train_env.step(env_action)

            # print("Terminated: ", terminated)
            # print("Truncated: ", truncated)
            # print("Reward: ", reward)

            # Combine termination and truncation arrays
            done = terminated | truncated

            # Track rewards (sum across all environments for logging)
            rollout_reward_sum += np.sum(reward)

            # 3. Add to buffer (Pass the arrays directly, no need to wrap in brackets [])
            buffer.add(
                obs=state_tensor,
                action=action,
                reward=torch.tensor(reward, dtype=torch.float32, device=buffer.device),
                value=value,
                log_prob=log_prob,
                done=torch.tensor(done, dtype=torch.float32, device=buffer.device),
            )

            train_obs = next_obs

            # 4. Handle Resets
            # Gymnasium's Vector Envs automatically reset finished sub-environments.
            # We just need to track how many finished for our logging.
            episodes_completed += int(np.sum(done))

            # # Steps completed in the episodes that were completed
            # step_completed_per_completed_episode +=
            # gates_passed_per_completed_episode +=
            if step % 100 == 0:
                rollout_bar.set_postfix(
                    {"Current Reward Sum": f"{rollout_reward_sum:.2f}"}
                )

        # Compute Advantages (GAE)
        state_tensor = model.format_state(
            train_obs["pos"],
            train_obs["quat"],
            train_obs["vel"],
            train_obs["ang_vel"],
            train_obs["target_gate"],
            train_obs["gates_pos"],
            train_obs["gates_quat"],
            use_pass_through=True,
        ).to(train_device)

        with torch.no_grad():
            _, last_value = model(state_tensor)

        # Pass the batched 'done' array as a tensor
        buffer.compute_returns_and_advantages(
            last_value, torch.tensor(done, dtype=torch.float32, device=buffer.device)
        )

        # ------------------------------------------
        # PHASE 2: Optimize Networks
        # ------------------------------------------
        # --- Tracking Variables for Training ---
        total_pg_loss = 0.0
        total_v_loss = 0.0
        total_entropy = 0.0
        batches_processed = 0.0

        train_bar = tqdm(range(ppo_epochs), desc="Training Phase ", leave=False)
        for epoch in train_bar:
            for batch in buffer.get_generator(batch_size):
                b_obs, b_actions, b_values, b_log_probs, b_advantages, b_returns = batch

                new_dist, new_values = model(b_obs)
                new_log_probs = new_dist.log_prob(b_actions).sum(axis=-1)
                entropy = new_dist.entropy().sum(axis=-1).mean()

                logratio = new_log_probs - b_log_probs
                ratio = logratio.exp()

                pg_loss1 = -b_advantages * ratio
                pg_loss2 = -b_advantages * torch.clamp(
                    ratio, 1.0 - clip_coef, 1.0 + clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = 0.5 * ((new_values.squeeze() - b_returns) ** 2).mean()
                loss = pg_loss + v_loss - (current_ent_coef * entropy)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

                with torch.no_grad():
                    # Clamp log_std to prevent infinite entropy or completely deterministic collapse
                    model.actor_log_std.clamp_(min=-20.0, max=0.5)

                # Accumulate stats
                total_pg_loss += pg_loss.item()
                total_v_loss += v_loss.item()
                total_entropy += entropy.item()
                batches_processed += 1

            # Update the progress bar dynamically
            train_bar.set_postfix(
                {
                    "Actor Loss": f"{total_pg_loss/max(1, batches_processed):.4f}",
                    "Critic Loss": f"{total_v_loss/max(1, batches_processed):.4f}",
                }
            )

        buffer.reset()

        # ------------------------------------------
        # PRINT PERMANENT STATS TO CONSOLE
        # ------------------------------------------
        avg_reward = rollout_reward_sum / max(1, episodes_completed)
        avg_pg_loss = total_pg_loss / max(1, batches_processed)
        avg_v_loss = total_v_loss / max(1, batches_processed)
        avg_ent = total_entropy / max(1, batches_processed)

        print(
            f"Done! | Rollout Avg Reward: {avg_reward:.2f} | Episodes: {episodes_completed}"
        )
        print(
            f"Stats | Actor Loss: {avg_pg_loss:.4f} | Critic Loss: {avg_v_loss:.4f} | Entropy: {avg_ent:.4f}"
        )
        # if avg_reward > best_avg_reward and episodes_completed > 100:
        #     best_avg_reward = avg_reward
        #     print(
        #         f"*** New Best Validation Reward: {best_avg_reward:.2f}! Saving best model... ***"
        #     )

        #     best_path = Path(checkpoint_dir) / "best_model.pth"
        #     torch.save(
        #         {
        #             "iteration": iteration,
        #             "model_state_dict": model.state_dict(),
        #             "optimizer_state_dict": optimizer.state_dict(),
        #             "best_avg_reward": best_avg_reward,
        #         },
        #         best_path,
        #     )

        # ------------------------------------------
        # PHASE 3: Validation (Rendered, Deterministic)
        # ------------------------------------------
        if iteration % render_sim_freq == 0:
            print(f"\n>>> Running Validation Episode (Iteration {iteration}) <<<")

            model.eval()

            # 1. Initialize the environment fresh
            val_env = create_env(config, num_envs=1, render=True)
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

            model.train()

        # ------------------------------------------
        # PHASE 3: Multi-Episode Validation
        # ------------------------------------------
        if iteration % val_freq == 0:
            num_val_episodes = 10
            print(f"\n>>> Validating over {num_val_episodes} episodes...")

            val_results = run_validation(
                model, config, num_episodes=num_val_episodes, device=train_device
            )

            print(f"--- Validation Results (Iter {iteration}) ---")
            print(f"Avg Reward: {val_results['avg_reward']:.2f}")
            print(f"Avg Gates:  {val_results['avg_gates']:.2f} / {num_gates}")
            print(f"Avg Steps:  {val_results['avg_steps']:.1f}")
            print(f"SR (Finish): {val_results['success_rate']}%")
            print("-" * 40)

            # Check for best model based on avg_reward or avg_gates
            if val_results["avg_reward"] > best_avg_reward:
                best_avg_reward = val_results["avg_reward"]
                print(f"*** New Best Model Found! ***")
                torch.save(
                    {
                        "iteration": iteration,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_avg_reward": best_avg_reward,
                    },
                    Path(checkpoint_dir) / "best_model.pth",
                )

            best_path = Path(routine_checkpoint_dir) / f"best_model_{iteration}.pth"
            torch.save(
                {
                    "iteration": iteration,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_avg_reward": best_avg_reward,
                },
                best_path,
            )

            model.train()  # Switch back to training mode

    train_env.close()


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger("lsy_drone_racing").setLevel(logging.INFO)
    logger.setLevel(logging.INFO)

    fire.Fire(train, serialize=lambda _: None)
