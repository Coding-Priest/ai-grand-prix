from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import fire
import gymnasium
import jax.numpy as jp
import numpy as np
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from lsy_drone_racing.utils import load_config
from visualize import VisualizeSim
from rollout_buffer import RolloutBuffer
from models.droneac import DroneActorCritic

if TYPE_CHECKING:
    from ml_collections import ConfigDict
    from lsy_drone_racing.envs.drone_race import DroneRaceEnv

logger = logging.getLogger(__name__)
# sim_visualizer = VisualizeSim()

# Environment Action Bounds
ACTION_LOW = np.array([-1.5707964, -1.5707964, -1.5707964, 0.08545052])
ACTION_HIGH = np.array([1.5707964, 1.5707964, 1.5707964, 0.8])


def scale_action(model_action):
    """Maps the [-1, 1] network output to the physical drone bounds."""
    normalized = (model_action + 1.0) / 2.0
    return ACTION_LOW + normalized * (ACTION_HIGH - ACTION_LOW)


def create_env(config, render=False):
    """Helper to instantiate environments cleanly."""
    # Override render config for the specific environment instance
    sim_config = config.sim
    sim_config.render = render

    env = gymnasium.make(
        config.env.id,
        freq=config.env.freq,
        sim_config=sim_config,
        sensor_range=config.env.sensor_range,
        control_mode=config.env.control_mode,
        track=config.env.track,
        disturbances=config.env.get("disturbances"),
        randomizations=config.env.get("randomizations"),
        seed=config.env.seed,
    )
    return JaxToNumpy(env)


def train(
    config_file: str = "level0_trial_flight.toml",
    iterations: int = 500,
    val_freq: int = 5,  # Run validation every 10 iterations
):
    """Train the Drone Actor-Critic using PPO with Validation."""

    config = load_config(Path(__file__).parents[1] / "config" / config_file)

    # 1. Setup Separate Environments
    # Training env runs as fast as possible (no rendering)
    train_env = create_env(config, render=False)
    # Validation env is used to watch the drone (rendering enabled)

    # 2. PPO Hyperparameters
    num_steps = 8
    batch_size = 1024
    ppo_epochs = 10
    learning_rate = 4e-4
    clip_coef = 0.2
    ent_coef = 0.01
    num_envs = 100000

    # 3. Initialize Models and Buffer
    num_gates = len(config.env.track.gates)
    model = DroneActorCritic(num_gates=num_gates)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    buffer = RolloutBuffer(
        num_steps=num_steps,
        num_envs=num_envs,
        obs_dim=model.state_dim,
        action_dim=model.action_dim,
    )

    train_obs, _ = train_env.reset()

    # ==========================================
    # MAIN TRAINING LOOP
    # ==========================================
    for iteration in range(1, iterations + 1):
        print(f"\n--- Iteration {iteration}/{iterations} ---")

        # --- Tracking Variables for this Iteration ---
        rollout_reward_sum = 0.0
        episodes_completed = 0

        # ------------------------------------------
        # PHASE 1: Data Collection
        # ------------------------------------------
        rollout_bar = tqdm(range(num_steps), desc="Rollout Phase", leave=False)
        for step in rollout_bar:
            state_tensor = model.format_state(
                train_obs["pos"],
                train_obs["quat"],
                train_obs["vel"],
                train_obs["ang_vel"],
                train_obs["target_gate"],
                train_obs["gates_pos"],
                train_obs["gates_quat"],
            )

            with torch.no_grad():
                action_dist, value = model(state_tensor)
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action).sum(axis=-1)

            numpy_action = action.cpu().numpy()[0]
            env_action = scale_action(numpy_action)

            next_obs, reward, terminated, truncated, _ = train_env.step(env_action)
            done = terminated or truncated

            # Track rewards
            rollout_reward_sum += reward

            buffer.add(
                obs=state_tensor,
                action=action,
                reward=torch.tensor([reward], dtype=torch.float32),
                value=value,
                log_prob=log_prob,
                done=torch.tensor([done], dtype=torch.float32),
            )

            train_obs = next_obs
            if done:
                episodes_completed += 1
                train_obs, _ = train_env.reset()

            # Update bar occasionally to show we are collecting points
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
        )

        with torch.no_grad():
            _, last_value = model(state_tensor)
        buffer.compute_returns_and_advantages(
            last_value, torch.tensor([done], dtype=torch.float32)
        )

        # ------------------------------------------
        # PHASE 2: Optimize Networks
        # ------------------------------------------
        # --- Tracking Variables for Training ---
        total_pg_loss = 0.0
        total_v_loss = 0.0
        total_entropy = 0.0
        batches_processed = 0

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
                loss = pg_loss + v_loss - (ent_coef * entropy)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

                # Accumulate stats
                total_pg_loss += pg_loss.item()
                total_v_loss += v_loss.item()
                total_entropy += entropy.item()
                batches_processed += 1

            # Update the progress bar dynamically
            train_bar.set_postfix(
                {
                    "Actor Loss": f"{total_pg_loss/batches_processed:.4f}",
                    "Critic Loss": f"{total_v_loss/batches_processed:.4f}",
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

        # ------------------------------------------
        # PHASE 3: Validation (Rendered, Deterministic)
        # ------------------------------------------
        if iteration % val_freq == 0:
            print(f"\n>>> Running Validation Episode (Iteration {iteration}) <<<")

            # 1. Initialize the environment fresh
            val_env = create_env(config, render=True)
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
                )

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

    train_env.close()


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger("lsy_drone_racing").setLevel(logging.INFO)
    logger.setLevel(logging.INFO)

    fire.Fire(train, serialize=lambda _: None)
