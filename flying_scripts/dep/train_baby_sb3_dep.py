import os

# MUST be set before any other imports, especially scipy or gymnasium
os.environ["SCIPY_ARRAY_API"] = "1"
# os.environ["JAX_PLATFORMS"] = "cpu"
import logging
from pathlib import Path
import fire

import gymnasium as gym
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy
import numpy as np
from scipy.spatial.transform import Rotation as R

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from lsy_drone_racing.utils import load_config

logger = logging.getLogger(__name__)


def create_env(config, render=False):
    """Helper to instantiate environments cleanly."""
    # Override render config for the specific environment instance
    sim_config = config.sim
    sim_config.render = render
    mode = "human" if render else None
    env = gym.make(
        config.env.id,
        render_mode=mode,
        freq=config.env.freq,
        sim_config=sim_config,
        sensor_range=config.env.sensor_range,
        control_mode=config.env.control_mode,
        track=config.env.track,
        disturbances=config.env.get("disturbances"),
        randomizations=config.env.get("randomizations"),
        seed=config.env.seed,
        disable_termination=True,
        disable_collisions=True,
    )
    return JaxToNumpy(env)


# ==========================================
# 1. ENVIRONMENT WRAPPERS
# ==========================================


class DroneObservationWrapper(gym.ObservationWrapper):
    """
    Replaces the custom `format_state` method.
    Converts the complex dict observation into a flat 1D NumPy array for SB3.
    """

    def __init__(self, env, window_size=3):
        super().__init__(env)
        self.window_size = window_size

        # Calculate Input Dimension: Quat(4) + Vel(3) + AngVel(3) + [Pos(3) + Quat(4)] * window_size
        self.state_dim = 4 + 3 + 3 + (3 * self.window_size) + (4 * self.window_size)

        # Override the observation space so SB3 knows the exact shape it will receive
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )

    def observation(self, obs):
        # Extract variables from the raw dictionary observation
        drone_pos = np.array(obs["pos"]).flatten()
        drone_quat = np.array(obs["quat"]).flatten()
        vel = np.array(obs["vel"]).flatten()
        ang_vel = np.array(obs["ang_vel"]).flatten()
        gates_pos = np.array(obs["gates_pos"])
        gates_quat = np.array(obs["gates_quat"])
        target_idx = int(np.array(obs["target_gate"]).item())

        total_gates = len(gates_pos)
        if target_idx == -1:
            target_idx = total_gates - 1  # Race is over, use final gate

        # Extract Lookahead Window
        end_idx = min(target_idx + self.window_size, total_gates)
        lookahead_pos = list(gates_pos[target_idx:end_idx])
        lookahead_quat = list(gates_quat[target_idx:end_idx])

        # Goal Duplication Padding
        while len(lookahead_pos) < self.window_size:
            lookahead_pos.append(lookahead_pos[-1])
            lookahead_quat.append(lookahead_quat[-1])

        lookahead_pos = np.array(lookahead_pos)
        lookahead_quat = np.array(lookahead_quat)

        # Ego-Centric Conversion
        r_drone = R.from_quat(drone_quat)
        r_drone_inv = r_drone.inv()

        rel_pos = lookahead_pos - drone_pos
        ego_gates_pos = r_drone_inv.apply(rel_pos)

        r_gates = R.from_quat(lookahead_quat)
        r_ego_gates = r_drone_inv * r_gates
        ego_gates_quat = r_ego_gates.as_quat()

        # Tensor Conversion & Normalization (keeping pure NumPy)
        norm_vel = vel / 10.0
        norm_ang_vel = ang_vel / 10.0
        norm_gates_pos = ego_gates_pos.flatten() / 10.0
        flat_gates_quat = ego_gates_quat.flatten()

        # Final Concatenation
        state = np.concatenate(
            [drone_quat, norm_vel, norm_ang_vel, norm_gates_pos, flat_gates_quat]
        ).astype(np.float32)

        return state


class DroneActionWrapper(gym.ActionWrapper):
    """
    Replaces the custom `scale_action` method.
    SB3 PPO outputs actions in [-1, 1]. This maps them to the physical drone bounds.
    """

    def __init__(self, env):
        super().__init__(env)
        self.action_low = np.array(
            [-1.5707964, -1.5707964, -1.5707964, 0.08545052], dtype=np.float32
        )
        self.action_high = np.array(
            [1.5707964, 1.5707964, 1.5707964, 0.8], dtype=np.float32
        )

        # Tell SB3 the policy is constrained to [-1, 1]
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )

    def action(self, act):
        normalized = (act + 1.0) / 2.0
        scaled_action = self.action_low + normalized * (
            self.action_high - self.action_low
        )
        return scaled_action


# ==========================================
# 2. MAIN TRAINING SCRIPT
# ==========================================


def make_wrapped_env(config, render=False):
    """Helper to create and wrap the environment."""
    # create_env is your original function from the previous script
    env = create_env(config, render=render)

    # Apply our custom wrappers
    env = DroneObservationWrapper(env, window_size=3)
    env = DroneActionWrapper(env)

    # SB3 requires Monitor to track episode returns and lengths
    env = Monitor(env)
    return env


def train(
    config_file: str = "level0_baby_steps.toml",
    iterations: int = 500,
    val_freq: int = 1,
):
    """Train the Drone using Stable Baselines3."""

    config = load_config(Path(__file__).parents[1] / "config" / config_file)

    def make_env():
        return make_wrapped_env(config, render=False)

    # 1. Setup Environments
    train_env = make_wrapped_env(config, render=False)

    # Validation env should stay as 1 single environment
    val_env = make_wrapped_env(config, render=True)

    # 2. Replicate Custom Actor-Critic Architecture
    # pi = Actor, vf = Critic. This perfectly matches your custom PyTorch class.
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        log_std_init=-0.5,  # Your std_init
    )

    # 3. Initialize PPO
    model = PPO(
        "MlpPolicy",
        env=train_env,
        n_steps=128,  # From your manual loop
        batch_size=1024,  # From your manual loop
        n_epochs=10,  # From your manual loop
        learning_rate=4e-4,  # From your manual loop
        clip_range=0.2,  # From your manual loop
        ent_coef=0.01,  # From your manual loop
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log="./ppo_drone_tensorboard/",
    )

    # 4. Setup Evaluation Callback
    # Evaluates the model periodically and saves the best version automatically
    eval_callback = EvalCallback(
        val_env,
        best_model_save_path="./logs/best_model",
        log_path="./logs/results",
        eval_freq=val_freq * 16,  # Note: eval_freq is in timesteps, not episodes
        deterministic=True,  # Uses action_dist.mean() just like your script
        render=True,
    )

    # 5. Train!
    # SB3 counts in total timesteps, not loop iterations.
    total_timesteps = iterations * 16
    print(f"Starting training for {total_timesteps} timesteps...")

    model.learn(
        total_timesteps=total_timesteps, callback=eval_callback, progress_bar=True
    )

    print("Training complete! Best model saved to ./logs/best_model")
    train_env.close()
    val_env.close()


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger("lsy_drone_racing").setLevel(logging.INFO)
    logger.setLevel(logging.INFO)

    fire.Fire(train, serialize=lambda _: None)
