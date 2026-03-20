"""A naive RL pipeline for drone racing."""

import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import fire
import gymnasium as gym
import jax
import jax.numpy as jp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from crazyflow.envs.norm_actions_wrapper import NormalizeActions
from gymnasium import spaces
from gymnasium.spaces import flatten_space
from gymnasium.vector import VectorEnv, VectorObservationWrapper, VectorRewardWrapper, VectorWrapper
from gymnasium.vector.utils import batch_space
from gymnasium.wrappers.vector.jax_to_torch import JaxToTorch
from jax import Array
from jax.scipy.spatial.transform import Rotation as R
from ml_collections import ConfigDict
from torch import Tensor
from torch.distributions.normal import Normal

from lsy_drone_racing.envs.drone_race import VecDroneRaceEnv
from lsy_drone_racing.utils import load_config


# region Arguments
@dataclass
class Args:
    """Class to store configurations."""

    seed: int = 42
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    jax_device: str = "gpu"
    """environment device"""
    wandb_project_name: str = "ADR-PPO-Racing"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""

    # Algorithm specific arguments
    total_timesteps: int = 15_000_000
    """total timesteps of the experiments"""
    learning_rate: float = 1.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1024
    """the number of parallel game environments"""

    # FIX 1: num_steps 8 -> 128
    # With num_steps=8, GAE cannot propagate gate-pass bonus signal backward
    # through a window shorter than the inter-gate distance in timesteps.
    # This was the root cause of explained_variance ~0 and gate bonus dying at 250k.
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""

    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""

    # FIX 2: gamma 0.94 -> 0.99
    # With gamma=0.94 and num_steps=8, effective horizon was ~17 steps.
    # This cuts off most of the episode return signal.
    gamma: float = 0.99
    """the discount factor gamma"""

    gae_lambda: float = 0.97
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 8
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.26
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""

    # FIX 3: ent_coef 0.05 -> 0.15
    # Entropy was collapsing too fast (3.5 -> 0.5 over 3M steps) while critic
    # was still broken (explained_variance ~0). Higher entropy keeps the policy
    # exploring while the value function catches up.
    # MonoRace uses ent_coef=0.003-0.005. Set to 0.005 (M17 value).
    ent_coef: float = 0.005
    """coefficient of the entropy"""

    vf_coef: float = 0.7
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

    checkpoint_freq: float = 0.1
    """checkpoint saving frequency (fraction of total iterations)"""
    resume: bool = False
    """whether to resume training from the latest checkpoint"""

    # Wrapper settings
    n_obs: int = 2
    rpy_coef: float = 0.06
    d_act_th_coef: float = 0.01
    d_act_xy_coef: float = 0.05
    act_coef: float = 0.001
    look_at_coef: float = 0.05
    global_scale = 0.01
    """reward coefficients for training"""

    @staticmethod
    def create(**kwargs: Any) -> "Args":
        """Create arguments class."""
        args = Args(**kwargs)
        args.batch_size = int(args.num_envs * args.num_steps)
        args.minibatch_size = int(args.batch_size // args.num_minibatches)
        args.num_iterations = args.total_timesteps // args.batch_size
        return args


# region Environment
class RaceTrainEnv(VecDroneRaceEnv):
    """Vectorized drone racing environment for RL training with compact ego-centric observations."""

    def __init__(self, **kwargs):
        """Init – delegates to parent then overrides the observation space."""
        super().__init__(**kwargs)
        self.prev_target_gate = jp.zeros((self.num_envs, 1), dtype=int)
        self._prev_dist_to_gate = jp.ones((self.num_envs,)) * 100.0  # large init distance
        obs_spec = {
            "rpy":            spaces.Box(-np.pi, np.pi,  shape=(3,), dtype=np.float32),
            "vel_body":       spaces.Box(-5.0,   5.0,    shape=(3,), dtype=np.float32),
            "dist_target":    spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32),
            "dist_next":      spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32),
            "gate_vec_body":  spaces.Box(-1.0,  1.0,   shape=(3,), dtype=np.float32),
            "gate_alignment": spaces.Box(-np.pi, np.pi,  shape=(1,), dtype=np.float32),
        }
        self.single_observation_space = spaces.Dict(obs_spec)
        self.observation_space = batch_space(self.single_observation_space, self.num_envs)

    def obs(self) -> dict[str, Array]:
        """Compact ego-centric observation."""
        base = super().obs()

        drone_pos  = base["pos"][:, 0, :]
        drone_quat = base["quat"][:, 0, :]
        vel_world  = base["vel"][:, 0, :]

        body_R = R.from_quat(drone_quat)
        rpy = body_R.as_euler("xyz")
        yaw = rpy[:, 2]
        vel_body = body_R.inv().apply(vel_world)

        n_gates    = len(self.gates["pos"])
        gates_pos  = base["gates_pos"][:, 0, :, :]
        gates_quat = base["gates_quat"][:, 0, :, :]

        target_idx = jp.clip(base["target_gate"][:, 0], 0, n_gates - 1)
        next_idx   = jp.clip(target_idx + 1, 0, n_gates - 1)

        env_ids = jp.arange(self.num_envs)
        target_gate_pos  = gates_pos[env_ids, target_idx]
        next_gate_pos    = gates_pos[env_ids, next_idx]
        target_gate_quat = gates_quat[env_ids, target_idx]

        dist_target_raw = jp.linalg.norm(drone_pos - target_gate_pos, axis=-1, keepdims=True)
        dist_next_raw   = jp.linalg.norm(drone_pos - next_gate_pos,   axis=-1, keepdims=True)
        dist_target = 2.0 * jp.exp(-2.0 * dist_target_raw) - 1.0
        dist_next   = 2.0 * jp.exp(-2.0 * dist_next_raw) - 1.0

        gate_vec_world = target_gate_pos - drone_pos
        gate_vec_body_raw = body_R.inv().apply(gate_vec_world)
        gate_vec_body = gate_vec_body_raw / (jp.linalg.norm(gate_vec_body_raw, axis=-1, keepdims=True) + 1e-6)

        local_x     = jp.broadcast_to(jp.array([1.0, 0.0, 0.0]), (self.num_envs, 3))
        gate_normal = R.from_quat(target_gate_quat).apply(local_x)
        gate_yaw    = jp.arctan2(gate_normal[:, 1], gate_normal[:, 0])
        align_diff  = gate_yaw - yaw
        gate_alignment = jp.arctan2(jp.sin(align_diff), jp.cos(align_diff))[:, None]

        return {
            "rpy":            rpy[:, None, :],
            "vel_body":       vel_body[:, None, :],
            "dist_target":    dist_target[:, None, :],
            "dist_next":      dist_next[:, None, :],
            "gate_vec_body":  gate_vec_body[:, None, :],
            "gate_alignment": gate_alignment[:, None, :],
        }

    def _reset(self, mask: Array | None = None, **kwargs) -> tuple[dict, dict]:
        """Reset the environment and the target gate tracker."""
        obs, info = super()._reset(mask=mask, **kwargs)
        n_gates = len(self.gates["pos"])
        clamped = jp.clip(self.data.target_gate[:, 0], 0, n_gates - 1)
        gate_pos = self.sim.mjx_data.mocap_pos[jp.arange(self.num_envs), self.data.gate_mj_ids[clamped]]
        drone_pos = self.sim.data.states.pos[:, 0, :]
        dist_now = jp.linalg.norm(drone_pos - gate_pos, axis=-1)
        if mask is None:
            self.prev_target_gate = self.data.target_gate
            self._prev_dist_to_gate = dist_now
        else:
            self.prev_target_gate = jp.where(mask[..., None], self.data.target_gate, self.prev_target_gate)
            self._prev_dist_to_gate = jp.where(mask, dist_now, self._prev_dist_to_gate)
        return obs, info

    def _step(self, action: Array) -> tuple[dict, Array, Array, Array, dict]:
        """Step the environment, update trackers, log reward decomposition."""
        obs, reward, terminated, truncated, info = super()._step(action)

        # --- REWARD DECOMPOSITION FOR LOGGING ---
        n_gates = len(self.gates["pos"])
        clamped = jp.clip(self.data.target_gate[:, 0], 0, n_gates - 1)
        gate_pos = self.sim.mjx_data.mocap_pos[jp.arange(self.num_envs), self.data.gate_mj_ids[clamped]]
        drone_pos = self.sim.data.states.pos[:, 0, :]
        dist_now = jp.linalg.norm(drone_pos - gate_pos, axis=-1)  # (num_envs,)

        progress = jp.clip(self._prev_dist_to_gate - dist_now, -jp.inf, 2.0)  # (num_envs,)

        # Squeeze passed to (num_envs,) — target_gate is (num_envs,1) so comparison
        # gives (num_envs,1); without squeeze it broadcasts wrong with is_crashed.
        passed = ((self.data.target_gate > self.prev_target_gate) | (
            (self.prev_target_gate == n_gates - 1) & (self.data.target_gate == -1)
        ))[:, 0]  # (num_envs,)

        gate_bonus = jp.where(passed, 10.0, 0.0)  # (num_envs,)

        omega = self.sim.data.states.ang_vel[:, 0, :]
        rate_penalty = -0.001 * jp.sum(omega ** 2, axis=-1)  # (num_envs,)

        disabled = self.data.disabled_drones[:, 0]             # (num_envs,)
        is_finished = self.data.target_gate[:, 0] == -1        # (num_envs,)
        is_crashed = disabled & ~is_finished                    # (num_envs,)
        crash_penalty = jp.where(is_crashed, 0.0, 0.0)       # (num_envs,) — no crash penalty

        # Add [:, None] so parent's info = {k: v[:, 0] ...} slice works correctly
        info["reward_progress"] = jp.where(is_crashed, 0.0, progress)[:, None]
        info["reward_gate"] = jp.where(is_crashed, 0.0, gate_bonus)[:, None]
        info["reward_rate_penalty"] = rate_penalty[:, None]
        info["penalty_crash"] = crash_penalty[:, None]
        info["dist_to_gate"] = dist_now[:, None]
        # --- END DECOMPOSITION ---

        # Update trackers — reset dist to large value when gate is passed
        self._prev_dist_to_gate = jp.where(passed, 100.0, dist_now)
        self.prev_target_gate = self.data.target_gate
        return obs, reward, terminated, truncated, info

    def reward(self) -> Array:
        """Distance-reduction progress + gate bonus. No survival bonus (causes hover optimum).
        
        Napkin check:
          Hover 500 steps:              progress=0,   gate=0  → total=0
          Approach gate 0.5m/s 500steps: progress=+25, gate=0  → total=+25
          Pass gate:                     progress=+25, gate=+10 → total=+35
        Hovering gives 0. Approaching dominates. No local optimum.
        Crash ends episode → loses all future progress reward. Implicit penalty.
        """
        n_gates = len(self.gates["pos"])
        clamped = jp.clip(self.data.target_gate[:, 0], 0, n_gates - 1)

        gate_pos = self.sim.mjx_data.mocap_pos[jp.arange(self.num_envs), self.data.gate_mj_ids[clamped]]
        drone_pos = self.sim.data.states.pos[:, 0, :]
        dist_now = jp.linalg.norm(drone_pos - gate_pos, axis=-1)

        progress = 5.0 * jp.clip(self._prev_dist_to_gate - dist_now, -jp.inf, 2.0)

        passed = ((self.data.target_gate > self.prev_target_gate) | (
            (self.prev_target_gate == n_gates - 1) & (self.data.target_gate == -1)
        ))[:, 0]

        reward = progress + jp.where(passed, 10.0, 0.0)

        disabled = self.data.disabled_drones[:, :1]
        is_finished = self.data.target_gate == -1
        return jp.where(disabled & ~is_finished, 0.0, reward)


# region Wrappers
class StackObs(VectorObservationWrapper):
    """Wrapper to stack history observations."""

    def __init__(self, env: VectorEnv, n_obs: int = 0):
        super().__init__(env)
        self.n_obs = n_obs
        if self.n_obs > 0:
            spec = {k: v for k, v in self.single_observation_space.items()}
            spec["prev_obs"] = spaces.Box(-np.inf, np.inf, shape=(6 * self.n_obs,))
            self.single_observation_space = spaces.Dict(spec)
            self.observation_space = batch_space(self.single_observation_space, self.num_envs)
            init_obs = env.unwrapped.obs()
            self._prev_obs = jp.zeros((self.num_envs, self.n_obs, 6))
            for _ in range(n_obs):
                self._prev_obs = self._update_prev_obs(self._prev_obs, init_obs)

    def observations(self, observations: dict) -> dict:
        if self.n_obs > 0:
            observations["prev_obs"] = self._prev_obs.reshape(self.num_envs, -1)
            self._prev_obs = self._update_prev_obs(self._prev_obs, observations)
        return observations

    @staticmethod
    @jax.jit
    def _update_prev_obs(prev_obs: Array, obs: dict) -> Array:
        basic_obs_key = ["rpy", "vel_body"]
        basic_obs = jp.concatenate(
            [jp.reshape(obs[k], (obs[k].shape[0], -1)) for k in basic_obs_key], axis=-1
        )
        prev_obs = jp.concatenate([prev_obs[:, 1:, :], basic_obs[:, None, :]], axis=1)
        return prev_obs


class ActionPenalty(VectorObservationWrapper):
    """Wrapper to apply action penalty.

    FIX 4: Now accepts a penalty_scale in [0, 1] that is updated externally
    by the training loop to implement the penalty curriculum. Set to 0.0 at
    init so penalties have no effect early in training.
    """

    def __init__(
        self,
        env: VectorEnv,
        act_coef: float = 0.01,
        d_act_th_coef: float = 0.2,
        d_act_xy_coef: float = 0.4,
    ):
        super().__init__(env)
        spec = {k: v for k, v in self.single_observation_space.items()}
        spec["last_action"] = spaces.Box(-np.inf, np.inf, shape=(4,))
        self.single_observation_space = spaces.Dict(spec)
        self.observation_space = batch_space(self.single_observation_space, self.num_envs)
        self._last_action = jp.zeros((self.num_envs, 4))
        self.act_coef = act_coef
        self.d_act_th_coef = d_act_th_coef
        self.d_act_xy_coef = d_act_xy_coef
        # Curriculum scale: updated externally by training loop
        self.penalty_scale = 0.0

    def step(self, action: Array) -> tuple[Array, Array, Array, Array, dict]:
        obs, reward, terminated, truncated, info = super().step(action)
        action_diff = action - self._last_action

        pen_energy = -self.act_coef * action[..., -1] ** 2
        pen_smooth_th = -self.d_act_th_coef * action_diff[..., -1] ** 2
        pen_smooth_xy = -self.d_act_xy_coef * jp.sum(action_diff[..., :3] ** 2, axis=-1)

        # Apply curriculum scale: penalties ramp in gradually
        reward = reward + self.penalty_scale * (pen_energy + pen_smooth_th + pen_smooth_xy)

        info["penalty_energy"] = pen_energy
        info["penalty_smooth_th"] = pen_smooth_th
        info["penalty_smooth_xy"] = pen_smooth_xy
        info["penalty_scale"] = self.penalty_scale

        self._last_action = action
        return self.observations(obs), reward, terminated, truncated, info

    def observations(self, observations: dict) -> dict:
        observations["last_action"] = self._last_action
        return observations


class LookAtPenalty(VectorObservationWrapper):
    """Wrapper to apply penalty if drone is not looking at the gate.

    FIX 4: Also accepts penalty_scale for curriculum.
    """

    def __init__(self, env: VectorEnv, look_at_coef: float = 0.05):
        super().__init__(env)
        self.look_at_coef = look_at_coef
        self.penalty_scale = 0.0

    def step(self, action: Array) -> tuple[Array, Array, Array, Array, dict]:
        obs, reward, terminated, truncated, info = super().step(action)

        gate_vec_body = obs["gate_vec_body"]
        if gate_vec_body.ndim == 3:
            gate_vec_body = gate_vec_body[:, 0, :]

        cos_theta = gate_vec_body[:, 0]
        angle = jp.acos(jp.clip(cos_theta, -1.0, 1.0))

        pen_look = -jp.where(angle > (jp.pi / 3), self.look_at_coef, 0.0)
        reward = reward + self.penalty_scale * pen_look

        info["penalty_look_at"] = pen_look

        return self.observations(obs), reward, terminated, truncated, info

    def observations(self, observations: dict) -> dict:
        return observations


class FlattenJaxObservation(VectorObservationWrapper):
    """Wrapper to flatten the observations."""

    def __init__(self, env: VectorEnv):
        super().__init__(env)
        self.single_observation_space = flatten_space(env.single_observation_space)
        self.observation_space = flatten_space(env.observation_space)

    def observations(self, observations: dict) -> dict:
        return jp.concatenate(
            [jp.reshape(v, (v.shape[0], -1)) for k, v in observations.items()], axis=-1
        )


class RunningMeanStd:
    """Tracks the running mean and variance of a data stream."""
    def __init__(self, epsilon: float = 1e-4, shape: tuple = ()):
        self.mean = jp.zeros(shape, "float32")
        self.var = jp.ones(shape, "float32")
        self.count = epsilon

    def update(self, x: Array):
        batch_mean = jp.mean(x, axis=0)
        batch_var = jp.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jp.square(delta) * self.count * batch_count / tot_count
        self.mean = new_mean
        self.var = M2 / tot_count
        self.count = tot_count


class VecNormalize(VectorWrapper):
    """Vectorized wrapper that normalizes observations and returns."""

    def __init__(
        self, venv: VectorEnv, ob=True, ret=True, clipob=10.0, cliprew=10.0, gamma=0.99, epsilon=1e-8, training=True
    ):
        super().__init__(venv)
        self.ob_rms = RunningMeanStd(shape=self.single_observation_space.shape) if ob else None
        self.ret_rms = RunningMeanStd(shape=()) if ret else None
        self.clipob = clipob
        self.cliprew = cliprew
        self.ret = jp.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon
        self.training = training

    def step(self, actions):
        obs, rews, terminations, truncations, infos = self.env.step(actions)
        dones = terminations | truncations
        infos["reward_raw"] = rews
        self.ret = self.ret * self.gamma + rews
        obs = self._obfilt(obs)
        if self.ret_rms:
            if self.training:
                self.ret_rms.update(self.ret)
            rews = jp.clip(rews / jp.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
        self.ret = jp.where(dones, 0.0, self.ret)
        return obs, rews, terminations, truncations, infos

    def _obfilt(self, obs):
        if self.ob_rms:
            if self.training:
                self.ob_rms.update(obs)
            obs = jp.clip((obs - self.ob_rms.mean) / jp.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.ret = jp.zeros(self.num_envs)
        return self._obfilt(obs), info

    def get_stats(self):
        return {
            "ob_rms_mean": self.ob_rms.mean if self.ob_rms else None,
            "ob_rms_var": self.ob_rms.var if self.ob_rms else None,
            "ob_rms_count": self.ob_rms.count if self.ob_rms else None,
            "ret_rms_mean": self.ret_rms.mean if self.ret_rms else None,
            "ret_rms_var": self.ret_rms.var if self.ret_rms else None,
            "ret_rms_count": self.ret_rms.count if self.ret_rms else None,
        }

    def set_stats(self, stats):
        if self.ob_rms and stats.get("ob_rms_mean") is not None:
            self.ob_rms.mean = jp.array(stats["ob_rms_mean"])
            self.ob_rms.var = jp.array(stats["ob_rms_var"])
            self.ob_rms.count = stats["ob_rms_count"]
        if self.ret_rms and stats.get("ret_rms_mean") is not None:
            self.ret_rms.mean = jp.array(stats["ret_rms_mean"])
            self.ret_rms.var = jp.array(stats["ret_rms_var"])
            self.ret_rms.count = stats["ret_rms_count"]


def get_vec_normalize(env: VectorEnv) -> VecNormalize | None:
    while hasattr(env, "env"):
        if isinstance(env, VecNormalize):
            return env
        env = env.env
    return None


def get_action_penalty(env: VectorEnv) -> ActionPenalty | None:
    """Walk wrapper stack to find ActionPenalty for curriculum updates."""
    while hasattr(env, "env"):
        if isinstance(env, ActionPenalty):
            return env
        env = env.env
    return None


def get_look_at_penalty(env: VectorEnv) -> LookAtPenalty | None:
    """Walk wrapper stack to find LookAtPenalty for curriculum updates."""
    while hasattr(env, "env"):
        if isinstance(env, LookAtPenalty):
            return env
        env = env.env
    return None


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# region MakeEnvs
def make_envs(
    config: str = "level0.toml",
    num_envs: int = None,
    jax_device: str = "cpu",
    torch_device: torch.device = torch.device("cpu"),
    coefs: dict = {},
) -> VectorEnv:
    cfg = load_config(Path(__file__).parents[2] / "config" / config)
    env = RaceTrainEnv(
        num_envs=num_envs,
        freq=cfg.env.freq,
        sim_config=cfg.sim,
        track=cfg.env.track,
        sensor_range=cfg.env.sensor_range,
        control_mode=cfg.env.control_mode,
        disturbances=cfg.env.disturbances,
        device=jax_device,
    )

    env = NormalizeActions(env)
    # MonoRace perception penalty: -0.05 * θ if θ > π/3. No curriculum — full scale always.
    look_at = LookAtPenalty(env, look_at_coef=coefs.get("look_at_coef", 0.05))
    look_at.penalty_scale = 1.0
    env = look_at
    env = StackObs(env, n_obs=coefs.get("n_obs", 0))
    # No ActionPenalty wrapper — motor smoothness is handled inside the reward function
    # via the angular rate penalty (MonoRace style), not as a wrapper.

    env = FlattenJaxObservation(env)
    env = VecNormalize(env, training=coefs.get("training", True), gamma=0.99)
    env = JaxToTorch(env, torch_device)
    return env


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# region Agent
class Agent(nn.Module):
    """RL Agent."""

    def __init__(self, obs_shape: tuple, action_shape: tuple):
        super().__init__()
        obs_dim = torch.tensor(obs_shape).prod()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )
        # Initialize final actor layer with near-zero weights but positive thrust bias.
        # Without this, zero-mean policy outputs zero thrust → drone falls before learning anything.
        # atanh(0.5) ≈ 0.55 → after Tanh → 0.5 normalized thrust → roughly hover.
        # RPY biases stay 0 (no initial rotation preference).
        final_actor_layer = nn.Linear(256, torch.tensor(action_shape).prod())
        torch.nn.init.orthogonal_(final_actor_layer.weight, 0.01)
        torch.nn.init.constant_(final_actor_layer.bias, 0.0)
        final_actor_layer.bias.data[3] = 0.55  # thrust dimension → Tanh(0.55) ≈ 0.5
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            final_actor_layer,
            nn.Tanh(),
        )
        self.actor_logstd = nn.Parameter(
            torch.Tensor([[-1, -1, -1, 1]])
        )

    def get_value(self, x: Tensor) -> Tensor:
        return self.critic(x)

    def get_action_and_value(
        self, x: Tensor, action: Tensor | None = None, deterministic: bool = False
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample() if not deterministic else action_mean
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


# region Train
def train_ppo(
    args: Args, device: torch.device, jax_device: str, wandb_enabled: bool = False
) -> None:
    if wandb_enabled and wandb.run is None:
        wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, config=vars(args))
    train_start_time = time.time()
    set_seeds(args.seed)
    print("Training on device:", device, "| Environment device:", jax_device)

    r_coefs = {
        "n_obs": args.n_obs,
        "rpy_coef": args.rpy_coef,
        "d_act_xy_coef": args.d_act_xy_coef,
        "d_act_th_coef": args.d_act_th_coef,
        "act_coef": args.act_coef,
        "look_at_coef": args.look_at_coef,
        "global_scale": args.global_scale,
    }
    envs = make_envs(
        num_envs=args.num_envs, jax_device=jax_device, torch_device=device, coefs=r_coefs
    )
    vec_norm = get_vec_normalize(envs)

    assert isinstance(envs.single_action_space, gym.spaces.Box), (
        "only continuous action space is supported"
    )

    agent = Agent(envs.single_observation_space.shape, envs.single_action_space.shape).to(device)
    optimizer = optim.AdamW(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # region Checkpoint Handling
    checkpoint_dir = Path(__file__).parent / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    start_iteration = 1
    global_step = 0
    best_mean_reward = -float("inf")

    if args.resume:
        latest_checkpoint = checkpoint_dir / "latest.ckpt"
        if not latest_checkpoint.exists():
            latest_checkpoint = checkpoint_dir / "checkpoint.ckpt"

        if latest_checkpoint.exists():
            print(f"Resuming from {latest_checkpoint}")
            checkpoint = torch.load(latest_checkpoint, map_location=device, weights_only=False)
            agent.load_state_dict(checkpoint["agent_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if vec_norm and "vec_normalize_stats" in checkpoint:
                vec_norm.set_stats(checkpoint["vec_normalize_stats"])
            global_step = checkpoint["global_step"]
            start_iteration = checkpoint["iteration"] + 1
            best_mean_reward = checkpoint.get("best_mean_reward", -float("inf"))
        else:
            print(f"No checkpoint found at {latest_checkpoint} to resume from.")

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    # Setup accumulators for logging
    sum_rewards = torch.zeros((args.num_envs)).to(device)
    sum_rewards_raw = torch.zeros((args.num_envs)).to(device)
    sum_reward_progress = torch.zeros((args.num_envs)).to(device)
    sum_reward_gate = torch.zeros((args.num_envs)).to(device)
    sum_penalty_crash = torch.zeros((args.num_envs)).to(device)

    sum_rewards_hist = []

    for iteration in range(start_iteration, args.num_iterations + 1):
        start_time = time.time()

        # Entropy annealing: decays 90% over training (MonoRace uses 0.003-0.005, fixed).
        ent_coef_now = args.ent_coef * (1.0 - (iteration / args.num_iterations) * 0.9)

        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # No penalty curriculum — MonoRace uses fixed reward coefficients from step 1.
        # LookAtPenalty is always at scale 1.0.

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, terminations, truncations, infos = envs.step(action)
            rewards[step] = reward
            sum_rewards += reward

            # Accumulate info metrics
            if "reward_raw" in infos:
                sum_rewards_raw += torch.as_tensor(infos["reward_raw"]).to(device)
            if "reward_progress" in infos:
                sum_reward_progress += torch.as_tensor(infos["reward_progress"]).to(device)
                sum_reward_gate += torch.as_tensor(infos["reward_gate"]).to(device)
                sum_penalty_crash += torch.as_tensor(infos["penalty_crash"]).to(device)

            sum_rewards_hist.extend(sum_rewards[next_done.bool()].tolist())

            if wandb_enabled and next_done.any():
                done_idx = next_done.bool()
                wandb.log(
                    {
                        "train/reward_total": sum_rewards[done_idx].mean().item(),
                        "train/reward_raw": sum_rewards_raw[done_idx].mean().item(),
                        "components/reward_progress": sum_reward_progress[done_idx].mean().item(),
                        "components/reward_gate_bonus": sum_reward_gate[done_idx].mean().item(),
                        "components/penalty_crash": sum_penalty_crash[done_idx].mean().item(),
                        "charts/ent_coef": ent_coef_now,
                    },
                    step=global_step,
                )

            sum_rewards[next_done.bool()] = 0
            sum_rewards_raw[next_done.bool()] = 0
            sum_reward_progress[next_done.bool()] = 0
            sum_reward_gate[next_done.bool()] = 0
            sum_penalty_crash[next_done.bool()] = 0

            next_done = terminations | truncations

        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done.float()
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef_now * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        if wandb_enabled:
            wandb.log(
                {
                    "charts/learning_rate": optimizer.param_groups[0]["lr"],
                    "losses/value_loss": v_loss.item(),
                    "losses/policy_loss": pg_loss.item(),
                    "losses/entropy": entropy_loss.item(),
                    "losses/old_approx_kl": old_approx_kl.item(),
                    "losses/approx_kl": approx_kl.item(),
                    "losses/clipfrac": np.mean(clipfracs),
                    "losses/explained_variance": explained_var,
                    "charts/SPS": int(global_step / (time.time() - start_time)),
                    "charts/ent_coef": ent_coef_now,
                },
                step=global_step,
            )

        if iteration % max(1, int(args.num_iterations * args.checkpoint_freq)) == 0:
            avg_reward = (
                np.mean(sum_rewards_hist[-100:]) if sum_rewards_hist else -float("inf")
            )
            latest_path = checkpoint_dir / "latest.ckpt"
            state = {
                "agent_state_dict": agent.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "vec_normalize_stats": vec_norm.get_stats() if vec_norm else None,
                "iteration": iteration,
                "global_step": global_step,
                "best_mean_reward": best_mean_reward,
            }
            torch.save(state, latest_path)
            print(f"Latest checkpoint saved to {latest_path} (Average Reward: {avg_reward:.2f})")

            if avg_reward > best_mean_reward:
                best_mean_reward = avg_reward
                best_path = checkpoint_dir / "best.ckpt"
                torch.save(state, best_path)
                print(f"New best model saved to {best_path} (Reward: {best_mean_reward:.2f})")

        end_time = time.time()
        print(f"Iter {iteration}/{args.num_iterations} took {end_time - start_time:.2f} seconds")
    train_end_time = time.time()
    print(f"Training for {global_step} steps took {train_end_time - train_start_time:.2f} seconds.")
    envs.close()

    return sum_rewards_hist


# region Evaluate
def evaluate_ppo(args: Args, n_eval: int) -> tuple[float, float]:
    set_seeds(args.seed)
    device = torch.device("cpu")
    r_coefs = {
        "n_obs": args.n_obs,
        "rpy_coef": args.rpy_coef,
        "d_act_xy_coef": args.d_act_xy_coef,
        "d_act_th_coef": args.d_act_th_coef,
        "act_coef": args.act_coef,
        "look_at_coef": args.look_at_coef,
        "training": False,
    }
    eval_env = make_envs(num_envs=1, coefs=r_coefs)
    vec_norm = get_vec_normalize(eval_env)

    agent = Agent(eval_env.single_observation_space.shape, eval_env.single_action_space.shape).to(
        device
    )
    checkpoint_dir = Path(__file__).parent / "checkpoints"
    best_path = checkpoint_dir / "best.ckpt"
    latest_path = checkpoint_dir / "latest.ckpt"
    old_path = checkpoint_dir / "checkpoint.ckpt"

    if best_path.exists():
        checkpoint_path = best_path
    elif latest_path.exists():
        checkpoint_path = latest_path
    elif old_path.exists():
        checkpoint_path = old_path
    else:
        print("No checkpoint found in checkpoints/")
        return [], []

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "agent_state_dict" in checkpoint:
        agent.load_state_dict(checkpoint["agent_state_dict"])
        if vec_norm and "vec_normalize_stats" in checkpoint:
            vec_norm.set_stats(checkpoint["vec_normalize_stats"])
    else:
        agent.load_state_dict(checkpoint)
    with torch.no_grad():
        episode_rewards = []
        episode_lengths = []
        ep_seed = args.seed
        for episode in range(n_eval):
            obs, _ = eval_env.reset(seed=(ep_seed := ep_seed + 1))
            done = torch.zeros(10, dtype=bool, device=device)
            episode_reward = 0
            steps = 0
            while not done.any():
                act, _, _, _ = agent.get_action_and_value(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(act)
                eval_env.render()
                done = terminated | truncated
                episode_reward += reward[0].item()
                steps += 1
            episode_rewards.append(episode_reward)
            episode_lengths.append(steps)
            print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Length = {steps}")

        print(
            f"Average Reward = {np.mean(episode_rewards):.2f}, Length = {np.mean(episode_lengths)}"
        )
        eval_env.close()

        return episode_rewards, episode_lengths


# region Main
def main(
    wandb_enabled: bool = True,
    train: bool = True,
    eval: int = 1,
    checkpoint_freq: float = 0.1,
    resume: bool = False,
):
    args = Args.create(checkpoint_freq=checkpoint_freq, resume=resume)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    jax_device = args.jax_device

    if train:
        train_ppo(args, device, jax_device, wandb_enabled)

    if eval > 0:
        episode_rewards, episode_lengths = evaluate_ppo(args, eval)
        if wandb_enabled and train:
            wandb.log(
                {
                    "eval/mean_rewards": np.mean(episode_rewards),
                    "eval/mean_steps": np.mean(episode_lengths),
                }
            )
            wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)