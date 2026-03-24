import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import random
import time
from collections import deque
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
# Pre-fetch utils
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
    total_timesteps: int = 150_000_000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 512
    """the number of parallel game environments"""
    num_steps: int = 2048
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.995
    """the discount factor gamma"""
    gae_lambda: float = 0.97
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.20
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.007
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
    d_act_th_coef: float = 0.01
    d_act_xy_coef: float = 0.01
    act_coef: float = 0.01
    look_at_coef: float = 0.01
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
    """Vectorized drone racing environment for RL training with compact ego-centric observations.

    Compact observation (19 values total):
        rpy (3)                 – roll, pitch, yaw of the drone (radians)
        vel_body (3)            – velocity in the drone's body frame
        ang_vel (3)             – angular velocity in the world frame
        dist_target (1)         – exponential distance to target: 2*exp(-2*dist)-1 (mapped to [-1, 1])
        dist_next (1)           – exponential distance to next gate: 2*exp(-2*dist)-1 (mapped to [-1, 1])
        gate_vec_body (3)       – unit vector from drone to target gate in body frame (range [-1, 1])
        gate_alignment (1)      – angle between target gate's normal and drone yaw
        gate_vec_body_next (3)  – unit vector from drone to next gate in body frame (range [-1, 1])
        gate_alignment_next (1) – angle between next gate's normal and drone yaw
    """

    def __init__(self, **kwargs):
        """Init – delegates to parent then overrides the observation space."""
        super().__init__(**kwargs)
        self.autoreset = False  # Handle autoreset manually to fix termination swallowing
        self.prev_target_gate = self.data.target_gate[:, 0]
        self.segment_dist = jp.zeros((self.num_envs,))
        self.prev_dist = jp.zeros((self.num_envs,))
        obs_spec = {
            # Euler angles: always in [-pi, pi]
            "rpy":                 spaces.Box(-np.pi, np.pi,  shape=(3,), dtype=np.float32),
            # Velocities: CF2.1B hardware max ~2.5 m/s or rad/s; allow 5.0 for sim headroom
            "vel_body":            spaces.Box(-5.0,   5.0,    shape=(3,), dtype=np.float32),
            "ang_vel":             spaces.Box(-5.0,   5.0,    shape=(3,), dtype=np.float32),
            # Exponential distances: 2*exp(-2*dist)-1 maps [0, inf) to [-1, 1]
            "dist_target":         spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32),
            "dist_next":           spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32),
            # Body-frame unit vector to gate: [x,y,z] components in [-1, 1]
            "gate_vec_body":       spaces.Box(-1.0,  1.0,   shape=(3,), dtype=np.float32),
            "gate_alignment":      spaces.Box(-np.pi, np.pi,  shape=(1,), dtype=np.float32),
            # Next gate info
            "gate_vec_body_next":  spaces.Box(-1.0,  1.0,   shape=(3,), dtype=np.float32),
            "gate_alignment_next": spaces.Box(-np.pi, np.pi,  shape=(1,), dtype=np.float32),
        }
        self.single_observation_space = spaces.Dict(obs_spec)
        self.observation_space = batch_space(self.single_observation_space, self.num_envs)

    def obs(self) -> dict[str, Array]:
        """Compact ego-centric observation.

        All angles are in radians and wrapped to [-pi, pi].
        The drone body frame uses: x=forward, y=left, z=up.
        """
        base = super().obs()  # world-frame obs from RaceCoreEnv

        # ── raw state ──────────────────────────────────────────────────────────
        drone_pos  = base["pos"][:, 0, :]   # (N, 3)  world pos
        drone_quat = base["quat"][:, 0, :]  # (N, 4)  scipy (x,y,z,w)
        vel_world  = base["vel"][:, 0, :]   # (N, 3)  world-frame velocity
        ang_vel_world = base["ang_vel"][:, 0, :] # (N, 3) world-frame angular velocity

        body_R = R.from_quat(drone_quat)    # batch rotation object

        # 1. Roll / pitch / yaw
        rpy = body_R.as_euler("xyz")        # (N, 3) radians
        yaw = rpy[:, 2]                     # (N,)

        # 2. Velocity and Angular Velocity
        vel_body = body_R.inv().apply(vel_world)  # (N, 3) body frame
        ang_vel = base["ang_vel"][:, 0, :]        # (N, 3) world frame

        # ── gate geometry ──────────────────────────────────────────────────────
        n_gates    = len(self.gates["pos"])
        gates_pos  = base["gates_pos"][:, 0, :, :]   # (N, n_gates, 3)  squeeze drone dim
        gates_quat = base["gates_quat"][:, 0, :, :]  # (N, n_gates, 4)

        target_gate = base["target_gate"][:, 0]
        target_idx = jp.clip(target_gate, 0, n_gates - 1)  # (N,)
        next_idx   = jp.clip(target_idx + 1, 0, n_gates - 1)             # (N,) wraps at last gate

        env_ids = jp.arange(self.num_envs)
        target_gate_pos  = gates_pos[env_ids, target_idx]   # (N, 3)
        next_gate_pos    = gates_pos[env_ids, next_idx]     # (N, 3)
        target_gate_quat = gates_quat[env_ids, target_idx]  # (N, 4)
        next_gate_quat   = gates_quat[env_ids, next_idx]    # (N, 4)

        # 3. Exponential distance to target gate and the one after it: 2*exp(-2.0 * dist) - 1
        #    This maps [0, inf) to [-1, 1]. Close = 1.0, Far = -1.0.
        dist_target_raw = jp.linalg.norm(drone_pos - target_gate_pos, axis=-1, keepdims=True)
        dist_next_raw   = jp.linalg.norm(drone_pos - next_gate_pos,   axis=-1, keepdims=True)
        dist_target = 2.0 * jp.exp(-2.0 * dist_target_raw) - 1.0
        dist_next   = 2.0 * jp.exp(-2.0 * dist_next_raw) - 1.0

        # 4. Unit vector from drone to target and next gate expressed in the drone's body frame.
        def get_gate_vec_body(target_pos):
            vec_world = target_pos - drone_pos
            vec_body_raw = body_R.inv().apply(vec_world)
            return vec_body_raw / (jp.linalg.norm(vec_body_raw, axis=-1, keepdims=True) + 1e-6)

        gate_vec_body = get_gate_vec_body(target_gate_pos)
        gate_vec_body_next = get_gate_vec_body(next_gate_pos)

        # 5. Gate alignment: angle between gate's fly-through axis and drone yaw
        #    The gate's local +x axis is its normal (the axis you fly along to pass through).
        def get_gate_alignment(g_quat):
            local_x     = jp.broadcast_to(jp.array([1.0, 0.0, 0.0]), (self.num_envs, 3))
            gate_normal = R.from_quat(g_quat).apply(local_x)  # (N, 3)
            gate_yaw    = jp.arctan2(gate_normal[:, 1], gate_normal[:, 0])  # (N,) world bearing
            align_diff  = gate_yaw - yaw  # (N,)
            return jp.arctan2(jp.sin(align_diff), jp.cos(align_diff))[:, None]  # (N, 1)

        gate_alignment = get_gate_alignment(target_gate_quat)
        gate_alignment_next = get_gate_alignment(next_gate_quat)

        # ── pack with drone dim so VecDroneRaceEnv.step() can squeeze with [:, 0] ──
        return {
            "rpy":                 rpy[:, None, :],             # (N, 1, 3)
            "vel_body":            vel_body[:, None, :],        # (N, 1, 3)
            "ang_vel":             ang_vel[:, None, :],         # (N, 1, 3)
            "dist_target":         dist_target_raw[:, None, :],     # (N, 1, 1)
            "dist_next":           dist_next_raw[:, None, :],       # (N, 1, 1)
            "gate_vec_body":       gate_vec_body[:, None, :],   # (N, 1, 3)
            "gate_alignment":      gate_alignment[:, None, :],  # (N, 1, 1)
            "gate_vec_body_next":  gate_vec_body_next[:, None, :],   # (N, 1, 3)
            "gate_alignment_next": gate_alignment_next[:, None, :],  # (N, 1, 1)
        }

    def _reset(self, mask: Array | None = None, **kwargs) -> tuple[dict, dict]:
        """Reset the environment and the target gate tracker."""
        obs, info = super()._reset(mask=mask, **kwargs)
        target_gate = self.data.target_gate[:, 0]
        
        # Calculate initial segment distance (distance to gate 0 from start pos)
        drone_pos = self.sim.data.states.pos[:, 0, :]
        gate_pos = self.sim.mjx_data.mocap_pos[jp.arange(self.num_envs), self.data.gate_mj_ids[0]]
        dist = jp.linalg.norm(gate_pos - drone_pos, axis=-1)
        
        if mask is None:
            self.prev_target_gate = target_gate
            self.segment_dist = dist
            self.prev_dist = dist
        else:
            self.prev_target_gate = jp.where(mask, target_gate, self.prev_target_gate)
            self.segment_dist = jp.where(mask, dist, self.segment_dist)
            self.prev_dist = jp.where(mask, dist, self.prev_dist)
        return obs, info

    def _step(self, action: Array) -> tuple[dict, Array, Array, Array, dict]:
        """Step the environment and update the target gate tracker with manual autoreset."""
        # 1. Step the base (RaceCoreEnv._step) which now has autoreset=False
        obs, reward, terminated, truncated, info = super()._step(action)
        
        # 2. Fix Warping Lag: manually warp drones that just crashed in THIS step
        # Note: reward/terminated already reflect the latest self.data
        self.sim.data = self._warp_disabled_drones(self.sim.data, self.data.disabled_drones)

        # 3. Update tracker for reward calculation in NEXT step (using latest data)
        target_gate = self.data.target_gate[:, 0]
        passed = (target_gate > self.prev_target_gate) | (
            (self.prev_target_gate == len(self.gates["pos"]) - 1) & (target_gate == -1)
        )
        
        if passed.any():
            # Update segment_dist for the NEW target gate
            drone_pos = self.sim.data.states.pos[:, 0, :]
            n_gates = len(self.gates["pos"])
            clamped = jp.clip(target_gate, 0, n_gates - 1)
            gate_pos = self.sim.mjx_data.mocap_pos[jp.arange(self.num_envs), self.data.gate_mj_ids[clamped]]
            new_dist = jp.linalg.norm(gate_pos - drone_pos, axis=-1)
            # Only update for environments that just passed a gate
            self.segment_dist = jp.where(passed, new_dist, self.segment_dist)

        self.prev_target_gate = target_gate
        mask = self.data.marked_for_reset
        self.prev_dist = jp.where(mask, self.prev_dist, self._last_dist_raw)

        # 4. Manual autoreset: ensures we returned the terminal state BEFORE wiping it
        if mask.any():
            self._reset(mask=mask)

        # 5. Add reward components to info for logging
        if hasattr(self, "_last_reward_components"):
            info.update(self._last_reward_components)

        return obs, reward, terminated, truncated, info

    def reward(self) -> Array:
        """Velocity-projection (progress) reward with gate-pass bonus.

        reward = dot(vel, unit_vec_to_gate) + 100 * passed_gate

        Positive when moving toward the gate, negative when drifting away,
        Crash penalty = -10.0 when the drone is disabled.
        """
        n_gates = len(self.gates["pos"])
        target_gate = self.data.target_gate[:, 0]
        clamped = jp.clip(target_gate, 0, n_gates - 1)

        gate_pos = self.sim.mjx_data.mocap_pos[jp.arange(self.num_envs), self.data.gate_mj_ids[clamped]]
        drone_pos = self.sim.data.states.pos[:, 0, :]  # (N, 3)  from state
        drone_vel = self.sim.data.states.vel[:, 0, :]  # (N, 3)  from state — no prev needed
        to_gate = gate_pos - drone_pos
        dist_raw = jp.linalg.norm(to_gate, axis=-1, keepdims=False)
        self._last_dist_raw = dist_raw

        # Progress reward: change in distance towards goal
        progress = (self.prev_dist - dist_raw) * 100
        
        # Detect gate pass: target gate incremented OR reached end (target_gate == -1)
        passed = (target_gate > self.prev_target_gate) | (
            (self.prev_target_gate == n_gates - 1) & (target_gate == -1)
        )
        
        # Reset progress on gate pass to avoid massive reward jump from target change
        progress = jp.where(passed, 0.0, progress)
        progress = jp.clip(progress, -1.0, 1.0)

        gate_reward = jp.where(passed, 100.0, 0.0)
        reward = progress + gate_reward

        disabled = self.data.disabled_drones[:, 0]
        is_finished = target_gate == -1
        # Crash penalty only if disabled but NOT finished. Return (N, 1) for VecDroneRaceEnv.
        crash_penalty = jp.where(disabled & ~is_finished, -50.0, 0.0)
        
        # Store components for info()
        self._last_reward_components = {
            "reward_progress": progress[:, None],
            "reward_gate": gate_reward[:, None],
            "penalty_crash": crash_penalty[:, None],
        }

        return (reward + crash_penalty)[:, None]


# region Wrappers
class StackObs(VectorObservationWrapper):
    """Wrapper to stack history observations."""

    def __init__(self, env: VectorEnv, n_obs: int = 0):
        """Init."""
        super().__init__(env)
        self.n_obs = n_obs
        if self.n_obs > 0:
            # Update observation space
            spec = {k: v for k, v in self.single_observation_space.items()}
            spec["prev_obs"] = spaces.Box(-np.inf, np.inf, shape=(6 * self.n_obs,))
            self.single_observation_space = spaces.Dict(spec)
            self.observation_space = batch_space(self.single_observation_space, self.num_envs)
            # Init obs buffer
            init_obs = env.unwrapped.obs()
            self._prev_obs = jp.zeros((self.num_envs, self.n_obs, 6))
            for _ in range(n_obs):
                self._prev_obs = self._update_prev_obs(self._prev_obs, init_obs)

    def observations(self, observations: dict) -> dict:
        """Override observation."""
        if self.n_obs > 0:
            observations["prev_obs"] = self._prev_obs.reshape(self.num_envs, -1)
            self._prev_obs = self._update_prev_obs(self._prev_obs, observations)
        return observations

    @staticmethod
    @jax.jit
    def _update_prev_obs(prev_obs: Array, obs: dict) -> Array:
        """Update previous observations."""
        basic_obs_key = ["rpy", "vel_body"]  # 3+3 = 6 values per step
        basic_obs = jp.concatenate(
            [jp.reshape(obs[k], (obs[k].shape[0], -1)) for k in basic_obs_key], axis=-1
        )
        prev_obs = jp.concatenate([prev_obs[:, 1:, :], basic_obs[:, None, :]], axis=1)
        return prev_obs


class ActionPenalty(VectorObservationWrapper):
    """Wrapper to apply action penalty."""

    def __init__(
        self,
        env: VectorEnv,
        act_coef: float = 0.01,
        d_act_th_coef: float = 0.2,
        d_act_xy_coef: float = 0.4,
    ):
        """Init."""
        super().__init__(env)
        # Update observation space
        spec = {k: v for k, v in self.single_observation_space.items()}
        spec["last_action"] = spaces.Box(-np.inf, np.inf, shape=(4,))
        self.single_observation_space = spaces.Dict(spec)
        self.observation_space = batch_space(self.single_observation_space, self.num_envs)
        self._last_action = jp.zeros((self.num_envs, 4))
        self.act_coef = act_coef
        self.d_act_th_coef = d_act_th_coef
        self.d_act_xy_coef = d_act_xy_coef

    def step(self, action: Array) -> tuple[Array, Array, Array, Array, dict]:
        """Override step."""
        obs, reward, terminated, truncated, info = super().step(action)
        # penalty on actions
        action_diff = action - self._last_action
        # energy
        act_penalty = self.act_coef * action[..., -1] ** 2
        reward -= act_penalty
        # smoothness
        smoothness_th_penalty = self.d_act_th_coef * action_diff[..., -1] ** 2
        smoothness_xy_penalty = self.d_act_xy_coef * jp.sum(action_diff[..., :3] ** 2, axis=-1)
        reward -= smoothness_th_penalty
        reward -= smoothness_xy_penalty
        
        info["penalty_action"] = -act_penalty
        info["penalty_smoothness_thrust"] = -smoothness_th_penalty
        info["penalty_smoothness_xy"] = -smoothness_xy_penalty
        
        self._last_action = action
        return self.observations(obs), reward, terminated, truncated, info

    def observations(self, observations: dict) -> dict:
        """Override observation."""
        observations["last_action"] = self._last_action
        return observations


class LookAtPenalty(VectorObservationWrapper):
    """Wrapper to apply penalty if drone is not looking at the gate."""

    def __init__(self, env: VectorEnv, look_at_coef: float = 0.05):
        """Init."""
        super().__init__(env)
        self.look_at_coef = look_at_coef

    def step(self, action: Array) -> tuple[Array, Array, Array, Array, dict]:
        """Override step."""
        obs, reward, terminated, truncated, info = super().step(action)
        # ── angle penalty ───────────────────────────────────────────────────────
        # gate_vec_body is unit vector from drone to target gate in body frame.
        # Check for both (N, 3) and (N, 1, 3) shapes.
        gate_vec_body = obs["gate_vec_body"]
        if gate_vec_body.ndim == 3:
            gate_vec_body = gate_vec_body[:, 0, :]

        # In body frame, drone forward is [1, 0, 0], so dot is just the X component.
        cos_theta = gate_vec_body[:, 0]
        angle = jp.acos(jp.clip(cos_theta, -1.0, 1.0))
        # apply penalty if angle > 60 degrees (pi/3)
        look_at_penalty = jp.where(angle > (jp.pi / 2), self.look_at_coef, 0.0)
        reward -= look_at_penalty
        
        info["penalty_look_at"] = -look_at_penalty

        return self.observations(obs), reward, terminated, truncated, info

    def observations(self, observations: dict) -> dict:
        """Override observation."""
        return observations


class FlattenJaxObservation(VectorObservationWrapper):
    """Wrapper to flatten the observations."""

    def __init__(self, env: VectorEnv):
        """Init."""
        super().__init__(env)
        self.single_observation_space = flatten_space(env.single_observation_space)
        self.observation_space = flatten_space(env.observation_space)

    def observations(self, observations: dict) -> dict:
        """Flatten observations."""
        return jp.concatenate(
            [jp.reshape(v, (v.shape[0], -1)) for k, v in observations.items()], axis=-1
        )


class RunningMeanStd:
    """Tracks the running mean and variance of a data stream."""

    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon: float = 1.0, shape: tuple = ()):
        """Init."""
        self.mean = jp.zeros(shape, "float32")
        self.var = jp.ones(shape, "float32")
        self.count = epsilon

    def update(self, x: Array):
        """Update statistics."""
        batch_mean = jp.mean(x, axis=0)
        batch_var = jp.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        """Update from moments."""
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
    """
    A vectorized wrapper that normalizes the observations
    and returns from an environment.
    """

    def __init__(
        self,
        venv: VectorEnv,
        ob=True,
        ret=True,
        clipob=100.0,
        cliprew=100.0,
        gamma=0.94,
        epsilon=1e-4,
        training=True,
    ):
        """Init."""
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
        """Override step."""
        obs, rews, terminations, truncations, infos = self.env.step(actions)
        dones = terminations | truncations

        # Store raw rewards before normalization
        infos["reward_raw"] = rews

        self.ret = self.ret * self.gamma + rews
        obs = self._obfilt(obs)
        if self.ret_rms:
            if self.training:
                self.ret_rms.update(self.ret)
            rews = jp.clip(
                rews / jp.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew
            )
        self.ret = jp.where(dones, 0.0, self.ret)
        return obs, rews, terminations, truncations, infos

    def _obfilt(self, obs):
        """Filter observations."""
        if self.ob_rms:
            if self.training:
                self.ob_rms.update(obs)
            obs = jp.clip(
                (obs - self.ob_rms.mean) / jp.sqrt(self.ob_rms.var + self.epsilon),
                -self.clipob,
                self.clipob,
            )
            return obs
        else:
            return obs

    def reset(self, **kwargs):
        """Override reset."""
        obs, info = self.env.reset(**kwargs)
        self.ret = jp.zeros(self.num_envs)
        return self._obfilt(obs), info

    def get_stats(self):
        """Get statistics."""
        return {
            "ob_rms_mean": self.ob_rms.mean if self.ob_rms else None,
            "ob_rms_var": self.ob_rms.var if self.ob_rms else None,
            "ob_rms_count": self.ob_rms.count if self.ob_rms else None,
            "ret_rms_mean": self.ret_rms.mean if self.ret_rms else None,
            "ret_rms_var": self.ret_rms.var if self.ret_rms else None,
            "ret_rms_count": self.ret_rms.count if self.ret_rms else None,
        }

    def set_stats(self, stats):
        """Set statistics."""
        if self.ob_rms and stats.get("ob_rms_mean") is not None:
            self.ob_rms.mean = jp.array(stats["ob_rms_mean"])
            self.ob_rms.var = jp.array(stats["ob_rms_var"])
            self.ob_rms.count = stats["ob_rms_count"]
        if self.ret_rms and stats.get("ret_rms_mean") is not None:
            self.ret_rms.mean = jp.array(stats["ret_rms_mean"])
            self.ret_rms.var = jp.array(stats["ret_rms_var"])
            self.ret_rms.count = stats["ret_rms_count"]


def get_vec_normalize(env: VectorEnv) -> VecNormalize | None:
    """Find VecNormalize wrapper in the environment stack."""
    while hasattr(env, "env"):
        if isinstance(env, VecNormalize):
            return env
        env = env.env
    return None


class GlobalRewardScale(VectorRewardWrapper):
    """Scales the final accumulated reward from the base env and all prior wrappers."""

    def __init__(self, env: VectorEnv, scale: float = 0.01):
        """Init."""
        super().__init__(env)
        self.scale = scale

    def rewards(self, reward: Array) -> Array:
        """Multiply the final reward by the scaling factor."""
        return reward * self.scale


def set_seeds(seed: int):
    """Seed everything."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# region MakeEnvs
def make_envs(
    config: str = "level3.toml",
    num_envs: int = None,
    jax_device: str = "cpu",
    torch_device: torch.device = torch.device("cpu"),
    coefs: dict = {},
) -> VectorEnv:
    """Make environments for training RL policy."""
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
    env = LookAtPenalty(env, look_at_coef=coefs.get("look_at_coef", 0.04))
    env = StackObs(env, n_obs=coefs.get("n_obs", 0))
    env = ActionPenalty(
        env,
        act_coef=coefs.get("act_coef", 0.04),
        d_act_th_coef=coefs.get("d_act_th_coef", 0.04),
        d_act_xy_coef=coefs.get("d_act_xy_coef", 0.04),
    )

    # env = GlobalRewardScale(env, scale=coefs.get("global_scale", 0.01))
    env = FlattenJaxObservation(env)
    env = VecNormalize(env, training=coefs.get("training", True), gamma=coefs.get("gamma", 0.99),)
    env = JaxToTorch(env, torch_device)
    return env


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    """Initialize layer."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# region Agent
class Agent(nn.Module):
    """RL Agent."""

    def __init__(self, obs_shape: tuple, action_shape: tuple):
        """Init network structures."""
        super().__init__()
        obs_dim = torch.tensor(obs_shape).prod()
        self.critic = nn.Sequential(
            # nn.LayerNorm(obs_dim),
            layer_init(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            # nn.LayerNorm(obs_dim),
            layer_init(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, torch.tensor(action_shape).prod()), std=0.01),
        )
        self.actor_logstd = nn.Parameter(
            torch.Tensor([[0, 0, 0, 1]])  # start with smaller std for roll, pitch, yaw
        )
        # self.actor_logstd = nn.Parameter(128
        #     torch.zeros(1, torch.tensor(action_shape).prod()) 
        # )

    def get_value(self, x: Tensor) -> Tensor:
        """Value estimation."""
        return self.critic(x)

    def get_action_and_value(
        self, x: Tensor, action: Tensor | None = None, deterministic: bool = False
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Action output."""
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        # During learning the agent explores the environment by sampling actions from a Normal
        # distribution. The standard deviation is a learnable parameter that should decrease during
        # training as the agent gets more confident in its actions.
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample() if not deterministic else action_mean
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


# region Train
def train_ppo(
    args: Args, device: torch.device, jax_device: str, wandb_enabled: bool = False
) -> None:
    """Train.

    An implementation of PPO from cleanrl, see https://docs.cleanrl.dev/.
    """
    # train setup
    if wandb_enabled and wandb.run is None:
        wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, config=vars(args))
    train_start_time = time.time()
    set_seeds(args.seed)  # TRY NOT TO MODIFY: seeding
    print("Training on device:", device, "| Environment device:", jax_device)

    # env setup
    r_coefs = {
        "n_obs": args.n_obs,
        "d_act_xy_coef": args.d_act_xy_coef,
        "d_act_th_coef": args.d_act_th_coef,
        "act_coef": args.act_coef,
        "look_at_coef": args.look_at_coef,
        "global_scale": args.global_scale,
        "gamma": args.gamma,
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
    best_raw_reward = -float("inf")

    if args.resume:
        latest_checkpoint = checkpoint_dir / "latest.ckpt"
        if not latest_checkpoint.exists():
            latest_checkpoint = checkpoint_dir / "checkpoint.ckpt"  # fallback
            
        if latest_checkpoint.exists():
            print(f"Resuming from {latest_checkpoint}")
            checkpoint = torch.load(latest_checkpoint, map_location=device, weights_only=False)
            agent.load_state_dict(checkpoint["agent_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if vec_norm and "vec_normalize_stats" in checkpoint:
                vec_norm.set_stats(checkpoint["vec_normalize_stats"])
            global_step = checkpoint["global_step"]
            start_iteration = checkpoint["iteration"] + 1
            best_raw_reward = checkpoint.get("best_raw_reward", -float("inf"))
        else:
            print(f"No checkpoint found at {latest_checkpoint} to resume from.")

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(
        device
    )
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(
        device
    )
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    sum_rewards = torch.zeros((args.num_envs)).to(device)
    sum_rewards_raw = torch.zeros((args.num_envs)).to(device)
    sum_steps = torch.zeros((args.num_envs)).to(device)
    # Components accumulators
    component_keys = [
        "reward_progress", "reward_gate", "penalty_crash",
        "penalty_action", "penalty_smoothness_thrust", "penalty_smoothness_xy",
        "penalty_look_at"
    ]
    sum_components = {k: torch.zeros((args.num_envs)).to(device) for k in component_keys}

    # Create buffers to hold episodic stats for the entire rollout (no-sync tracking)
    ep_rewards_buffer = torch.zeros((args.num_steps, args.num_envs), device=device)
    ep_raw_rewards_buffer = torch.zeros((args.num_steps, args.num_envs), device=device)
    ep_lengths_buffer = torch.zeros((args.num_steps, args.num_envs), device=device)
    ep_components_buffer = {
        k: torch.zeros((args.num_steps, args.num_envs), device=device) for k in component_keys
    }
    
    sum_rewards_hist = deque(maxlen=1000)
    sum_rewards_raw_hist = deque(maxlen=1000)

    for iteration in range(start_iteration, args.num_iterations + 1):
        start_time = time.time()

        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                # Sanity check: prevent NaN observations from producing NaN actions
                next_obs = torch.nan_to_num(next_obs, nan=0.0, posinf=10.0, neginf=-10.0)
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action)
            
            # Sanity check: prevent NaN from crashing training
            reward = torch.nan_to_num(reward, nan=0.0, posinf=1.0, neginf=-1.0)
            
            rewards[step] = reward
            sum_rewards += reward
            sum_steps += 1
            if "reward_raw" in infos:
                # Ensure zero-copy if coming from JaxToTorch dlpack, avoid .to(device) if already on GPU
                r_raw = infos["reward_raw"]
                if not isinstance(r_raw, torch.Tensor):
                    r_raw = torch.as_tensor(r_raw, device=device)
                sum_rewards_raw += r_raw
            
            # --- THE NEW TRACKING LOGIC ---
            # Store completed episode stats. next_done (from prev step) acts as multiplier.
            ep_rewards_buffer[step] = sum_rewards * next_done
            ep_raw_rewards_buffer[step] = sum_rewards_raw * next_done
            ep_lengths_buffer[step] = sum_steps * next_done
            for k in component_keys:
                if k in infos:
                    v = infos[k]
                    if not isinstance(v, torch.Tensor):
                        v = torch.as_tensor(v, device=device).flatten()
                    sum_components[k] += v
                ep_components_buffer[k][step] = sum_components[k] * next_done

            # Reset tracking variables without boolean indexing (pure math)
            mask_alive = 1.0 - next_done.float()
            sum_rewards *= mask_alive
            sum_rewards_raw *= mask_alive
            sum_steps *= mask_alive

            for k in component_keys:
                sum_components[k] *= mask_alive
            
            next_done = terminations | truncations

        # --- BATCHED LOGGING (1 Sync per rollout) ---
        done_mask = dones.bool()
        completed_rewards = ep_rewards_buffer[done_mask].cpu().tolist()
        completed_raw_rewards = ep_raw_rewards_buffer[done_mask].cpu().tolist()
        completed_lengths = ep_lengths_buffer[done_mask].cpu().tolist()

        sum_rewards_hist.extend(completed_rewards)
        sum_rewards_raw_hist.extend(completed_raw_rewards)

        if wandb_enabled and len(completed_rewards) > 0:
            log_dict = {
                "train/reward": np.mean(completed_rewards),
                "train/reward_raw": np.mean(completed_raw_rewards),
                "charts/avg_episode_length": np.mean(completed_lengths),
            }
            
            # Track crash/success rates and components
            for k in component_keys:
                comp_vals = ep_components_buffer[k][done_mask].cpu()
                if k == "penalty_crash":
                    crashes = (comp_vals <= -10.0).float()
                    log_dict["charts/crash_rate"] = crashes.mean().item()
                elif k == "reward_gate":
                    successes = (comp_vals > 0).float()
                    log_dict["charts/success_rate"] = successes.mean().item()
                
                # per-step components (averaged over finished episodes)
                log_dict[f"train/{k}"] = comp_vals.mean().item()
            
            wandb.log(log_dict, step=global_step)

        # bootstrap value if not done
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

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
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
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
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
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                # Extra safety: ensure no NaN/Inf gradients survive
                for param in agent.parameters():
                    if param.grad is not None:
                        torch.nan_to_num(param.grad, nan=0.0, posinf=1.0, neginf=-1.0, out=param.grad)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # region Logging and Checkpointing
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
                },
                step=global_step,
            )

        # Iterative checkpointing every checkpoint_freq * total_iterations
        if iteration % max(1, int(args.num_iterations * args.checkpoint_freq)) == 0:
            avg_reward = (
                np.mean(list(sum_rewards_hist)[-100:]) if sum_rewards_hist else -float("inf")
            )
            avg_raw_reward = (
                np.mean(list(sum_rewards_raw_hist)[-100:]) if sum_rewards_raw_hist else -float("inf")
            )
            latest_path = checkpoint_dir / "latest.ckpt"
            state = {
                "agent_state_dict": agent.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "vec_normalize_stats": vec_norm.get_stats() if vec_norm else None,
                "iteration": iteration,
                "global_step": global_step,
                "best_raw_reward": best_raw_reward,
            }
            torch.save(state, latest_path)
            print(f"Latest checkpoint saved to {latest_path} (Avg Raw Reward: {avg_raw_reward:.2f})")

            if avg_raw_reward > best_raw_reward:
                best_raw_reward = avg_raw_reward
                best_path = checkpoint_dir / "best.ckpt"
                torch.save(state, best_path)
                print(f"New best model saved to {best_path} (Raw Reward: {best_raw_reward:.2f})")
        # endregion

        end_time = time.time()
        print(f"Iter {iteration}/{args.num_iterations} took {end_time - start_time:.2f} seconds")
    train_end_time = time.time()
    print(f"Training for {global_step} steps took {train_end_time - train_start_time:.2f} seconds.")
    envs.close()

    return sum_rewards_hist


# region Evaluate
def evaluate_ppo(args: Args, n_eval: int) -> tuple[float, float]:
    """Evaluate."""
    set_seeds(args.seed)
    device = torch.device("cpu")
    r_coefs = {
        "n_obs": args.n_obs,
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
        # Evaluate the policy
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
    resume: bool = True,
):
    """Main."""
    args = Args.create(checkpoint_freq=checkpoint_freq, resume=resume)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    jax_device = args.jax_device

    if train:  # use "--train False" to skip training
        train_ppo(args, device, jax_device, wandb_enabled)

    if eval > 0:  # use "--eval <N>" to perform N evaluation episodes
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
