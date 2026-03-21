"""Visualize multiple drones running a trained RL policy in a single simulation.

Run as:
    $ python scripts/multi_drone_visualizer.py --config level0.toml --checkpoint lsy_drone_racing/control/checkpoints/latest.ckpt --n_drones 5
"""

import time
from pathlib import Path
from typing import Literal

import fire
import gymnasium as gym
import jax
import jax.numpy as jp
import numpy as np
import torch
import jax.dlpack
from crazyflow.envs.norm_actions_wrapper import NormalizeActions
from gymnasium import spaces
from gymnasium.vector import VectorEnv, VectorObservationWrapper, VectorWrapper
from gymnasium.vector.utils import batch_space
from jax import Array
from jax.scipy.spatial.transform import Rotation as R

from lsy_drone_racing.envs.multi_drone_race import VecMultiDroneRaceEnv
from lsy_drone_racing.envs.race_core import build_action_space
from lsy_drone_racing.utils import load_config
from lsy_drone_racing.control.train_rl_custom import (
    Agent, 
    VecNormalize, 
    get_vec_normalize, 
    FlattenJaxObservation, 
    StackObs, 
    ActionPenalty, 
    LookAtPenalty,
    Args
)

# region MultiRaceTrainEnv
class MultiRaceTrainEnv(VecMultiDroneRaceEnv):
    """Multi-agent drone racing environment for RL evaluation with compact ego-centric observations.
    
    This version returns observations as (n_drones, ...) directly to satisfy training wrappers
    like StackObs that bypass the wrapper stack and call .unwrapped.obs().
    """

    def __init__(self, **kwargs):
        """Init."""
        # We still want n_worlds=1 in the simulation
        super().__init__(num_envs=1, **kwargs)
        # But we want to look like we have num_envs = n_drones for the outside world
        # We set it after super().__init__ to avoid the property setter error
        self.num_envs = self.sim.n_drones
        
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
        
        self.single_action_space = build_action_space(
            self.sim.control, 
            self.sim.params.drone_model if hasattr(self.sim, "params") else "cf21B_500"
        )
        self.action_space = batch_space(self.single_action_space, self.num_envs)

    def obs(self) -> dict[str, Array]:
        """Compact ego-centric observation for all drones (flattened)."""
        base = super().obs()  # world-frame obs from RaceCoreEnv (shape: n_envs=1, n_drones, ...)

        # Extract all drones (shape: D, ...)
        drone_pos  = base["pos"][0]    # (D, 3) 
        drone_quat = base["quat"][0]   # (D, 4) 
        vel_world  = base["vel"][0]    # (D, 3) 

        body_R = R.from_quat(drone_quat)

        # 1. Roll / pitch / yaw
        rpy = body_R.as_euler("xyz")        # (D, 3)
        yaw = rpy[:, 2]

        # 2. Velocity in drone body frame
        vel_body = body_R.inv().apply(vel_world)

        # ── gate geometry ──────────────────────────────────────────────────────
        n_gates    = len(self.gates["pos"])
        gates_pos  = base["gates_pos"][0]
        gates_quat = base["gates_quat"][0]

        target_idx = jp.clip(base["target_gate"][0], 0, n_gates - 1)
        next_idx   = jp.clip(target_idx + 1, 0, n_gates - 1)

        drone_ids = jp.arange(self.sim.n_drones)
        target_gate_pos  = gates_pos[drone_ids, target_idx]
        next_gate_pos    = gates_pos[drone_ids, next_idx]
        target_gate_quat = gates_quat[drone_ids, target_idx]

        # 3. Distance (raw, matching RaceTrainEnv)
        dist_target_raw = jp.linalg.norm(drone_pos - target_gate_pos, axis=-1, keepdims=True)
        dist_next_raw   = jp.linalg.norm(drone_pos - next_gate_pos,   axis=-1, keepdims=True)

        # 4. Unit vector to gate
        gate_vec_world = target_gate_pos - drone_pos
        gate_vec_body_raw = body_R.inv().apply(gate_vec_world)
        gate_vec_body = gate_vec_body_raw / (jp.linalg.norm(gate_vec_body_raw, axis=-1, keepdims=True) + 1e-6)

        # 5. Gate alignment
        local_x     = jp.broadcast_to(jp.array([1.0, 0.0, 0.0]), (self.sim.n_drones, 3))
        gate_normal = R.from_quat(target_gate_quat).apply(local_x)
        gate_yaw    = jp.arctan2(gate_normal[:, 1], gate_normal[:, 0])
        align_diff  = gate_yaw - yaw
        gate_alignment = jp.arctan2(jp.sin(align_diff), jp.cos(align_diff))[:, None]

        # Returns (D, ...) directly
        return {
            "rpy":            rpy,
            "vel_body":       vel_body,
            "dist_target":    dist_target_raw,
            "dist_next":      dist_next_raw,
            "gate_vec_body":  gate_vec_body,
            "gate_alignment": gate_alignment,
        }


    def reset(self, **kwargs):
        """Reset the environment."""
        obs, info = super().reset(**kwargs)
        # obs is from self.obs() which is (D, ...) already
        # reward, terminated, truncated are (1, D) from _step.
        # But VecMultiDroneRaceEnv.reset() returns self.obs() and self.info().
        return obs, info

    def step(self, action: Array):
        """Step the environment."""
        # action is (D, action_dim). VecMultiDroneRaceEnv expects (1, D, action_dim).
        obs, reward, terminated, truncated, info = self._step(action[None, ...])
        # Reshape returns from (1, D, ...) to (D, ...)
        reward = reward[0]
        terminated = terminated[0]
        truncated = truncated[0]
        return obs, reward, terminated, truncated, info

    def __getattr__(self, name):
        """Proxy any missing attributes to the wrapped env or core env."""
        # This helps when wrappers like NormalizeActions look for env.device
        if hasattr(super(), name):
            return getattr(super(), name)
        return getattr(self.unwrapped, name)

# region Visualization
def visualize(
    config: str = "level0.toml",
    checkpoint: str = "lsy_drone_racing/control/checkpoints/best.ckpt",
    n_eval: int = 1,
    n_drones: int = 5,
    stochastic: bool = False,
    seed: int = 42,
):
    """Run visualization."""
    # Load config etc
    cfg_path = Path(__file__).parents[1] / "config" / config
    cfg = load_config(cfg_path)
    
    # Defaults from Args if not provided
    default_args = Args.create()
    
    if len(cfg.env.track.drones) < n_drones:
        base_drone = cfg.env.track.drones[0]
        cfg.env.track.drones = [base_drone] * n_drones
    
    # 1. Create the environment (looks like n_drones parallel single-drone worlds)
    env = MultiRaceTrainEnv(
        freq=cfg.env.freq,
        sim_config=cfg.sim,
        track=cfg.env.track,
        sensor_range=cfg.env.sensor_range,
        control_mode=cfg.env.control_mode,
        disturbances=cfg.env.disturbances,
        device="gpu", # User prefers GPU
    )
    
    # 2. Apply the same wrapper stack as in make_envs (train_rl_custom.py)
    # No MultiToSingleAgentWrapper needed anymore as MultiRaceTrainEnv handles it!
    env = NormalizeActions(env)
    env = LookAtPenalty(env, look_at_coef=default_args.look_at_coef)
    env = StackObs(env, n_obs=default_args.n_obs)
    env = ActionPenalty(
        env,
        act_coef=default_args.act_coef,
        d_act_th_coef=default_args.d_act_th_coef,
        d_act_xy_coef=default_args.d_act_xy_coef,
    )
    env = FlattenJaxObservation(env)
    env = VecNormalize(env, training=False, gamma=default_args.gamma)
    
    # Load checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint from {checkpoint}")
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    
    # Reconstruct Agent automatically using the env shapes
    obs_shape = env.single_observation_space.shape
    action_shape = env.single_action_space.shape
    print(f"Agent observation shape: {obs_shape}, action shape: {action_shape}")
    
    agent = Agent(obs_shape, action_shape).to(device)
    agent.load_state_dict(ckpt["agent_state_dict"])
    agent.eval()
    
    # Apply normalization stats from checkpoint
    vec_norm = get_vec_normalize(env)
    if vec_norm and "vec_normalize_stats" in ckpt:
        print("Applying VecNormalize stats from checkpoint...")
        vec_norm.set_stats(ckpt["vec_normalize_stats"])
    
    # Evaluate the policy
    episode_rewards = []
    episode_lengths = []
    ep_seed = seed
    fps = 60
    
    print(f"Starting simulation with {n_drones} drones at {cfg.env.freq} Hz...")
    
    with torch.no_grad():
        for episode in range(n_eval):
            # Reset with incremental seed
            next_obs, info = env.reset(seed=(ep_seed := ep_seed + 1))
            # Zero-copy conversion for JAX to Torch (stays on GPU)
            next_obs = torch.from_dlpack(next_obs).to(device)
            
            done = torch.zeros(n_drones, dtype=torch.bool, device=device)
            episode_reward = 0
            steps = 0
            start_time = time.time()
            
            while not done.any():
                # 1. Synchronize with real time for smooth playback
                elapsed = time.time() - start_time
                target_elapsed = steps / cfg.env.freq
                if target_elapsed > elapsed:
                    time.sleep(target_elapsed - elapsed)

                # 2. Update camera (using unwrapped to access physics)
                unwrapped = env.unwrapped
                active_drones = ~unwrapped.data.disabled_drones[0] # (D,)
                if active_drones.any():
                    avg_pos = unwrapped.sim.data.states.pos[0, active_drones].mean(axis=0)
                    unwrapped.cam_config["lookat"] = np.array(avg_pos)
                    if active_drones.sum() > 1:
                        spread = jp.linalg.norm(unwrapped.sim.data.states.pos[0, active_drones].max(axis=0) - unwrapped.sim.data.states.pos[0, active_drones].min(axis=0))
                        unwrapped.cam_config["distance"] = max(2.5, spread * 1.5)
                
                # 3. Get actions from Agent
                act, _, _, _ = agent.get_action_and_value(next_obs, deterministic=not stochastic)
                    
                # 4. Step environment
                # Zero-copy conversion for Torch to JAX (stays on GPU)
                action_jax = jax.dlpack.from_dlpack(act)
                next_obs, reward, terminated, truncated, info = env.step(action_jax)
                # Zero-copy conversion for JAX back to Torch (stays on GPU)
                next_obs = torch.from_dlpack(next_obs).to(device)
                
                # 5. Render
                if ((steps * fps) % cfg.env.freq) < fps:
                    unwrapped.render()
                    
                done = torch.from_numpy(np.array(terminated | truncated)).to(device)
                episode_reward += reward[0].item() # Track first drone's reward
                steps += 1
                
                if steps % 10 == 0:
                    active_count = active_drones.sum()
                    print(f"Episode {episode+1} | Step {steps}: {active_count}/{n_drones} drones active (FPS: {steps / (time.time() - start_time):.1f})", end="\r")
                    
            episode_rewards.append(episode_reward)
            episode_lengths.append(steps)
            print(f"\nEpisode {episode + 1}: Reward = {episode_reward:.2f}, Length = {steps}")

        print(
            f"\nAverage Reward = {np.mean(episode_rewards):.2f}, Length = {np.mean(episode_lengths)}"
        )
    
    env.close()

if __name__ == "__main__":
    fire.Fire(visualize)
