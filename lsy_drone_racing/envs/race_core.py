"""Core environment for drone racing simulations.

This module provides the shared logic for simulating drone racing environments. It defines a core
environment class that wraps our drone simulation, drone control, gate tracking, and collision
detection. The module serves as the base for both single-drone and multi-drone racing environments.

The environment is designed to be configurable, supporting:

* Different control modes (state or attitude)
* Customizable tracks with gates and obstacles
* Various randomization options for robust policy training
* Disturbance modeling for realistic flight conditions
* Vectorized execution for parallel training

This module is primarily used as a base for the higher-level environments in
:mod:`~lsy_drone_racing.envs.drone_race` and :mod:`~lsy_drone_racing.envs.multi_drone_race`,
which provide Gymnasium-compatible interfaces for reinforcement learning, MPC and other control
techniques.
"""

from __future__ import annotations

import copy as copy
import logging
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from crazyflow.sim import Sim
from crazyflow.sim.sim import use_box_collision
from drone_controllers.mellinger.params import ForceTorqueParams
from flax.struct import dataclass
from gymnasium import spaces

from lsy_drone_racing.envs.randomize import (
    randomize_drone_inertia_fn,
    randomize_drone_mass_fn,
    randomize_drone_pos_fn,
    randomize_drone_quat_fn,
    randomize_gate_pos_fn,
    randomize_gate_rpy_fn,
    randomize_obstacle_pos_fn,
)
from lsy_drone_racing.envs.utils import gate_passed, generate_random_track, load_track

if TYPE_CHECKING:
    from crazyflow.sim.data import SimData
    from jax import Array, Device
    from ml_collections import ConfigDict
    from mujoco import MjSpec
    from mujoco.mjx import Data
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# region EnvData


@dataclass
class EnvData:
    """Struct holding the data of all auxiliary variables for the environment.

    This dataclass stores the dynamic and static state of the environment that is not directly
    part of the physics simulation. It includes information about gate progress, drone status,
    and environment boundaries. Static variables are initialized once and do not change during the
    episode.

    Args:
        target_gate: Current target gate index for each drone in each environment
        gates_visited: Boolean flags indicating which gates have been visited by each drone
        obstacles_visited: Boolean flags indicating which obstacles have been detected
        last_drone_pos: Previous positions of drones, used for gate passing detection
        marked_for_reset: Flags indicating which environments need to be reset
        disabled_drones: Flags indicating which drones have crashed or are otherwise disabled
        contact_masks: Masks for contact detection between drones and objects
        pos_limit_low: Lower position limits for the environment
        pos_limit_high: Upper position limits for the environment
        gate_mj_ids: MuJoCo IDs for the gates
        obstacle_mj_ids: MuJoCo IDs for the obstacles
        max_episode_steps: Maximum number of steps per episode
        sensor_range: Range at which drones can detect gates and obstacles
    """

    # Dynamic variables
    target_gate: Array
    gates_visited: Array
    obstacles_visited: Array
    last_drone_pos: Array
    marked_for_reset: Array
    disabled_drones: Array
    steps: Array
    # Static variables
    contact_masks: Array
    pos_limit_low: Array
    pos_limit_high: Array
    gate_mj_ids: Array
    obstacle_mj_ids: Array
    max_episode_steps: Array
    sensor_range: Array
    last_drone_vel: Array
    accel_buffer: Array
    gyro_buffer: Array
    last_target_gate: Array

    @classmethod
    def create(
        cls,
        n_envs: int,
        n_drones: int,
        n_gates: int,
        n_obstacles: int,
        contact_masks: Array,
        gate_mj_ids: Array,
        obstacle_mj_ids: Array,
        max_episode_steps: int,
        sensor_range: float,
        pos_limit_low: Array,
        pos_limit_high: Array,
        device: Device,
        last_drone_vel: Array,
        imu_steps_per_env: int,
    ) -> EnvData:
        """Create a new environment data struct with default values."""
        return cls(
            target_gate=jp.zeros((n_envs, n_drones), dtype=int, device=device),
            gates_visited=jp.zeros(
                (n_envs, n_drones, n_gates), dtype=bool, device=device
            ),
            obstacles_visited=jp.zeros(
                (n_envs, n_drones, n_obstacles), dtype=bool, device=device
            ),
            last_drone_pos=jp.zeros(
                (n_envs, n_drones, 3), dtype=np.float32, device=device
            ),
            marked_for_reset=jp.zeros(n_envs, dtype=bool, device=device),
            disabled_drones=jp.zeros((n_envs, n_drones), dtype=bool, device=device),
            contact_masks=jp.array(contact_masks, dtype=bool, device=device),
            steps=jp.zeros(n_envs, dtype=int, device=device),
            pos_limit_low=jp.array(pos_limit_low, dtype=np.float32, device=device),
            pos_limit_high=jp.array(pos_limit_high, dtype=np.float32, device=device),
            gate_mj_ids=jp.array(gate_mj_ids, dtype=int, device=device),
            obstacle_mj_ids=jp.array(obstacle_mj_ids, dtype=int, device=device),
            max_episode_steps=jp.array([max_episode_steps], dtype=int, device=device),
            sensor_range=jp.array([sensor_range], dtype=jp.float32, device=device),
            last_drone_vel=jp.tile(
                jp.array(last_drone_vel, dtype=np.float32, device=device),
                (n_envs, n_drones, 1),
            ),
            accel_buffer=jp.zeros(
                (n_envs, n_drones, imu_steps_per_env, 3),
                dtype=np.float32,
                device=device,
            ),
            gyro_buffer=jp.zeros(
                (n_envs, n_drones, imu_steps_per_env, 3),
                dtype=np.float32,
                device=device,
            ),
            last_target_gate=jp.zeros((n_envs, n_drones), dtype=int, device=device),
        )


def build_action_space(
    control_mode: Literal["state", "attitude"], drone_model: str
) -> spaces.Box:
    """Create the action space for the environment.

    Args:
        control_mode: The control mode to use. Either "state" for full-state control
            or "attitude" for attitude control.
        drone_model: Drone model of the environment.

    Returns:
        A Box space representing the action space for the specified control mode.
    """
    if control_mode == "state":
        return spaces.Box(low=-1, high=1, shape=(13,))
    elif control_mode == "attitude":
        params = ForceTorqueParams.load(drone_model)
        thrust_min, thrust_max = params.thrust_min * 4, params.thrust_max * 4
        return spaces.Box(
            np.array(
                [-np.pi / 2, -np.pi / 2, -np.pi / 2, thrust_min], dtype=np.float32
            ),
            np.array([np.pi / 2, np.pi / 2, np.pi / 2, thrust_max], dtype=np.float32),
        )
    else:
        raise ValueError(f"Invalid control mode: {control_mode}")


def build_observation_space(
    n_gates: int, n_obstacles: int, imu_steps_per_env: int
) -> spaces.Dict:
    """Create the observation space for the environment.

    The observation space is a dictionary containing the drone state, gate information,
    and obstacle information.

    Args:
        n_gates: Number of gates in the environment.
        n_obstacles: Number of obstacles in the environment.
    """
    obs_spec = {
        "pos": spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
        "quat": spaces.Box(low=-1, high=1, shape=(4,)),
        "vel": spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
        "ang_vel": spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
        "accel": spaces.Box(low=-np.inf, high=np.inf, shape=(imu_steps_per_env, 3)),
        "gyro": spaces.Box(low=-np.inf, high=np.inf, shape=(imu_steps_per_env, 3)),
        "target_gate": spaces.Discrete(n_gates, start=-1),
        "gates_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(n_gates, 3)),
        "gates_quat": spaces.Box(low=-1, high=1, shape=(n_gates, 4)),
        "gates_visited": spaces.Box(low=0, high=1, shape=(n_gates,), dtype=bool),
        "obstacles_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(n_obstacles, 3)),
        "obstacles_visited": spaces.Box(
            low=0, high=1, shape=(n_obstacles,), dtype=bool
        ),
    }
    return spaces.Dict(obs_spec)


# region Core Env


class RaceCoreEnv:
    """The core environment for drone racing simulations.

    This environment simulates a drone racing scenario where a single drone navigates through a
    series of gates in a predefined track. It supports various configuration options for
    randomization, disturbances, and physics models.

    The environment provides:

    * A customizable track with gates and obstacles
    * Configurable simulation and control frequencies
    * Support for different physics models (e.g., identified dynamics, analytical dynamics)
    * Randomization of drone properties and initial conditions
    * Disturbance modeling for realistic flight conditions
    * Symbolic expressions for advanced control techniques (optional)

    The environment tracks the drone's progress through the gates and provides termination
    conditions based on gate passages and collisions.

    The observation space is a dictionary with the following keys:

    * pos: Drone position
    * quat: Drone orientation as a quaternion (x, y, z, w)
    * vel: Drone linear velocity
    * ang_vel: Drone angular velocity
    * gates_pos: Positions of the gates
    * gates_quat: Orientations of the gates
    * gates_visited: Flags indicating if the drone already was/ is in the sensor range of the
      gates and the true position is known
    * obstacles_pos: Positions of the obstacles
    * obstacles_visited: Flags indicating if the drone already was/ is in the sensor range of the
      obstacles and the true position is known
    * target_gate: The current target gate index

    The action space consists of a desired full-state command
    [x, y, z, vx, vy, vz, ax, ay, az, yaw, rrate, prate, yrate] that is tracked by the drone's
    low-level controller, or a desired collective thrust and attitude command [collective thrust,
    roll, pitch, yaw].
    """

    gate_spec_path = Path(__file__).parent / "assets/gate.xml"
    obstacle_spec_path = Path(__file__).parent / "assets/obstacle.xml"

    def __init__(
        self,
        n_envs: int,
        n_drones: int,
        freq: int,
        sim_config: ConfigDict,
        sensor_range: float,
        track: ConfigDict,
        control_mode: Literal["state", "attitude"] = "state",
        disturbances: ConfigDict | None = None,
        randomizations: ConfigDict | None = None,
        seed: str | int = "random",
        max_episode_steps: int = 1000,
        device: Literal["cpu", "gpu"] = "cpu",
        disable_termination: bool = False,
        disable_collisions: bool = True,
    ):
        """Initialize the DroneRacingEnv.

        Args:
            n_envs: Number of worlds in the vectorized environment.
            n_drones: Number of drones.
            freq: Environment step frequency.
            sim_config: Configuration dictionary for the simulation.
            sensor_range: Sensor range for gate and obstacle detection.
            control_mode: Control mode for the drones. See `build_action_space` for details.
            track: Track configuration.
            disturbances: Disturbance configuration.
            randomizations: Randomization configuration.
            seed: "random" for a generated seed or the random seed directly.
            max_episode_steps: Maximum number of steps per episode. Needs to be tracked manually for
                vectorized environments.
            device: Device used for the environment and the simulation.
        """
        super().__init__()
        if type(seed) is str:
            seed: int = (
                np.random.SeedSequence().entropy if seed == "random" else hash(seed)
            )
            seed &= 0xFFFFFFFF  # Limit seed to 32 bit for jax.random
        self.sim = Sim(
            n_worlds=n_envs,
            n_drones=n_drones,
            physics=sim_config.physics,
            drone_model=sim_config.drone_model,
            control=control_mode,
            freq=sim_config.freq,
            state_freq=freq,
            attitude_freq=sim_config.attitude_freq,
            rng_key=seed,
            device=device,
        )
        use_box_collision(self.sim, True)
        self.cam_config = {
            "distance": sim_config.camera_view[0],
            "azimuth": sim_config.camera_view[1],
            "elevation": sim_config.camera_view[2],
            "lookat": sim_config.camera_view[3:],
        }

        self.disable_termination = disable_termination
        self.disable_collisions = disable_collisions

        # Sanitize args
        if sim_config.freq % freq != 0:
            raise ValueError(f"({sim_config.freq=}) is no multiple of ({freq=})")

        # Env settings
        self.freq = freq
        self.imu_freq = sim_config.get("imu_freq", 500)  # Default to 500Hz

        # Ensure frequencies divide cleanly
        if self.sim.freq % self.imu_freq != 0:
            raise ValueError(
                f"Simulation freq ({self.sim.freq}) must be a multiple of IMU freq ({self.imu_freq})"
            )
        if self.imu_freq % self.freq != 0:
            raise ValueError(
                f"IMU freq ({self.imu_freq}) must be a multiple of Env/SLAM freq ({self.freq})"
            )

        self.sim_steps_per_imu = self.sim.freq // self.imu_freq
        self.imu_steps_per_env = self.imu_freq // self.freq

        self.seed = seed
        self.autoreset = True  # Can be overridden by subclasses
        self.device = jax.devices(device)[0]
        self.sensor_range = sensor_range
        self.track = track
        self.gates, self.obstacles, self.drone = load_track(track)
        specs = {} if disturbances is None else disturbances
        self.disturbances = {mode: rng_spec2fn(spec) for mode, spec in specs.items()}
        specs = {} if randomizations is None else randomizations
        randomizations = {mode: rng_spec2fn(spec) for mode, spec in specs.items()}

        # Load the track into the simulation and compile the reset and step functions with hooks
        self._setup_sim(randomizations)

        # Create the environment data struct.
        n_gates, n_obstacles = len(track.gates), len(track.obstacles)
        contact_masks = self._load_contact_masks(
            self.sim, disable_collisions=disable_collisions
        )
        m = self.sim.mj_model
        gate_ids = [int(m.body(f"gate:{i}").mocapid.squeeze()) for i in range(n_gates)]
        obstacle_ids = [
            int(m.body(f"obstacle:{i}").mocapid.squeeze()) for i in range(n_obstacles)
        ]
        self.data = EnvData.create(
            n_envs=n_envs,
            n_drones=n_drones,
            n_gates=n_gates,
            n_obstacles=n_obstacles,
            contact_masks=contact_masks,
            gate_mj_ids=gate_ids,
            obstacle_mj_ids=obstacle_ids,
            max_episode_steps=max_episode_steps,
            sensor_range=sensor_range,
            pos_limit_low=[-3, -3, -1e-3],
            pos_limit_high=[3, 3, 2.5],
            device=self.device,
            last_drone_vel=self.drone["vel"],
            imu_steps_per_env=self.imu_steps_per_env,
        )
        self.randomize_track = build_track_randomization_fn(
            randomizations, gate_ids, obstacle_ids
        )

    @staticmethod
    @jax.jit
    def _rotate_vector(q: Array, v: Array) -> Array:
        """Rotates a batched vector v by a batched quaternion q (scipy format: x, y, z, w)."""
        q_xyz = q[..., :3]
        q_w = q[..., 3:4]  # Keep dimension for broadcasting

        t = 2.0 * jp.cross(q_xyz, v, axis=-1)
        return v + q_w * t + jp.cross(q_xyz, t, axis=-1)

    @staticmethod
    @jax.jit
    def _quat_conjugate(q: Array) -> Array:
        """Returns the conjugate of a batched quaternion [x, y, z, w]."""
        return jp.concatenate([-q[..., :3], q[..., 3:4]], axis=-1)

    def _reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
        mask: Array | None = None,
    ) -> tuple[dict[str, Array], dict]:
        """Reset the environment.

        Args:
            seed: Random seed.
            options: Additional reset options. Not used.
            mask: Mask of worlds to reset.

        Returns:
            Observation and info.
        """
        if seed is not None:
            self.sim.seed(seed)
            self._np_random = np.random.default_rng(seed)  # Also update gymnasium's rng
        # Randomization of the drone is compiled into the sim reset pipeline, so we don't need to
        # explicitly do it here
        self.sim.reset(mask=mask)
        key, subkey, subkey2 = jax.random.split(self.sim.data.core.rng_key, 3)
        # Generate random track
        track = (
            generate_random_track(self.track, subkey2)
            if self.track.randomize
            else self.track
        )
        self.gates, self.obstacles, self.drone = load_track(track)
        # Randomize the track
        self.sim.data = self.sim.data.replace(
            core=self.sim.data.core.replace(rng_key=key)
        )

        @jax.jit
        def update_sim_data(
            data: SimData, mjx_data: Data, key: jax.random.PRNGKey
        ) -> tuple[SimData, Data]:
            # Randomized drone pos
            # pos = data.states.pos.at[...].set(self.drone["pos"])
            # data = data.replace(states=data.states.replace(pos=pos))

            mjx_data = self.randomize_track(
                mjx_data,
                mask,
                self.gates["nominal_pos"],
                self.gates["nominal_quat"],
                self.obstacles["nominal_pos"],
                key,
            )
            return data, mjx_data

        self.sim.data, self.sim.mjx_data = update_sim_data(
            self.sim.data, self.sim.mjx_data, subkey
        )

        # Reset the environment data
        self.data = self._reset_env_data(
            self.data,
            self.sim.data.states.pos,
            self.sim.data.states.vel,
            self.sim.mjx_data.mocap_pos,
            mask,
        )

        return self.obs(), self.info()

    def _step(self, action: Array) -> tuple[dict[str, Array], float, bool, bool, dict]:
        """Step the firmware_wrapper class and its environment.

        This function should be called once at the rate of ctrl_freq. Step processes and high level
        commands, and runs the firmware loop and simulator according to the frequencies set.

        Args:
            action: Full-state command [x, y, z, vx, vy, vz, ax, ay, az, yaw, rrate, prate, yrate]
                to follow.
        """
        # Check if the physics engine is registering ANY contacts at all

        self.apply_action(action)

        accel_list = []
        gyro_list = []
        dt_imu = 1.0 / self.imu_freq
        last_vel_for_imu = self.data.last_drone_vel

        # Step the physics engine in smaller chunks to gather IMU data
        for _ in range(self.imu_steps_per_env):
            self.sim.step(self.sim_steps_per_imu)

            current_vel = self.sim.data.states.vel
            current_quat = self.sim.data.states.quat
            current_ang_vel = self.sim.data.states.ang_vel

            # Compute IMU for this specific micro-step
            accel, gyro = self.compute_imu(
                current_vel=current_vel,
                last_vel=last_vel_for_imu,
                current_quat=current_quat,
                current_ang_vel=current_ang_vel,
                dt=dt_imu,
            )

            accel_list.append(accel)
            gyro_list.append(gyro)
            last_vel_for_imu = current_vel

        self.data = self.data.replace(
            accel_buffer=jp.stack(accel_list, axis=-2),
            gyro_buffer=jp.stack(gyro_list, axis=-2),
        )

        # self.sim.step(self.sim.freq // self.freq)
        # Warp drones that have crashed outside the track to prevent them from interfering with
        # other drones still in the race
        self.sim.data = self._warp_disabled_drones(
            self.sim.data, self.data.disabled_drones
        )
        # Apply the environment logic. Check which drones are now disabled, check which gates have
        # been passed, and update the target gate.
        drone_pos = self.sim.data.states.pos
        drone_vel = self.sim.data.states.vel
        mocap_pos, mocap_quat = (
            self.sim.mjx_data.mocap_pos,
            self.sim.mjx_data.mocap_quat,
        )
        contacts = self.sim.contacts()
        import jax

        contact_dists = self.sim.mjx_data.contact.dist
        active_contacts = jp.sum(contact_dists <= 0.0, axis=-1)
        jax.debug.print("Active contacts per world: {}", active_contacts)
        jax.debug.print("Disabled drones flag: {}", self.data.disabled_drones)
        # ----------------------------------
        # Get marked_for_reset before it is updated, because the autoreset needs to be based on the
        # previous flags, not the ones from the current step
        marked_for_reset = self.data.marked_for_reset
        # Apply the environment logic with updated simulation data.
        self.data = self._step_env(
            self.data,
            drone_pos,
            drone_vel,
            mocap_pos,
            mocap_quat,
            contacts,
            self.sim.freq,
            stay_on_last=self.disable_termination,
        )
        # Auto-reset envs. Add configuration option to disable for single-world envs
        step_reward = self.reward()
        step_terminated = self.terminated()
        step_truncated = self.truncated()
        step_info = self.info()

        # Auto-reset envs
        if self.autoreset and marked_for_reset.any():
            self._reset(mask=marked_for_reset)

        return (
            self.obs(),  # Observation is from AFTER reset (Correct for Gym API)
            step_reward,  # Reward is from BEFORE reset (The actual crash)
            step_terminated,  # Flags are from BEFORE reset
            step_truncated,
            step_info,
        )

    def apply_action(self, action: Array):
        """Apply the commanded state action to the simulation."""
        # Convert to a buffer that meets XLA's alginment restrictions to prevent warnings. See
        # https://github.com/jax-ml/jax/discussions/6055
        # Tracking issue:
        # https://github.com/jax-ml/jax/issues/29810
        # Forcing a copy here is less efficient, but avoids the warning.
        action = np.reshape(
            action, (self.sim.n_worlds, self.sim.n_drones, -1), copy=True
        )
        if "action" in self.disturbances:
            key, subkey = jax.random.split(self.sim.data.core.rng_key)
            action += self.disturbances["action"](subkey, action.shape)
            self.sim.data = self.sim.data.replace(
                core=self.sim.data.core.replace(rng_key=key)
            )
        match self.sim.control:
            case "attitude":
                self.sim.attitude_control(action)
            case "state":
                self.sim.state_control(action)
            case _:
                raise ValueError(f"Unsupported control mode: {self.sim.control}")

    def render(self):
        """Render the environment."""
        self.sim.render(cam_config=self.cam_config)

    def close(self):
        """Close the environment by stopping the drone and landing back at the starting position."""
        self.sim.close()

    def obs(self) -> dict[str, Array]:
        """Return the observation of the environment."""
        # Add the gate and obstacle poses to the info. If gates or obstacles are in sensor range,
        # use the actual pose, otherwise use the nominal pose.
        gates_pos, gates_quat, obstacles_pos = self._obs(
            self.sim.mjx_data.mocap_pos,
            self.sim.mjx_data.mocap_quat,
            self.data.gates_visited,
            self.data.gate_mj_ids,
            self.gates["nominal_pos"],
            self.gates["nominal_quat"],
            self.data.obstacles_visited,
            self.data.obstacle_mj_ids,
            self.obstacles["nominal_pos"],
        )

        # # Calculate IMU values
        # dt = 1.0 / self.freq
        # accel, gyro = self.compute_imu(
        #     current_vel=self.sim.data.states.vel,
        #     last_vel=self.data.last_drone_vel,
        #     current_quat=self.sim.data.states.quat,
        #     current_ang_vel=self.sim.data.states.ang_vel,
        #     dt=dt,
        # )

        obs = {
            "pos": self.sim.data.states.pos,
            "quat": self.sim.data.states.quat,
            "vel": self.sim.data.states.vel,
            "ang_vel": self.sim.data.states.ang_vel,
            "accel": self.data.accel_buffer,  # Now returns the full batch of readings
            "gyro": self.data.gyro_buffer,
            "target_gate": self.data.target_gate,
            "gates_pos": gates_pos,
            "gates_quat": gates_quat,
            "gates_visited": self.data.gates_visited,
            "obstacles_pos": obstacles_pos,
            "obstacles_visited": self.data.obstacles_visited,
        }
        return obs

    def reward(self) -> Array:
        """Compute the composite reward for the vectorized environment."""
        n_gates = len(self.data.gate_mj_ids)
        target_idx = jp.maximum(self.data.last_target_gate, 0)

        # 1. Identify Phase
        is_hover_phase = target_idx == (n_gates - 1)

        # Get positions
        gate_ids = self.data.gate_mj_ids[target_idx % n_gates]
        mocap_pos = self.sim.mjx_data.mocap_pos
        current_gate_pos = mocap_pos[jp.arange(self.sim.n_worlds)[:, None], gate_ids]
        drone_pos = self.sim.data.states.pos

        # Calculate distances
        dist_old = jp.linalg.norm(self.data.last_drone_pos - current_gate_pos, axis=-1)
        dist_new = jp.linalg.norm(drone_pos - current_gate_pos, axis=-1)

        # Get velocities
        drone_vel = self.sim.data.states.vel
        vel_mag = jp.linalg.norm(drone_vel, axis=-1)
        ang_vel = self.sim.data.states.ang_vel

        # Determine active status
        is_active = ~self.data.disabled_drones

        # ==========================================
        # PHASE A: NAVIGATION REWARDS
        # ==========================================
        # 1. Passage Reward (+10.0 per gate)
        gate_passed = (self.data.target_gate > self.data.last_target_gate) & (
            self.data.target_gate != -1
        )
        reward_pass = jp.where(~is_hover_phase & is_active, 10.0 * gate_passed, 0.0)

        # 2. Progress Reward (Scaled down to prevent dominating the gate pass)
        progress = dist_old - dist_new
        reward_prog = jp.where(~is_hover_phase, 5.0 * progress, 0.0)

        # 3. Time Penalty (Push it to fly fast)
        reward_time = jp.where(~is_hover_phase, -0.01, 0.0)

        # ==========================================
        # PHASE B: HOVER REWARDS
        # ==========================================
        # 1. The "Reach" Bonus (+5.0 once for entering the zone)
        just_reached_hover = (dist_new < 0.2) & (dist_old >= 0.2)
        reward_reach = jp.where(
            is_hover_phase & is_active, 5.0 * just_reached_hover, 0.0
        )

        # 2. The "Magnet" Reward (+0.1 max per step -> +5.0 per sec at 50Hz)
        alpha = 5.0
        hover_magnet = jp.exp(-alpha * (dist_new**2))

        # 3. The "Brake" Penalty
        hover_brake = -0.05 * vel_mag

        reward_hover = jp.where(is_hover_phase, (0.1 * hover_magnet) + hover_brake, 0.0)

        # ==========================================
        # UNIVERSAL PENALTIES (Active in all phases)
        # ==========================================
        reward_smooth = -0.005 * jp.linalg.norm(ang_vel, axis=-1)

        # Soft Boundary Penalty
        threshold = 0.05
        dist_to_low = drone_pos - self.data.pos_limit_low
        dist_to_high = self.data.pos_limit_high - drone_pos
        in_danger_zone = jp.any(dist_to_low < threshold, axis=-1) | jp.any(
            dist_to_high < threshold, axis=-1
        )
        reward_boundary = -0.5 * in_danger_zone

        # Crash Penalty (Only apply ONCE on the exact frame it becomes disabled)
        # Assuming you have access to `marked_for_reset` or similar to check the transition.
        # If not, we just apply a flat -10.0 when it dies.
        just_died = self.data.disabled_drones & is_active  # Transition check
        reward_crash = -10.0 * just_died

        # ==========================================
        # FINAL SUMMATION & SCALING
        # ==========================================
        # Only accumulate dense continuous rewards while the drone is actually alive
        dense_rewards = jp.where(
            is_active,
            reward_prog + reward_time + reward_hover + reward_smooth + reward_boundary,
            0.0,
        )

        total_reward = reward_pass + reward_reach + reward_crash + dense_rewards

        # GLOBAL SCALER: Bring everything down to a neural-network-friendly [-1, 1] range.
        scaled_reward = total_reward / 10.0

        return scaled_reward

    # def reward(self) -> Array:
    #     """Compute the composite reward for the vectorized environment."""

    #     # 0. Identify the Current Phase (Navigating vs Hovering)
    #     n_gates = len(self.data.gate_mj_ids)
    #     target_idx = jp.maximum(self.data.last_target_gate, 0)

    #     # Create a boolean mask: True if this specific environment is on the final waypoint
    #     is_last_gate = target_idx == (n_gates - 1)

    #     # 1. Gate Passage Reward
    #     # This still fires for intermediate gates
    #     gate_passed = (self.data.target_gate > self.data.last_target_gate) & (
    #         self.data.target_gate != -1
    #     )
    #     reward_pass = 100.0 * gate_passed

    #     # 2. Crash / Termination Penalty
    #     # (Keeping this active in case boundary hits or other logic still disables drones)
    #     reward_crash = -50.0 * self.data.disabled_drones

    #     # 3. Distance Calculations
    #     gate_ids = self.data.gate_mj_ids[target_idx % n_gates]
    #     mocap_pos = self.sim.mjx_data.mocap_pos
    #     current_gate_pos = mocap_pos[jp.arange(self.sim.n_worlds)[:, None], gate_ids]

    #     drone_pos = self.sim.data.states.pos
    #     dist_old = jp.linalg.norm(self.data.last_drone_pos - current_gate_pos, axis=-1)
    #     dist_new = jp.linalg.norm(drone_pos - current_gate_pos, axis=-1)

    #     # 4. Phase-Dependent Position & Progress Rewards
    #     # Progress (dist_old - dist_new) is great for racing, but equals 0 when hovering perfectly.
    #     # Absolute distance penalty (-dist_new) is terrible for racing (too punitive), but perfect for hovering.
    #     progress = dist_old - dist_new

    #     reward_prog = jp.where(~is_last_gate, 10.0 * progress, 0.0)
    #     reward_hover_pos = jp.where(is_last_gate, -5.0 * dist_new, 0.0)

    #     # 5. NEW: Hover Velocity Penalty
    #     # We must penalize speed heavily at the end so it learns to brake.
    #     # (Assuming linear velocity is stored at self.sim.data.states.vel)
    #     drone_vel = self.sim.data.states.vel
    #     vel_mag = jp.linalg.norm(drone_vel, axis=-1)

    #     # Apply a harsh -1.0 penalty multiplier only when at the final gate
    #     reward_hover_vel = jp.where(is_last_gate, -1.0 * vel_mag, 0.0)

    #     # 6. Smoothness Penalty (Active everywhere)
    #     ang_vel = self.sim.data.states.ang_vel
    #     reward_smooth = -0.01 * jp.linalg.norm(ang_vel, axis=-1)

    #     # 7. Time Penalty
    #     # If we leave the time penalty active while hovering, the agent accumulates infinite
    #     # negative reward and might just fly out of bounds to escape. We turn it off at the end.
    #     reward_time = jp.where(~is_last_gate, -0.05, 0.0)

    #     # 8. Soft Boundary Penalty (Kept exact same as your logic)
    #     threshold = 0.05
    #     dist_to_low = drone_pos - self.data.pos_limit_low
    #     dist_to_high = self.data.pos_limit_high - drone_pos

    #     near_low = jp.any(dist_to_low < threshold, axis=-1)
    #     near_high = jp.any(dist_to_high < threshold, axis=-1)
    #     in_danger_zone = near_low | near_high

    #     reward_boundary = -5.0 * in_danger_zone

    #     drone_z = self.sim.data.states.pos[..., 2]

    #     # Define a "floor threshold" (e.g., z < 0.05)
    #     floor_threshold = 0.05
    #     hit_ground = drone_z < floor_threshold

    #     # Apply a heavy penalty for hitting the ground, EVEN IF collisions are disabled
    #     reward_ground_collision = -50.0 * hit_ground

    #     # Combine dense rewards...
    #     is_active = ~self.data.disabled_drones

    #     # Combine dense rewards and mask them out if the drone is disabled
    #     is_active = ~self.data.disabled_drones
    #     dense_rewards = jp.where(
    #         is_active,
    #         reward_prog
    #         + reward_hover_pos
    #         + reward_hover_vel
    #         + reward_smooth
    #         + reward_time
    #         + reward_boundary
    #         + reward_ground_collision,
    #         0.0,
    #     )

    #     # Total sum
    #     total_reward = reward_pass + reward_crash + dense_rewards
    #     total_reward = total_reward / 100.0
    #     return total_reward

    def terminated(self) -> Array:
        """Check if the episode is terminated.

        Returns:
            True if all drones have been disabled, else False.
        """
        # if self.disable_termination:
        # return np.zeros_like(self.data.disabled_drones, dtype=bool)
        return self.data.disabled_drones

    def truncated(self) -> Array:
        """Array of booleans indicating if the episode is truncated."""
        return self._truncated(
            self.data.steps, self.data.max_episode_steps, self.sim.n_drones
        )

    def info(self) -> dict:
        """Return an info dictionary containing additional information about the environment."""
        return {}

    @property
    def drone_mass(self) -> NDArray[np.floating]:
        """The mass of the drones in the environment."""
        return np.asarray(self.sim.default_data.params.mass[..., 0])

    @staticmethod
    @jax.jit
    def _reset_env_data(
        data: EnvData,
        drone_pos: Array,
        drone_vel: Array,
        mocap_pos: Array,
        mask: Array | None = None,
    ) -> EnvData:
        """Reset auxiliary variables of the environment data."""
        mask = jp.ones(data.steps.shape, dtype=bool) if mask is None else mask
        target_gate = jp.where(mask[..., None], 0, data.target_gate)
        last_drone_pos = jp.where(mask[..., None, None], drone_pos, data.last_drone_pos)
        disabled_drones = jp.where(mask[..., None], False, data.disabled_drones)
        steps = jp.where(mask, 0, data.steps)
        # Check which gates are in range of the drone
        gates_pos = mocap_pos[:, data.gate_mj_ids]
        dpos = drone_pos[..., None, :2] - gates_pos[:, None, :, :2]
        gates_visited = jp.linalg.norm(dpos, axis=-1) < data.sensor_range
        gates_visited = jp.where(
            mask[..., None, None], gates_visited, data.gates_visited
        )
        # And which obstacles are in range
        obstacles_pos = mocap_pos[:, data.obstacle_mj_ids]
        dpos = drone_pos[..., None, :2] - obstacles_pos[:, None, :, :2]
        obstacles_visited = jp.linalg.norm(dpos, axis=-1) < data.sensor_range
        obstacles_visited = jp.where(
            mask[..., None, None], obstacles_visited, data.obstacles_visited
        )
        last_drone_vel = jp.where(mask[..., None, None], drone_vel, data.last_drone_vel)
        accel_buffer = jp.where(mask[..., None, None, None], 0.0, data.accel_buffer)
        gyro_buffer = jp.where(mask[..., None, None, None], 0.0, data.gyro_buffer)

        return data.replace(
            target_gate=target_gate,
            last_drone_pos=last_drone_pos,
            disabled_drones=disabled_drones,
            gates_visited=gates_visited,
            obstacles_visited=obstacles_visited,
            steps=steps,
            marked_for_reset=jp.where(
                mask, 0, data.marked_for_reset
            ),  # Unmark after env reset
            last_drone_vel=last_drone_vel,
            accel_buffer=accel_buffer,  # Update state
            gyro_buffer=gyro_buffer,  # Update state
            last_target_gate=jp.where(mask[..., None], 0, data.last_target_gate),
        )

    @staticmethod
    @jax.jit
    def _step_env(
        data: EnvData,
        drone_pos: Array,
        drone_vel: Array,
        mocap_pos: Array,
        mocap_quat: Array,
        contacts: Array,
        freq: int,
        stay_on_last: bool = False,
    ) -> EnvData:
        """Step the environment data."""
        n_gates = len(data.gate_mj_ids)
        taken_off_drones = (data.steps > freq // 5)[
            :, None
        ]  # Only activate check after 0.2s
        disabled_drones = taken_off_drones & RaceCoreEnv._disabled_drones(
            drone_pos, contacts, data
        )
        gates_pos = mocap_pos[:, data.gate_mj_ids]
        obstacles_pos = mocap_pos[:, data.obstacle_mj_ids]
        # We need to convert the mocap quat from MuJoCo order to scipy order
        gates_quat = mocap_quat[:, data.gate_mj_ids][..., [1, 2, 3, 0]]
        # Extract the gate poses of the current target gates and check if the drones have passed
        # them between the last and current position
        gate_ids = data.gate_mj_ids[data.target_gate % n_gates]
        gate_pos = gates_pos[jp.arange(gates_pos.shape[0])[:, None], gate_ids]
        gate_quat = gates_quat[jp.arange(gates_quat.shape[0])[:, None], gate_ids]
        passed = gate_passed(
            drone_pos, data.last_drone_pos, gate_pos, gate_quat, (0.45, 0.45)
        )
        # Update the target gate index. Increment by one if drones have passed a gate
        new_target_idx = data.target_gate + passed * ~disabled_drones

        target_gate = jp.where(
            stay_on_last,
            jp.clip(new_target_idx, 0, n_gates - 1),
            jp.where(new_target_idx >= n_gates, -1, new_target_idx),
        )
        steps = data.steps + 1
        truncated = steps >= data.max_episode_steps
        marked_for_reset = jp.all(disabled_drones | truncated[..., None], axis=-1)
        # Update which gates and obstacles are or have been in range of the drone
        sensor_range = data.sensor_range
        dpos = drone_pos[..., None, :2] - gates_pos[:, None, :, :2]
        gates_visited = data.gates_visited | (
            jp.linalg.norm(dpos, axis=-1) < sensor_range
        )
        dpos = drone_pos[..., None, :2] - obstacles_pos[:, None, :, :2]
        obstacles_visited = data.obstacles_visited | (
            jp.linalg.norm(dpos, axis=-1) < sensor_range
        )
        data = data.replace(
            last_drone_pos=drone_pos,
            last_drone_vel=drone_vel,
            target_gate=target_gate,
            disabled_drones=disabled_drones,
            marked_for_reset=marked_for_reset,
            gates_visited=gates_visited,
            obstacles_visited=obstacles_visited,
            steps=steps,
            last_target_gate=data.target_gate,
        )
        return data

    @staticmethod
    @jax.jit
    def _obs(
        mocap_pos: Array,
        mocap_quat: Array,
        gates_visited: Array,
        gate_mocap_ids: Array,
        nominal_gate_pos: NDArray,
        nominal_gate_quat: NDArray,
        obstacles_visited: Array,
        obstacle_mocap_ids: Array,
        nominal_obstacle_pos: NDArray,
    ) -> tuple[Array, Array]:
        """Get the nominal or real gate positions and orientations depending on the sensor range."""
        mask, real_pos = gates_visited[..., None], mocap_pos[:, gate_mocap_ids]
        real_quat = mocap_quat[:, gate_mocap_ids][..., [1, 2, 3, 0]]
        gates_pos = jp.where(mask, real_pos[:, None], nominal_gate_pos[None, None])
        gates_quat = jp.where(mask, real_quat[:, None], nominal_gate_quat[None, None])
        mask, real_pos = obstacles_visited[..., None], mocap_pos[:, obstacle_mocap_ids]
        obstacles_pos = jp.where(
            mask, real_pos[:, None], nominal_obstacle_pos[None, None]
        )
        return gates_pos, gates_quat, obstacles_pos

    @staticmethod
    @partial(jax.jit, static_argnames="n_drones")
    def _truncated(steps: Array, max_episode_steps: Array, n_drones: int) -> Array:
        return jp.tile((steps >= max_episode_steps)[..., None], (1, n_drones))

    @staticmethod
    def _disabled_drones(pos: Array, contacts: Array, data: EnvData) -> Array:
        disabled = data.disabled_drones | jp.any(pos < data.pos_limit_low, axis=-1)
        disabled = disabled | jp.any(pos > data.pos_limit_high, axis=-1)

        disabled = disabled | (data.target_gate == -1)
        contacts = jp.any(contacts[:, None, :] & data.contact_masks, axis=-1)
        disabled = disabled | contacts
        return disabled

    @staticmethod
    @jax.jit
    def _warp_disabled_drones(data: SimData, mask: Array) -> SimData:
        """Warp the disabled drones below the ground."""
        pos = jax.numpy.where(mask[..., None], -1, data.states.pos)
        return data.replace(states=data.states.replace(pos=pos))

    @staticmethod
    @jax.jit
    def compute_imu(
        current_vel: Array,
        last_vel: Array,
        current_quat: Array,
        current_ang_vel: Array,
        dt: float,
    ) -> tuple[Array, Array]:
        # --- Accelerometer ---
        # 1. Kinematic acceleration (World Frame)
        accel_world = (current_vel - last_vel) / dt

        # 2. Add gravity (World Frame)
        gravity = jp.array([0.0, 0.0, 9.81])
        proper_accel_world = accel_world + gravity

        # 3. Rotate to Body Frame
        q_inv = RaceCoreEnv._quat_conjugate(current_quat)
        accel_body = RaceCoreEnv._rotate_vector(q_inv, proper_accel_world)

        # --- Gyroscope ---
        # MuJoCo's qvel provides angular velocity ALREADY in the local body frame.
        # We do not need to rotate it!
        gyro_body = current_ang_vel

        return accel_body, gyro_body

    def _setup_sim(self, randomizations: dict):
        """Setup the simulation data and build the reset and step functions with custom hooks."""
        gate_spec = mujoco.MjSpec.from_file(str(self.gate_spec_path))
        obstacle_spec = mujoco.MjSpec.from_file(str(self.obstacle_spec_path))
        self._load_track_into_sim(gate_spec, obstacle_spec)
        # Set the initial drone states
        pos = self.sim.data.states.pos.at[...].set(self.drone["pos"])
        quat = self.sim.data.states.quat.at[...].set(self.drone["quat"])
        vel = self.sim.data.states.vel.at[...].set(self.drone["vel"])
        ang_vel = self.sim.data.states.ang_vel.at[...].set(self.drone["ang_vel"])
        states = self.sim.data.states.replace(
            pos=pos, quat=quat, vel=vel, ang_vel=ang_vel
        )
        self.sim.data = self.sim.data.replace(states=states)
        self.sim.build_default_data()
        # Build the reset randomizations and disturbances into the sim itself
        self.sim.reset_pipeline = self.sim.reset_pipeline + (
            build_reset_fn(randomizations),
        )
        self.sim.build_reset_fn()
        if "dynamics" in self.disturbances:
            disturbance_fn = build_dynamics_disturbance_fn(
                self.disturbances["dynamics"]
            )
            self.sim.step_pipeline = (
                self.sim.step_pipeline[:2]
                + (disturbance_fn,)
                + self.sim.step_pipeline[2:]
            )
            self.sim.build_step_fn()

    def _load_track_into_sim(self, gate_spec: MjSpec, obstacle_spec: MjSpec):
        """Load the track into the simulation."""
        frame = self.sim.spec.worldbody.add_frame()
        n_gates, n_obstacles = len(self.gates["pos"]), len(self.obstacles["pos"])
        for i in range(n_gates):
            gate_body = gate_spec.body("gate")
            if gate_body is None:
                raise ValueError("Gate body not found in gate spec")
            gate = frame.attach_body(gate_body, "", f":{i}")
            gate.pos = self.gates["pos"][i]
            # Convert from scipy order to MuJoCo order
            gate.quat = self.gates["quat"][i][[3, 0, 1, 2]]
            gate.mocap = (
                True  # Make mocap to modify the position of static bodies during sim
            )
        for i in range(n_obstacles):
            obstacle_body = obstacle_spec.body("obstacle")
            if obstacle_body is None:
                raise ValueError("Obstacle body not found in obstacle spec")
            obstacle = frame.attach_body(obstacle_body, "", f":{i}")
            obstacle.pos = self.obstacles["pos"][i]
            obstacle.mocap = True
        self.sim.build_mjx()

    @staticmethod
    def _load_contact_masks(sim: Sim, disable_collisions: bool = False) -> Array:
        """Load contact masks for the simulation that zero out irrelevant contacts per drone."""

        sim.contacts()  # Trigger initial contact information computation
        contact = sim.mjx_data._impl.contact
        n_contacts = len(contact.geom1[0])
        masks = np.zeros((sim.n_drones, n_contacts), dtype=bool)

        geom1, geom2 = (contact.geom1[0], contact.geom2[0])

        # 1. Unconditionally identify the "world" (ground) geoms
        world_id = sim.mj_model.body("world").id
        w_start = sim.mj_model.body_geomadr[world_id]
        w_count = sim.mj_model.body_geomnum[world_id]

        world_active = (geom1 >= w_start) & (geom1 < w_start + w_count) | (
            geom2 >= w_start
        ) & (geom2 < w_start + w_count)

        for i in range(sim.n_drones):
            drone_id = sim.mj_model.body(f"drone:{i}").id
            d_start = sim.mj_model.body_geomadr[drone_id]
            d_count = sim.mj_model.body_geomnum[drone_id]

            # 2. Unconditionally identify geoms belonging to THIS drone
            drone_active = (geom1 >= d_start) & (geom1 < d_start + d_count) | (
                geom2 >= d_start
            ) & (geom2 < d_start + d_count)

            # 3. Apply the conditional logic for the mask
            if disable_collisions:
                # If collisions are "disabled", ONLY allow Drone <-> Floor collisions
                masks[i, :] = drone_active & world_active
            else:
                # If collisions are enabled, allow the Drone to collide with ANYTHING
                masks[i, :] = drone_active

        masks = np.tile(masks[None, ...], (sim.n_worlds, 1, 1))
        return masks


# region Factories


def rng_spec2fn(fn_spec: dict) -> Callable:
    """Convert a function spec to a wrapped and scaled function from jax.random."""
    offset, scale = np.array(fn_spec.get("offset", 0)), np.array(
        fn_spec.get("scale", 1)
    )
    kwargs = fn_spec.get("kwargs", {})
    if "shape" in kwargs:
        raise KeyError("Shape must not be specified for randomization functions.")
    kwargs = {k: np.array(v) if isinstance(v, list) else v for k, v in kwargs.items()}
    jax_fn = partial(getattr(jax.random, fn_spec["fn"]), **kwargs)

    def random_fn(*args: Any, **kwargs: Any) -> Array:
        return jax_fn(*args, **kwargs) * scale + offset

    return random_fn


def build_reset_fn(randomizations: dict) -> Callable[[SimData, Array], SimData]:
    """Build the reset hook for the simulation."""
    randomization_fns = ()
    for target, rng in sorted(randomizations.items()):
        match target:
            case "drone_pos":
                randomization_fns += (randomize_drone_pos_fn(rng),)
            case "drone_rpy":
                randomization_fns += (randomize_drone_quat_fn(rng),)
            case "drone_mass":
                randomization_fns += (randomize_drone_mass_fn(rng),)
            case "drone_inertia":
                randomization_fns += (randomize_drone_inertia_fn(rng),)
            case "gate_pos" | "gate_rpy" | "obstacle_pos":
                pass
            case _:
                raise ValueError(f"Invalid target: {target}")

    def reset_fn(data: SimData, mask: Array) -> SimData:
        for randomize_fn in randomization_fns:
            data = randomize_fn(data, mask)
        return data

    return reset_fn


def build_track_randomization_fn(
    randomizations: dict, gate_mocap_ids: list[int], obstacle_mocap_ids: list[int]
) -> Callable[[Data, Array, jax.random.PRNGKey], Data]:
    """Build the track randomization function for the simulation."""
    randomization_fns = ()
    for target, rng in sorted(randomizations.items()):
        match target:
            case "gate_pos":
                randomization_fns += (randomize_gate_pos_fn(rng, gate_mocap_ids),)
            case "gate_rpy":
                randomization_fns += (randomize_gate_rpy_fn(rng, gate_mocap_ids),)
            case "obstacle_pos":
                randomization_fns += (
                    randomize_obstacle_pos_fn(rng, obstacle_mocap_ids),
                )
            case "drone_pos" | "drone_rpy" | "drone_mass" | "drone_inertia":
                pass
            case _:
                raise ValueError(f"Invalid target: {target}")

    @jax.jit
    def track_randomization(
        data: Data,
        mask: Array,
        nominal_gate_pos: Array,
        nominal_gate_quat: Array,
        nominal_obstacle_pos: Array,
        key: jax.random.PRNGKey,
    ) -> Data:
        gate_quat = jp.roll(
            nominal_gate_quat, 1, axis=-1
        )  # Convert from scipy to MuJoCo order

        # Reset to default track positions first
        data = data.replace(
            mocap_pos=data.mocap_pos.at[:, gate_mocap_ids].set(nominal_gate_pos)
        )
        data = data.replace(
            mocap_quat=data.mocap_quat.at[:, gate_mocap_ids].set(gate_quat)
        )
        data = data.replace(
            mocap_pos=data.mocap_pos.at[:, obstacle_mocap_ids].set(nominal_obstacle_pos)
        )
        keys = jax.random.split(key, len(randomization_fns))
        for key, randomize_fn in zip(keys, randomization_fns, strict=True):
            data = randomize_fn(data, mask, key)
        return data

    return track_randomization


def build_dynamics_disturbance_fn(
    fn: Callable[[jax.random.PRNGKey, tuple[int]], jax.Array],
) -> Callable[[SimData], SimData]:
    """Build the dynamics disturbance function for the simulation."""

    def dynamics_disturbance(data: SimData) -> SimData:
        key, subkey = jax.random.split(data.core.rng_key)
        states = data.states
        states = states.replace(force=fn(subkey, states.force.shape))  # World frame
        return data.replace(states=states, core=data.core.replace(rng_key=key))

    return dynamics_disturbance
