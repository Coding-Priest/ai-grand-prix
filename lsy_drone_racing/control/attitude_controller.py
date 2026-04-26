"""This module implements an AttitudeController for quadrotor control.

It utilizes the collective thrust interface for drone control to compute control commands based on
current state observations and desired waypoints. The attitude control is handled by computing a
PID control law for position tracking, incorporating gravity compensation in thrust calculations.

The waypoints are generated using cubic spline interpolation from a set of predefined waypoints.
Note that the trajectory uses pre-defined waypoints instead of dynamically generating a good path.
"""

from __future__ import annotations  # Python 3.10 type hints

import math
from typing import TYPE_CHECKING

import numpy as np
from drone_models.core import load_params
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R

import lsy_drone_racing.utils as utils
from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray

class Buffer:
    def __init__(self):
        self.obs = []
        self.act = []

    def consume(self, act, obs):
        self.act.append(act)
        self.obs.append(obs)

    def save(self, file):

        obs_arr = np.array(self.obs)
        act_arr = np.array(self.act)
        np.savez_compressed(file, obs=obs_arr, act=act_arr)

        self.obs = []
        self.act = []

class AttitudeController(Controller):
    """Example of a controller using the collective thrust and attitude interface."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize the attitude controller.

        Args:
            obs: The initial observation of the environment's state. See the environment's
                observation space for details.
            info: Additional environment information from the reset.
            config: The configuration of the environment.
        """
        super().__init__(obs, info, config)
        self._freq = config.env.freq

        self.buffer = Buffer()

        drone_params = load_params(config.sim.physics, config.sim.drone_model)
        self.drone_mass = drone_params["mass"]  # alternatively from sim.drone_mass

        self.kp = np.array([0.4, 0.4, 1.25])
        self.ki = np.array([0.05, 0.05, 0.05])
        self.kd = np.array([0.2, 0.2, 0.4])
        self.ki_range = np.array([2.0, 2.0, 0.4])
        self.i_error = np.zeros(3)
        self.g = 9.81

        # Same waypoints as in the position controller. Determined by trial and error.
        waypoints = np.array(
            [
                [-1.5, 0.75, 0.05],
                [-1.0, 0.55, 0.4],
                [0.3, 0.35, 0.7],
                [1.3, -0.15, 0.9],
                [0.85, 0.85, 1.2],
                [-0.5, -0.05, 0.7],
                [-1.2, -0.2, 0.8],
                [-1.2, -0.2, 1.2],
                [-0.0, -0.7, 1.2],
                [0.5, -0.75, 1.2],
            ]
        )
        self._t_total = 15  # s
        t = np.linspace(0, self._t_total, len(waypoints))
        self._des_pos_spline = CubicSpline(t, waypoints)
        self._des_vel_spline = self._des_pos_spline.derivative()

        self._tick = 0
        self._finished = False

        self.episode_n = 0

    def compute_control(
            self, obs: dict[str, NDArray[np.floating]], info: dict | None = None) -> NDArray[np.floating]:
        """Compute the next desired collective thrust and roll/pitch/yaw of the drone.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                for details.
            info: Optional additional information as a dictionary.

        Returns:
            The orientation as roll, pitch, yaw angles, and the collective thrust
            [r_des, p_des, y_des, t_des] as a numpy array.
        """
        t = min(self._tick / self._freq, self._t_total)
        if t >= self._t_total:  # Maximum duration reached
            self._finished = True

        des_pos = self._des_pos_spline(t)
        des_vel = self._des_vel_spline(t)
        des_yaw = 0.0

        # Calculate the deviations from the desired trajectory
        pos_error = des_pos - obs["pos"]
        vel_error = des_vel - obs["vel"]

        # Update integral error
        self.i_error += pos_error * (1 / self._freq)
        self.i_error = np.clip(self.i_error, -self.ki_range, self.ki_range)

        # Compute target thrust
        target_thrust = np.zeros(3)
        target_thrust += self.kp * pos_error
        target_thrust += self.ki * self.i_error
        target_thrust += self.kd * vel_error
        target_thrust[2] += self.drone_mass * self.g

        # Update z_axis to the current orientation of the drone
        z_axis = R.from_quat(obs["quat"]).as_matrix()[:, 2]

        # update current thrust
        thrust_desired = target_thrust.dot(z_axis)

        # update z_axis_desired
        z_axis_desired = target_thrust / np.linalg.norm(target_thrust)
        x_c_des = np.array([math.cos(des_yaw), math.sin(des_yaw), 0.0])
        y_axis_desired = np.cross(z_axis_desired, x_c_des)
        y_axis_desired /= np.linalg.norm(y_axis_desired)
        x_axis_desired = np.cross(y_axis_desired, z_axis_desired)

        R_desired = np.vstack([x_axis_desired, y_axis_desired, z_axis_desired]).T
        euler_desired = R.from_matrix(R_desired).as_euler("xyz", degrees=False)

        action = np.concatenate([euler_desired, [thrust_desired]], dtype=np.float32)

        return action

    def step_callback(
            self, action: NDArray[np.floating],
            obs: dict[str, NDArray[np.floating]],
            reward: float, terminated: bool,
            truncated: bool, info: dict
        ) -> bool:
        """Increment the tick counter.

        Returns:
            True if the controller is finished, False otherwise.
        """
        x,  y,  z  = obs["pos"]
        r,  p,  ya = utils.tr.quat2rpy(*obs["quat"])
        vx, vy, vz = obs["vel"]
        wx, wy, wz = obs["ang_vel"]

        # external state
        _i = obs["target_gate"]

        gx, gy, gz  = obs["gates_pos"][_i]
        gr, gp, gya = utils.tr.quat2rpy(*obs["gates_quat"][_i])
        del _i, gr, gp

        _dp  = np.array([x, y, z])
        _obp = np.array(obs["obstacles_pos"])
        _ds  = np.sum((_obp - _dp)**2, axis=1)        
        _o   = int(np.argmin(_ds))

        ox, oy, oz = obs["obstacles_pos"][_o]
        del _dp, _obp, _ds, _o

        statev = np.array([[
            x,  y,  z,  r,  p,  ya,
            vx, vy, vz, wx, wy, wz,
            gx, gy, gz, gya,
            ox, oy, oz
        ]], dtype=np.float32) # shape = (1, 19)

        self.buffer.consume(action, statev)
        self._tick += 1
        return self._finished

    def episode_callback(self):
        """Reset the internal state."""
        self.i_error[:] = 0
        self._tick = 0
        self.buffer.save(f"episode{self.episode_n}.npz")
        self.episode_n += 1
