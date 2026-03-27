import os
import sys

import math
import random
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from lsy_drone_racing.control import Controller
import lsy_drone_racing.reward as reward
import lsy_drone_racing.model as model
import lsy_drone_racing.utils as utils

from drone_models.core import load_params
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R

class FullMetalRL(Controller):
    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        super().__init__(obs, info, config)

        self.train  = config.rl.train
        self.reward = getattr(reward, config.rl.reward, None)
        self.model  = getattr(model, config.rl.model, None)

        assert not self.train or (self.reward is not None), f"invalid reward '{config.rl.reward}'"
        assert self.model is not None, f"invalid model '{config.rl.model}'"
 
        _ckpt = Path(__file__).parent.parent / ".ckpt" / config.rl.checkpoint
        self.agent = self.model.Agent(self.train, 19, alpha=0.01, gamma=0.8, ckpt=_ckpt)
        self.save = Path(__file__).parent.parent / ".ckpt" / config.rl.save
        self.ep = 0

        # immitation learning
        self.il = False
        self.il_prob = config.rl.il_prob

        self._freq = config.env.freq
        drone_params = load_params(config.sim.physics, config.sim.drone_model)
        self.drone_mass = drone_params["mass"]  # alternatively from sim.drone_mass

        self.kp = np.array([0.4, 0.4, 1.25])
        self.ki = np.array([0.05, 0.05, 0.05])
        self.kd = np.array([0.2, 0.2, 0.4])
        self.ki_range = np.array([2.0, 2.0, 0.4])
        self.i_error = np.zeros(3)
        self.g = 9.81

        # Same waypoints as in the position controller. Determined by trial and error.
        waypoints = np.array([
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
        ])
        self._t_total = 15  # s
        t = np.linspace(0, self._t_total, len(waypoints))
        self._des_pos_spline = CubicSpline(t, waypoints)
        self._des_vel_spline = self._des_pos_spline.derivative()
        self._tick = 0
        self._finished = False

    def il_control(
            self, 
            obs: dict[str, NDArray[np.floating]],
            info: dict | None = None) -> NDArray[np.floating]:
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

    def compute_control(
            self, 
            obs: dict[str, NDArray[np.floating]],
            info: dict | None = None) -> NDArray[np.floating]:

        # obs
        # pos              : [x, y, z]
        # quat             : [qx, qy, qz, qw]
        # vel              : [vx, vy, vz]
        # ang_vel          : [wx, wy, wz]
        # target_gate      : i                                (index of the next gate)
        # gates_pos        : [[x0, y0, z0], [x1, y1, z1] ...]
        # gates_quat       : [[qx0, qy0, qz0, qw0], ...]
        # gates_visited    : [b0, b1, b2...]                  (boolean)
        # obstacles_pos    : [[x0, y0, z0], [x1, y1, z1] ...]
        # obstacles_visited: [b0, b1, b2...]                  (boolean)

        # act
        # act[0]: r_des
        # act[1]: p_des
        # act[2]: y_des
        # act[3]: t_des

        # internal state
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

        if self.il:
            iact = self.il_control(obs, info)
            act, _ = self.agent.forward(statev, iact)
        else:
            act, confidence = self.agent.forward(statev)
        return act

    def step_callback(
            self,
            act: NDArray[np.floating],
            obs: dict[str, NDArray[np.floating]],
            reward: float,
            terminated: bool,
            truncated: bool,
            info: dict) -> bool:

        self._tick += 1
        r = self.reward(obs, act, terminated)
        self.agent.consume(r)
        return False

    def episode_callback(self):

        self.i_error[:] = 0
        self._tick = 0

        self.il = random.random() < self.il_prob
        if self.il:
            print("immitation learning")

        if not self.train:
            return 
        loss, ret = self.agent.backward()
        print(f"episode{self.ep} loss: {loss:.6f} \treturn: {ret:.6f}")
        self.ep += 1
        if not self.save.is_dir():
            self.agent.save(self.save)
