import os
import sys

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from lsy_drone_racing.control import Controller
import lsy_drone_racing.reward as reward
import lsy_drone_racing.model as model
import lsy_drone_racing.utils as utils

class FullMetalRL(Controller):
    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        super().__init__(obs, info, config)

        self.train  = config.rl.train
        self.reward = getattr(reward, config.rl.reward, None)
        self.model = getattr(model, config.rl.model, None)

        assert not self.train or (self.reward is not None), f"invalid reward '{config.rl.reward}'"
        assert self.model is not None, f"invalid model '{config.rl.model}'"
 
        _ckpt = Path(__file__).parent.parent / ".ckpt" / config.rl.checkpoint
        self.agent = self.model.Agent(self.train, 19, alpha=0.001, gamma=0.5, ckpt=_ckpt)
        
        self.save = Path(__file__).parent.parent / ".ckpt" / config.rl.save

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
        ]]) # shape = (1, 19)

        act = self.agent.forward(statev).squeeze()
        return act

    def step_callback(
            self,
            act: NDArray[np.floating],
            obs: dict[str, NDArray[np.floating]],
            reward: float,
            terminated: bool,
            truncated: bool,
            info: dict) -> bool:
        r = self.reward(obs, act, terminated)
        self.agent.consume(r)
        return False

    def episode_callback(self):
        if not self.train:
            return 
        loss, ret = self.agent.backward()
        print(f"episode loss: {loss:.6f} \tepisode return: {ret:.6f}")
        if not self.save.is_dir():
            self.agent.save(self.save)
