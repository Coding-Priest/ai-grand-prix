import numpy as np
from numpy.typing import NDArray

from lsy_drone_racing.control import Controller

class FullMetalRL(Controller):
    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        super().__init__(obs, info, config)
        
    def compute_control(
            self, 
            obs: dict[str, NDArray[np.floating]],
            info: dict | None = None) -> NDArray[np.floating]:

        # obs
        # pos              : [x, y, z]
        # quat             : [qx, qy, qz, qw]
        # vel              : [vx, vy, vz]
        # ang_vel          : [wx, wy, wz]
        # target_gate      : i (index of the next gate)
        # gates_pos        : [[x0, y0, z0], [x1, y1, z1] ...]
        # gates_quat       : [[qx0, qy0, qz0, qw0], [qx1, qy1, qz1, qw1] ...]
        # gates_visited    : [b0, b1, b2...] (boolean)
        # obstacles_pos    : [[x0, y0, z0], [x1, y1, z1] ...]
        # obstacles_visited: [b0, b1, b2...] (boolean)

        # act
        # act[0] = r_des
        # act[1] = p_des
        # act[2] = y_des
        # act[3] = t_des

        act = np.array([0.0, 0.0, 0.0, 0.6], dtype=np.float32)
        return act

    def step_callback(
            self,
            act: NDArray[np.floating],
            obs: dict[str, NDArray[np.floating]],
            reward: float,
            terminated: bool,
            truncated: bool,
            info: dict) -> bool:
        print(terminated, truncated)
        return False

    def episode_callback(self):
        pass
