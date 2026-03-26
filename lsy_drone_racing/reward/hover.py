# reward

# positional
# dx = (gates_pos[i][0] - x)
# dy = (gates_pos[i][1] - y)
# dz = (gates_pos[i][2] - z)
# dr = dx^2 + dy^2 + dz^2

# orientational
# yv  = atan2(qw, qz) `yaw vector`
# ur  = cos(yv)       `up right cosine` 
# yaw = atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))

import math
import numpy as np
from numpy.typing import NDArray
import lsy_drone_racing.utils as utils

def reward(
        obs: dict[str, NDArray[np.floating]],
        act: NDArray[np.floating],
        terminated: bool) -> float:

    if terminated:
        return -50.0 

    sbo = 1.0 

    tz = obs["gates_pos"][0][2]
    cz = obs["pos"][2]
    dz_sq = (cz - tz) ** 2 

    qx, qy, qz, qw = obs["quat"]
    tp = (qx**2 + qy**2) 

    step_reward = sbo - (10.0 * dz_sq) - (10.0 * tp)

    return step_reward
