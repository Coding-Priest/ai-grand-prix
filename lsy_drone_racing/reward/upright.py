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

    qx, qy, qz, qw = obs["quat"]
    yv = math.atan2(qw, qz)
    dz = obs["pos"][2] - obs["gates_pos"][0][2]
    
    return -math.tanh(abs(math.cos(yv)) + 10 * terminated)
