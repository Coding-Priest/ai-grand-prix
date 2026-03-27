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

I = 0

def reward(
        obs: dict[str, NDArray[np.floating]],
        act: NDArray[np.floating],
        terminated: bool) -> float:

    global I

    if terminated:
        return -50.0 

    sbo = 0.0 

    i = obs["target_gate"]
    dx = (obs["pos"][0] - obs["gates_pos"][i][0]) 
    dy = (obs["pos"][1] - obs["gates_pos"][i][1]) 
    dz = (obs["pos"][2] - obs["gates_pos"][i][2]) 

    dr2 = dx*dx + dy*dy + dz*dz

    qx, qy, qz, qw = obs["quat"]
    tp = qx**2 + qy**2 

    if I != i:
        I = i
        return 50

    step_reward = sbo - (2.0 * dr2)

    return step_reward
