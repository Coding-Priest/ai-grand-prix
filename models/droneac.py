import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
from scipy.spatial.transform import Rotation as R


class DroneActorCritic(nn.Module):
    def __init__(self, num_gates=4, hidden_layers=[256, 256, 256], std_init=-0.5):
        super(DroneActorCritic, self).__init__()

        self.num_gates = num_gates

        # --- Calculate Input Dimension ---
        # Quat(4) + Vel(3) + AngVel(3) + Target_Gate_OneHot(N)
        # + Ego_Gates_Pos(3 * N) + Ego_Gates_Quat(4 * N)
        self.state_dim = (
            4 + 3 + 3 + self.num_gates + (3 * self.num_gates) + (4 * self.num_gates)
        )
        self.action_dim = 4

        # --- Build Actor Network ---
        actor_layers = []
        in_dim = self.state_dim
        for h_dim in hidden_layers:
            actor_layers.append(nn.Linear(in_dim, h_dim))
            actor_layers.append(nn.ReLU())
            in_dim = h_dim

        actor_layers.append(nn.Linear(in_dim, self.action_dim))
        actor_layers.append(nn.Tanh())
        self.actor_mean = nn.Sequential(*actor_layers)

        self.actor_log_std = nn.Parameter(torch.ones(1, self.action_dim) * std_init)

        # --- Build Critic Network ---
        critic_layers = []
        in_dim = self.state_dim
        for h_dim in hidden_layers:
            critic_layers.append(nn.Linear(in_dim, h_dim))
            critic_layers.append(nn.ReLU())
            in_dim = h_dim

        critic_layers.append(nn.Linear(in_dim, 1))
        self.critic = nn.Sequential(*critic_layers)

    def forward(self, state):
        value = self.critic(state)
        action_mean = self.actor_mean(state)
        action_std = self.actor_log_std.exp().expand_as(action_mean)
        dist = Normal(action_mean, action_std)
        return dist, value

    def format_state(
        self,
        drone_pos,
        drone_quat,
        vel,
        ang_vel,
        target_gate_idx,
        gates_pos,
        gates_quat,
    ):
        """
        Converts global observations to ego-centric tensors and flattens them.
        Assumes quaternions are in [x, y, z, w] format.
        """
        # Ensure inputs are numpy arrays for Scipy processing
        drone_pos = np.array(drone_pos)
        gates_pos = np.array(gates_pos)
        gates_quat = np.array(gates_quat)

        # 1. --- Ego-Centric Conversion ---
        if len(gates_pos) > 0:
            # Create a Scipy Rotation object for the drone and its inverse
            r_drone = R.from_quat(drone_quat)
            r_drone_inv = r_drone.inv()

            # Translate and Rotate Positions
            rel_pos = gates_pos - drone_pos
            ego_gates_pos = r_drone_inv.apply(rel_pos)

            # Rotate Quaternions
            r_gates = R.from_quat(gates_quat)
            r_ego_gates = r_drone_inv * r_gates
            ego_gates_quat = r_ego_gates.as_quat()
        else:
            ego_gates_pos = np.array([])
            ego_gates_quat = np.array([])

        # 2. --- Tensor Conversion & Padding ---
        t_drone_quat = torch.as_tensor(drone_quat, dtype=torch.float32)
        t_vel = torch.as_tensor(vel, dtype=torch.float32)
        t_ang_vel = torch.as_tensor(ang_vel, dtype=torch.float32)

        # Target Gate One-Hot
        t_target_one_hot = torch.zeros(self.num_gates, dtype=torch.float32)
        if 0 <= target_gate_idx < self.num_gates:
            t_target_one_hot[target_gate_idx] = 1.0

        # Flatten and pad Ego Positions
        t_gates_pos = torch.as_tensor(ego_gates_pos, dtype=torch.float32).flatten()
        if len(t_gates_pos) < (3 * self.num_gates):
            t_gates_pos = torch.cat(
                [t_gates_pos, torch.zeros((3 * self.num_gates) - len(t_gates_pos))]
            )

        # Flatten and pad Ego Quaternions
        t_gates_quat = torch.as_tensor(ego_gates_quat, dtype=torch.float32).flatten()
        if len(t_gates_quat) < (4 * self.num_gates):
            # Quaternions pad with [0, 0, 0, 1] (identity) instead of pure zeros
            pad_count = self.num_gates - (len(t_gates_quat) // 4)
            identity_pad = torch.tensor([0.0, 0.0, 0.0, 1.0]).repeat(pad_count)
            t_gates_quat = torch.cat([t_gates_quat, identity_pad])

        # 3. --- Final Concatenation ---
        state = torch.cat(
            [
                t_drone_quat,
                t_vel,
                t_ang_vel,
                t_target_one_hot,
                t_gates_pos,
                t_gates_quat,
            ]
        )

        return state.unsqueeze(0)
