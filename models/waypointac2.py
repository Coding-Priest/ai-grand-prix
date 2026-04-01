import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
from scipy.spatial.transform import Rotation as R


class WaypointActorCritic2(nn.Module):
    def __init__(self, window_size=3, hidden_layers=[256, 256], std_init=-0.5):
        """
        Args:
            window_size: The number of lookahead waypoints the drone can see.
            hidden_layers: List defining the size of the hidden layers.
            std_init: Initial value for the log standard deviation of the actions.
        """
        super(WaypointActorCritic2, self).__init__()

        self.window_size = window_size

        # --- Calculate Input Dimension ---
        # Quat(4) + Vel(3) + AngVel(3) + [Pos(3) + Quat(4)] * window_size
        self.state_dim = 4 + 3 + 3 + (3 * self.window_size) + (4 * self.window_size) + 1
        self.action_dim = 4  # Roll, Pitch, Yaw, Thrust

        # --- Build Actor Network ---
        actor_layers = []
        in_dim = self.state_dim
        for h_dim in hidden_layers:
            actor_layers.append(nn.Linear(in_dim, h_dim))
            actor_layers.append(nn.ReLU())  # Mish or Tanh also work well here
            in_dim = h_dim

        actor_layers.append(nn.Linear(in_dim, self.action_dim))
        actor_layers.append(nn.Tanh())  # Bound actions to [-1, 1]
        self.actor_mean = nn.Sequential(*actor_layers)

        # Action standard deviation (learned parameter, independent of state)
        self.actor_log_std = nn.Parameter(torch.ones(1, self.action_dim) * std_init)

        # --- Build Critic Network ---
        critic_layers = []
        in_dim = self.state_dim
        for h_dim in hidden_layers:
            critic_layers.append(nn.Linear(in_dim, h_dim))
            critic_layers.append(nn.ReLU())
            in_dim = h_dim

        critic_layers.append(nn.Linear(in_dim, 1))  # Critic outputs a single Value
        self.critic = nn.Sequential(*critic_layers)

        # --- Initialization ---
        self._init_weights()

    def _init_weights(self):
        """Orthogonal initialization for stable PPO training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                module.bias.data.zero_()

    def forward(self, state):
        """Calculates both the action distribution and the state value."""
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
        use_pass_through=False,
    ):
        from scipy.spatial.transform import Rotation as R

        drone_pos = np.array(drone_pos)
        drone_quat = np.array(drone_quat)
        vel = np.array(vel)
        ang_vel = np.array(ang_vel)
        target_gate_idx = np.array(target_gate_idx)
        gates_pos = np.array(gates_pos)
        gates_quat = np.array(gates_quat)

        is_batched = drone_pos.ndim > 1
        batch_size = drone_pos.shape[0] if is_batched else 1

        if not is_batched:
            drone_pos = np.expand_dims(drone_pos, axis=0)
            drone_quat = np.expand_dims(drone_quat, axis=0)
            vel = np.expand_dims(vel, axis=0)
            ang_vel = np.expand_dims(ang_vel, axis=0)
            target_gate_idx = np.expand_dims(target_gate_idx, axis=0)

        has_batched_gates = gates_pos.ndim == 3
        processed_states = []

        for i in range(batch_size):
            d_pos = drone_pos[i]
            d_quat = drone_quat[i]
            v = vel[i]
            a_vel = ang_vel[i]
            t_idx = int(target_gate_idx[i].item())

            g_pos = gates_pos[i] if has_batched_gates else gates_pos
            g_quat = gates_quat[i] if has_batched_gates else gates_quat

            # --- Lookahead Extraction & Final Gate Logic ---
            total_gates = len(g_pos)
            is_final_gate = 0.0

            if t_idx == -1:
                t_idx = total_gates - 1
                is_final_gate = 1.0  # FIX: Network now knows this is the end!

            end_idx = min(t_idx + self.window_size, total_gates)
            lookahead_pos = list(g_pos[t_idx:end_idx])
            lookahead_quat = list(g_quat[t_idx:end_idx])

            # --- Ghost Gate Padding ---
            while len(lookahead_pos) < self.window_size:
                if use_pass_through:
                    last_pos = np.array(lookahead_pos[-1])
                    last_quat = np.array(lookahead_quat[-1])

                    gate_rot = R.from_quat(last_quat)
                    forward_vec = gate_rot.apply(np.array([1.0, 0.0, 0.0]))

                    ghost_pos = last_pos + (forward_vec * 2.0)
                    lookahead_pos.append(ghost_pos)
                    lookahead_quat.append(last_quat)
                else:
                    lookahead_pos.append(lookahead_pos[-1])
                    lookahead_quat.append(lookahead_quat[-1])

            lookahead_pos = np.array(lookahead_pos)
            lookahead_quat = np.array(lookahead_quat)

            # --- Ego-Centric Conversion ---
            r_drone = R.from_quat(d_quat)
            r_drone_inv = r_drone.inv()

            # 1. Gates to Ego Frame
            rel_pos = lookahead_pos - d_pos
            ego_gates_pos = r_drone_inv.apply(rel_pos)

            r_gates = R.from_quat(lookahead_quat)
            r_ego_gates = r_drone_inv * r_gates
            ego_gates_quat = r_ego_gates.as_quat()

            # 2. Velocities to Ego Frame (CRITICAL FIX)
            ego_vel = r_drone_inv.apply(v)
            ego_ang_vel = r_drone_inv.apply(a_vel)

            # --- Tensor Conversion ---
            # NOTE: I left t_drone_quat in here so your visualization parser doesn't break,
            # but consider replacing this with an ego-projected gravity vector in the future.
            t_drone_quat = torch.as_tensor(d_quat, dtype=torch.float32)

            t_vel = torch.as_tensor(ego_vel, dtype=torch.float32) / 10.0
            t_ang_vel = torch.as_tensor(ego_ang_vel, dtype=torch.float32) / 10.0

            t_gates_pos = torch.as_tensor(ego_gates_pos, dtype=torch.float32).flatten()
            t_gates_quat = torch.as_tensor(
                ego_gates_quat, dtype=torch.float32
            ).flatten()
            t_is_final = torch.tensor([is_final_gate], dtype=torch.float32)

            state = torch.cat(
                [t_drone_quat, t_vel, t_ang_vel, t_gates_pos, t_gates_quat, t_is_final]
            )
            processed_states.append(state)

        return torch.stack(processed_states, dim=0)
