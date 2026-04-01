import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


class VisualizeEgoSimV3:
    def __init__(self, window_size=3):
        self.window_size = window_size

        # 1. Setup the Figure for Side-by-Side
        plt.ion()
        self.fig = plt.figure(figsize=(16, 7))

        # --- LEFT: WORLD VIEW ---
        self.ax_world = self.fig.add_subplot(121, projection="3d")
        self.ax_world.set_title("World View (Absolute)")

        (self.trajectory_line,) = self.ax_world.plot(
            [], [], [], "b-", label="Drone Path", alpha=0.5
        )
        self.gates_scatter = self.ax_world.scatter(
            [], [], [], c="green", marker="s", s=40, label="Track Gates", alpha=0.3
        )
        self.target_scatter = self.ax_world.scatter(
            [], [], [], c="magenta", marker="*", s=200, label="Target Gate"
        )

        self.world_lookahead_scatter = self.ax_world.scatter(
            [],
            [],
            [],
            c="yellow",
            marker="o",
            s=80,
            edgecolor="black",
            label="Tensor Lookahead",
            zorder=5,
        )

        self.drone_arrow = None
        self.ax_world.set_xlim([-5, 5])
        self.ax_world.set_ylim([-5, 5])
        self.ax_world.set_zlim([0, 10])
        self.ax_world.legend(loc="upper right")

        # --- RIGHT: EGO VIEW ---
        self.ax_ego = self.fig.add_subplot(122, projection="3d")
        self.ax_ego.set_title("Model Ego View (From Tensor)")

        # Drone orientation indicators
        self.ax_ego.quiver(
            0,
            0,
            0,
            1,
            0,
            0,
            length=1.0,
            color="red",
            linewidth=3,
            label="Ego Forward (+X)",
        )
        self.gravity_arrow = None  # Will be updated from tensor

        self.ego_gates_scatter = self.ax_ego.scatter(
            [],
            [],
            [],
            c="cyan",
            marker="s",
            s=60,
            edgecolor="black",
            label="Lookahead Gates",
        )

        self.ax_ego.set_xlim([-5, 5])
        self.ax_ego.set_ylim([-5, 5])
        self.ax_ego.set_zlim([-5, 5])
        self.ax_ego.set_xlabel("Forward (+X)")
        self.ax_ego.set_ylabel("Left/Right (+Y)")
        self.ax_ego.set_zlabel("Up/Down (+Z)")
        self.ax_ego.legend(loc="upper right")

        self.path_x, self.path_y, self.path_z = [], [], []
        self.world_texts = []
        self.ego_texts = []
        self.world_fwd_arrows = None
        self.ego_fwd_arrows = None
        self.ego_up_arrows = None

    def plot_obs(self, obs, formatted_state_tensor):
        """
        Parses WaypointActorCritic3 tensor format:
        [Grav(3), Vel(3), AngVel(3), Pos(3*W), Fwd(3*W), Up(3*W), is_final(1)]
        """
        # 1. Extract Data
        pos = np.array(obs["pos"])
        t_idx = int(obs["target_gate"])
        gates_pos = np.array(obs["gates_pos"])
        drone_quat_xyzw = np.array(obs["quat"])

        if formatted_state_tensor.ndim > 1:
            state = formatted_state_tensor[0].cpu().numpy()
        else:
            state = formatted_state_tensor.cpu().numpy()

        # 2. Parse Tensor (WaypointActorCritic3 Structure)
        # Header: Gravity(0-2), Vel(3-5), AngVel(6-8)
        ego_gravity = state[0:3]

        # Gates:
        w = self.window_size
        pos_start = 9
        fwd_start = pos_start + (w * 3)
        up_start = fwd_start + (w * 3)

        # NOTE: We multiply by 10.0 here to undo the normalization for visualization
        ego_gates_pos = state[pos_start : pos_start + (w * 3)].reshape(w, 3) * 10.0
        ego_gates_fwd = state[fwd_start : fwd_start + (w * 3)].reshape(w, 3)
        ego_gates_up = state[up_start : up_start + (w * 3)].reshape(w, 3)

        # 3. Transform to World View
        r_drone = R.from_quat(drone_quat_xyzw)
        world_lookahead_pos = r_drone.apply(ego_gates_pos) + pos
        world_lookahead_fwd = r_drone.apply(ego_gates_fwd)

        # 4. Update Ego Plot
        if self.ego_fwd_arrows:
            self.ego_fwd_arrows.remove()
        if self.ego_up_arrows:
            self.ego_up_arrows.remove()
        if self.gravity_arrow:
            self.gravity_arrow.remove()

        # Gravity (Points 'Down' in ego frame)
        self.gravity_arrow = self.ax_ego.quiver(
            0,
            0,
            0,
            ego_gravity[0],
            ego_gravity[1],
            ego_gravity[2],
            length=1.5,
            color="green",
            linewidth=2,
            label="Ego Gravity",
        )

        # Gate Forward (Cyan) and Up (Yellow)
        self.ego_fwd_arrows = self.ax_ego.quiver(
            ego_gates_pos[:, 0],
            ego_gates_pos[:, 1],
            ego_gates_pos[:, 2],
            ego_gates_fwd[:, 0],
            ego_gates_fwd[:, 1],
            ego_gates_fwd[:, 2],
            length=0.8,
            color="cyan",
            linewidth=2,
        )
        self.ego_up_arrows = self.ax_ego.quiver(
            ego_gates_pos[:, 0],
            ego_gates_pos[:, 1],
            ego_gates_pos[:, 2],
            ego_gates_up[:, 0],
            ego_gates_up[:, 1],
            ego_gates_up[:, 2],
            length=0.5,
            color="yellow",
            linewidth=1,
        )

        # 5. Update World Plot
        if self.world_fwd_arrows:
            self.world_fwd_arrows.remove()
        self.world_fwd_arrows = self.ax_world.quiver(
            world_lookahead_pos[:, 0],
            world_lookahead_pos[:, 1],
            world_lookahead_pos[:, 2],
            world_lookahead_fwd[:, 0],
            world_lookahead_fwd[:, 1],
            world_lookahead_fwd[:, 2],
            length=0.8,
            color="orange",
            linewidth=2,
        )

        # Text labels and Trajectory
        self._update_common_elements(
            pos, r_drone, t_idx, gates_pos, world_lookahead_pos, ego_gates_pos
        )

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def _update_common_elements(
        self, pos, r_drone, t_idx, gates_pos, world_lookahead_pos, ego_gates_pos
    ):
        # Update Path
        self.path_x.append(pos[0])
        self.path_y.append(pos[1])
        self.path_z.append(pos[2])
        self.trajectory_line.set_data(self.path_x, self.path_y)
        self.trajectory_line.set_3d_properties(self.path_z)

        # Drone Heading in World
        if self.drone_arrow:
            self.drone_arrow.remove()
        fwd = r_drone.apply([1, 0, 0])
        self.drone_arrow = self.ax_world.quiver(
            pos[0],
            pos[1],
            pos[2],
            fwd[0],
            fwd[1],
            fwd[2],
            length=1.0,
            color="red",
            linewidth=2,
        )

        # Scatters
        self.ego_gates_scatter._offsets3d = (
            ego_gates_pos[:, 0],
            ego_gates_pos[:, 1],
            ego_gates_pos[:, 2],
        )
        self.world_lookahead_scatter._offsets3d = (
            world_lookahead_pos[:, 0],
            world_lookahead_pos[:, 1],
            world_lookahead_pos[:, 2],
        )

        if len(gates_pos) > 0:
            if 0 <= t_idx < len(gates_pos):
                tg = gates_pos[t_idx]
                self.target_scatter._offsets3d = ([tg[0]], [tg[1]], [tg[2]])
            self.gates_scatter._offsets3d = (
                gates_pos[:, 0],
                gates_pos[:, 1],
                gates_pos[:, 2],
            )
