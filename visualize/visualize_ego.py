import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


class VisualizeEgoSim:
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
            label="Tensor Projected to World",
            zorder=5,
        )

        self.drone_arrow = None

        self.ax_world.set_xlim([-2.5, 2.5])
        self.ax_world.set_ylim([-2.5, 2.5])
        self.ax_world.set_zlim([0, 5])
        self.ax_world.set_xlabel("X")
        self.ax_world.set_ylabel("Y")
        self.ax_world.set_zlabel("Altitude")
        self.ax_world.legend(loc="upper right")

        # --- RIGHT: EGO VIEW ---
        self.ax_ego = self.fig.add_subplot(122, projection="3d")
        self.ax_ego.set_title("Model View (Direct from Tensor)")

        self.ax_ego.quiver(
            0,
            0,
            0,
            1,
            0,
            0,
            length=1.0,
            color="red",
            normalize=True,
            linewidth=3,
            label="Ego Drone (+X Forward)",
        )

        self.ego_gates_scatter = self.ax_ego.scatter(
            [],
            [],
            [],
            c="cyan",
            marker="s",
            s=60,
            edgecolor="black",
            label="Tensor Lookahead Gates",
        )

        self.ax_ego.set_xlim([-5, 5])
        self.ax_ego.set_ylim([-5, 5])
        self.ax_ego.set_zlim([-5, 5])
        self.ax_ego.set_xlabel("Ego Forward (+X)")
        self.ax_ego.set_ylabel("Ego Left/Right (+Y)")
        self.ax_ego.set_zlabel("Ego Up/Down (+Z)")
        self.ax_ego.legend(loc="upper right")

        self.path_x, self.path_y, self.path_z = [], [], []
        self.world_texts = []
        self.ego_texts = []
        self.world_gate_arrows = None
        self.ego_gate_arrows = None

    def plot_obs(self, obs, formatted_state_tensor):
        # ==========================================
        # 1. EXTRACT DATA & STANDARDIZE
        # ==========================================
        pos = np.array(obs["pos"])
        t_idx = int(obs["target_gate"])
        gates_pos = np.array(obs["gates_pos"])

        # We now know definitively this is [x, y, z, w]
        drone_quat_xyzw = np.array(obs["quat"])

        if formatted_state_tensor.ndim > 1:
            state = formatted_state_tensor[0].cpu().numpy()
        else:
            state = formatted_state_tensor.cpu().numpy()

        # ==========================================
        # 2. PARSE THE TENSOR
        # ==========================================
        start_idx = 10
        pos_end_idx = start_idx + (self.window_size * 3)
        quat_end_idx = pos_end_idx + (self.window_size * 4)

        ego_gates_pos = state[start_idx:pos_end_idx].reshape(self.window_size, 3)
        # Tensor is also natively outputting [x, y, z, w]
        ego_gates_quat_xyzw = state[pos_end_idx:quat_end_idx].reshape(
            self.window_size, 4
        )

        base_forward_vec = np.array([1.0, 0.0, 0.0])

        # Get the gates' forward directions in the Ego frame
        gate_rots_ego = R.from_quat(ego_gates_quat_xyzw)
        ego_fwd_vectors = gate_rots_ego.apply(base_forward_vec)

        # ==========================================
        # 3. TRANSFORM TO WORLD VIEW
        # ==========================================
        r_drone = R.from_quat(drone_quat_xyzw)

        # Apply the drone's rotation and translation to move the gates to the world frame
        world_lookahead_pos = r_drone.apply(ego_gates_pos) + pos
        world_fwd_vectors = r_drone.apply(ego_fwd_vectors)

        # ==========================================
        # 4. UPDATE PLOTS & ARROWS
        # ==========================================
        if self.world_gate_arrows:
            self.world_gate_arrows.remove()
        if self.ego_gate_arrows:
            self.ego_gate_arrows.remove()

        # Draw Ego Arrows (Cyan)
        self.ego_gate_arrows = self.ax_ego.quiver(
            ego_gates_pos[:, 0],
            ego_gates_pos[:, 1],
            ego_gates_pos[:, 2],
            ego_fwd_vectors[:, 0],
            ego_fwd_vectors[:, 1],
            ego_fwd_vectors[:, 2],
            length=0.8,
            color="cyan",
            linewidth=2,
            label="Gate Heading",
        )

        # Draw World Arrows (Orange)
        self.world_gate_arrows = self.ax_world.quiver(
            world_lookahead_pos[:, 0],
            world_lookahead_pos[:, 1],
            world_lookahead_pos[:, 2],
            world_fwd_vectors[:, 0],
            world_fwd_vectors[:, 1],
            world_fwd_vectors[:, 2],
            length=0.8,
            color="orange",
            linewidth=2,
        )

        # Clean up old text labels
        for txt in getattr(self, "world_texts", []):
            txt.remove()
        for txt in getattr(self, "ego_texts", []):
            txt.remove()
        self.world_texts = []
        self.ego_texts = []

        # Re-draw text labels
        for i in range(self.window_size):
            te = self.ax_ego.text(
                ego_gates_pos[i, 0],
                ego_gates_pos[i, 1],
                ego_gates_pos[i, 2],
                f"Index {i}",
                color="cyan",
                fontsize=10,
                fontweight="bold",
            )
            self.ego_texts.append(te)

            tw = self.ax_world.text(
                world_lookahead_pos[i, 0],
                world_lookahead_pos[i, 1],
                world_lookahead_pos[i, 2],
                f"Local {i}",
                color="black",
                fontsize=9,
            )
            self.world_texts.append(tw)

        # Update Path
        x, y, z = pos
        self.path_x.append(x)
        self.path_y.append(y)
        self.path_z.append(z)

        self.trajectory_line.set_data(self.path_x, self.path_y)
        self.trajectory_line.set_3d_properties(self.path_z)

        # Redraw Drone Arrow
        if self.drone_arrow is not None:
            self.drone_arrow.remove()

        fwd = r_drone.apply(base_forward_vec)
        self.drone_arrow = self.ax_world.quiver(
            x,
            y,
            z,
            fwd[0],
            fwd[1],
            fwd[2],
            length=0.6,
            color="red",
            normalize=True,
            linewidth=2,
        )

        # Update Base Track Gates
        if len(gates_pos) > 0:
            if t_idx < len(gates_pos) and t_idx != -1:
                tg = gates_pos[t_idx]
                other_gates = np.delete(gates_pos, t_idx, axis=0)
                self.target_scatter._offsets3d = ([tg[0]], [tg[1]], [tg[2]])
            else:
                other_gates = gates_pos
                self.target_scatter._offsets3d = ([], [], [])

            if len(other_gates) > 0:
                self.gates_scatter._offsets3d = (
                    other_gates[:, 0],
                    other_gates[:, 1],
                    other_gates[:, 2],
                )

        # Update Lookahead Scatters
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

        # Refresh plot
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)
