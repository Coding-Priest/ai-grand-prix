import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


class VisualizeSim:

    def __init__(self):
        # 1. Setup the Figure
        plt.ion()  # Turn on interactive mode
        self.fig = plt.figure(figsize=(10, 7))
        self.ax = self.fig.add_subplot(111, projection="3d")

        # 2. Initialize Plot Objects
        (self.trajectory_line,) = self.ax.plot(
            [], [], [], "b-", label="Drone Path", alpha=0.5
        )

        # Initialize scatter plots for gates (Empty for now)
        self.gates_scatter = self.ax.scatter(
            [], [], [], c="green", marker="s", s=40, label="Gates", alpha=0.6
        )
        self.target_scatter = self.ax.scatter(
            [], [], [], c="magenta", marker="*", s=200, label="Target Gate"
        )

        self.target_text = self.ax.text2D(
            0.05,
            0.95,
            "Target Gate: --",
            transform=self.ax.transAxes,
            fontsize=12,
            fontweight="bold",
            color="magenta",
            bbox=dict(
                facecolor="white", alpha=0.8, edgecolor="gray", boxstyle="round,pad=0.5"
            ),
        )

        # We will store the quiver (arrow) object here to remove/redraw it each frame
        self.drone_arrow = None

        # Set labels and limits (Crucial for 3D stability)
        self.ax.set_xlim([-2.5, 2.5])  # Adjusted based on your gate coordinates
        self.ax.set_ylim([-2.5, 2.5])
        self.ax.set_zlim([0, 5])
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Altitude")
        self.ax.legend(loc="upper right")

        # 3. The Simulation Loop
        self.path_x, self.path_y, self.path_z = [], [], []

    def plot_obs(self, obs):
        # global drone_arrow  # Declare global to modify the arrow object

        # ['pos', 'quat', 'vel', 'ang_vel',

        # 'accel', 'gyro',

        # 'target_gate', 'gates_pos', 'gates_quat', 'gates_visited', 'obstacles_pos', 'obstacles_visited',

        # 'camera_frame'])

        pos = obs["pos"]
        quat_wxyz = obs["quat"]
        target_gate_idx = obs["target_gate"]
        gates_pos = np.array(obs["gates_pos"])

        x, y, z = pos

        # --- Update Trajectory ---
        self.path_x.append(x)
        self.path_y.append(y)
        self.path_z.append(z)

        self.trajectory_line.set_data(self.path_x, self.path_y)
        self.trajectory_line.set_3d_properties(self.path_z)

        # --- Update Drone Arrow (Orientation) ---
        # Remove the old arrow if it exists
        if self.drone_arrow is not None:
            self.drone_arrow.remove()

        qw, qx, qy, qz = quat_wxyz

        # Convert quaternion to forward direction vector (Assuming local +X is forward)
        u = 1 - 2 * (qy**2 + qz**2)
        v = 2 * (qx * qy + qw * qz)
        w = 2 * (qx * qz - qw * qy)

        # Draw the new arrow
        self.drone_arrow = self.ax.quiver(
            x,
            y,
            z,
            u,
            v,
            w,
            length=0.5,
            color="red",
            normalize=True,
            linewidth=2,
            label="Current Pos",
        )

        # --- Update Gates ---
        if len(gates_pos) > 0:
            # Separate the target gate from the rest of the gates
            if target_gate_idx < len(gates_pos):
                tg = gates_pos[target_gate_idx]
                other_gates = np.delete(gates_pos, target_gate_idx, axis=0)

                # Update target gate position (using _offsets3d is the standard Matplotlib 3D workaround)
                self.target_scatter._offsets3d = ([tg[0]], [tg[1]], [tg[2]])
            else:
                other_gates = gates_pos
                self.target_scatter._offsets3d = (
                    [],
                    [],
                    [],
                )  # Hide if target is out of bounds

            # Update remaining gates
            if len(other_gates) > 0:
                self.gates_scatter._offsets3d = (
                    other_gates[:, 0],
                    other_gates[:, 1],
                    other_gates[:, 2],
                )
        self.target_text.set_text(f"Target Gate Index: {target_gate_idx}")
        # --- Refresh the canvas ---
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)
