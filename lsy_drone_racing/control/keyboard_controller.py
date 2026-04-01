from typing import TYPE_CHECKING

from lsy_drone_racing.control import Controller
from numpy.typing import NDArray

import math
import cv2
import numpy as np
from drone_models.core import load_params


def quat_to_euler(quat: NDArray) -> tuple[float, float, float]:
    """Convert quaternion (x, y, z, w) to euler angles (roll, pitch, yaw) in radians."""
    x, y, z, w = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)
    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


class KeyboardController(Controller):
    """HUD-style manual controller for the drone using keyboard inputs.

    Displays a fighter-jet style heads-up display (HUD) overlay on the FPV
    camera feed, including artificial horizon, altitude tape, heading compass,
    target gate indicator, and a control state panel.

    Uses a height P-controller instead of raw thrust for easier altitude management.

    Controls:
    - W/S: Pitch forward/backward
    - A/D: Yaw left/right
    - Up/Down arrows: Increase/decrease target height
    - R: Reset attitude to level
    - Esc: Quit episode
    """

    # ─── HUD color palette (BGR) ───
    HUD_CYAN = (255, 255, 0)
    HUD_GREEN = (0, 255, 0)
    HUD_YELLOW = (0, 255, 255)
    HUD_RED = (0, 80, 255)
    HUD_WHITE = (255, 255, 255)
    HUD_DIM = (180, 180, 100)
    HUD_BG = (0, 0, 0)

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize the HUD keyboard controller."""
        super().__init__(obs, info, config)
        # print("╔══════════════════════════════════════════╗")
        # print("║   HUD Keyboard Controller Initialized    ║")
        # print("╠══════════════════════════════════════════╣")
        # print("║  W/S      = Pitch forward/backward       ║")
        # print("║  A/D      = Yaw left/right                ║")
        # print("║  ↑/↓      = Increase/decrease height      ║")
        # print("║  R        = Level out                     ║")
        # print("║  Esc      = Quit                          ║")
        # print("╚══════════════════════════════════════════╝")
        self._freq = config.env.freq

        drone_params = load_params(config.sim.physics, config.sim.drone_model)
        self.drone_mass = drone_params["mass"]
        self.g = 9.81
        self.hover_thrust = self.drone_mass * self.g

        # ─── Control state ───
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0

        # Height controller
        start_z = float(obs["pos"][2]) if "pos" in obs else 1.0
        self.target_height = max(start_z, 0.05)  # Start at current or 0.5m minimum
        self.height_step = 0.02  # meters per key press
        self.height_min = 0.1
        self.height_max = 2.4
        self.Kp_height = 0.5  # P-gain for height controller
        self.Kd_height = 0.8  # D-gain for height controller (damps oscillation)
        self._prev_height = start_z
        self._prev_height_error = 0.0

        # Control sensitivities
        self.pitch_sens = 0.02
        self.yaw_sens = 0.03
        self.pitch_decay = (
            0.85  # pitch springs back to 0 each frame (lower = faster return)
        )

        # Limits
        self.max_pitch = math.pi / 4
        self.max_yaw_rate = math.pi
        self.min_thrust = 0.0
        self.max_thrust = 2.0 * self.hover_thrust

        self._tick = 0
        self._finished = False

        # Telemetry state for HUD (updated each frame)
        self._current_height = start_z
        self._current_roll = 0.0
        self._current_pitch = 0.0
        self._current_yaw = 0.0
        self._current_vel = np.zeros(3)
        self._current_thrust = self.hover_thrust
        self._target_gate_id = -1
        self._gate_distance = 0.0
        self._gate_direction = 0.0  # angle in degrees

        # Create OpenCV window
        cv2.namedWindow("Drone HUD", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Drone HUD", 960, 540)

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute control from keyboard input and display HUD overlay."""
        # ─── Update telemetry from observations ───
        self._update_telemetry(obs)

        # ─── Display HUD ───
        if "camera_frame" in obs:
            frame = obs["camera_frame"]
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            hud_frame = self._draw_hud(bgr_frame)
            cv2.imshow("Drone HUD", hud_frame)

        # ─── Poll keyboard (waitKeyEx for arrow key support) ───
        raw_key = cv2.waitKeyEx(1)

        # Handle arrow keys (platform-dependent codes)
        # Linux: Up=65362, Down=65364, Left=65361, Right=65363
        if raw_key == 65362:  # Up arrow
            self.target_height = min(
                self.target_height + self.height_step, self.height_max
            )
        elif raw_key == 65364:  # Down arrow
            self.target_height = max(
                self.target_height - self.height_step, self.height_min
            )

        # Handle ASCII keys
        key = raw_key & 0xFF

        # W/S — Pitch
        pitch_active = False
        if key == ord("w"):
            self.pitch = min(self.pitch + self.pitch_sens, self.max_pitch)
            pitch_active = True
        elif key == ord("s"):
            self.pitch = max(self.pitch - self.pitch_sens, -self.max_pitch)
            pitch_active = True

        # Auto-decay pitch back to zero when no key is held
        if not pitch_active:
            self.pitch *= self.pitch_decay
            if abs(self.pitch) < 0.001:
                self.pitch = 0.0

        # A/D — Yaw
        if key == ord("a"):
            self.yaw += self.yaw_sens
        elif key == ord("d"):
            self.yaw -= self.yaw_sens

        # R — Reset attitude
        if key == ord("r"):
            self.roll = 0.0
            self.pitch = 0.0
            self.yaw = 0.0

        # Esc — Quit
        if key == 27:
            self._finished = True

        # ─── Height PD-controller → thrust ───
        height_error = self.target_height - self._current_height
        # Derivative: rate of change of error (approx via velocity)
        vz = float(self._current_vel[2]) if len(self._current_vel) > 2 else 0.0
        d_term = -self.Kd_height * vz  # negative vz = falling, needs more thrust
        thrust = self.hover_thrust + self.Kp_height * height_error + d_term
        thrust = np.clip(thrust, self.min_thrust, self.max_thrust)
        self._current_thrust = float(thrust)

        action = np.array([self.roll, self.pitch, self.yaw, thrust], dtype=np.float32)
        return action

    def _update_telemetry(self, obs: dict[str, NDArray[np.floating]]):
        """Extract telemetry from the observation dict for HUD rendering."""
        if "pos" in obs:
            pos = np.asarray(obs["pos"])
            self._current_height = float(pos[2])

        if "quat" in obs:
            quat = np.asarray(obs["quat"])
            r, p, y = quat_to_euler(quat)
            self._current_roll = r
            self._current_pitch = p
            self._current_yaw = y

        if "vel" in obs:
            self._current_vel = np.asarray(obs["vel"])

        if "target_gate" in obs:
            self._target_gate_id = int(obs["target_gate"])

        # Compute distance and direction to next gate
        if self._target_gate_id >= 0 and "gates_pos" in obs and "pos" in obs:
            gates_pos = np.asarray(obs["gates_pos"])
            pos = np.asarray(obs["pos"])
            if self._target_gate_id < len(gates_pos):
                gate_pos = gates_pos[self._target_gate_id]
                delta = gate_pos - pos
                self._gate_distance = float(np.linalg.norm(delta))
                self._gate_direction = float(
                    math.degrees(math.atan2(delta[1], delta[0]))
                )

    # ═══════════════════════════════════════════════════════════════
    #  HUD RENDERING
    # ═══════════════════════════════════════════════════════════════

    def _draw_hud(self, frame: np.ndarray) -> np.ndarray:
        """Draw full HUD overlay on the FPV frame."""
        h, w = frame.shape[:2]
        overlay = frame.copy()

        self._draw_artificial_horizon(overlay, w, h)
        self._draw_altitude_tape(overlay, w, h)
        self._draw_heading_indicator(overlay, w, h)
        self._draw_velocity_vector(overlay, w, h)
        self._draw_target_gate_indicator(overlay, w, h)
        self._draw_control_panel(overlay, w, h)
        self._draw_key_hints(overlay, w, h)

        # Blend overlay with original for semi-transparency on drawn elements
        result = cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)
        return result

    def _draw_artificial_horizon(self, frame: np.ndarray, w: int, h: int):
        """Draw artificial horizon with pitch ladder, rotated by roll."""
        cx, cy = w // 2, h // 2
        roll_deg = math.degrees(self._current_roll)
        pitch_deg = math.degrees(self._current_pitch)

        # Pitch offset in pixels (positive pitch = nose up = line moves down)
        pitch_px_per_deg = h / 90.0
        pitch_offset = pitch_deg * pitch_px_per_deg

        # Rotation matrix for roll
        cos_r = math.cos(-self._current_roll)
        sin_r = math.sin(-self._current_roll)

        def rot_point(dx, dy):
            """Rotate point (dx, dy) by roll around center, with pitch offset."""
            dy_shifted = dy + pitch_offset
            rx = cos_r * dx - sin_r * dy_shifted
            ry = sin_r * dx + cos_r * dy_shifted
            return int(cx + rx), int(cy + ry)

        # ── Horizon line ──
        half_w = w // 2
        p1 = rot_point(-half_w, 0)
        p2 = rot_point(half_w, 0)
        cv2.line(frame, p1, p2, self.HUD_GREEN, 2, cv2.LINE_AA)

        # ── Pitch ladder ──
        for deg in range(-30, 35, 10):
            if deg == 0:
                continue
            dy = (
                -deg * pitch_px_per_deg
            )  # pixels offset from center for this pitch line
            line_half = 70 if abs(deg) <= 20 else 45

            pl = rot_point(-line_half, dy)
            pr = rot_point(line_half, dy)
            color = self.HUD_GREEN if deg > 0 else self.HUD_CYAN
            thickness = 1
            cv2.line(frame, pl, pr, color, thickness, cv2.LINE_AA)

            # Label
            label = f"{deg}"
            lpos = rot_point(-line_half - 45, dy)
            cv2.putText(
                frame,
                label,
                lpos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                1,
                cv2.LINE_AA,
            )

        # ── Center crosshair (fixed) ──
        size = 25
        cv2.line(frame, (cx - size, cy), (cx - 7, cy), self.HUD_YELLOW, 2, cv2.LINE_AA)
        cv2.line(frame, (cx + 7, cy), (cx + size, cy), self.HUD_YELLOW, 2, cv2.LINE_AA)
        cv2.line(frame, (cx, cy - size), (cx, cy - 7), self.HUD_YELLOW, 2, cv2.LINE_AA)
        cv2.line(frame, (cx, cy + 7), (cx, cy + size), self.HUD_YELLOW, 2, cv2.LINE_AA)

        # ── Roll indicator arc at top ──
        radius = min(w, h) // 3
        cv2.ellipse(
            frame,
            (cx, cy),
            (radius, radius),
            0,
            -150,
            -30,
            self.HUD_DIM,
            1,
            cv2.LINE_AA,
        )
        # Tick mark for current roll
        roll_rad = self._current_roll
        tick_x = int(cx + radius * math.cos(-math.pi / 2 + roll_rad))
        tick_y = int(cy + radius * math.sin(-math.pi / 2 + roll_rad))
        cv2.circle(frame, (tick_x, tick_y), 6, self.HUD_YELLOW, -1, cv2.LINE_AA)

    def _draw_altitude_tape(self, frame: np.ndarray, w: int, h: int):
        """Draw altitude tape on the right side."""
        # Tape area
        tape_x = w - 120
        tape_w = 100
        tape_top = h // 4
        tape_bot = 3 * h // 4
        tape_h = tape_bot - tape_top

        # Semi-transparent background
        sub = frame[tape_top:tape_bot, tape_x : tape_x + tape_w]
        dark = np.zeros_like(sub)
        cv2.addWeighted(sub, 0.4, dark, 0.6, 0, sub)

        # Border
        cv2.rectangle(
            frame,
            (tape_x, tape_top),
            (tape_x + tape_w, tape_bot),
            self.HUD_CYAN,
            1,
            cv2.LINE_AA,
        )

        # Scale: map altitude to tape position
        alt = self._current_height
        alt_range = 3.0  # meters visible in tape
        px_per_m = tape_h / alt_range

        # Draw tick marks
        alt_center = alt
        for tick_alt_10 in range(0, 30):  # 0.0 to 3.0 in 0.1 steps
            tick_alt = tick_alt_10 * 0.1
            dy = (alt_center - tick_alt) * px_per_m
            y = int((tape_top + tape_bot) / 2 + dy)
            if tape_top <= y <= tape_bot:
                if tick_alt_10 % 5 == 0:
                    # Major tick
                    cv2.line(
                        frame,
                        (tape_x, y),
                        (tape_x + 18, y),
                        self.HUD_CYAN,
                        1,
                        cv2.LINE_AA,
                    )
                    label = f"{tick_alt:.1f}"
                    cv2.putText(
                        frame,
                        label,
                        (tape_x + 22, y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        self.HUD_CYAN,
                        1,
                        cv2.LINE_AA,
                    )
                else:
                    cv2.line(
                        frame,
                        (tape_x, y),
                        (tape_x + 10, y),
                        self.HUD_DIM,
                        1,
                        cv2.LINE_AA,
                    )

        # Current altitude pointer (center)
        ptr_y = (tape_top + tape_bot) // 2
        pts = np.array(
            [
                [tape_x - 12, ptr_y],
                [tape_x, ptr_y - 12],
                [tape_x + tape_w // 2 + 10, ptr_y - 12],
                [tape_x + tape_w // 2 + 10, ptr_y + 12],
                [tape_x, ptr_y + 12],
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(frame, [pts], self.HUD_BG)
        cv2.polylines(frame, [pts], True, self.HUD_GREEN, 2, cv2.LINE_AA)
        cv2.putText(
            frame,
            f"{alt:.2f}",
            (tape_x + 4, ptr_y + 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            self.HUD_GREEN,
            1,
            cv2.LINE_AA,
        )

        # Target height marker
        target_dy = (alt_center - self.target_height) * px_per_m
        target_y = int((tape_top + tape_bot) / 2 + target_dy)
        if tape_top <= target_y <= tape_bot:
            cv2.line(
                frame,
                (tape_x, target_y),
                (tape_x + tape_w, target_y),
                self.HUD_YELLOW,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"TGT {self.target_height:.2f}",
                (tape_x - 100, target_y + 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                self.HUD_YELLOW,
                1,
                cv2.LINE_AA,
            )

        # Label
        cv2.putText(
            frame,
            "ALT m",
            (tape_x + 5, tape_top - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            self.HUD_CYAN,
            1,
            cv2.LINE_AA,
        )

    def _draw_heading_indicator(self, frame: np.ndarray, w: int, h: int):
        """Draw heading compass bar at the top."""
        bar_y = 25
        bar_h = 36
        bar_x = w // 5
        bar_w = 3 * w // 5

        # Background
        sub = frame[bar_y : bar_y + bar_h, bar_x : bar_x + bar_w]
        dark = np.zeros_like(sub)
        cv2.addWeighted(sub, 0.4, dark, 0.6, 0, sub)
        cv2.rectangle(
            frame,
            (bar_x, bar_y),
            (bar_x + bar_w, bar_y + bar_h),
            self.HUD_CYAN,
            1,
            cv2.LINE_AA,
        )

        # Current heading in degrees
        hdg = math.degrees(self._current_yaw) % 360
        # Compass labels
        compass_labels = {
            0: "N",
            45: "NE",
            90: "E",
            135: "SE",
            180: "S",
            225: "SW",
            270: "W",
            315: "NW",
        }

        px_per_deg = bar_w / 60.0  # 60 degrees visible

        for offset_deg in range(-30, 31):
            disp_deg = (hdg + offset_deg) % 360
            x = int(bar_x + bar_w / 2 + offset_deg * px_per_deg)
            if bar_x <= x <= bar_x + bar_w:
                int_deg = int(round(disp_deg))
                if int_deg % 10 == 0:
                    cv2.line(
                        frame,
                        (x, bar_y + bar_h),
                        (x, bar_y + bar_h - 14),
                        self.HUD_CYAN,
                        1,
                        cv2.LINE_AA,
                    )
                    label = compass_labels.get(int_deg, f"{int_deg}")
                    cv2.putText(
                        frame,
                        label,
                        (x - 8, bar_y + bar_h - 16),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        self.HUD_CYAN,
                        1,
                        cv2.LINE_AA,
                    )
                elif int_deg % 5 == 0:
                    cv2.line(
                        frame,
                        (x, bar_y + bar_h),
                        (x, bar_y + bar_h - 7),
                        self.HUD_DIM,
                        1,
                        cv2.LINE_AA,
                    )

        # Center pointer triangle
        center_x = bar_x + bar_w // 2
        pts = np.array(
            [
                [center_x, bar_y + bar_h + 10],
                [center_x - 8, bar_y + bar_h],
                [center_x + 8, bar_y + bar_h],
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(frame, [pts], self.HUD_YELLOW)

        # Heading readout
        cv2.putText(
            frame,
            f"HDG {hdg:.0f}",
            (bar_x + bar_w + 10, bar_y + bar_h - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            self.HUD_GREEN,
            1,
            cv2.LINE_AA,
        )

    def _draw_velocity_vector(self, frame: np.ndarray, w: int, h: int):
        """Draw speed tape on the left side."""
        tape_x = 20
        tape_w = 90
        tape_top = h // 4
        tape_bot = 3 * h // 4
        tape_h = tape_bot - tape_top

        speed = float(np.linalg.norm(self._current_vel))
        vz = float(self._current_vel[2]) if len(self._current_vel) > 2 else 0.0

        # Background
        sub = frame[tape_top:tape_bot, tape_x : tape_x + tape_w]
        dark = np.zeros_like(sub)
        cv2.addWeighted(sub, 0.4, dark, 0.6, 0, sub)
        cv2.rectangle(
            frame,
            (tape_x, tape_top),
            (tape_x + tape_w, tape_bot),
            self.HUD_CYAN,
            1,
            cv2.LINE_AA,
        )

        # Speed indicator at center
        ptr_y = (tape_top + tape_bot) // 2
        cv2.putText(
            frame,
            f"{speed:.2f}",
            (tape_x + 6, ptr_y + 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            self.HUD_GREEN,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "m/s",
            (tape_x + 6, ptr_y + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            self.HUD_DIM,
            1,
            cv2.LINE_AA,
        )

        # Vertical speed arrow
        arrow_len = int(np.clip(abs(vz) * 40, 8, tape_h // 3))
        arrow_x = tape_x + tape_w // 2
        if vz > 0.05:
            cv2.arrowedLine(
                frame,
                (arrow_x, ptr_y + 30),
                (arrow_x, ptr_y + 30 - arrow_len),
                self.HUD_GREEN,
                2,
                cv2.LINE_AA,
                tipLength=0.3,
            )
        elif vz < -0.05:
            cv2.arrowedLine(
                frame,
                (arrow_x, ptr_y - 30),
                (arrow_x, ptr_y - 30 + arrow_len),
                self.HUD_RED,
                2,
                cv2.LINE_AA,
                tipLength=0.3,
            )

        cv2.putText(
            frame,
            f"VS {vz:+.1f}",
            (tape_x, ptr_y - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            self.HUD_CYAN,
            1,
            cv2.LINE_AA,
        )

        # Label
        cv2.putText(
            frame,
            "SPD",
            (tape_x + 10, tape_top - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            self.HUD_CYAN,
            1,
            cv2.LINE_AA,
        )

    def _draw_target_gate_indicator(self, frame: np.ndarray, w: int, h: int):
        """Draw target gate info panel."""
        if self._target_gate_id < 0:
            # All gates passed
            cv2.putText(
                frame,
                "ALL GATES PASSED",
                (w // 2 - 80, h // 2 - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                self.HUD_GREEN,
                2,
                cv2.LINE_AA,
            )
            return

        # Info box at top-left
        box_x, box_y = 20, 70
        box_w, box_h = 220, 65

        sub = frame[box_y : box_y + box_h, box_x : box_x + box_w]
        dark = np.zeros_like(sub)
        cv2.addWeighted(sub, 0.4, dark, 0.6, 0, sub)
        cv2.rectangle(
            frame,
            (box_x, box_y),
            (box_x + box_w, box_y + box_h),
            self.HUD_YELLOW,
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            f"GATE {self._target_gate_id + 1}",
            (box_x + 8, box_y + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            self.HUD_YELLOW,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"DIST: {self._gate_distance:.2f}m",
            (box_x + 8, box_y + 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            self.HUD_CYAN,
            1,
            cv2.LINE_AA,
        )

        # Direction indicator — small diamond pointing towards gate
        bearing = self._gate_direction - math.degrees(self._current_yaw)
        bearing_rad = math.radians(bearing)
        ind_cx = w // 2
        ind_cy = h - 90
        ind_r = 35
        dx = int(ind_r * math.sin(bearing_rad))
        dy = int(-ind_r * math.cos(bearing_rad))
        cv2.circle(frame, (ind_cx, ind_cy), ind_r + 5, self.HUD_DIM, 1, cv2.LINE_AA)
        cv2.circle(frame, (ind_cx, ind_cy), 2, self.HUD_CYAN, -1, cv2.LINE_AA)
        cv2.line(
            frame,
            (ind_cx, ind_cy),
            (ind_cx + dx, ind_cy + dy),
            self.HUD_YELLOW,
            2,
            cv2.LINE_AA,
        )
        cv2.circle(
            frame, (ind_cx + dx, ind_cy + dy), 6, self.HUD_YELLOW, -1, cv2.LINE_AA
        )

    def _draw_control_panel(self, frame: np.ndarray, w: int, h: int):
        """Draw current control state at the bottom."""
        panel_h = 50
        panel_y = h - panel_h
        panel_x = w // 4
        panel_w = w // 2

        # Background
        sub = frame[panel_y:h, panel_x : panel_x + panel_w]
        dark = np.zeros_like(sub)
        cv2.addWeighted(sub, 0.3, dark, 0.7, 0, sub)
        cv2.rectangle(
            frame,
            (panel_x, panel_y),
            (panel_x + panel_w, h),
            self.HUD_CYAN,
            1,
            cv2.LINE_AA,
        )

        # Control values
        pitch_deg = math.degrees(self.pitch)
        yaw_deg = math.degrees(self.yaw) % 360
        texts = [
            f"PIT: {pitch_deg:+.1f}°",
            f"YAW: {yaw_deg:.0f}°",
            f"THR: {self._current_thrust:.2f}N",
            f"TGT-H: {self.target_height:.2f}m",
        ]
        spacing = panel_w // len(texts)
        for i, txt in enumerate(texts):
            x = panel_x + 10 + i * spacing
            cv2.putText(
                frame,
                txt,
                (x, panel_y + 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                self.HUD_GREEN,
                1,
                cv2.LINE_AA,
            )

    def _draw_key_hints(self, frame: np.ndarray, w: int, h: int):
        """Draw small key hint legend at bottom-right."""
        hints = [
            "W/S: Pitch",
            "A/D: Yaw",
            chr(0x2191) + "/" + chr(0x2193) + ": Height",
            "R: Level  ESC: Quit",
        ]
        x = w - 190
        y_start = h - 80
        for i, hint in enumerate(hints):
            cv2.putText(
                frame,
                hint,
                (x, y_start + i * 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                self.HUD_DIM,
                1,
                cv2.LINE_AA,
            )

    # ═══════════════════════════════════════════════════════════════
    #  LIFECYCLE
    # ═══════════════════════════════════════════════════════════════

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Increment the tick counter and check for quit."""
        self._tick += 1
        return self._finished

    def episode_callback(self):
        """Reset the internal state."""
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.target_height = 1.0
        self._tick = 0
        self._finished = False

    def episode_reset(self):
        """Reset for next episode."""
        pass

    def close(self):
        """Clean up the OpenCV window."""
        cv2.destroyAllWindows()
