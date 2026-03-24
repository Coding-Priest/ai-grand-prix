"""Simulate the competition as in the IROS 2022 Safe Robot Learning competition.

Run as:

    $ python scripts/sim.py --config level0.toml

Look for instructions in `README.md` and in the official documentation.
"""

from __future__ import annotations

import logging
from pathlib import Path

import fire
import gymnasium
import jax.numpy as jp
import numpy as np
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy
import time
from lsy_drone_racing.envs.drone_race import DroneRaceEnv
from lsy_drone_racing.utils import load_config, load_controller

from lsy_drone_racing.slam.slop_slam_streamer import SLAMStreamer

import cv2 as cv

logger = logging.getLogger(__name__)

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


def plot_imu_path(accel_data, gyro_data, freq):
    """
    Plots the 3D trajectory using pure IMU dead reckoning.

    accel_data: List or array of (N, 3) accelerations in m/s^2
    gyro_data: List or array of (N, 3) angular velocities in rad/s
    freq: The frequency of the IMU data (Hz)
    """
    dt = 1.0 / freq
    num_steps = len(accel_data)

    # Initialize state arrays
    positions = np.zeros((num_steps, 3))
    velocities = np.zeros((num_steps, 3))

    # Start with an identity rotation (facing perfectly forward/level)
    current_rot = R.from_quat([0, 0, 0, 1])

    # Gravity vector in the WORLD frame (Assuming Z is up)
    gravity = np.array([0.0, 0.0, 9.81])

    for i in range(1, num_steps):
        a_body = np.array(accel_data[i])
        w_body = np.array(gyro_data[i])

        # 1. Update Rotation using Gyroscope
        # Create a rotation from the angular velocity vector * dt
        delta_rot = R.from_rotvec(w_body * dt)
        current_rot = current_rot * delta_rot

        # 2. Rotate Acceleration to World Frame and Remove Gravity
        a_world = current_rot.apply(a_body)
        a_linear = a_world - gravity

        # 3. Integrate to Velocity and Position
        velocities[i] = velocities[i - 1] + (a_linear * dt)
        positions[i] = (
            positions[i - 1] + (velocities[i - 1] * dt) + (0.5 * a_linear * dt**2)
        )

    # Plotting
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        label="IMU Dead Reckoning Path",
        color="b",
    )
    ax.scatter(
        positions[0, 0],
        positions[0, 1],
        positions[0, 2],
        color="g",
        s=100,
        label="Start",
    )
    ax.scatter(
        positions[-1, 0],
        positions[-1, 1],
        positions[-1, 2],
        color="r",
        s=100,
        label="End",
    )

    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.set_zlabel("Z Position (m)")
    ax.set_title("Pure IMU Trajectory")
    ax.legend()

    # Auto-scale axes to be equal
    max_range = (
        np.array(
            [
                positions[:, 0].max() - positions[:, 0].min(),
                positions[:, 1].max() - positions[:, 1].min(),
                positions[:, 2].max() - positions[:, 2].min(),
            ]
        ).max()
        / 2.0
    )
    mid_x = (positions[:, 0].max() + positions[:, 0].min()) * 0.5
    mid_y = (positions[:, 1].max() + positions[:, 1].min()) * 0.5
    mid_z = (positions[:, 2].max() + positions[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.show()


def simulate(
    config: str = "level0_slam_start.toml",
    controller: str | None = None,
    n_runs: int = 1,
    render: bool | None = None,
) -> list[float]:
    """Evaluate the drone controller over multiple episodes.

    Args:
        config: The path to the configuration file. Assumes the file is in `config/`.
        controller: The name of the controller file in `lsy_drone_racing/control/` or None. If None,
            the controller specified in the config file is used.
        n_runs: The number of episodes.
        render: Enable/disable rendering the simulation.

    Returns:
        A list of episode times.
    """
    # Load configuration and check if firmare should be used.
    config = load_config(Path(__file__).parents[1] / "config" / config)
    if render is None:
        render = config.sim.render
    else:
        config.sim.render = render
    # Load the controller module
    control_path = Path(__file__).parents[1] / "lsy_drone_racing/control"
    controller_path = control_path / (controller or config.controller.file)
    controller_cls = load_controller(
        controller_path
    )  # This returns a class, not an instance
    # Create the racing environment
    env: DroneRaceEnv = gymnasium.make(
        config.env.id,
        freq=config.env.freq,
        sim_config=config.sim,
        sensor_range=config.env.sensor_range,
        control_mode=config.env.control_mode,
        track=config.env.track,
        disturbances=config.env.get("disturbances"),
        randomizations=config.env.get("randomizations"),
        seed=config.env.seed,
    )
    env = JaxToNumpy(env)

    # slam_bridge = OrbSlamBridge()
    # fps = 30
    # width, height = 1920, 1080  # Match your env dimensions
    # fourcc = cv.VideoWriter_fourcc(*"mp4v")
    # video_writer = cv.VideoWriter("drone_fly.mp4", fourcc, fps, (width, height))

    env_freq = config.env.freq  # e.g., 50 Hz or 60 Hz
    imu_freq = 500  # Must match your YAML!
    dt_imu = 1.0 / imu_freq
    imu_steps_per_env = imu_freq // env_freq

    all_accel_points = []
    all_gyro_points = []

    ep_times = []
    with SLAMStreamer(record_path="basics.bin") as slam:
        for _ in range(n_runs):  # Run n_runs episodes with the controller
            obs, info = env.reset()
            controller: Controller = controller_cls(obs, info, config)
            i = 0
            fps = 60

            while True:
                curr_time = i / config.env.freq
                base_time = i / env_freq

                # print(curr_time)
                action = controller.compute_control(obs, info)

                action = np.asarray(jp.asarray(action), copy=True)

                obs, reward, terminated, truncated, info = env.step(action)

                # What is even the point of this?
                # Update the controller internal state and models.
                controller_finished = controller.step_callback(
                    action, obs, reward, terminated, truncated, info
                )
                # Add up reward, collisions
                if terminated or truncated or controller_finished:
                    break

                frame = obs["camera_frame"]
                accel_batch = obs["accel"]
                gyro_batch = obs["gyro"]

                all_accel_points.extend(accel_batch)
                all_gyro_points.extend(gyro_batch)

                # 1. Figure out how many IMU readings we got in this single step
                num_imu = len(accel_batch)

                # 2. Calculate the time window.
                # Assuming config.env.freq is your camera/step frequency (e.g., 20Hz)
                step_duration = 1.0 / config.env.freq

                # The timestamp of the PREVIOUS frame
                prev_time = curr_time - step_duration

                # 3. Send all IMU readings FIRST, with interpolated timestamps
                for j in range(num_imu):
                    # Distribute timestamps evenly across the step's time window.
                    # By doing (j + 1), the very last IMU reading lands exactly on curr_time.
                    imu_ts = base_time + (j + 1) * dt_imu

                    slam.send_imu(accel_batch[j], gyro_batch[j], timestamp=imu_ts)

                # time.sleep(0.1)
                t_frame = base_time + (imu_steps_per_env * dt_imu)
                # 4. THEN send the frame, closing out the time window
                # if i % 10 == 0:
                #     print("Sending frame")
                slam.send_frame(frame, timestamp=t_frame)

                if config.sim.render:  # Render the sim if selected.
                    if ((i * fps) % config.env.freq) < fps:
                        env.render()
                i += 1

        controller.episode_callback()  # Update the controller internal state and models.
        log_episode_stats(obs, info, config, curr_time)
        controller.episode_reset()
        ep_times.append(curr_time if obs["target_gate"] == -1 else None)

    plot_imu_path(all_accel_points, all_gyro_points, imu_freq)

    # Close the environment
    env.close()
    return ep_times


def log_episode_stats(obs: dict, info: dict, config: ConfigDict, curr_time: float):
    """Log the statistics of a single episode."""
    gates_passed = obs["target_gate"]
    if gates_passed == -1:  # The drone has passed the final gate
        gates_passed = len(config.env.track.gates)
    finished = gates_passed == len(config.env.track.gates)
    logger.info(
        f"Flight time (s): {curr_time}\nFinished: {finished}\nGates passed: {gates_passed}\n"
    )


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger("lsy_drone_racing").setLevel(logging.INFO)
    logger.setLevel(logging.INFO)
    fire.Fire(simulate, serialize=lambda _: None)
