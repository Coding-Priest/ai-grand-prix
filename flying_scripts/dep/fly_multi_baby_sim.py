"""Simulate the competition as in the IROS 2022 Safe Robot Learning competition.

Run as:

    $ python scripts/sim.py --config level0.toml

Look for instructions in `README.md` and in the official documentation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import fire
import gymnasium
import jax.numpy as jp
import numpy as np
from gymnasium.wrappers.vector import JaxToNumpy

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from lsy_drone_racing.utils import load_config, load_controller
from .visualize import VisualizeSim
from .rollout_buffer import RolloutBuffer
from models.droneac import DroneActorCritic


if TYPE_CHECKING:
    from ml_collections import ConfigDict

    from lsy_drone_racing.control.controller import Controller
    from lsy_drone_racing.envs.drone_race import VecDroneRaceEnv


logger = logging.getLogger(__name__)

# sim_visualizer = VisualizeSim()


def simulate(
    config: str = "multi_baby_level0.toml",  # "level0_baby_steps.toml",
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
    env: VecMultiDroneRaceEnv = gymnasium.make_vec(
        config.env.id,
        num_envs=10,
        freq=config.env.freq,
        sim_config=config.sim,
        sensor_range=config.env.sensor_range,
        control_mode=config.env.control_mode,
        track=config.env.track,
        disturbances=config.env.get("disturbances"),
        randomizations=config.env.get("randomizations"),
        seed=config.env.seed,
        disable_termination=True,
        disable_collisions=True,
        device="gpu",
    )

    env = JaxToNumpy(env)

    ep_times = []
    for _ in range(n_runs):  # Run n_runs episodes
        batched_obs, batched_info = env.reset()
        # Print all the positions of the drone
        # 1. Figure out how many parallel environments we are running
        # Assuming the state array is shaped (num_envs, feature_dim)
        num_envs = batched_obs["target_gate"].shape[0]
        print(f"Number of parallel environments: {num_envs}")

        # 2. Instantiate a SEPARATE controller for each environment
        controllers = []
        for j in range(num_envs):
            # Slice the batched obs/info for this specific environment
            single_obs = {k: v[j][0] for k, v in batched_obs.items()}
            single_info = {k: v[j][0] for k, v in batched_info.items()}
            controllers.append(controller_cls(single_obs, single_info, config))

        i = 0
        fps = 60

        while True:
            curr_time = i / config.env.freq

            actions = []
            controller_finished_list = []

            # 3. Compute control for each environment independently
            for j, ctrl in enumerate(controllers):
                single_obs = {k: v[j][0] for k, v in batched_obs.items()}
                single_info = {k: v[j][0] for k, v in batched_info.items()}

                action = ctrl.compute_control(single_obs, single_info)

                actions.append(action)

            # Stack the individual actions back into a batched array for the env
            batched_action = np.array(actions, dtype=np.float32)
            batched_action = np.expand_dims(batched_action, axis=1)

            (
                batched_obs,
                batched_reward,
                batched_terminated,
                batched_truncated,
                batched_info,
            ) = env.step(batched_action)

            print("Reward: ", batched_reward)

            # 4. Step callbacks for each controller
            for j, ctrl in enumerate(controllers):
                single_obs = {k: v[j] for k, v in batched_obs.items()}
                single_info = {k: v[j] for k, v in batched_info.items()}

                # Note: You might need to handle scalar vs array indexing for reward/terminated depending on your env wrapper
                r = (
                    batched_reward[j]
                    if isinstance(batched_reward, np.ndarray)
                    else batched_reward
                )
                term = (
                    batched_terminated[j]
                    if isinstance(batched_terminated, np.ndarray)
                    else batched_terminated
                )
                trunc = (
                    batched_truncated[j]
                    if isinstance(batched_truncated, np.ndarray)
                    else batched_truncated
                )

                c_finished = ctrl.step_callback(
                    actions[j], single_obs, r, term, trunc, single_info
                )
                controller_finished_list.append(c_finished)

            # Check if ALL environments are done
            # (If you want to reset them independently, you'll need an auto-reset wrapper)
            if np.all(batched_terminated) or np.all(batched_truncated):
                break

            if config.sim.render:  # Render the sim if selected.
                if ((i * fps) % config.env.freq) < fps:
                    env.render()

            i += 1

        # End of episode callbacks
        for ctrl in controllers:
            ctrl.episode_callback()
            ctrl.episode_reset()

        # Logging might need to be adjusted to handle batched info
        log_episode_stats(batched_obs, batched_info, config, curr_time)

        # Example of appending times for the first environment
        ep_times.append(curr_time if batched_obs["target_gate"][0] == -1 else None)

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
