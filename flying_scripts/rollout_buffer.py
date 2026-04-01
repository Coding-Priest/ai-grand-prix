import torch
import numpy as np
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class RolloutBuffer:
    def __init__(self, num_steps, num_envs, obs_dim, action_dim, device="cpu"):
        """
        Args:
            num_steps: Number of steps to collect per environment before updating.
            num_envs: Number of parallel environments running.
            obs_dim: Dimension of the flattened observation state.
            action_dim: Dimension of the action space.
            device: 'cpu' or 'cuda'.
        """
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = torch.device(device)

        # Buffer pointers and state
        self.step = 0
        self.full = False

        # Initialize memory tensors
        # Shapes are usually (num_steps, num_envs, dimension)
        self.observations = torch.zeros(
            (num_steps, num_envs, obs_dim), dtype=torch.float32
        ).to(self.device)
        self.actions = torch.zeros(
            (num_steps, num_envs, action_dim), dtype=torch.float32
        ).to(self.device)
        self.rewards = torch.zeros((num_steps, num_envs), dtype=torch.float32).to(
            self.device
        )
        self.values = torch.zeros((num_steps, num_envs), dtype=torch.float32).to(
            self.device
        )
        self.log_probs = torch.zeros((num_steps, num_envs), dtype=torch.float32).to(
            self.device
        )

        # Dones are used to mask out values when an episode terminates
        self.dones = torch.zeros((num_steps, num_envs), dtype=torch.float32).to(
            self.device
        )

        # Computed during the advantage estimation phase
        self.advantages = torch.zeros((num_steps, num_envs), dtype=torch.float32).to(
            self.device
        )
        self.returns = torch.zeros((num_steps, num_envs), dtype=torch.float32).to(
            self.device
        )

    def reset(self):
        """Clears the buffer for the next rollout phase."""
        self.step = 0
        self.full = False

    def add(self, obs, action, reward, value, log_prob, done):
        """
        Inserts a single step of data from all parallel environments.
        All inputs should be PyTorch tensors of shape (num_envs, ...)
        """
        if self.full:
            raise RuntimeError(
                "RolloutBuffer is full. Call compute_returns_and_advantages() and train before adding more."
            )

        self.observations[self.step] = obs.detach()
        self.actions[self.step] = action.detach()
        self.rewards[self.step] = reward.detach()
        self.values[self.step] = value.detach().squeeze(-1)
        self.log_probs[self.step] = log_prob.detach().squeeze(-1)
        self.dones[self.step] = done.detach()

        self.step += 1
        if self.step == self.num_steps:
            self.full = True

    def compute_returns_and_advantages(
        self, last_value, last_done, gamma=0.99, gae_lambda=0.95
    ):
        """
        Computes Generalized Advantage Estimation (GAE).
        """
        last_value = last_value.clone().detach().squeeze(-1)
        last_done = last_done.clone().detach()

        last_gae_lam = 0

        # Iterate backwards through the buffer to compute advantages
        for step in reversed(range(self.num_steps)):
            if step == self.num_steps - 1:
                next_non_terminal = 1.0 - last_done
                next_values = last_value
            else:
                # FIX: Check if the CURRENT step ended the episode to mask the NEXT value
                next_non_terminal = 1.0 - self.dones[step]
                next_values = self.values[step + 1]

            # TD Error
            delta = (
                self.rewards[step]
                + gamma * next_values * next_non_terminal
                - self.values[step]
            )

            # GAE Calculation
            last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam

        # Returns are simply Advantages + Values
        self.returns = self.advantages + self.values

    def get_generator(self, batch_size):
        """
        Yields flattened mini-batches from the rollout buffer for network updates.
        """
        if not self.full:
            raise RuntimeError("RolloutBuffer is not full. Cannot generate batches.")

        # Flatten the buffer from (num_steps, num_envs, dim) to (num_steps * num_envs, dim)
        flat_size = self.num_steps * self.num_envs

        flat_obs = self.observations.view(flat_size, -1)
        flat_actions = self.actions.view(flat_size, -1)
        flat_values = self.values.view(flat_size)
        flat_log_probs = self.log_probs.view(flat_size)
        flat_advantages = self.advantages.view(flat_size)
        flat_returns = self.returns.view(flat_size)

        # Normalize advantages (Standard practice to stabilize training)
        # flat_advantages = (flat_advantages - flat_advantages.mean()) / (
        #     flat_advantages.std() + 1e-8
        # )

        # Create a random sampler to shuffle the data
        sampler = BatchSampler(
            SubsetRandomSampler(range(flat_size)), batch_size, drop_last=True
        )

        for indices in sampler:
            mb_advantages = flat_advantages[indices]
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                mb_advantages.std(unbiased=False) + 1e-8
            )
            yield (
                flat_obs[indices],
                flat_actions[indices],
                flat_values[indices],
                flat_log_probs[indices],
                flat_advantages[indices],
                flat_returns[indices],
            )
