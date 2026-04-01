import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path

# Import your model (adjust the import path based on your project structure)
from models.waypointac3 import WaypointActorCritic3


class ExpertDataset(Dataset):
    def __init__(self, data_dir):
        """Loads and concatenates all .npz files found in the specified directory."""
        all_states = []
        all_actions = []

        # Find all .npz files in the directory
        data_path = Path(data_dir)
        data_files = list(data_path.glob("*.npz"))

        if not data_files:
            raise FileNotFoundError(f"No .npz files found in directory: {data_dir}")

        print(f"Found {len(data_files)} data files in '{data_dir}'. Loading...")

        for file_path in data_files:
            data = np.load(file_path)
            all_states.append(data["states"])
            all_actions.append(data["actions"])
            print(f" - Loaded {file_path.name}: {len(data['states'])} transitions")

        # Concatenate all loaded data along the first axis (batch dimension)
        combined_states = np.concatenate(all_states, axis=0)
        combined_actions = np.concatenate(all_actions, axis=0)

        self.states = torch.tensor(combined_states, dtype=torch.float32)
        self.actions = torch.tensor(combined_actions, dtype=torch.float32)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]


def pretrain_actor(
    data_dir="/home/homefree/Development/anduril-drone-race/ai-grand-prix/bootstrap_data/version3/two_Gates",
    epochs=400,
    batch_size=512,
    lr=1e-3,
):
    # 1. Setup Device & Data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Initialize the dataset using the directory path
    dataset = ExpertDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"Total transitions across all files: {len(dataset)}")

    # 2. Initialize Model
    model = WaypointActorCritic3().to(device)

    # We only want to train the actor_mean network for Behavioral Cloning
    optimizer = optim.Adam(model.actor_mean.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # 3. Training Loop
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_states, batch_actions in dataloader:
            batch_states = batch_states.to(device)
            batch_actions = batch_actions.to(device)

            # Forward pass: get deterministic action from the actor
            predicted_actions = model.actor_mean(batch_states)

            # Compute MSE Loss against expert
            loss = loss_fn(predicted_actions, batch_actions)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")

    # 4. Save the Pre-trained weights
    save_path = "pretrained_waypoint_ac3_bc0.pth"
    # torch.save(model.state_dict(), save_path)

    torch.save(
        {
            # "iteration": iteration,
            "model_state_dict": model.state_dict(),
            # "optimizer_state_dict": optimizer.state_dict(),
            # "best_avg_reward": best_avg_reward,
        },
        "/home/homefree/Development/anduril-drone-race/ai-grand-prix/model_weights/3_waypoints_ac/"
        + save_path,
    )
    print(f"Pre-training complete. Model saved to {save_path}")


if __name__ == "__main__":
    pretrain_actor()
