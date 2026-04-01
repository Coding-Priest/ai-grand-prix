import torch
import os

from models.waypointac import WaypointActorCritic
from models.waypointac2 import WaypointActorCritic2


def upgrade_checkpoint(old_checkpoint_path, new_checkpoint_path):
    """
    Transforms weights from an old checkpoint (without the is_final_gate flag)
    to be compatible with the new architecture (with the flag).
    """
    if not os.path.exists(old_checkpoint_path):
        raise FileNotFoundError(f"Could not find checkpoint at {old_checkpoint_path}")

    print(f"Loading old checkpoint from: {old_checkpoint_path}")
    checkpoint = torch.load(old_checkpoint_path, map_location="cpu", weights_only=False)

    # Extract the state dict
    if "model_state_dict" in checkpoint:
        old_state_dict = checkpoint["model_state_dict"]
        iteration = checkpoint.get("iteration", 0)
        best_reward = checkpoint.get("best_avg_reward", -float("inf"))
    else:
        old_state_dict = checkpoint
        iteration = 0
        best_reward = -float("inf")

    new_state_dict = {}

    print("\n--- Starting Network Surgery ---")
    for key, tensor in old_state_dict.items():
        # Intercept the first layer of both the Actor and the Critic networks
        if key in ["actor_mean.0.weight", "critic.0.weight"]:
            out_features, in_features = tensor.shape
            print(f"Patching {key} | Old Shape: [{out_features}, {in_features}]")

            # Create a column of zeros for the new input dimension (is_final_gate)
            # This ensures the new flag initially has zero impact on the drone's flight
            zero_column = torch.zeros(
                (out_features, 1), dtype=tensor.dtype, device=tensor.device
            )

            # Concatenate the old weights with the new zero column
            patched_tensor = torch.cat([tensor, zero_column], dim=1)
            print(
                f"  -> New Shape: [{patched_tensor.shape[0]}, {patched_tensor.shape[1]}]"
            )

            new_state_dict[key] = patched_tensor
        else:
            # All other hidden layers, biases, and output layers copy over perfectly
            new_state_dict[key] = tensor

    print("--- Surgery Complete ---\n")

    # Create the new checkpoint dictionary
    # NOTE: We intentionally leave out the 'optimizer_state_dict'.
    # Because we changed the shape of layer 0, the old Adam momentum buffers are invalid.
    new_checkpoint = {
        "iteration": iteration,
        "best_avg_reward": best_reward,
        "model_state_dict": new_state_dict,
    }

    # Save the upgraded checkpoint
    torch.save(new_checkpoint, new_checkpoint_path)
    print(f"Successfully saved upgraded checkpoint to: {new_checkpoint_path}")
    print(
        "You can now load this file in your training loop. Remember to let Adam re-initialize!"
    )


if __name__ == "__main__":
    # Define your paths here
    OLD_PATH = "/home/homefree/Development/anduril-drone-race/ai-grand-prix/checkpoints/best_model.pth"
    NEW_PATH = "/home/homefree/Development/anduril-drone-race/ai-grand-prix/model_weights/2_waypoint_ac/botox_model_1.pth"

    upgrade_checkpoint(OLD_PATH, NEW_PATH)
