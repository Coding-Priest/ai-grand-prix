import torch
import numpy as np
from train_rl_custom import make_envs

def test_thrust():
    """Test script to apply maximum thrust and print observations."""
    print("Initializing environment...")
    
    # Create a single environment instance on CPU
    # make_envs(config="level0.toml", num_envs=1, jax_device="cpu", torch_device=torch.device("cpu"), coefs={})
    device = torch.device("cpu")
    envs = make_envs(
        config="level0.toml",
        num_envs=1,
        jax_device="cpu",
        torch_device=device,
        coefs={"training": False} # Set training=False for VecNormalize
    )
    
    # Define observation feature names in order of flattening
    # From RaceTrainEnv and ActionPenalty wrapper
    obs_names = [
        "roll", "pitch", "yaw",
        "vel_x", "vel_y", "vel_z",
        "dist_target", "dist_next",
        "gate_vec_x", "gate_vec_y", "gate_vec_z",
        "gate_alignment",
        "last_act_roll", "last_act_pitch", "last_act_yaw", "last_act_thrust"
    ]
    
    # Reset the environment
    obs, info = envs.reset()
    print("\nInitial Observation:")
    if isinstance(obs, torch.Tensor):
        obs_np = obs.cpu().numpy().flatten()
        for name, val in zip(obs_names, obs_np):
            print(f"  {name:15}: {val:10.4f}")
    
    # Define actions for individual components
    actions = [
        ("Circular Flight", None), # Special case
        ("Mixed (Roll + Pitch)", torch.tensor([[ 0.0,  0.5,  1.0,  1.0]], dtype=torch.float32).to(device)),
    ]
    
    print("\nTesting control behaviors...")
    
    try:
        for name, action in actions:
            envs.reset() 
            print(f"\n>>> Testing {name}...")
            
            for i in range(200): # More steps to see circular motion
                if name == "Circular Flight":
                    # Increment yaw over time to create circular motion
                    # Action: [roll, pitch, yaw, thrust]
                    # We cycle yaw from -1 to 1 repeatedly
                    yaw_val = (i % 40) / 20.0 - 1.0 # Simple sawtooth for yaw
                    current_action = torch.tensor([[0.0, 0.3, yaw_val, 0.7]], dtype=torch.float32).to(device)
                else:
                    current_action = action
                
                obs, reward, terminated, truncated, info = envs.step(current_action)
                envs.render()
                
                if i % 20 == 0:
                    roll = obs[0, 0].item()
                    pitch = obs[0, 1].item()
                    yaw = obs[0, 2].item()
                    vel_z = obs[0, 5].item()
                    print(f"[{name}] Step {i+1:3d} | Roll: {roll:7.4f} | Pitch: {pitch:7.4f} | Yaw: {yaw:7.4f} | Vel_Z: {vel_z:7.4f}")
                
                if terminated.any() or truncated.any():
                    print(f"Episode ended during {name} test.")
                    break
    
    except Exception as e:
        print(f"An error occurred during stepping: {e}")
    finally:
        envs.close()
        print("\nEnvironment closed.")

if __name__ == "__main__":
    test_thrust()
