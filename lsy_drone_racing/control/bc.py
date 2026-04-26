import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import optax
import flax.serialization

import lsy_drone_racing.model as model

class DataLoader:
    def __init__(self, npz_files: list):
        all_obs = []
        all_act = []

        for file_path in npz_files:
            data = np.load(file_path, allow_pickle=True)
            all_obs.append(data['obs'][:-1, 0])
            act = data["act"]
            dact = act[1:] - act[:-1]
            all_act.append(dact)
            
        self.obs = np.concatenate(all_obs, axis=0)
        self.act = np.concatenate(all_act, axis=0)

        print(f"Observations shape: {self.obs.shape}")
        print(f"Actions shape: {self.act.shape}")

        self.len = self.obs.shape[0]
        self.idims = self.obs.shape[1]
        self.odims = self.act.shape[1]
        
    def __len__(self):
        return self.len

    def get_batch(self, batch_size: int):
        random_indices = np.random.randint(0, self.len, size=batch_size)
        
        batch_obs = self.obs[random_indices]
        batch_act = self.act[random_indices]
        
        return batch_obs, batch_act

def nll(mu, std, expert_acts):
    std = jnp.clip(std, 1e-6, 1e6)
    log_probs = jax.scipy.stats.norm.logpdf(expert_acts, loc=mu, scale=std)
    loss = -jnp.mean(jnp.sum(log_probs, axis=-1))
    return loss


if __name__ == "__main__":

    ckpt_name = sys.argv[1]
    steps     = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    batch     = int(sys.argv[3]) if len(sys.argv) > 3 else 64
    alpha     = 1e-3 
    
    train_loader = DataLoader([f"episode{r}.npz" for r in range(45)])
    val_loader   = DataLoader([f"episode{r}.npz" for r in range(45, 50)])
    policy = model.v0.Policy()

    key = jax.random.PRNGKey(np.random.randint(0, 1000))
    key, init_key = jax.random.split(key)
    
    params = policy.init(init_key, jnp.zeros((1, train_loader.idims)))

    optim = optax.adam(learning_rate=alpha)
    opts  = optim.init(params)

    print(f"checkpoint: {ckpt_name}.msgpack")
    print(f"steps     : {steps}")
    print(f"batch     : {batch}")

    @jax.jit
    def train_step(params, opt_state, obs, acts):
        def loss_fn(p):
            mu, std = policy.apply(p, obs)
            loss = nll(mu, std, acts)
            
            std_max = jnp.max(std)
            std_min = jnp.min(std)
            std_avg = jnp.mean(std)
            
            return loss, (std_max, std_min, std_avg)
            
        (loss, (s_max, s_min, s_avg)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        
        updates, new_opt_state = optim.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        
        return new_params, new_opt_state, loss, s_max, s_min, s_avg

    @jax.jit
    def val_step(params, obs, acts):
        mu, std = policy.apply(params, obs)
        loss = nll(mu, std, acts)
        return loss

    val_jxobs  = jax.device_put(val_loader.obs)
    val_jxacts = jax.device_put(val_loader.act)

    plt.ion()
    fig, ax = plt.subplots()
    ax.set_xlabel('Steps')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    train_line, = ax.plot([], [], label='Train Loss', color='blue')
    val_line, = ax.plot([], [], label='Val Loss', color='orange')
    ax.legend()
    
    step_hist, train_hist, val_hist = [], [], []

    loss = float('inf')  # Fallback in case process is killed before step 0 finishes

    try:
        for step in range(steps):
            bobs, bacts = train_loader.get_batch(batch) 
            jxobs  = jax.device_put(bobs)
            jxacts = jax.device_put(bacts)
            
            params, opts, loss, s_max, s_min, s_avg = train_step(params, opts, jxobs, jxacts)
            
            if step % 100 == 0:
                val_loss = val_step(params, val_jxobs, val_jxacts)
                print(f"[step {step:4d}] Train Loss: {loss:7.4f} | Val Loss: {val_loss:7.4f} | Std - Min: {s_min:6.4f}, Max: {s_max:6.4f}, Avg: {s_avg:6.4f}")

                step_hist.append(step)
                train_hist.append(float(loss))
                val_hist.append(float(val_loss))
                
                train_line.set_data(step_hist, train_hist)
                val_line.set_data(step_hist, val_hist)
                ax.relim()
                ax.autoscale_view()
                fig.canvas.draw()
                fig.canvas.flush_events()

    except KeyboardInterrupt:
        print("\n[!] Ctrl+C detected. Stopping training early...")

    finally:
        # This block executes whether the loop finishes naturally or is interrupted
        val_loss = val_step(params, val_jxobs, val_jxacts)
        print(f"\nFinal Train Loss: {loss:.4f} | Final Val Loss: {val_loss:.4f}")
        
        plt.ioff()
        plt.show(block=False)
        
        ckpt_path = f"lsy_drone_racing/.ckpt/{ckpt_name}.msgpack"
        
        # Ensure the directory exists so saving doesn't fail
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        
        with open(ckpt_path, "wb") as f:
            bytes_output = flax.serialization.to_bytes(params)
            f.write(bytes_output)
            
        print(f"Successfully saved checkpoint to {ckpt_path}")
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import optax
import flax.serialization

import lsy_drone_racing.model as model

class DataLoader:
    def __init__(self, npz_files: list):
        all_obs = []
        all_act = []

        for file_path in npz_files:
            data = np.load(file_path, allow_pickle=True)
            all_obs.append(data['obs'][:-1, 0])
            act = data["act"]
            dact = act[1:] - act[:-1]
            all_act.append(dact)
            
        self.obs = np.concatenate(all_obs, axis=0)
        self.act = np.concatenate(all_act, axis=0)

        print(f"Observations shape: {self.obs.shape}")
        print(f"Actions shape: {self.act.shape}")

        self.len = self.obs.shape[0]
        self.idims = self.obs.shape[1]
        self.odims = self.act.shape[1]
        
    def __len__(self):
        return self.len

    def get_batch(self, batch_size: int):
        random_indices = np.random.randint(0, self.len, size=batch_size)
        
        batch_obs = self.obs[random_indices]
        batch_act = self.act[random_indices]
        
        return batch_obs, batch_act

def nll(mu, std, expert_acts):
    std = jnp.clip(std, 1e-6, 1e6)
    log_probs = jax.scipy.stats.norm.logpdf(expert_acts, loc=mu, scale=std)
    loss = -jnp.mean(jnp.sum(log_probs, axis=-1))
    return loss


if __name__ == "__main__":

    ckpt_name = sys.argv[1]
    steps     = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    batch     = int(sys.argv[3]) if len(sys.argv) > 3 else 64
    alpha     = 1e-3 
    
    train_loader = DataLoader([f"episode{r}.npz" for r in range(45)])
    val_loader   = DataLoader([f"episode{r}.npz" for r in range(45, 50)])
    policy = model.v0.Policy()

    key = jax.random.PRNGKey(np.random.randint(0, 1000))
    key, init_key = jax.random.split(key)
    
    params = policy.init(init_key, jnp.zeros((1, train_loader.idims)))

    optim = optax.adam(learning_rate=alpha)
    opts  = optim.init(params)

    print(f"checkpoint: {ckpt_name}.msgpack")
    print(f"steps     : {steps}")
    print(f"batch     : {batch}")

    @jax.jit
    def train_step(params, opt_state, obs, acts):
        def loss_fn(p):
            mu, std = policy.apply(p, obs)
            loss = nll(mu, std, acts)
            
            std_max = jnp.max(std)
            std_min = jnp.min(std)
            std_avg = jnp.mean(std)
            
            return loss, (std_max, std_min, std_avg)
            
        (loss, (s_max, s_min, s_avg)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        
        updates, new_opt_state = optim.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        
        return new_params, new_opt_state, loss, s_max, s_min, s_avg

    @jax.jit
    def val_step(params, obs, acts):
        mu, std = policy.apply(params, obs)
        loss = nll(mu, std, acts)
        return loss

    val_jxobs  = jax.device_put(val_loader.obs)
    val_jxacts = jax.device_put(val_loader.act)

    plt.ion()
    fig, ax = plt.subplots()
    ax.set_xlabel('Steps')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    train_line, = ax.plot([], [], label='Train Loss', color='blue')
    val_line, = ax.plot([], [], label='Val Loss', color='orange')
    ax.legend()
    
    step_hist, train_hist, val_hist = [], [], []

    loss = float('inf')  # Fallback in case process is killed before step 0 finishes

    try:
        for step in range(steps):
            bobs, bacts = train_loader.get_batch(batch) 
            jxobs  = jax.device_put(bobs)
            jxacts = jax.device_put(bacts)
            
            params, opts, loss, s_max, s_min, s_avg = train_step(params, opts, jxobs, jxacts)
            
            if step % 100 == 0:
                val_loss = val_step(params, val_jxobs, val_jxacts)
                print(f"[step {step:4d}] Train Loss: {loss:7.4f} | Val Loss: {val_loss:7.4f} | Std - Min: {s_min:6.4f}, Max: {s_max:6.4f}, Avg: {s_avg:6.4f}")

                step_hist.append(step)
                train_hist.append(float(loss))
                val_hist.append(float(val_loss))
                
                train_line.set_data(step_hist, train_hist)
                val_line.set_data(step_hist, val_hist)
                ax.relim()
                ax.autoscale_view()
                fig.canvas.draw()
                fig.canvas.flush_events()

    except KeyboardInterrupt:
        print("\n[!] Ctrl+C detected. Stopping training early...")

    finally:
        val_loss = val_step(params, val_jxobs, val_jxacts)
        print(f"\nFinal Train Loss: {loss:.4f} | Final Val Loss: {val_loss:.4f}")
        plt.ioff()
        plt.show(block=False)
        ckpt_path = f"lsy_drone_racing/.ckpt/{ckpt_name}.msgpack"
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        with open(ckpt_path, "wb") as f:
            bytes_output = flax.serialization.to_bytes(params)
            f.write(bytes_output)
            
        print(f"Successfully saved checkpoint to {ckpt_path}")
