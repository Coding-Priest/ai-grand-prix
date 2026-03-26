from pathlib import Path

import functools
import numpy as np
from numpy.typing import NDArray

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import serialization
import optax

SQRT_PI  = 1.7724538509055159
SQRT_2   = 1.4142135623730951
SQRT_2PI = 2.5066282746310002
EPSILON  = 1e-6

class Policy(nn.Module):
    nslope: float = 0.01

    @nn.compact
    def __call__(self, v):

        h = nn.leaky_relu(nn.Dense(features=256)(v), negative_slope=self.nslope)
        h = nn.leaky_relu(nn.Dense(features=256)(h), negative_slope=self.nslope)
        h = nn.leaky_relu(nn.Dense(features=256)(h), negative_slope=self.nslope)
        
        m   = nn.leaky_relu(nn.Dense(features=256)(h), negative_slope=self.nslope)
        mu  = nn.tanh(nn.Dense(features=4)(m))

        s   = nn.leaky_relu(nn.Dense(features=256)(h), negative_slope=self.nslope)
        std = nn.softplus(nn.Dense(features=4)(s)) + EPSILON

        return mu, std

class Agent:
    def __init__(self,
            train: bool,
            obs_dim: int,
            alpha: float = 0.001,
            gamma: float = 0.99,
            ckpt: Path | str | None = None):

        self.train   = train
        self.obs_dim = obs_dim
        self.alpha   = alpha
        self.gamma   = gamma

        self.policy = Policy()

        self.key = jax.random.PRNGKey(42) 
        self.key, init_key = jax.random.split(self.key)
        self.params = self.policy.init(init_key, jnp.zeros((1, obs_dim))) 

        if ckpt and Path(ckpt).is_file():
            print(f"ckpt: {ckpt}")
            self.params = serialization.from_bytes(self.params, open(ckpt, "rb").read())

        if train:
            self.optim = optax.adam(learning_rate=self.alpha)
            self.opts = self.optim.init(self.params)

            self.jxobs   = []
            self.rewards = []
            self.jxacts  = []

    def consume(self, reward):
        if self.train:
            self.rewards.append(reward)

    def forward(self, ob: NDArray[np.floating]) -> np.ndarray:
        jxob   = jax.device_put(ob)
        mu, std = self.policy.apply(self.params, jxob)

        if self.train:
            self.key, ckey = jax.random.split(self.key)
            z = jax.random.normal(ckey, shape=mu.shape) 
            act = mu + z * std
            self.jxobs.append(jxob)
            self.jxacts.append(act)
        else:
            act = mu

        return np.asarray(act).squeeze(), np.asarray(1 / (std + EPSILON)).squeeze()

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(self, params, opts, jxobs, jxacts, jxG):
        def lfn(params):
            mus, stds = self.policy.apply(params, jxobs)
            nll  = (0.5 * ((mus - jxacts) / stds) ** 2 + jnp.log(stds * SQRT_2PI)).sum(axis=1)
            loss = (jxG * nll).mean()
            return loss

        loss, dw = jax.value_and_grad(lfn)(params)
        updates, opts = self.optim.update(dw, opts, params)
        params = optax.apply_updates(params, updates)
        return params, opts, loss

    def backward(self):
        lj = len(self.jxobs)
        lr = len(self.rewards)
        la = len(self.jxacts)

        assert lj == lr == la, f"size mismatch {lj} {lr} {la}"

        G = [] # return
        g = 0
        for r in reversed(self.rewards):
            g = r + self.gamma * g
            G.append([g])
        G = list(reversed(G))
        
        jxobs  = jnp.stack(self.jxobs).astype(jnp.float32)
        jxacts = jnp.stack(self.jxacts).astype(jnp.float32)
        jxG    = jnp.array(G, dtype=jnp.float32)
        jxG    = (jxG - jxG.mean()) / (jxG.std() + EPSILON)

        self.params, self.opts, loss = self.step(self.params, self.opts, jxobs, jxacts, jxG)
    
        self.jxobs.clear()
        self.rewards.clear()
        self.jxacts.clear()

        return float(loss), sum(g[0] for g in G)

    def save(self, path: Path | str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        bo = serialization.to_bytes(self.params)
        open(path, "wb").write(bo)
