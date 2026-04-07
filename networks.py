import jax.numpy as jnp
from typing import Any, Callable, Optional, Sequence
import flax.linen as nn
from utils import default_init
import distrax



class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: bool = False
    layer_norm: bool = False
    kernel_init: Callable[[Any, Sequence[int], Any], jnp.ndarray] = default_init()

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for i, hidden_dims in enumerate(self.hidden_dims):
            x = nn.Dense(hidden_dims, kernel_init=self.kernel_init)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                if self.layer_norm:  # Add layer norm before activation
                    x = nn.LayerNorm()(x)
                x = self.activations(x)
        return x

class Critic(nn.Module):
    """Don't use extra critic for values instead use the same with rest=[]"""
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    ensure_positive: bool = False
    layer_norm: bool = False

    @nn.compact
    def __call__(self, observations: jnp.ndarray, *rest: jnp.ndarray) -> jnp.ndarray:
        inputs0: Sequence[jnp.ndarray] = [observations]+list(rest)
        inputs = jnp.concatenate(inputs0, -1)
        critic = MLP((*self.hidden_dims, 1), activations=self.activations, layer_norm=self.layer_norm)(inputs)
        if self.ensure_positive:
            critic = jnp.abs(critic)
        return jnp.squeeze(critic, -1)

class MonotonicCritic(nn.Module):
    """Don't use extra critic for values instead use the same with rest=[]"""
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    layer_norm: bool = False

    @nn.compact
    def __call__(self, observations: jnp.ndarray, *rest: jnp.ndarray) -> jnp.ndarray:
        assert len(rest) > 0, "Extra input is required for monotonic critic"
        budgets = rest[-1] # Model is monotonic in the last input   
        rest = rest[:-1]
        final_size: int = self.hidden_dims[-1]
        hidden_dims: Sequence[int]  = self.hidden_dims[:-1]
        inputs = jnp.concatenate([observations]+list(rest), -1)
        dims = (*hidden_dims, (1+final_size*2))
        critic = MLP(dims, activations=self.activations, layer_norm=self.layer_norm)(inputs)
        base = critic[..., 0] 
        w, b = critic[..., 1:1+final_size], critic[..., 1+final_size:]
        w = jnp.abs(w) # Positive weights
        critic = base + jnp.max(w*budgets+b, axis=-1) # Max w.r.t budget
        return critic

class GaussianPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    log_std_min: Optional[float] = -20
    log_std_max: Optional[float] = 2
    droupout_rate: float = 0.25
    tanh_squash_distribution: bool  = True

    @nn.compact
    def __call__(
        self, observations: jnp.ndarray, cost_limits: Optional[jnp.ndarray]=None, temperature: float = 1.0,
        training: bool = False
    ) -> distrax.Distribution:
        
        if cost_limits is None:
            x = observations
        else:
            x = jnp.concatenate([observations, cost_limits], -1)

        outputs = MLP(
            self.hidden_dims,
            activate_final=True,
        )(x)
        if self.droupout_rate > 0:
            outputs = nn.Dropout(rate=self.droupout_rate, deterministic=not training)(outputs)

        means = nn.Dense(
            self.action_dim, kernel_init=default_init()
        )(outputs)
        log_stds = nn.Dense(self.action_dim, kernel_init=default_init())(outputs) #self.param("log_stds", nn.initializers.zeros, (self.action_dim,))
        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)
        if not self.tanh_squash_distribution:
            means = nn.tanh(means)

        base_dist = distrax.MultivariateNormalDiag(loc=means,
                                                scale_diag=jnp.exp(log_stds) * temperature)
        if self.tanh_squash_distribution:
            tanh_bijector = distrax.Block(distrax.Tanh(), ndims=1)
            return distrax.Transformed(base_dist, tanh_bijector)
        else:
            return base_dist


