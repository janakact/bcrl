# source https://github.com/ikostrikov/implicit_q_learning
# https://arxiv.org/abs/2110.06169
import os
from functools import partial
from typing import Any, NamedTuple, Tuple, Union, Dict, Sequence, Callable, Optional
from flax.core.frozen_dict import FrozenDict
import matplotlib.pyplot as plt
import flax.linen as nn
import chex

use_gymnasium = os.getenv("USE_GYMNASIUM", "1") == "1" # Default true

if use_gymnasium:
    import gymnasium as gym
else:
    import gym
import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
import optax
import distrax
import wandb
from flax.training.train_state import TrainState
from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict
import dsrl as _dsrl
from utils import (
    ensemblize, save_model, Transition,
    evaluate_for_multiple_threshodls, get_dataset, expectile_loss, set_common_jax_flags, 
    normalize_dataset, sample_batch, target_update, update_by_loss_grad, train_parallel,
    gumbel_rescale_loss, dict_hash_ignore_keys, get_task_name_from_env_name
)
from networks import Critic, MonotonicCritic, GaussianPolicy, MLP, default_init

set_common_jax_flags()
# jax.config.update("jax_disable_jit", True)

class IQLConfig(BaseModel):
    # GENERAL
    env_name: str  = "OfflinePointButton1Gymnasium-v0" # SafetyGym, The first raw in DSRL resutls
    tag: str = "empty"
    algo: str = "bcrl-det"
    project: str = "bcrl"
    use_wandb: bool = True
    log_dir_base: str = "tmp/bcrl-logs"
    seed: int = 42
    final_eval_episodes: int = 20
    log_interval: int = 1000
    eval_interval: int = 20_000
    batch_size: int = 512
    max_steps: int = int(1e5)
    n_jitted_updates: int = 10
    # DATASET
    data_size: int = int(1e10)
    # NETWORK
    hidden_dims: Tuple[int, int] = (256, 256)
    actor_lr: float = 3e-4
    value_lr: float = 3e-4
    critic_lr: float = 3e-4
    cost_value_lr: float = 3e-4
    cost_critic_lr: float = 3e-4
    gradient_clip: float = np.inf
    # IQL SPECIFIC
    expectile: float = 0.8  # FYI: for Hopper-me, 0.5 produce better result. (antmaze: expectile=0.9)
    beta: float = 8.0  # FYI: for Hopper-me, 6.0 produce better result. (antmaze: beta=10.0
    tau: float = 0.005
    discount: float = 0.99

    # Safe IQL specific
    cost_expectile: Union[None, float] = None
    cost_expectile_transition: float = 0.5 # Transition probabilities we should take max instead of min or mean 0.5 for mean
    cost_discount: float = 0.99
    cost_violation_penalty: Union[float, None] = None
    cost_violation_reward: float = -100.0
    eval_cost_thresholds: Union[tuple, int] = (20, 40, 80)
    rescale_remaining_budget_on_eval: bool = False
    cost_budget_sample_count: int = 1
    normalize_cost_and_reward: bool = False
    opex_step_size: float = 0.0 # Zero is no opex
    max_cost_to_learn: Union[float, None] = None
    actor_lr_decay: bool = True
    normalize_state: bool = True
    policy_dropout_rate: float = 0.0
    use_monotonic: bool = False
    min_cost_threhold_ratio: float = 0.0
    in_sample_mode: str = 'expectile' #  #'extreme', 'quantile'
    extreme_beta_cost: float = 0.1  # Beta for extreme loss
    extreme_beta_reward: float = 0.1  # Beta for extreme loss
    sparse_alpha_reward: float = 1
    sparse_alpha_cost: float = 1
    use_advantage: bool = True
    budget_exponential_scale: float = 10.0
    finite_horizon_min: Union[float, None] = None
    save_policy: bool = True
    tanh_squash_distribution: bool = False


    model_config = ConfigDict(extra='forbid')

    def __hash__(
        self,
    ):  # make config hashable to be specified as static_argnums in jax.jit.
        return hash(self.__repr__())




def logit_weighted_mse_loss(predicted, target, logits, negative_target):
    """Positive -> target, Negative -> infeasible_target"""
    p = jax.nn.sigmoid(logits)
    return p*((target-predicted)**2) + (1-p)*((negative_target-predicted)**2)


class IQLTrainState(NamedTuple):
    rng: jnp.ndarray
    critic: TrainState
    target_critic: TrainState
    value: TrainState
    cost_critic: TrainState
    cost_target_critic: TrainState
    cost_value: TrainState
    actor: TrainState

class GaussianHindSightPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    log_std_min: Optional[float] = -20
    log_std_max: Optional[float] = 2
    droupout_rate: float = 0.25
    tanh_squash_distribution: bool  = False

    @nn.compact
    def __call__(
        self, observations: jnp.ndarray, cost_limits: jnp.ndarray, temperature: float = 1.0,
        training: bool = False
    ) -> Tuple[distrax.Distribution,distrax.Distribution]:
        dist= GaussianPolicy(
            hidden_dims=self.hidden_dims,
            action_dim=self.action_dim,
            log_std_min=self.log_std_min,log_std_max=self.log_std_max, 
            droupout_rate=self.droupout_rate,
            tanh_squash_distribution=self.tanh_squash_distribution
        )(observations, cost_limits, temperature=temperature, training=training)
        cost_limits_dummy = jnp.zeros((*cost_limits.shape[:-1], 0))  # Dummy budget for safe distribution
        dist_c= GaussianPolicy(hidden_dims=self.hidden_dims,
                               action_dim=self.action_dim,
                               log_std_min=self.log_std_min,log_std_max=self.log_std_max, 
                               droupout_rate=self.droupout_rate,
                                tanh_squash_distribution=self.tanh_squash_distribution
                               )(observations, cost_limits_dummy, temperature=temperature, training=training)
        return dist, dist_c

def sparse_value_loss(v, q, alpha):
    sp_term = (q - v) / (2 * alpha) + 1.0
    sp_weight = jnp.where(sp_term > 0, 1., 0.)
    value_loss = (sp_weight * (sp_term**2) + v / alpha)
    return value_loss

class IQL(object):


    @classmethod
    def update_cost_critic(
        cls, train_state: IQLTrainState, batch: Transition, config: IQLConfig
    ) -> Tuple["IQLTrainState", jnp.ndarray]:
        def cost_critic_loss_fn(
            cost_critic_params: FrozenDict[str, Any]
        ) -> jnp.ndarray:
            next_v = train_state.cost_value.apply_fn(
                train_state.cost_value.params, batch.next_observations
            )
            target_q = batch.costs + config.cost_discount * (1 - batch.dones) * next_v
            q1, q2 = train_state.cost_critic.apply_fn(
                cost_critic_params, batch.observations, batch.actions
            )

            critic_loss = (((target_q-q1)**2 + (target_q-q2)**2))/2
            chex.assert_shape(critic_loss, (len(batch.observations),))
            critic_loss = critic_loss.mean()
            return critic_loss, {"critic_loss": critic_loss, "mean_q_cost": (q1+q2).mean()}

        new_critic, critic_loss, metrics = update_by_loss_grad(
            train_state.cost_critic, cost_critic_loss_fn
        )
        return train_state._replace(cost_critic=new_critic), metrics

    @classmethod
    def update_cost_value(
        cls, train_state: IQLTrainState, batch: Transition, config: IQLConfig
    ) -> Tuple["IQLTrainState", jnp.ndarray]:
        def cost_value_loss_fn(cost_value_params: FrozenDict[str, Any]) -> jnp.ndarray:
            q1, q2 = train_state.cost_target_critic.apply_fn(
                train_state.cost_target_critic.params, batch.observations, batch.actions
            )
            q = jax.lax.stop_gradient(jnp.maximum(q1, q2))
            v = train_state.cost_value.apply_fn(cost_value_params, batch.observations)
            if config.in_sample_mode == 'expectile':
                value_loss = expectile_loss(q - v, config.cost_expectile).mean()
            elif config.in_sample_mode == 'extreme':
                value_loss = gumbel_rescale_loss(v-q, config.extreme_beta_cost).mean()
            elif config.in_sample_mode == 'sparse':
                value_loss = sparse_value_loss(-v, -q, config.sparse_alpha_cost).mean()
            else:
                raise ValueError(f"Unknown in_sample_mode: {config.in_sample_mode}")
            return value_loss, {"mean_value_cost": v.mean(), "mean_q_cost": q.mean()}

        new_value, value_loss, metrics = update_by_loss_grad(train_state.cost_value, cost_value_loss_fn)
        return train_state._replace(cost_value=new_value), {**metrics, "value_loss": value_loss}

    @classmethod
    def sample_deltabar(cls, rng, shape, base_cost, max_cost):
        """Sample deltabar from uniform distribution between base_cost and max_cost."""
        base_cost = jnp.clip(base_cost, min=0, max=max_cost)
        uniform = jax.random.uniform(rng, shape, minval=0, maxval=1)
        scale = jnp.clip(max_cost - base_cost, min=0)
        return uniform * scale + base_cost

    @classmethod
    def update_critic(
        cls, train_state: IQLTrainState, batch: Transition, config: IQLConfig, 
        rng: jnp.ndarray, max_cost: float
    ) -> Tuple["IQLTrainState", jnp.ndarray]:
        def critic_loss_fn(
            critic_params: FrozenDict[str, Any]
        ) -> jnp.ndarray:
            # jax.debug.print("Re: {a} {b}", a=batch.rewards[:2], b=batch.actions[:2])
            observations = batch.observations.repeat(config.cost_budget_sample_count, axis=0)
            actions = batch.actions.repeat(config.cost_budget_sample_count, axis=0)
            next_observations = batch.next_observations.repeat(config.cost_budget_sample_count, axis=0)
            costs = batch.costs[..., None].repeat(config.cost_budget_sample_count, axis=0)
            rewards = batch.rewards.repeat(config.cost_budget_sample_count, axis=0)
            dones = batch.dones.repeat(config.cost_budget_sample_count, axis=0)

            # Get cost estimation
            cq1, cq2 = train_state.cost_target_critic.apply_fn(
                train_state.cost_target_critic.params, observations, actions
            )
            action_cost = jnp.maximum(cq1, cq2)
            next_state_cost = train_state.cost_value.apply_fn(train_state.cost_value.params, next_observations)
            next_state_cost = jnp.maximum(next_state_cost[..., None], (action_cost[...,None]-costs)/config.cost_discount)  
            # Ensure non-negative cost
            next_state_cost = jnp.clip(next_state_cost, min=0)

            deltabar_min = next_state_cost 
            deltabar_target = cls.sample_deltabar(rng, (len(observations), 1), deltabar_min, max_cost)
            weight = (max_cost - deltabar_min[:, 0]) / max_cost
            deltabar = costs +  deltabar_target*config.cost_discount

            chex.assert_shape(deltabar, (len(observations), 1))
            chex.assert_shape(action_cost, (len(observations), ))
            chex.assert_shape(next_state_cost, (len(observations), 1))
            chex.assert_shape(costs, (len(observations), 1))
            chex.assert_shape(deltabar_target, (len(observations), 1))
            chex.assert_shape(rewards, (len(observations),))
            chex.assert_shape(weight, (len(observations), ))

            next_v = train_state.value.apply_fn(
                train_state.value.params, next_observations, deltabar_target
            )
            target_q = rewards + config.discount * (1 - dones) * next_v
            q1, q2 = train_state.critic.apply_fn(
                critic_params, observations, actions, deltabar
            )
            critic_loss =  ((target_q - q1)**2 +  (target_q - q2)**2)/2

            chex.assert_shape(critic_loss, (len(observations),))
            critic_loss = critic_loss.mean()
            return critic_loss, {"critic_loss": critic_loss,}

        new_critic, critic_loss, metrics = update_by_loss_grad(
            train_state.critic, critic_loss_fn,
        )
        return train_state._replace(critic=new_critic), metrics 

    @classmethod
    def update_value(
        cls, train_state: IQLTrainState, batch: Transition, config: IQLConfig, 
        rng: jnp.ndarray, max_cost: float
    ) -> Tuple["IQLTrainState", jnp.ndarray]:
        def value_loss_fn(value_params: FrozenDict[str, Any]) -> jnp.ndarray:
            observations = batch.observations.repeat(config.cost_budget_sample_count, axis=0)
            actions = batch.actions.repeat(config.cost_budget_sample_count, axis=0)
            cq1, cq2 = train_state.cost_target_critic.apply_fn(
                train_state.cost_target_critic.params, observations, actions
            )
            action_cost = jnp.maximum(cq1, cq2)

            deltabar = cls.sample_deltabar(rng, (len(observations), 1), action_cost[...,None], max_cost)
            q1, q2 = train_state.target_critic.apply_fn(
                train_state.target_critic.params, observations, actions, deltabar
            )
            # jax.debug.print("{a}:{b}   / {c}", a=action_cost[0], b=action_cost.mean(), c=max_cost)
            q = jax.lax.stop_gradient(jnp.minimum(q1, q2))
            v = train_state.value.apply_fn(value_params, observations, deltabar)

            if config.in_sample_mode == 'expectile':
                loss_values = expectile_loss(q - v, config.expectile)
            elif config.in_sample_mode == 'extreme':
                loss_values = gumbel_rescale_loss(q-v, config.extreme_beta_reward)
            elif config.in_sample_mode == 'sparse':
                loss_values = sparse_value_loss(v, q, config.sparse_alpha_reward).mean()
            value_loss = loss_values.mean()
            return value_loss, {"value_loss": value_loss}

        new_value, value_loss, metrics = update_by_loss_grad(train_state.value, value_loss_fn)
        return train_state._replace(value=new_value), metrics

    @classmethod
    def update_actor(
        cls, train_state: IQLTrainState, batch: Transition, config: IQLConfig,
        rng: jnp.ndarray, max_cost: float
    ) -> Tuple["IQLTrainState", jnp.ndarray]:
        dropout_train_key, rng = jax.random.split(rng)
        def actor_loss_fn(actor_params: FrozenDict[str, Any]) -> jnp.ndarray:
            observations = batch.observations.repeat(config.cost_budget_sample_count, axis=0)
            actions = batch.actions.repeat(config.cost_budget_sample_count, axis=0)

            cq1, cq2 = train_state.cost_target_critic.apply_fn(
                train_state.cost_target_critic.params, observations, actions
            )
            action_cost = jnp.maximum(cq1, cq2)
            state_cost = train_state.cost_value.apply_fn(train_state.cost_value.params, observations)
            deltabar = cls.sample_deltabar(rng, (len(observations), 1), action_cost[...,None], max_cost)

            v = train_state.value.apply_fn(train_state.value.params, observations, deltabar)
            q1, q2 = train_state.critic.apply_fn(
                train_state.critic.params, 
                observations, actions, deltabar
            )
            q = jnp.minimum(q1, q2)


            dist, dist_safe = train_state.actor.apply_fn(actor_params, 
                                                                              observations, 
                                                                              deltabar,
                                                                              training=True,
                                                                              rngs={'dropout': dropout_train_key}
                                                                              )

            log_probs = dist.log_prob(actions)
            log_probs_safe = dist_safe.log_prob(actions)

            size = config.batch_size*config.cost_budget_sample_count
            chex.assert_shape(observations, (size, None))
            chex.assert_shape(log_probs, (size,))
            chex.assert_shape(v, (size,))
            chex.assert_shape(deltabar, (size,1))

            # q = (q1+q2)/2
            chex.assert_shape(q1, (size,))
            chex.assert_shape(q2, (size,))
            chex.assert_shape(v, (size,))


            # safety advantage
            if config.in_sample_mode == 'sparse':
                weight_a = jnp.maximum(q - v, 0)
                weight_s = jnp.maximum(state_cost-action_cost, 0)
            else:
                if config.use_advantage:
                    weight_a = jnp.exp((q - v) * config.beta)
                    weight_s = jnp.exp((state_cost-action_cost) * config.beta)
                else:
                    weight_a = jnp.exp(q*config.beta)
                    weight_s = jnp.exp(-action_cost*config.beta)

            weight_a = jnp.minimum(weight_a, 100.0)
            weight_s = jnp.minimum(weight_s, 100.0)
            loss_values = -(weight_a * log_probs)
            loss_values_safe = -(weight_s * log_probs_safe)
            chex.assert_shape(loss_values, (size,))
            actor_loss = loss_values.mean() + loss_values_safe.mean()
            return actor_loss, {"actor_loss": actor_loss.mean(), "mean_action": actions.mean(), 
                                "mean_advantage": (q-v).mean(), 
                                "mean_safety_advantage": (state_cost-action_cost).mean(),
                                "mean_q": (q1+q2).mean(),
                                "mean_v": v.mean(),
                                "mean_cost_q": action_cost.mean(),
                                "mean_cost_v": state_cost.mean(),
                                }

        new_actor, actor_loss, metrics = update_by_loss_grad(train_state.actor, actor_loss_fn)
        return train_state._replace(actor=new_actor), metrics


        
    @classmethod
    def update_n_times_all(
        cls,
        train_state: IQLTrainState,
        dataset: Transition,
        rng: jnp.ndarray,
        config: IQLConfig,
        max_cost: float
    ) -> Tuple["IQLTrainState", Dict]:
        """
        Perform n_jitted_updates of IQL updates using jax.lax.scan instead of a Python loop.
        Returns the final train state and the last iteration losses.
        """
        def _step(carry, _):
            state, key = carry
            # split rng
            key, sample_key, subk_v, subk_c, subk_p = jax.random.split(key, 5)
            # sample a batch
            batch = sample_batch(dataset, sample_key, config.batch_size)

            # cost critic update
            state, m1 = cls.update_cost_value(state, batch, config)
            state, m2 = cls.update_cost_critic(state, batch, config)
            new_cost_target = target_update(
                state.cost_critic, state.cost_target_critic, config.tau
            )
            state = state._replace(cost_target_critic=new_cost_target)

            # reward critic update
            state, m3 = cls.update_value(state, batch, config, subk_v, max_cost)
            state, m4 = cls.update_critic(state, batch, config, subk_c, max_cost)
            new_target = target_update(
                state.critic, state.target_critic, config.tau
            )
            state = state._replace(target_critic=new_target)

            # policy update
            state, m5 = cls.update_actor(state, batch, config, subk_p, max_cost)


            return (state, key), {**m1, **m2, **m3, **m4, **m5}

        # run scan over n updates
        (final_state, _), metrics = lax.scan(
            _step,
            (train_state, rng),
            None,
            length=config.n_jitted_updates
        )

        return final_state,  {k:jnp.mean(v) for k, v in metrics.items()}

    @classmethod
    def update_n_times_all_for(
        cls,
        train_state: IQLTrainState,
        dataset: Transition,
        rng: jnp.ndarray,
        config: IQLConfig,
        max_cost: float
    ) -> Tuple["IQLTrainState", Dict]:
        cost_value_loss = 0.0
        cost_critic_loss = 0.0
        value_loss = 0.0
        critic_loss = 0.0
        actor_loss = 0.0
        for _ in range(config.n_jitted_updates):
            rng, sample_key, subk_v, subk_c, subk_p= jax.random.split(rng, 5)
            batch = sample_batch(dataset, sample_key, config.batch_size)

            # Update cost critic
            train_state, metrics = cls.update_cost_value(train_state, batch, config)
            train_state, cost_critic_loss = cls.update_cost_critic(train_state, batch, config)
            new_cost_target_critic = target_update(
                train_state.cost_critic, train_state.cost_target_critic, config.tau
            )
            train_state = train_state._replace(cost_target_critic=new_cost_target_critic)

            # Update reward critic
            train_state, value_loss = cls.update_value(train_state, batch, config, subk_v, max_cost)
            train_state, critic_loss = cls.update_critic(train_state, batch, config, subk_c, max_cost)
            new_target_critic = target_update(
                train_state.critic, train_state.target_critic, config.tau
            )
            train_state = train_state._replace(target_critic=new_target_critic)

            # Update policy
            train_state, actor_loss = cls.update_actor(train_state, batch, config, subk_p, max_cost)

        return train_state, {
            **metrics,
            "cost_value_loss": cost_value_loss,
            "cost_critic_loss": cost_critic_loss,
            "value_loss": value_loss,
            "critic_loss": critic_loss,
            "actor_loss": actor_loss,
        }


    @classmethod
    def get_action(
        cls,
        train_state: IQLTrainState,
        observations: np.ndarray,
        seed: jnp.ndarray,
        cost_limits: jnp.ndarray,
        temperature: float = 1.0,
        max_action: float = 1.0,  # In D4RL, the action space is [-1, 1]
        cost_normalizing_factor: float = 1.0,
        config:IQLConfig  = None,
        obs_mean: Union[np.ndarray, None] = None,
        obs_std: Union[np.ndarray, None] = None,
        remaining_steps: float = 1000,
    ) -> jnp.ndarray:
        cost_limits = jnp.clip(cost_limits, min=0)/cost_normalizing_factor

        # remaining_steps = jnp.clip(remaining_steps, min=config.finite_horizon_min)
        # cost_limits = cost_limits / ((1-config.cost_discount) * remaining_steps)

        cost_limits = jnp.clip(cost_limits, min=1, max=config.max_cost_to_learn)
        observations = (observations - obs_mean) / obs_std
        state_cost = train_state.cost_value.apply_fn(train_state.cost_value.params, observations)
        # cost_margin = config.cost_margin / cost_normalizing_factor

        deltabar = jnp.clip(cost_limits, min=0)

        dist, dist_safe = train_state.actor.apply_fn(
            train_state.actor.params, observations, deltabar, temperature=temperature
        )

        r_actions = dist.sample(seed=seed)
        safe_actions = dist_safe.sample(seed=seed)
        cost_lb = state_cost
        actions = jnp.where(cost_limits > cost_lb, r_actions, safe_actions)
        actions = jnp.clip(actions, -max_action, max_action)
        return actions

    @classmethod
    def get_policy_from_train_state(cls, train_state, config, normalizing_factor_cost: float, obs_mean: np.ndarray, obs_std: np.ndarray):
        return jax.jit(partial(
            cls.get_action,
            temperature=0.0,
            seed=jax.random.PRNGKey(0),
            train_state=train_state,
            cost_normalizing_factor=normalizing_factor_cost,
            obs_mean=obs_mean,
            obs_std=obs_std,
            config=config
        ))
    
    @classmethod
    def plot_action(cls, train_state, batch, env, epoch):
        state_cost = train_state.cost_value.apply_fn(
            train_state.cost_value.params, batch.observations
        )

        action_cost1, action_costs2 = train_state.cost_critic.apply_fn(
            train_state.cost_critic.params, batch.observations, batch.actions
        )
        
        fig, axs = plt.subplots(3, 2, figsize=(12, 10), layout='tight')
        # Plot sigmoid functions
        action_cost = jnp.maximum(action_cost1, action_costs2)
        x = np.linspace(0, 100, 200)
        obs = batch.observations[None,...].repeat(len(x), axis=0)
        acts = batch.actions[None,...].repeat(len(x), axis=0)
        deltabar_state = jnp.clip(x[...,None] - state_cost[None, :], min=0)[..., None]  # Cost needs an aditional dim
        deltabar_action = jnp.clip(x[...,None] - action_cost[None, :], min=0)[..., None]  # Cost needs an aditional dim 
        q1, q2 = train_state.critic.apply_fn(
            train_state.critic.params, obs, acts, 
            deltabar_action, # Cost needs an aditional dim
        )

        v = train_state.value.apply_fn(
            train_state.value.params, obs, deltabar_state# Cost needs an aditional dim
        )

        fig.suptitle(f'Epoch {epoch}')
        def plot_step(ax, x0, title, ls='-'):
            y = np.where(x<x0, 0, 1)
            ax.plot(x, y, label=f'{title} ={x0:.2f}', ls=ls)

        for i in range(len(batch.observations)):
            a = np.array(batch.actions[i])
            plot_step(axs[0][0], action_cost1[i], f'QC1 Action{i+1}')
            plot_step(axs[0][1], action_costs2[i], f'QC2 Action{i+1}')

            axs[1][0].plot(x, q1[:,i], label=f'Q1 Action{i+1}')
            axs[1][1].plot(x, q2[:,i],  label=f'Q2 Action{i+1}')

 
        

        for i in range(len(batch.observations)):
            plot_step(axs[0][0], state_cost[i], f'VC ({i+1})', ls='--')
            plot_step(axs[0][1], state_cost[i], f'VC ({i+1})', ls='--')
            axs[1][0].plot(x, v[:,i],  label=f'Value{i}', ls='--')
            axs[1][1].plot(x, v[:,i],  label=f'Value{i}', ls='--')

            axs[2][0].plot(x, q1[:,i]-v[:,i],  label=f'Adv{i}', ls='--')
            axs[2][1].plot(x, q2[:,i]-v[:,i],  label=f'Adv{i}', ls='--')

        for axv in axs:
            for ax in axv:
                if hasattr(env, 'compute_a_b_v'):
                    ymin, ymax = ax.get_ylim()
                    a, b, v = env.compute_a_b_v(config.cost_expectile, config.cost_discount)
                    ax.vlines(x=a, ymin=ymin, ymax=ymax, colors='red', linestyles='dotted', label=f'QC1-Analytical = {a:.2f}')
                    ax.vlines(x=b, ymin=ymin, ymax=ymax, colors='blue', linestyles='dotted', label=f'QC2-Analytical = {b:.2f}')
                    ax.vlines(x=v, ymin=ymin, ymax=ymax, colors='green', linestyles='dotted', label=f'VC-Analytical = {v:.2f}')
                ax.legend()
                ax.grid(True)
        plt.tight_layout()
        return fig




def create_iql_train_state(
    rng: jnp.ndarray,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    config: IQLConfig,
) -> IQLTrainState:
    rng, actor_rng, critic_rng, value_rng, cost_critic_rng, cost_value_rng = jax.random.split(rng, 6)
    # initialize actor
    action_dim = actions.shape[-1]
    actor_model = GaussianHindSightPolicy(
        config.hidden_dims,
        action_dim=action_dim,
        log_std_min=-20.0,
        droupout_rate=config.policy_dropout_rate,
    )
    if config.actor_lr_decay:
        schedule_fn = optax.cosine_decay_schedule(-config.actor_lr, config.max_steps)
        actor_tx = optax.chain(optax.scale_by_adam(), optax.scale_by_schedule(schedule_fn))
    else:
        actor_tx =   optax.chain(
            optax.clip_by_global_norm(config.gradient_clip),
            optax.adam(config.actor_lr))
    sample_cost_limits = jnp.zeros((1,))
    actor = TrainState.create(
        apply_fn=actor_model.apply,
        params=actor_model.init(actor_rng, observations, sample_cost_limits),
        tx=actor_tx,
    )

    cost_critic_tx, cost_value_tx, critic_tx, value_tx = [optax.chain(
        optax.clip_by_global_norm(config.gradient_clip),
        optax.adam(lr),
    ) for lr in [config.cost_critic_lr, config.cost_value_lr, config.critic_lr, config.value_lr]]

    # initialize cost-critic
    cost_critic_model = ensemblize(Critic, num_qs=2)(config.hidden_dims, ensure_positive=True)
    cost_critic = TrainState.create(
        apply_fn=cost_critic_model.apply,
        params=cost_critic_model.init(cost_critic_rng, observations, actions),
        tx=cost_critic_tx,
    )
    cost_target_critic = TrainState.create(
        apply_fn=cost_critic_model.apply,
        params=cost_critic_model.init(cost_critic_rng, observations, actions),
        tx=cost_critic_tx,
    )
    # initialize cost value
    cost_value_model = Critic(config.hidden_dims, ensure_positive=True, layer_norm=True)
    cost_value = TrainState.create(
        apply_fn=cost_value_model.apply,
        params=cost_value_model.init(cost_value_rng, observations),
        tx=cost_value_tx,
    )

    # initialize critic
    if config.use_monotonic:
        critic_model = ensemblize(MonotonicCritic, num_qs=2)(config.hidden_dims)
        value_model = MonotonicCritic(config.hidden_dims, layer_norm=True)
    else:
        critic_model = ensemblize(Critic, num_qs=2)(config.hidden_dims)
        value_model = Critic(config.hidden_dims, layer_norm=True)
    critic = TrainState.create(
        apply_fn=critic_model.apply,
        params=critic_model.init(critic_rng, observations, actions, sample_cost_limits),
        tx=critic_tx,
    )
    target_critic = TrainState.create(
        apply_fn=critic_model.apply,
        params=critic_model.init(critic_rng, observations, actions, sample_cost_limits),
        tx=critic_tx,
    )
    # initialize value
    value = TrainState.create(
        apply_fn=value_model.apply,
        params=value_model.init(value_rng, observations, sample_cost_limits),
        tx=value_tx,
    )

    return IQLTrainState(
        rng,
        cost_critic=cost_critic,
        cost_target_critic=cost_target_critic,
        cost_value=cost_value,
        critic=critic,
        target_critic=target_critic,
        value=value,
        actor=actor,
    )







def run_for_config(config: IQLConfig):
    print("Use gymnasium:", use_gymnasium)
    if isinstance(config.eval_cost_thresholds, int):
        config.eval_cost_thresholds = (config.eval_cost_thresholds,)
    if config.cost_violation_penalty is not None:
        config.cost_violation_reward = -config.cost_violation_penalty
    

    name = f"{config.algo}-{config.env_name}-{config.seed}"
    np.random.seed(config.seed)

    if config.use_wandb:
        wandb.init(config=config.model_dump(), project=config.project, name=name, tags=[config.tag])


    config_hash = dict_hash_ignore_keys(config.dict(), ['env_name', 'seed', 'eval_cost_thresholds'])
    log_dir = config.log_dir_base + f"/{config_hash}/{config.algo}/{config.env_name}/seed_{config.seed}/"
    os.makedirs(log_dir, exist_ok=True)
    # Save to JSON file
    with open(f"{log_dir}/config.json", "w") as f:
        f.write(config.model_dump_json(indent=4))


    rng = jax.random.PRNGKey(config.seed)
    env = gym.make(config.env_name)
    if config.finite_horizon_min is None:
        config.finite_horizon_min = 1 #env.spec.max_episode_steps*0.1
    print("Finite horizon min", config.finite_horizon_min)

    if use_gymnasium:
        env.reset(seed=config.seed)
    else:
        env.seed(config.seed)
        env.reset()
    print("Max action", env.action_space.high[0])

    dataset = get_dataset(env, config.data_size)
    print("Dataset loaded")
    for k, v in dataset._asdict().items():
        print(k, v.shape, f"[{jnp.min(v)}], [{jnp.max(v)}]")
    
    dataset, normalizing_factor_reward, normalizing_factor_cost, obs_mean, obs_std = normalize_dataset(dataset,config.normalize_cost_and_reward, config.normalize_state)
    # Shuffle
    # rng, rng_permute= jax.random.split(rng, 2)
    # perm = jax.random.permutation(rng_permute, len(dataset.observations))
    # dataset = jax.tree_util.tree_map(lambda x: x[perm], dataset)

    if config.cost_expectile is None:
        length = len(dataset.observations)
        coverage = length/2e6
        new_cost_expectile = np.clip(0.5-coverage*0.3, a_max=0.4, a_min=0.2)
        print("Updated expectile", new_cost_expectile, config.cost_expectile)
        config.cost_expectile = new_cost_expectile
    
    if config.max_cost_to_learn is None:
        max_cost = dataset.costs.max()/(1-config.cost_discount)
    else:
        max_cost = config.max_cost_to_learn
        max_cost = max_cost/normalizing_factor_cost
    print("Max cost: ", max_cost)


    # normalize max cost threshold also
    config.max_cost_to_learn = max_cost
    # create train_state
    rng, subkey = jax.random.split(rng)
    example_batch: Transition = jax.tree_util.tree_map(lambda x: x[0], dataset)
    train_state = create_iql_train_state(
        subkey,
        example_batch.observations,
        example_batch.actions,
        config,
    )
    train_state: IQLTrainState = train_state
    print("Train State Created")

    algo = IQL()
    update_func = jax.jit(partial(
        algo.update_n_times_all,
        config=config,
        max_cost=max_cost,
        dataset=dataset,
    ))

    get_policy_from_train_state = partial(algo.get_policy_from_train_state, 
                                          config=config, 
                                          normalizing_factor_cost=normalizing_factor_cost, 
                                          obs_mean=obs_mean, obs_std=obs_std)
    def evaluate_func(train_state, epoch):
        policy_fn = get_policy_from_train_state(train_state)
        small_dataset = jax.tree_util.tree_map(lambda x: x[:2], dataset)  # Use a small dataset for evaluation
        # fig = algo.plot_action(train_state, small_dataset, env, epoch)
        # plt.savefig(f"{log_dir}/action_plot_{(epoch//config.eval_interval):07d}.png")
        # plt.close(fig)
        print("cost_violation_reward", config.cost_violation_reward)
        rescalse_remaining_budget_by = config.cost_discount if config.rescale_remaining_budget_on_eval else 1
        return evaluate_for_multiple_threshodls(
            policy_fn,
            env,
            num_episodes=config.final_eval_episodes,
            cost_thresholds=config.eval_cost_thresholds,
            min_cost_threhold_ratio=config.min_cost_threhold_ratio,
            rescalse_remaining_budget_by=rescalse_remaining_budget_by
        )
    save_func = partial(save_model, config=config, cost_normalizing_factor=normalizing_factor_cost, obs_mean=obs_mean, obs_std=obs_std) if config.save_policy else (lambda x,y: None)

    train_info, train_state = train_parallel(
        rng,
        train_state,
        update_func,
        evaluate_func,
        save_func,
        save_folder=log_dir,
        epochs=config.max_steps,
        n_jitted_updates=config.n_jitted_updates,
        log_interval=config.log_interval,
        eval_interval=config.eval_interval,
        use_wandb=config.use_wandb,
    )
    if config.use_wandb:
        # wandb.log
        wandb.log({**train_info,
            'normalizing_factor_reward': normalizing_factor_reward,
            'normalizing_factor_cost': normalizing_factor_cost,
            'new_cost_expectile': config.cost_expectile,
            'len_dataset': len(dataset.observations),
        }, step=train_info['total_steps'])
        wandb.finish()
    print("Done Training, Total Steps: ", train_info)


if __name__ == "__main__":
    conf_dict = OmegaConf.from_cli()

    (taskname, group) = get_task_name_from_env_name(conf_dict['env_name'])
    group_overrides = {
        "MetaDrive": {
            "eval_cost_thresholds": (10, 20, 40),
            "expectile": 0.6,
            "cost_expectile": 0.4,
            "beta": 8.0,
            "max_steps": 200_000
        },
    }
    if group in group_overrides:
        conf_dict.update(group_overrides[group])

    config = IQLConfig(**conf_dict)
    run_for_config(config)
