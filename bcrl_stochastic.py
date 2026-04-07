# source https://github.com/ikostrikov/implicit_q_learning
# https://arxiv.org/abs/2110.06169
import os
from functools import partial
from typing import Any, NamedTuple, Tuple, Union, Dict, Sequence, Callable, Optional, List
from flax.core.frozen_dict import FrozenDict
import matplotlib.pyplot as plt
from collections import defaultdict
import flax.linen as nn
import chex
from envs.maritime.env import MaritimeEnv
import pandas as pd

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
import tqdm
from flax.training.train_state import TrainState
from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict
import dsrl as _dsrl
from utils import (
    ensemblize, get_initial_states, save_model, Transition,
    get_dataset, expectile_loss, set_common_jax_flags, percentile_loss,
    normalize_dataset, sample_batch, target_update, update_by_loss_grad, train_parallel,
    gumbel_rescale_loss, dict_hash_ignore_keys, get_task_name_from_env_name
)
from networks import Critic, MonotonicCritic, GaussianPolicy, MLP, default_init

class CostCritic(nn.Module):
    """Don't use extra critic for values instead use the same with rest=[]"""
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    layer_norm: bool = False
    max_cost: float = 1.0  # Maximum cost to learn

    @nn.compact
    def __call__(self, observations: jnp.ndarray, *rest: jnp.ndarray) -> jnp.ndarray:
        inputs0: Sequence[jnp.ndarray] = [observations]+list(rest)
        inputs = jnp.concatenate(inputs0, -1)
        critic = MLP((*self.hidden_dims, 1), activations=self.activations, layer_norm=self.layer_norm)(inputs)
        critic = nn.sigmoid(critic)*self.max_cost  # Use sigmoid to ensure critic is bounded by [0, max_cost]
        return jnp.squeeze(critic, -1)



set_common_jax_flags()
# jax.config.update("jax_disable_jit", True)

class IQLConfig(BaseModel):
    # GENERAL
    env_name: str  = "OfflinePointButton1Gymnasium-v0" # SafetyGym, The first raw in DSRL resutls
    tag: str = "empty"
    algo: str = "bcrl-stoch"
    project: str = "bcrl"
    use_wandb: bool = True
    wandb_run: str = "" # Don't input it as param. Just used for logging.
    log_dir_base: str = "tmp/bcrl-logs"
    seed: int = 42
    final_eval_episodes: int = 20
    log_interval: int = 1000
    eval_interval: int = 100_000
    batch_size: int = 512
    max_steps: int = int(1e5)
    n_jitted_updates: int = 10
    # DATASET
    data_size: int = int(1e10)
    # NETWORK
    hidden_dims: Tuple[int, int] = (512,512)
    actor_lr: float = 3e-4
    value_lr: float = 3e-4
    critic_lr: float = 3e-4
    cost_value_lr: float = 3e-4
    cost_critic_lr: float = 3e-4
    gradient_clip: float = 10.0
    # IQL SPECIFIC
    expectile: float = 0.8
    beta: float = 3.0
    tau: float = 0.005
    discount: float = 0.99

    # Safe IQL specific
    cost_expectile: Union[None, float] = None
    cost_expectile_transition: float = 0.5 # Transition probabilities we should take max instead of min or mean 0.5 for mean
    cost_discount: float = 0.99
    cost_discount_eval: float = 1.0
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
    policy_dropout_rate: float = 0.1
    use_monotonic: bool = False
    min_cost_threhold_ratio: float = 0.0
    in_sample_mode: str = 'expectile' #  #'extreme', 'quantile'
    in_sample_mode_cost: str = 'expectile' #  #'extreme', 'quantile'
    extreme_beta_cost: float = 0.1  # Beta for extreme loss
    extreme_beta_reward: float = 0.1  # Beta for extreme loss
    sparse_alpha_reward: float = 1
    sparse_alpha_cost: float = 1
    use_advantage: bool = True
    budget_exponential_scale: float = 10.0
    finite_horizon_min: Union[float, None] = None
    save_policy: bool = True
    tanh_squash_distribution: bool = False
    bottom_weight: float = 0.0
    discount_budget: bool = False
    eval_temperature: float = 0.0
    min_reward: float = -1.

    model_config = ConfigDict(extra='forbid')

    def __hash__(
        self,
    ):  # make config hashable to be specified as static_argnums in jax.jit.
        return hash(self.__repr__())



def evaluate(
    policy_fn, env: gym.Env, num_episodes: int, cost_threshold: float, init_cost: float, discount_budget:bool, eval_for_hard_const:bool=False
) -> dict:
    # Finite-horizon budget tracking (cf. CAPS https://github.com/yassineCh/CAPS):
    # at each step the remaining budget is rescaled by both the remaining steps and
    # a geometric factor, so the per-step delta_hat accounts for how many steps are
    # left to spend the remaining cost allowance.
    episode_returns = []
    episode_costs = []
    episode_lengths = []
    env.reset()
    print("Setting cost threshold", cost_threshold)
    env.set_target_cost(cost_threshold)
    episode_max_length:int = env.spec.max_episode_steps if hasattr(env.spec, 'max_episode_steps') else 1001
    print(episode_max_length, "is the max episode length")

    gamma = 0.99
    # modified_threshold = cost_threshold/(normalizing_factor_cost*(1-0.99)*episode_max_length)
    print("before after threhold", cost_threshold)
    episode_infos = defaultdict(list) 
    if eval_for_hard_const:
        cost_threshold = 0 # after setting the target_cost in env

    for _ in tqdm.tqdm(range(num_episodes), desc=f"Evaluation C={cost_threshold}"):
        episode_return = 0
        episode_cost = 0
        episode_length = 0
        (observation, info), done = env.reset(), False
        truncated = False
        while (not done) and (not truncated):
            obs_jnp = jnp.array([observation])
            steps_remain = episode_max_length-episode_length
            budget_remain = jnp.clip(cost_threshold-episode_cost, min=0)
            delta_hat = budget_remain  * (1-gamma**steps_remain)/(1-gamma) / steps_remain
            delta_hat_jnp = jnp.array([[delta_hat]])
            action, delta_hat_jnp = policy_fn(observations=obs_jnp, delta_hat=delta_hat_jnp) #pisode_max_length-episode_length)
            action = np.array(action)
            action = action.squeeze()
            observation, reward, done, truncated, info = env.step(action)
            episode_cost += info["cost"]
            episode_return += reward
            episode_length += 1
        episode_returns.append(episode_return)
        episode_costs.append(episode_cost)
        episode_lengths.append(episode_length)
        for k, v in info.items():
            episode_infos[k].append(v)
            
        print("\nNew episode", episode_cost, episode_return, episode_length)
    print(episode_costs, "cost in eval")
    normalize_r, normalized_c = zip(*[env.get_normalized_score(r, c) for r, c in zip(episode_returns, episode_costs)])
    print(f"Evaluation C={cost_threshold} | cost:{np.mean(normalized_c)}, reward: {np.mean(normalize_r)}, lengths: {np.mean(episode_lengths)}")
    print("Info irmes", len(episode_infos), [len(v) for v in episode_infos.values()], num_episodes)
    episode_infos = {k:v for k, v in episode_infos.items() if len(v)==num_episodes}
    print("Keep keys", episode_infos.keys())
    

    # print("Ep mean", episode_infos_mean)

    info = {
        # **episode_infos_mean,
        **{f"episode_info/{k}": v for k,v in episode_infos.items()},
        "episode_rewards": episode_returns,
        "episode_costs": episode_costs,
        "episode_lengths": episode_lengths,
        "normalized_rewards": normalize_r,
        "normalized_costs": normalized_c,
        "mean_episode_length": np.mean(episode_lengths),
    }
    return info

def evaluate_for_multiple_threshodls(
    policy_fn, env: gym.Env, num_episodes: int, cost_thresholds: List[float], init_cost: float, discount_budget: float, eval_for_hard_const:bool=False
) -> pd.DataFrame:
    data = {cost_theshold: pd.DataFrame(evaluate(policy_fn, env, num_episodes, cost_theshold, init_cost, discount_budget, eval_for_hard_const))
        for cost_theshold in cost_thresholds}
    return pd.concat(data, axis=1)


def stochastic_evaluate(
    policy_fn, cost_value_fn, env: gym.Env, num_episodes: int, cost_threshold: float,
    mean_init_cost: float, eval_for_hard_const: bool = False
) -> dict:
    """Evaluate using paper's budget initialization (Eq. 10):
        δ_0 = V_C*(s_0) + δ_init - E_{s~μ_0}[V_C*(s)]
    Budget updates are handled inside policy_fn (get_action_and_new_delta_hat, Eq. 11).

    Args:
        cost_value_fn: maps unnormalized observations -> V_C*(s), shape (batch,)
        mean_init_cost: E_{s~μ_0}[V_C*(s)], precomputed over dataset initial states
    """
    episode_returns = []
    episode_costs = []
    episode_lengths = []
    env.reset()
    print("Setting cost threshold", cost_threshold)
    env.set_target_cost(cost_threshold)
    episode_max_length: int = env.spec.max_episode_steps if hasattr(env.spec, 'max_episode_steps') else 1001
    print(episode_max_length, "is the max episode length")
    print("before after threshold", cost_threshold)
    episode_infos = defaultdict(list)

    budget_threshold = 0.0 if eval_for_hard_const else cost_threshold

    for _ in tqdm.tqdm(range(num_episodes), desc=f"Stochastic Eval C={cost_threshold}"):
        episode_return = 0
        episode_cost = 0
        episode_length = 0
        (observation, info), done = env.reset(), False
        truncated = False

        # Initialize budget per paper Eq. 10: δ_0 = V_C*(s_0) + δ_init - E_{s~μ_0}[V_C*(s)]
        obs_jnp = jnp.array([observation])
        vc_s0 = cost_value_fn(obs_jnp)  # shape (1,)
        delta_0 = float(vc_s0[0]) + budget_threshold - mean_init_cost
        delta_hat_jnp = jnp.array([[max(delta_0, 0.0)]])

        while (not done) and (not truncated):
            obs_jnp = jnp.array([observation])
            action, delta_hat_jnp = policy_fn(observations=obs_jnp, delta_hat=delta_hat_jnp)
            action = np.array(action).squeeze()
            observation, reward, done, truncated, info = env.step(action)
            episode_cost += info["cost"]
            episode_return += reward
            episode_length += 1

        episode_returns.append(episode_return)
        episode_costs.append(episode_cost)
        episode_lengths.append(episode_length)
        for k, v in info.items():
            episode_infos[k].append(v)
        print("\nNew episode", episode_cost, episode_return, episode_length)

    print(episode_costs, "cost in stochastic eval")
    normalize_r, normalized_c = zip(*[env.get_normalized_score(r, c) for r, c in zip(episode_returns, episode_costs)])
    print(f"Stochastic Eval C={cost_threshold} | cost:{np.mean(normalized_c)}, reward: {np.mean(normalize_r)}, lengths: {np.mean(episode_lengths)}")
    print("Info items", len(episode_infos), [len(v) for v in episode_infos.values()], num_episodes)
    episode_infos = {k: v for k, v in episode_infos.items() if len(v) == num_episodes}
    print("Keep keys", episode_infos.keys())

    return {
        **{f"episode_info/{k}": v for k, v in episode_infos.items()},
        "episode_rewards": episode_returns,
        "episode_costs": episode_costs,
        "episode_lengths": episode_lengths,
        "normalized_rewards": normalize_r,
        "normalized_costs": normalized_c,
        "mean_episode_length": np.mean(episode_lengths),
    }


def stochastic_evaluate_for_multiple_thresholds(
    policy_fn, cost_value_fn, env: gym.Env, num_episodes: int, cost_thresholds: List[float],
    mean_init_cost: float, eval_for_hard_const: bool = False
) -> pd.DataFrame:
    data = {t: pd.DataFrame(stochastic_evaluate(policy_fn, cost_value_fn, env, num_episodes, t, mean_init_cost, eval_for_hard_const))
            for t in cost_thresholds}
    return pd.concat(data, axis=1)

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
    actor_safe: TrainState

def sparse_value_loss(v, q, alpha):
    sp_term = (q - v) / (2 * alpha) + 1.0
    sp_weight = jnp.where(sp_term > 0, 1., 0.)
    value_loss = (sp_weight * (sp_term**2) + v / alpha)
    return value_loss

class IQL(object):

    # @classmethod
    # def get_combined_value(cls, budget, estimated_cost, rest_value, min_value):
    #     # budget - Total remainig budget
    #     # qc - expected cost
    #     # extra budget = budget - qc 
    #     # chex.assert_trees_all_close(rest_value>=0, jnp.ones_like(rest_value, jnp.bool))
    #     # size = len(budget)
    #     # chex.assert_shape(budget, (size,))
    #     # chex.assert_shape(estimated_cost, (size,))
    #     # chex.assert_shape(rest_value, (size,))
    #     resp = jnp.where(budget > estimated_cost, min_value + rest_value, min_value)
    #     # chex.assert_shape(resp, (size,))
    #     return resp


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
            # jax.debug.print("q1: {a}, q2: {b}, target_q: {c}", a=q1[:2], b=q2[:2], c=target_q[:2])
            # q1_loss = optax.huber_loss(q1, target_q).mean()
            # q2_loss = optax.huber_loss(q2, target_q).mean()
            # q1_loss = expectile_loss(target_q-q1, config.cost_expectile_transition).mean()
            # q2_loss = expectile_loss(target_q-q2, config.cost_expectile_transition).mean()

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
            if config.in_sample_mode_cost == 'expectile':
                value_loss = expectile_loss(q - v, config.cost_expectile).mean()
            elif config.in_sample_mode_cost == 'extreme':
                value_loss = gumbel_rescale_loss(v-q, config.extreme_beta_cost).mean()
            elif config.in_sample_mode_cost == 'sparse':
                value_loss = sparse_value_loss(-v, -q, config.sparse_alpha_cost).mean()
            elif config.in_sample_mode_cost == 'percentile':
                value_loss = percentile_loss(q - v, config.cost_expectile).mean()
            else:
                raise ValueError(f"Unknown in_sample_mode_cost: {config.in_sample_mode_cost}")
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
            chex.assert_shape(next_state_cost, (len(observations), ))

            uniform = jax.random.uniform(rng, (len(observations), ), minval=0, maxval=1)
            delta_v = uniform * max_cost # Learn 0 - max_cost <- overflow is delta.
            delta = delta_v * config.cost_discount

            delta, delta_v = delta[..., None], delta_v[..., None]

            # target_delta = next_state_cost[...,None] + (delta - action_cost[...,None])/config.cost_discount
            # weight = (max_cost - delta[:,0]) / max_cost
            chex.assert_shape(delta_v, (len(observations), 1))
            chex.assert_shape(delta, (len(observations), 1))

            next_v = train_state.value.apply_fn(
                train_state.value.params, next_observations, delta_v
            )
            target_q = rewards + config.discount * (1 - dones) * next_v
            q1, q2 = train_state.critic.apply_fn(
                critic_params, observations, actions, delta
            )
            critic_loss =  ((target_q - q1)**2 +  (target_q - q2)**2)/2

            chex.assert_shape(critic_loss, (len(observations),))
            chex.assert_shape(delta, (len(observations), 1))
            chex.assert_shape(action_cost, (len(observations), ))
            chex.assert_shape(costs, (len(observations), 1))
            chex.assert_shape(rewards, (len(observations),))

            critic_loss = critic_loss.mean() 

            # Minimize at zero budget
            q1z, q2z = train_state.critic.apply_fn(
                critic_params, observations, actions, jnp.full_like(delta, -0.0)  # Something close to zero but from negative side
            )
            target_q_zero = (-1000) - action_cost # config.min_reward/(1-config.discount)
            critic_loss_zero =  ((target_q_zero - q1z)**2 +  (target_q_zero - q2z)**2)/2
            critic_loss = critic_loss + critic_loss_zero.mean()*config.bottom_weight

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
            state_cost = train_state.cost_value.apply_fn(train_state.cost_value.params, observations)
            uniform = jax.random.uniform(rng, (len(observations), ), minval=0, maxval=1)
            delta_abs = uniform * max_cost  + jnp.maximum(action_cost, state_cost) # In absolute terms. we need telative value to update nets.
            delta_q = delta_abs - action_cost
            delta_v = delta_abs - state_cost
            

            delta_q, delta_v = delta_q[..., None], delta_v[..., None]
            chex.assert_shape(delta_v, (len(observations), 1))
            chex.assert_shape(delta_q, (len(observations), 1))

            q1, q2 = train_state.target_critic.apply_fn(
                train_state.target_critic.params, observations, actions, delta_q
            )
            q = jax.lax.stop_gradient(jnp.minimum(q1, q2))
            v = train_state.value.apply_fn(value_params, observations, delta_v)


            if config.in_sample_mode == 'expectile':
                loss_values = expectile_loss(q - v, config.expectile)
            elif config.in_sample_mode == 'percentile':
                loss_values = percentile_loss(q - v, config.expectile).mean()
            elif config.in_sample_mode == 'extreme':
                loss_values = gumbel_rescale_loss(q-v, config.extreme_beta_reward)
            elif config.in_sample_mode == 'sparse':
                loss_values = sparse_value_loss(v, q, config.sparse_alpha_reward).mean()
            else:
                raise ValueError(f"Unknown in_sample_mode: {config.in_sample_mode}")

            # bottom_delta_ub = action_cost - state_cost - 1.0 # 1 is to have some gap
            # # bottom_delta_ub = jnp.clip(bottom_delta_ub, min=-1) 
            # bottom_delta  =  bottom_delta_ub - max_cost * uniform
            # bottom_delta = bottom_delta[..., None]
            # chex.assert_shape(bottom_delta, (len(observations), 1))

            # v_bottom = train_state.value.apply_fn(value_params, observations, bottom_delta)
            # bottom_target = config.min_reward/(1-config.discount)  - action_cost
            # chex.assert_shape(bottom_target, (len(observations),))
            # bottom_loss = expectile_loss(v_bottom - bottom_target, config.expectile).mean()

            value_loss = loss_values.mean() # + bottom_loss * config.bottom_weight
            return value_loss, {"value_loss": value_loss}

        new_value, value_loss, metrics = update_by_loss_grad(train_state.value, value_loss_fn)
        return train_state._replace(value=new_value), metrics

    @classmethod
    def update_actor(
        cls, train_state: IQLTrainState, batch: Transition, config: IQLConfig,
        rng: jnp.ndarray, max_cost: float
    ) -> Tuple["IQLTrainState", jnp.ndarray]:
        dropout_train_key, rng = jax.random.split(rng)
        observations = batch.observations.repeat(config.cost_budget_sample_count, axis=0)
        actions = batch.actions.repeat(config.cost_budget_sample_count, axis=0)

        cq1, cq2 = train_state.cost_target_critic.apply_fn(
            train_state.cost_target_critic.params, observations, actions
        )
        action_cost = jnp.maximum(cq1, cq2)
        state_cost = train_state.cost_value.apply_fn(train_state.cost_value.params, observations)

        uniform = jax.random.uniform(rng, (len(observations), ), minval=0, maxval=1)
        delta_abs = uniform * max_cost  + jnp.maximum(action_cost, state_cost) # In absolute terms. we need telative value to update nets.
        delta_q = delta_abs - action_cost
        delta_v = delta_abs - state_cost

        delta_v = jnp.clip(delta_v, min=0, max=max_cost)
        # jax.debug.print("delta_v: {a}\t {b}", a=delta_v[:2], b=action_cost[:2]-state_cost[:2])
        delta_q, delta_v = delta_q[..., None], delta_v[..., None]
        chex.assert_shape(delta_v, (len(observations), 1))
        chex.assert_shape(delta_q, (len(observations), 1))
        cost_advantage =  state_cost - action_cost

        v = train_state.value.apply_fn(train_state.value.params, observations, delta_v)
        q1, q2 = train_state.critic.apply_fn(
            train_state.critic.params, 
            observations, actions, delta_q
        )
        q = jnp.minimum(q1, q2)
        size = config.batch_size*config.cost_budget_sample_count
        chex.assert_shape(observations, (size, None))
        chex.assert_shape(v, (size,))

        # q = (q1+q2)/2
        chex.assert_shape(q1, (size,))
        chex.assert_shape(q2, (size,))
        chex.assert_shape(v, (size,))
        # jax.debug.print("advangate: {a}: {a_min}: {a_std}", a=(q-v)[:4], a_min=(q-v).min(), a_std=(q-v).std())

        if config.in_sample_mode == 'sparse':
            weight_a = jnp.maximum(q - v, 0)
        else:
            if config.use_advantage:
                weight_a = jnp.exp((q - v) * config.beta)
            else:
                weight_a = jnp.exp(q*config.beta)

        weight_a = jnp.minimum(weight_a, 100.0)

        weight_c = jnp.exp(cost_advantage * config.beta)
        weight_c = jnp.minimum(weight_c, 100.0)
        chex.assert_shape(weight_c, (size,))
        chex.assert_shape(weight_a, (size,))

        def actor_loss_fn(actor_params: FrozenDict[str, Any]):
            dist = train_state.actor.apply_fn(actor_params, 
                                              observations, 
                                              delta_v, # Should be train wrt to the state budget.
                                              training=True,
                                              rngs={'dropout': dropout_train_key}
                                              )

            log_probs = dist.log_prob(actions)
            chex.assert_shape(log_probs, (size,))
            loss_values = -(weight_a * log_probs)
            chex.assert_shape(loss_values, (size,))
            actor_loss = loss_values.mean()

            return actor_loss, {"actor_loss": actor_loss.mean(), 
                                "mean_action": actions.mean(), 
                                "mean_advantage": (q-v).mean(), 
                                "mean_cost_advantage": cost_advantage.mean(),
                                "mean_cost_v": state_cost.mean(),
                                "mean_q": (q1+q2).mean()/2,
                                "max_q": (q1+q2).max()/2,
                                "mean_v": v.mean(),
                                "mean_cost_q": action_cost.mean(),
                                "max_cost_q": action_cost.max(),
                                "max_cost_v": state_cost.max(),
                                }

        def actor_safe_loss_fn(actor_params: FrozenDict[str, Any]):
            dist = train_state.actor_safe.apply_fn(actor_params, 
                                              observations, 
                                              training=True,
                                              rngs={'dropout': dropout_train_key}
                                              )
            log_probs = dist.log_prob(actions)
            chex.assert_shape(log_probs, (size,))
            loss_values = -(weight_c*log_probs)
            chex.assert_shape(loss_values, (size,))
            actor_loss = loss_values.mean()
            return actor_loss, {"actor_loss_safe": actor_loss.mean()}

        new_actor, actor_loss, metrics = update_by_loss_grad(train_state.actor, actor_loss_fn)
        train_state =  train_state._replace(actor=new_actor)
        
        new_actor_safe, actor_loss_safe, metrics_safe = update_by_loss_grad(train_state.actor_safe, actor_safe_loss_fn)
        train_state =  train_state._replace(actor_safe=new_actor_safe)
        return train_state, {**metrics, **metrics_safe}


        
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

            # pack losses
            # losses = (metrics,
            #           cost_critic_loss,
            #           value_loss,
            #           critic_loss,
            #           actor_loss)


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
    def get_action_and_new_delta_hat(
        cls,
        train_state: IQLTrainState,
        observations: np.ndarray,
        seed: jnp.ndarray,
        delta_hat: jnp.ndarray, # The overflow budget. This plus state_cost is the actual budget
        temperature: float = 1.0,
        max_action: float = 1.0,  # In D4RL, the action space is [-1, 1]
        cost_normalizing_factor: float = 1.0,
        config:IQLConfig  = None,
        obs_mean: Union[np.ndarray, None] = None,
        obs_std: Union[np.ndarray, None] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Input is in the """
        delta_hat = jnp.clip(delta_hat, min=0, max=config.max_cost_to_learn)
        observations = (observations - obs_mean) / obs_std
        chex.assert_shape(observations, (1, None))
        chex.assert_shape(delta_hat, (1, 1))

        state_cost = train_state.cost_value.apply_fn(train_state.cost_value.params, observations)
        chex.assert_shape(state_cost, (1, ))
        delta_hat = jnp.clip(delta_hat - state_cost[:,None], min=0)

        # Sample action with new delta
        dist = train_state.actor.apply_fn(
            train_state.actor.params, observations, delta_hat, temperature=temperature
        )
        dist_c = train_state.actor_safe.apply_fn(
            train_state.actor_safe.params, observations, temperature=temperature
        )
        actions = dist.sample(seed=seed)
        actions_c = dist_c.sample(seed=seed)
        actions = jnp.where(delta_hat < 0.1, actions_c, actions)
        # actions = actions_c

        actions = jnp.clip(actions, -max_action, max_action)
        ac1, ac2 = train_state.cost_target_critic.apply_fn(train_state.cost_target_critic.params, observations, actions)
        action_cost = jnp.maximum(ac1, ac2)
        # jax.debug.print("State cost: {a}, Action cost: {b}, Delta_hat: {c}", a=state_cost, b=action_cost, c=delta_hat)
        action_cost = jnp.maximum(action_cost, state_cost)
        # jax.debug.print("Action cost new {a}", a=action_cost)
        # new_delta_hat = state_cost + delta_hat +  - action_cost
        new_delta_hat =delta_hat + (state_cost - action_cost)
        # jax.debug.print("Updated delta har new {a}", a=new_delta_hat)
        new_delta_hat = jnp.clip(new_delta_hat, min=0)/config.cost_discount_eval
        # jax.debug.print("Updated delta har new2 {a}", a=new_delta_hat)
        return actions, new_delta_hat

    @classmethod
    def get_policy_from_train_state(cls, train_state, config, normalizing_factor_cost: float, obs_mean: np.ndarray, obs_std: np.ndarray):
        return jax.jit(partial(
            cls.get_action_and_new_delta_hat,
            temperature=config.eval_temperature,
            seed=jax.random.PRNGKey(0),
            train_state=train_state,
            cost_normalizing_factor=normalizing_factor_cost,
            obs_mean=obs_mean,
            obs_std=obs_std,
            config=config
        ))
    
    @classmethod
    def get_cost_value_fn_from_train_state(cls, train_state, obs_mean: np.ndarray, obs_std: np.ndarray):
        """Returns a jitted function mapping unnormalized observations -> V_C*(s).
        Used to initialize the budget in stochastic_evaluate via Eq. 10."""
        def _cost_value_fn(observations: jnp.ndarray) -> jnp.ndarray:
            obs_normalized = (observations - obs_mean) / obs_std
            return train_state.cost_value.apply_fn(train_state.cost_value.params, obs_normalized)
        return jax.jit(_cost_value_fn)

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
    actor_model = GaussianPolicy(
        config.hidden_dims,
        action_dim=action_dim,
        log_std_min=-20.0,
        droupout_rate=config.policy_dropout_rate,
        tanh_squash_distribution=config.tanh_squash_distribution,
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
    actor_safe = TrainState.create(
        apply_fn=actor_model.apply,
        params=actor_model.init(actor_rng, observations),
        tx=actor_tx,
    )



    cost_critic_tx, cost_value_tx, critic_tx, value_tx = [optax.chain(
        optax.clip_by_global_norm(config.gradient_clip),
        optax.adam(lr),
    ) for lr in [config.cost_critic_lr, config.cost_value_lr, config.critic_lr, config.value_lr]]

    # initialize cost-critic
    # cost_critic_model = ensemblize(CostCritic, num_qs=2)(config.hidden_dims, max_cost=config.max_cost_to_learn)
    cost_critic_model = ensemblize(Critic, num_qs=2)(config.hidden_dims)
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
    cost_value_model = Critic(config.hidden_dims, layer_norm=True)
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
        actor_safe=actor_safe.replace()
    )







def run_for_config(config: IQLConfig):
    print("Use gymnasium:", use_gymnasium)
    if isinstance(config.eval_cost_thresholds, int):
        config.eval_cost_thresholds = (config.eval_cost_thresholds,)
    if config.cost_violation_penalty is not None:
        config.cost_violation_reward = -config.cost_violation_penalty
    

    name = f"{config.algo}-{config.env_name}-{config.seed}"
    np.random.seed(config.seed)

    run = None
    if config.use_wandb:
        run = wandb.init(config=config.model_dump(), project=config.project, name=name, tags=[config.tag])


    config_hash = dict_hash_ignore_keys(config.dict(), ['env_name', 'seed', 'eval_cost_thresholds'])
    log_dir = config.log_dir_base + f"/{config_hash}/{config.algo}/{config.env_name}/seed_{config.seed}/"
    print("Log dir", log_dir)
    os.makedirs(log_dir, exist_ok=True)
    # Save to JSON file
    with open(f"{log_dir}/config.json", "w") as f:
        config.wandb_run = run.path if config.use_wandb else ""
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
    initial_states = get_initial_states(dataset)
    print(len(initial_states), "initial states found")
    if config.cost_expectile is None:
        print("Must set cost expectile")
        exit()

    config.min_reward = float(dataset.rewards.min())
    print("Min reward in dataset", config.min_reward)
    
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
        initial_cost = train_state.cost_value.apply_fn(train_state.cost_value.params, initial_states)
        print(initial_cost, jnp.mean(initial_cost), jnp.std(initial_cost))
        policy_fn = get_policy_from_train_state(train_state)
        small_dataset = jax.tree_util.tree_map(lambda x: x[:2], dataset)  # Use a small dataset for evaluation
        # fig = algo.plot_action(train_state, small_dataset, env, epoch)
        # plt.savefig(f"{log_dir}/action_plot_{(epoch//config.eval_interval):07d}.png")
        # plt.close(fig)
        print("cost_violation_reward", config.cost_violation_reward)
        return evaluate_for_multiple_threshodls(
            policy_fn,
            env,
            num_episodes=config.final_eval_episodes,
            cost_thresholds=config.eval_cost_thresholds,
            init_cost=initial_cost.mean(),
            discount_budget=config.discount_budget
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
        "BulletGym": {
            "eval_cost_thresholds": (10, 20, 40),
            "expectile": 0.5,
            "cost_expectile": 0.2
        },
        "SafetyGym": {
            "eval_cost_thresholds": (20, 40, 80),
            "expectile": 0.6,
            "cost_expectile": 0.3
        },
    }

    task_overrides = {
        "CarCircle" : {
            "cost_expectile": 0.3,
        },
        "DroneCircle":{
            "cost_expectile": 0.3,
        }
    }

    if group in group_overrides:
        conf_dict.update(group_overrides[group])
    if taskname in task_overrides:
        conf_dict.update(task_overrides[taskname])

    config = IQLConfig(**conf_dict)
    print(config.expectile, config.cost_expectile, config.beta)
    run_for_config(config)
