import gymnasium as gym
from collections import defaultdict
import wandb
from functools import partial
from flax.training.train_state import TrainState
import jax
from flax import serialization
import cloudpickle
import numpy as np
import tqdm
import pandas as pd
from typing import NamedTuple, List, Callable, Tuple, Any
import os
import flax.linen as nn
import jax.numpy as jnp
import hashlib
import json


def dict_hash_ignore_keys(d: dict, ignore_keys=None) -> str:
    if ignore_keys is None:
        ignore_keys = []
    # Filter out keys to ignore
    filtered = {k: v for k, v in d.items() if k not in ignore_keys}
    # Serialize and hash
    encoded = json.dumps(filtered, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()

def default_init():
    return nn.initializers.orthogonal(2**0.5)
    # return nn.initializers.glorot_uniform() # xavier_uniform is same as  glorot_uniform

def ensemblize(cls, num_qs, out_axes=0, **kwargs):
    split_rngs = kwargs.pop("split_rngs", {})
    return nn.vmap(
        cls,
        variable_axes={"params": 0},
        split_rngs={**split_rngs, "params": True, "dropout": True},
        in_axes=None,
        out_axes=out_axes,
        axis_size=num_qs,
        **kwargs,
    )


def save_model(state, path, config, cost_normalizing_factor: float, obs_mean, obs_std):
    data = {
        "state": serialization.to_state_dict(state),
        "config": config.model_dump(),
        "cost_normalizing_factor": cost_normalizing_factor,
        "obs_mean": obs_mean,
        "obs_std": obs_std,
    }
    with open(path, "wb") as f:
        cloudpickle.dump(data, f)

def get_action_from_actor(
        actor_model,
        actor_params,
        observations: jnp.ndarray,
        seed: jnp.ndarray,
        cost_limits: jnp.ndarray,
        temperature: float = 1.0,
        max_action: float = 1.0,  # In D4RL, the action space is [-1, 1]
        cost_normalizing_factor: float = 1.0,
        obs_mean: np.ndarray = None,
        obs_std: np.ndarray = None,
    ) -> jnp.ndarray:
        observations = (observations - obs_mean) / obs_std
        cost_limits = jnp.clip(cost_limits, min=0)/cost_normalizing_factor
        actions = actor_model.apply(
            actor_params, observations, cost_limits, temperature=temperature
        ).sample(seed=seed)
        actions = jnp.clip(actions, -max_action, max_action)
        return actions


def load_model(path):
    with open(path, "rb") as f:
        data = cloudpickle.load(f)
        return data


class Transition(NamedTuple):
    observations: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    next_observations: jnp.ndarray
    dones: jnp.ndarray
    costs: jnp.ndarray
    truncations: jnp.ndarray

# batch_indices_counts = defaultdict(int)
def sample_batch(dataset, rng, batch_size):
    batch_indices = jax.random.randint(rng, (batch_size,), 0, len(dataset.observations))
    return jax.tree_util.tree_map(lambda x: x[batch_indices], dataset)


def get_dataset(
    env: gym.Env, limit: int, clip_to_eps: bool = True, eps: float = 1e-5
) -> Transition:
    dataset = env.get_dataset()

    if clip_to_eps:
        lim = 1 - eps
        dataset["actions"] = np.clip(dataset["actions"], -lim, lim)

    dones = dataset['terminals']
    truncations = dataset['timeouts']

    dataset = Transition(
        observations=jnp.array(dataset["observations"], dtype=jnp.float32),
        actions=jnp.array(dataset["actions"], dtype=jnp.float32),
        rewards=jnp.array(dataset["rewards"], dtype=jnp.float32),
        next_observations=jnp.array(dataset["next_observations"], dtype=jnp.float32),
        dones=jnp.array(dones, dtype=jnp.float32),
        costs=jnp.array(dataset["costs"], dtype=jnp.float32),
        truncations=jnp.array(truncations, dtype=jnp.float32),
    )

    data_size = min(limit, len(dataset.observations))
    assert len(dataset.observations) >= data_size
    dataset = jax.tree_util.tree_map(lambda x: x[:data_size], dataset)
    return dataset


def get_initial_states(dataset: Transition) -> jnp.ndarray:
    dataset = jax.tree_util.tree_map(lambda x: np.array(x), dataset)
    init_states  = []
    episode_ended = True
    for term, trunc, obs in zip(dataset.dones, dataset.truncations, dataset.observations):
        if episode_ended:
            init_states.append(obs)
        if term or trunc:
            episode_ended = True
        else:
            episode_ended = False
    return jnp.array(init_states, dtype=jnp.float32)


def get_normalization(dataset: Transition) -> tuple:
    # into numpy.ndarray
    dataset = jax.tree_util.tree_map(lambda x: np.array(x), dataset)
    returns = []
    cost_all = []
    ret = 0
    cost = 0
    for r, c, term, trunc in zip(dataset.rewards, dataset.costs, dataset.dones, dataset.truncations):
        ret += r
        cost += c
        if term or trunc:
            returns.append(ret)
            cost_all.append(cost)
            ret = 0
            cost = 0
    print("R", max(returns), min(returns), "C", max(cost_all), min(cost_all))
    reward_scale =  (max(returns) - min(returns)) / 1000 # max(returns)/1000 #
    cost_scale = (max(cost_all) - min(cost_all)) / 1000 # max(cost_all)/1000 # 

    print("Reward range:[", min(returns), max(returns), "]", min(dataset.rewards), max(dataset.rewards))
    print("Cost range: [", min(cost_all), max(cost_all), "]", min(dataset.costs), max(dataset.costs))
    obs_mean, obs_std = dataset.observations.mean(axis=0), dataset.observations.std(axis=0)
    print("Obs mean: ", obs_mean, obs_std)
    print("Obs std: ", dataset.observations.std(axis=0))
    return reward_scale, cost_scale, obs_mean, obs_std

def normalize_dataset(dataset, normalize_cost_and_reward: bool, normalize_state:bool):
    normalizing_factor_reward, normalizing_factor_cost, obs_mean, obs_std = get_normalization(dataset)
    obs_std += 1e-5 # to avoid division by zero
    if not normalize_cost_and_reward:
        normalizing_factor_cost = 1.0
        normalizing_factor_reward = 1.0
    
    if not normalize_state:
        obs_mean = 0.0
        obs_std = 1.0

    print("Normalization factors: reward, cost:", normalizing_factor_reward, normalizing_factor_cost)
    dataset = dataset._replace(rewards=dataset.rewards / normalizing_factor_reward)
    dataset = dataset._replace(costs=dataset.costs / normalizing_factor_cost)
    dataset = dataset._replace(observations=(dataset.observations - obs_mean) / obs_std)
    dataset = dataset._replace(next_observations=(dataset.next_observations - obs_mean) / obs_std)
    return dataset, normalizing_factor_reward, normalizing_factor_cost, obs_mean, obs_std

def evaluate(
    policy_fn, env: gym.Env, num_episodes: int, cost_threshold: float, min_cost_threhold_ratio: float, rescalse_remaining_budget_by: float
) -> dict:
    episode_returns = []
    episode_costs = []
    episode_lengths = []
    env.reset()
    assert rescalse_remaining_budget_by <= 1.0, "rescalse_remaining_budget_by should be leq to 1.0 to avoid infinite loop"
    env.set_target_cost(cost_threshold)
    episode_max_length = env.spec.max_episode_steps if hasattr(env.spec, 'max_episode_steps') else 1001
    print(episode_max_length, "is the max episode length")

    modified_threshold = cost_threshold /((1-0.99)*episode_max_length)
    episode_infos = defaultdict(list)
    print("before after threhold", cost_threshold, modified_threshold)
    for _ in tqdm.tqdm(range(num_episodes), desc=f"Evaluation C={cost_threshold}"):
        episode_return = 0
        episode_cost = 0
        episode_length = 0
        (observation, info), done = env.reset(), False
        truncated = False
        cost_v = jnp.array(modified_threshold)
        while (not done) and (not truncated):

            action = policy_fn(observations=observation, cost_limits=cost_v[...,None], remaining_steps=episode_max_length-episode_length )#episode_max_length-episode_length)
            action = np.array(action)
            observation, reward, done, truncated, info = env.step(action)
            episode_cost += info["cost"]
            cost_v = ((cost_v - info['cost'])/0.99).clip(max=modified_threshold, min=0)
            episode_return += reward
            episode_length += 1
        episode_returns.append(episode_return)
        episode_costs.append(episode_cost)
        episode_lengths.append(episode_length)
        for k, v in info.items():
            episode_infos[k].append(v)
    normalize_r, normalized_c = zip(*[env.get_normalized_score(r, c) for r, c in zip(episode_returns, episode_costs)])
    print(f"Evaluation C={cost_threshold} | cost:{np.mean(normalized_c)}, reward: {np.mean(normalize_r)}, lengths: {np.mean(episode_lengths)}")
    info = {
        **{f"episode_info/{k}": v for k,v in episode_infos.items()},
        "episode_rewards": episode_returns,
        "episode_costs": episode_costs,
        "episode_lengths": episode_lengths,
        "normalized_rewards": normalize_r,
        "normalized_costs": normalized_c,
    }
    return info

def evaluate_for_multiple_threshodls(
    policy_fn, env: gym.Env, num_episodes: int, cost_thresholds: List[float], min_cost_threhold_ratio: float, rescalse_remaining_budget_by: float=1
) -> pd.DataFrame:
    data = {cost_theshold: pd.DataFrame(evaluate(policy_fn, env, num_episodes, cost_theshold, min_cost_threhold_ratio, rescalse_remaining_budget_by)) 
        for cost_theshold in cost_thresholds}
    return pd.concat(data, axis=1)


def get_eval_results_summary(df: pd.DataFrame):
    budgets = df.columns.get_level_values(0)

    budget_results = {}
    for b in budgets:
        rewards = df[(b, 'normalized_rewards')]
        costs = df[(b, 'normalized_costs')]
        budget_results = {**budget_results, 
            f'{b}_normalized_rewards_mean': rewards.values.flatten().mean(),
            f'{b}_normalized_rewards_std': rewards.values.flatten().std(),
            f'{b}_normalized_costs_mean': costs.values.flatten().mean(),
            f'{b}_normalized_costs_std': costs.values.flatten().std(),
        }

    rewards = df.xs('normalized_rewards', level=1, axis=1)
    costs = df.xs('normalized_costs', level=1, axis=1)

    # print(df)
    # exit()

    return {**budget_results ,'normalized_rewards_mean': rewards.values.flatten().mean(),
            'normalized_rewards_std': rewards.values.flatten().std(),
            'normalized_costs_mean': costs.values.flatten().mean(),
            'normalized_costs_std': costs.values.flatten().std(),
            'feasible_reward': rewards.values.flatten().mean() if (costs.values.flatten().mean() < 1.0) else -costs.values.flatten().mean(),
            }

def train_parallel(
    rng,
    train_state: Any,
    update_func: Callable,
    evaluate_func: Callable,
    save_func: Callable,
    save_folder: str,
    epochs: int,
    n_jitted_updates: int,
    log_interval:int,
    eval_interval:int,
    use_wandb: bool
    ):
    num_steps = epochs // n_jitted_updates
    log_interval = log_interval // n_jitted_updates
    eval_interval = eval_interval // n_jitted_updates
    best_safe_reward = -np.inf
    summary_records = {}
    all_metrics = None
    all_keys = jax.random.split(rng, num_steps)
    for i in tqdm.tqdm(range(1, num_steps + 1), smoothing=0.1, dynamic_ncols=True, desc="Train Parallel"):
        epoch = i* n_jitted_updates
        subkey = all_keys[i-1]
        train_state, update_info = update_func(train_state=train_state, rng=subkey)
        if i % log_interval == 0:
            summary_records[epoch] = update_info

        is_last_iter = i == num_steps
        if i % eval_interval == 0 or is_last_iter:
            print(f"Evaluating at epoch: {epoch}")
            all_records = evaluate_func(train_state, epoch)
            all_records.to_csv(f"{save_folder}/eval_last.csv")
            all_metrics = get_eval_results_summary(all_records)
            is_best_safe_reward = all_metrics["normalized_rewards_mean"] > best_safe_reward and  all_metrics["normalized_costs_mean"] < 1.0
            summary_records[epoch] = {**all_metrics, **update_info, "is_safe_best": is_best_safe_reward, "best_safe_reward": best_safe_reward}

            save_func(train_state, f"{save_folder}/model_last.pt")
            for k, v in all_metrics.items():
                print(f"{k}: {v:.4f}")

            if is_best_safe_reward:
                best_safe_reward = all_metrics["normalized_rewards_mean"]
                save_func(train_state, f"{save_folder}/model_best.pt")
                print("Best safe reward updated:", best_safe_reward)
                all_records.to_csv(f"{save_folder}/eval_best.csv")
        if (i%log_interval == 0 or i%eval_interval == 0 or is_last_iter):
            if use_wandb:
                wandb.log(summary_records[epoch], step=epoch)
            # if i%eval_interval == 0 or is_last_iter:
            pd.DataFrame.from_dict(summary_records, orient='index').to_csv(f"{save_folder}/progress.csv")

    total_steps = (num_steps*n_jitted_updates)

    if all_metrics is None:
        raise ValueError("During training it was never evaluated")

    return_info = {
        "total_steps": total_steps,
        "best_safe_reward": best_safe_reward,
        "final_reward": all_metrics["normalized_rewards_mean"],
        "final_cost": all_metrics["normalized_costs_mean"],
    }
    return return_info, train_state


# diff=Target - current_prediction
def expectile_loss(diff, expectile) -> jnp.ndarray:
    weight = jnp.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)

def percentile_loss(diff, percentile) -> jnp.ndarray:
    weight = jnp.where(diff > 0, percentile, (1 - percentile))
    return weight * jnp.abs(diff)

def gumbel_rescale_loss(diff, alpha, max_clip=None):
    """Gumbel loss J: E[e^x - x - 1]. For stability to outliers, we scale the gradients with the max value over a batch
    and optionally clip the exponent. This has the effect of training with an adaptive lr.
    Source: https://github.com/nissymori/JAX-CORL/blob/main/algos/xql.py
    """
    z = diff / alpha
    if max_clip is not None:
        z = jnp.minimum(z, max_clip)  # clip max value
    max_z = jnp.max(z, axis=0)
    max_z = jnp.where(max_z < -1.0, -1.0, max_z)
    max_z = jax.lax.stop_gradient(max_z)  # Detach the gradients
    loss = (
        jnp.exp(z - max_z) - z * jnp.exp(-max_z) - jnp.exp(-max_z)
    )  # scale by e^max_z
    return loss



def update_by_loss_grad(
    train_state: TrainState, loss_fn: Callable
) -> Tuple[TrainState, jnp.ndarray, dict]:
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, metrics), grad = grad_fn(train_state.params)
    new_train_state = train_state.apply_gradients(grads=grad)
    return new_train_state, loss, metrics

def target_update(
    model: TrainState, target_model: TrainState, tau: float
) -> TrainState:
    new_target_params = jax.tree_util.tree_map(
        lambda p, tp: p * tau + tp * (1 - tau), model.params, target_model.params
    )
    return target_model.replace(params=new_target_params)


def set_common_jax_flags():
    os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=false --xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0" 
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = "false"

def get_env_name_from_task_name(task_name):
    if "Average" in task_name:
        raise "Average task names are not supported"
    if "Velocity" in task_name:
        return f"Offline{task_name}Gymnasium-v1", "SafetyGym"
    if task_name.islower():
        return f"OfflineMetadrive-{task_name}-v0", "MetaDrive"
    if task_name[-1] == "1" or task_name[-1] == "2":
        return f"Offline{task_name}Gymnasium-v0", "SafetyGym"
    return f"Offline{task_name}-v0", "BulletGym"

def get_task_name_from_env_name(env_name):
    if "Meta" in env_name:
        return env_name.split("-")[-2], "MetaDrive"
    if "Gymnasium" in env_name:
        return env_name.split("Gymnasium")[0][len("Offline"):], "SafetyGym"
    else:
        return env_name.split("-")[0][len("Offline"):], "BulletGym"


