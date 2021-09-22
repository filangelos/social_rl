# Copyright 2021 Angelos Filos. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Agent-environment interaction loops."""

from typing import Tuple

import haiku as hk
import jax
import tqdm

from social_rl import parts
from social_rl import tree_utils


def run_actor_loop(
    rng_key,
    params: hk.Params,
    actor_step,
    initial_actor_state,
    env: parts.Environment,
    num_episodes: int,
    evaluation: bool,
) -> parts.InfoDict:
  """Run the `agent's` actor on the `env`, using `params`."""

  # Container for holding the logged objects.
  stats = list()

  # Interaction loop.
  with tqdm.tqdm(range(num_episodes)) as pbar_episode:
    for _ in pbar_episode:
      env_output = env.reset()
      # Initialise the agent's actor.
      rng_key, actor_key = jax.random.split(rng_key, num=2)
      actor_state = initial_actor_state(actor_key)
      episode_returns = 0.0
      episode_length = 0
      while not env_output.last():
        rng_key, actor_key = jax.random.split(rng_key, num=2)
        agent_output, actor_state, logging_dict = actor_step(
            params, env_output, actor_state, actor_key, evaluation=evaluation)
        env_output = env.step(int(agent_output.action))
        episode_returns += env_output.reward
        episode_length += 1
      stats.append({'returns': episode_returns, 'length': episode_length})

  # Homogenise the logs.
  stats = tree_utils.stack(stats)

  return stats


def run_offline_learner_loop(
    rng_key: parts.PRNGKey,
    params: hk.Params,
    learner_state: parts.State,
    learner_step,
    replay_buffer: parts.ReplayBuffer,
    *,
    num_steps: int,
    batch_size: int,
    evaluate_every: int,
) -> Tuple[hk.Params, parts.State, parts.InfoDict]:
  """Run the `agent`'s learner by sampling data from `replay_buffer`."""

  # Container for holding the logged objects.
  stats_train = list()
  stats_eval = list()

  # Interaction loop.
  with tqdm.tqdm(range(num_steps)) as pbar_step:
    for step in pbar_step:
      # Training step.
      train_transition = replay_buffer.sample(
          batch_size=batch_size, evaluation=False)
      rng_key, learner_key = jax.random.split(rng_key, num=2)
      params, learner_state, train_logging_dict = learner_step(
          params, train_transition, learner_state, learner_key)
      train_logging_dict = tree_utils.to_numpy(train_logging_dict)
      stats_train.append({'step': step, 'stats': train_logging_dict})
      description = 'step {:6d} | TRAIN: '.format(step) + ' | '.join(
          ['{}: {:.2f}'.format(k, v) for k, v in train_logging_dict.items()])
      # Evaluation step.
      if step % evaluate_every == 0:
        eval_transition = replay_buffer.sample(
            batch_size=batch_size, evaluation=True)
        _, _, eval_logging_dict = learner_step( # Do **not** update the learner.
            params, eval_transition, learner_state, learner_key)
        eval_logging_dict = tree_utils.to_numpy(eval_logging_dict)
        stats_eval.append({'step': step, 'stats': eval_logging_dict})
        description += ' || EVAL : ' + ' | '.join(
            ['{}: {:.2f}'.format(k, v) for k, v in eval_logging_dict.items()])
      pbar_step.set_description(description)

  # Homogenise the logs.
  stats_train = tree_utils.stack(stats_train)
  stats_eval = tree_utils.stack(stats_eval)

  return params, learner_state, dict(train=stats_train, eval=stats_eval)
