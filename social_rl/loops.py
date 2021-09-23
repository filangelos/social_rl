# Copyright 2020 Angelos Filos. All Rights Reserved.
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

from typing import Optional, Sequence, Tuple

import haiku as hk
import jax
import tqdm

from social_rl import parts
from social_rl import tree_utils


class AgentEnvironmentLoop(parts.Loop):
  """The interaction loop between an `agent` and an `environment`."""

  def __init__(
      self,
      environment: parts.Environment,
      agent: parts.Agent,
      replay_buffer: parts.ReplayBuffer,
      name: Optional[str] = None,
  ) -> None:
    """Construct an interaction loop for an `agent`-`environment` pair."""
    self._agent = agent
    self._environment = environment
    self._replay_buffer = replay_buffer
    self._name = name

  def run(
      self,
      rng_key: parts.PRNGKey,
      num_iterations: int,
      *,
      params: Optional[hk.Params] = None,
      actor_state: Optional[parts.State] = None,
      learner_state: Optional[parts.State] = None,
      evaluate_every: int = 1_000,
  ) -> Tuple[hk.Params, parts.State, parts.State, parts.InfoDict]:
    """Run the loop for `num_iterations` and return the updated
    `(params, actor_state, learner_state)`.

    Args:
      rng_key: The random number generator's initial seed.
      num_iterations: The number of environment steps performed.
      params: The agent's parameters. If `None` then the `agent.initial_params`
        function is used for initialising them.
      actor_state: The actor's initial state. If `None` then the
        `agent.initial_actor_state` function is used for initialising it.
      learner_state: The learner's initial state. If `None` then the
        `agent.initial_learner_state` function is used for initialising it.
    """
    del evaluate_every

    # Maybe initialise agent's components.
    if params is None:
      rng_key, params_key = jax.random.split(rng_key, num=2)
      params = self._agent.initial_params(params_key)
    if actor_state is None:
      rng_key, actor_key = jax.random.split(rng_key, num=2)
      actor_state = self._agent.initial_actor_state(actor_key)
    if learner_state is None:
      rng_key, learner_key = jax.random.split(rng_key, num=2)
      learner_state = self._agent.initial_learner_state(learner_key, params)

    # Container for holding the logged objects.
    stats_actor = list()
    stats_learner = list()
    stats_train_episode_metrics = list()
    stats_eval_episode_metrics = list()

    # The `agent`-`environment` interaction loop.
    env_output = self._environment.reset()
    self._replay_buffer.add(env_output)
    episode_returns = 0.0
    episode_length = 0
    for step in tqdm.trange(num_iterations):
      rng_key, actor_key = jax.random.split(rng_key, num=2)
      agent_output, actor_state, actor_logging_dict = self._agent.actor_step(
          params, env_output, actor_state, actor_key, evaluation=False)
      stats_actor.append(
          dict(step=step, stats=tree_utils.to_numpy(actor_logging_dict)))
      env_output = self._environment.step(int(agent_output.action))
      episode_returns += env_output.reward
      episode_length += 1
      self._replay_buffer.add(env_output, agent_output)
      if env_output.last():
        stats_train_episode_metrics.append(
            dict(
                step=step,
                stats=dict(returns=episode_returns, length=episode_length)))
        env_output = self._environment.reset()
        self._replay_buffer.add(env_output)
        episode_returns = 0.0
        episode_length = 0
      if self._replay_buffer.can_sample(
          evaluation=False) and self._agent.should_learn(
              learner_state,
              actor_state,
          ):
        transition = self._replay_buffer.sample(evaluation=False)
        rng_key, learner_key = jax.random.split(rng_key, num=2)
        params, learner_state, learner_logging_dict = self._agent.learner_step(
            params, *[transition], learner_state, learner_key)
        stats_learner.append(
            dict(step=step, stats=tree_utils.to_numpy(learner_logging_dict)))

    # Homogenise the logs.
    stats_actor = tree_utils.stack(stats_actor)
    stats_learner = tree_utils.stack(stats_learner)
    stats_train_episode_metrics = tree_utils.stack(stats_train_episode_metrics)
    # stats_eval_episode_metrics = tree_utils.stack(stats_eval_episode_metrics)
    stats = dict(
        actor=stats_actor,
        learner=stats_learner,
        episode_metrics=dict(
            train=stats_train_episode_metrics, eval=stats_eval_episode_metrics))

    return params, actor_state, learner_state, stats


class ActorEnvironmentLoop(parts.Loop):
  """The interaction loop between an `actor` and an `environment`."""

  def __init__(
      self,
      environment: parts.Environment,
      agent: parts.Agent,
      name: Optional[str] = None,
  ) -> None:
    """Construct an interaction loop for an `actor`-`environment` pair."""
    self._agent = agent
    self._environment = environment
    self._name = name

  def run(
      self,
      rng_key: parts.PRNGKey,
      num_iterations: int,
      *,
      params: Optional[hk.Params] = None,
      actor_state: Optional[parts.State] = None,
      evaluation: bool = True,
  ) -> Tuple[parts.State, parts.InfoDict]:
    """Run the loop for `num_iterations` and return the updated `actor_state`.

    Args:
      rng_key: The random number generator's initial seed.
      num_iterations: The number of episodes run.
      params: The agent's parameters.
      actor_state: The actor's initial state. If `None` then the
        `agent.initial_actor_state` function is used for initialising it.
      evaluation: Whether we run the actor in the `evaluation` mode.
    """

    # Maybe initialise agent's components.
    if actor_state is None:
      rng_key, actor_key = jax.random.split(rng_key, num=2)
      actor_state = self._agent.initial_actor_state(actor_key)

    # Container for holding the logged objects.
    stats = list()

    # The `actor`-`environment` interaction loop.
    for _ in tqdm.trange(num_iterations):
      env_output = self._environment.reset()
      episode_returns = 0.0
      episode_length = 0
      while not env_output.last():
        rng_key, actor_key = jax.random.split(rng_key, num=2)
        agent_output, actor_state, _ = self._agent.actor_step(
            params, env_output, actor_state, actor_key, evaluation=evaluation)
        env_output = self._environment.step(int(agent_output.action))
        episode_returns += env_output.reward
        episode_length += 1
      stats.append({'returns': episode_returns, 'length': episode_length})

    # Homogenise the logs.
    stats = tree_utils.stack(stats)

    return actor_state, stats


class LearnerBufferLoop(parts.Loop):
  """The interaction loop between a `learner` and a fixed `buffer`."""

  def __init__(
      self,
      replay_buffers: Sequence[parts.ReplayBuffer],
      agent: parts.Agent,
      name: Optional[str] = None,
  ) -> None:
    """Construct an interaction loop for a `learner`-`replay_buffer` pair."""
    self._replay_buffers = replay_buffers
    self._agent = agent
    self._name = name

  def run(
      self,
      rng_key: parts.PRNGKey,
      num_iterations: int,
      *,
      params: Optional[hk.Params] = None,
      learner_state: Optional[parts.State] = None,
      evaluate_every: int = 50,
  ) -> Tuple[hk.Params, parts.State, parts.InfoDict]:
    """Run the loop for `num_iterations` and return the updated
    `(params, learner_state)`.

    Args:
      rng_key: The random number generator's initial seed.
      num_iterations: The number of learning steps.
      params: The agent's parameters. If `None` then the `agent.initial_params`
        function is used for initialising them.
      learner_state: The learner's initial state. If `None` then the
        `agent.initial_learner_state` function is used for initialising it.
    """

    # Maybe initialise agent's components.
    if params is None:
      rng_key, params_key = jax.random.split(rng_key, num=2)
      params = self._agent.initial_params(params_key)
    if learner_state is None:
      rng_key, learner_key = jax.random.split(rng_key, num=2)
      learner_state = self._agent.initial_learner_state(learner_key, params)

    # Container for holding the logged objects.
    stats_train = list()
    stats_eval = list()

    # The `learner`-`replay_buffer` interaction loop.
    for step in tqdm.trange(num_iterations):
      # Training step.
      train_transitions = [
          rb.sample(evaluation=False) for rb in self._replay_buffers
      ]
      rng_key, learner_key = jax.random.split(rng_key, num=2)
      params, learner_state, train_logging_dict = self._agent.learner_step(
          params,
          *train_transitions,
          learner_state=learner_state,
          rng_key=learner_key)
      train_logging_dict = tree_utils.to_numpy(train_logging_dict)
      stats_train.append({'step': step, 'stats': train_logging_dict})
      # Evaluation step.
      if evaluate_every > 0 and step % evaluate_every == 0:
        eval_transitions = [
            rb.sample(evaluation=False) for rb in self._replay_buffers
        ]
        _, _, eval_logging_dict = self._agent.learner_step(
            params,
            *eval_transitions,
            learner_state=learner_state,
            rng_key=learner_key)
        eval_logging_dict = tree_utils.to_numpy(eval_logging_dict)
        stats_eval.append({'step': step, 'stats': eval_logging_dict})

    # Homogenise the logs.
    if len(stats_train) > 0:
      stats_train = tree_utils.stack(stats_train)
    if len(stats_eval) > 0:
      stats_eval = tree_utils.stack(stats_eval)

    return params, learner_state, dict(train=stats_train, eval=stats_eval)
