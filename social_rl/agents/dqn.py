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
"""A simple DQN agent."""

import copy
import functools as ft
from typing import NamedTuple, Optional, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import ml_collections
import optax
import rlax

from social_rl import parts
from social_rl.losses import DQNLoss
from social_rl.networks import GridWorldConvEncoder, LayerNormMLP


class DQNetwork(hk.Module):
  """A simple deep q-network."""

  def __init__(self, num_actions: int, name: Optional[str] = None) -> None:
    """Construct a simple deep q-network network.

    Args:
      num_actions: The number of available discrete actions.
    """
    super().__init__(name=name)
    self._num_actions = num_actions

  def __call__(self, pixels_observation: jnp.ndarray) -> jnp.ndarray:
    """Return the action values, conditioned on the `pixels_observation`."""
    embedding = GridWorldConvEncoder()(pixels_observation)
    action_values = LayerNormMLP(
        output_sizes=(256, 128, self._num_actions),
        activate_final=False)(embedding)
    return action_values  # [B, A]


class DQNLearnerState(NamedTuple):
  """Container for holding the DQN learner's state."""
  target_params: hk.Params
  opt_state: optax.OptState
  num_unique_steps: int


class DQNActorState(NamedTuple):
  """Container for holding the DQN actor's state."""
  network_state: parts.State
  num_unique_steps: int


def get_config() -> ml_collections.ConfigDict:
  """Return the default config for the DQN agent."""
  config = ml_collections.ConfigDict()
  config.learning_rate = 1e-4
  config.gamma = 0.9
  config.min_actor_steps_before_learning = 1_000
  config.train_every = 1
  config.train_epsilon = ml_collections.ConfigDict()
  config.train_epsilon.init_value = 1.0
  config.train_epsilon.end_value = 0.05
  config.train_epsilon.transition_steps = 50_000
  config.eval_epsilon = 0.00
  config.update_target_every = 1_000
  return config


class DQNAgent(parts.Agent):
  """A simple DQN agent."""

  def __init__(
      self,
      env: parts.Environment,
      *,
      config: parts.Config = get_config(),
  ) -> None:
    """Construct an agent."""
    super().__init__(env, config=config)

    # Initialise the network used by the agent.
    self._network = hk.without_apply_rng(
        hk.transform(lambda x: DQNetwork(self._action_spec.num_values)(x)))
    # Initialise the optimizer used by the learner.
    self._optimizer = optax.adam(learning_rate=self._cfg.learning_rate)
    # Initialise the loss function used by the learner.
    self._loss_fn = DQNLoss(
        network_fn=self._network.apply, gamma=self._cfg.gamma)

    # Epsilon linear schedule for data collection.
    self._train_epsilon = optax.linear_schedule(**self._cfg.train_epsilon)
    self._eval_epsilon = self._cfg.eval_epsilon

  def should_learn(
      self,
      learner_state: DQNLearnerState,
      actor_state: DQNActorState,
  ) -> bool:
    """Whether the agent is ready to call `learner_step`."""
    del learner_state
    return (
        actor_state.num_unique_steps >=
        self._cfg.min_actor_steps_before_learning) and (
            actor_state.num_unique_steps % self._cfg.train_every == 0)

  def initial_params(self, rng_key: parts.PRNGKey) -> hk.Params:
    """Return the agent's initial parameters."""
    dummy_observation = self._observation_spec['pixels'].generate_value()[None]
    return self._network.init(rng_key, dummy_observation)

  def initial_learner_state(
      self,
      rng_key: parts.PRNGKey,
      params: hk.Params,
  ) -> DQNLearnerState:
    """Return the agent's initial learner state."""
    del rng_key
    target_params = copy.deepcopy(params)
    opt_state = self._optimizer.init(params)
    return DQNLearnerState(
        target_params=target_params, opt_state=opt_state, num_unique_steps=0)

  def initial_actor_state(self, rng_key: parts.PRNGKey) -> DQNActorState:
    """Return the agent's initial actor state."""
    del rng_key
    network_state = ()
    num_unique_steps = 0
    return DQNActorState(network_state, num_unique_steps)

  @ft.partial(jax.jit, static_argnums=0)
  def actor_step(
      self,
      params: hk.Params,
      env_output: parts.EnvOutput,  # Unbatched.
      actor_state: DQNActorState,
      rng_key: parts.PRNGKey,
      evaluation: bool,
  ) -> Tuple[parts.AgentOutput, DQNActorState, parts.InfoDict]:
    """Perform an actor step.

    Args:
      params: The agent's parameters.
      env_output: The environment's **unbatched** output, with `.observation`
        of shape [...].
      actor_state: The actor's state.
      rng_key: The random number generators key.
      evaluation: Whether this is the actor is used for data collection or
        evaluation.

    Returns:
      agent_output: The (possibly nested) agent's output, with at least a
        `.action` field.
      new_actor_state: The updated actor state after the application of one
        step.
      logging_dict: The auxiliary information used for logging purposes.
    """
    action_value = self._network.apply( # [B]
        params, env_output.observation['pixels'][None])
    epsilon = jax.lax.select(
        evaluation, self._eval_epsilon,
        self._train_epsilon(actor_state.num_unique_steps))
    policy = rlax.epsilon_greedy(epsilon)
    action = policy.sample(key=rng_key, preferences=action_value)  # [B]
    action = jnp.squeeze(action)  # []
    new_actor_state = actor_state._replace(
        num_unique_steps=actor_state.num_unique_steps + 1)
    return parts.AgentOutput(action=action), new_actor_state, dict(
        epsilon=epsilon)

  @ft.partial(jax.jit, static_argnums=0)
  def learner_step(
      self,
      params: hk.Params,
      *transitions: parts.Transition,  # [B, ...]
      learner_state: DQNLearnerState,
      rng_key: parts.PRNGKey,
  ) -> Tuple[hk.Params, DQNLearnerState, parts.InfoDict]:
    """Peform a single learning step and return the new agent state and
    auxiliary data, e.g., used for logging.

    Args:
      params: The agent's parameters.
      transition: The **batched** transition used for updating the `params` and
        `learner_state`, with shape [B, ...].
      learner_state: The learner's state.
      rng_key: The random number generators key.

    Returns:
      new_params: The agent's updated parameters after a learner's step.
      new_learner_state: The learner's updated parameters.
      logging_dict: The auxiliary information used for logging purposes.
    """
    del rng_key
    assert len(transitions) == 1
    transition = transitions[0]

    new_target_params = rlax.periodic_update(
        params,
        learner_state.target_params,
        steps=learner_state.num_unique_steps,
        update_period=self._cfg.update_target_every)

    (loss, logging_dict), grads = jax.value_and_grad(
        self._loss_fn,
        has_aux=True)(params, learner_state.target_params, transition)
    updates, new_opt_state = self._optimizer.update(
        grads, learner_state.opt_state)
    logging_dict['global_gradient_norm'] = optax.global_norm(updates)

    new_params = optax.apply_updates(params, updates)

    new_learner_state = learner_state._replace(
        target_params=new_target_params,
        opt_state=new_opt_state,
        num_unique_steps=learner_state.num_unique_steps + 1)

    return new_params, new_learner_state, dict(loss=loss, **logging_dict)
