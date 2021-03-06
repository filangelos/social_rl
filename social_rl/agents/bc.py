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
"""A simple behavioural cloning (BC) agent."""

import functools as ft
from typing import NamedTuple, Optional, Tuple

import distrax
import haiku as hk
import jax
import jax.numpy as jnp
import ml_collections
import optax

from social_rl import parts
from social_rl.losses import BCLoss
from social_rl.networks import GridWorldConvEncoder, LayerNormMLP


class BCNetwork(hk.Module):
  """A simple behavioural cloning neural network."""

  def __init__(self, num_actions: int, name: Optional[str] = None) -> None:
    """Construct a simple behavioural cloning neural network for discrete
    action spaces.

    Args:
      num_actions: The number of available discrete actions.
    """
    super().__init__(name=name)
    self._num_actions = num_actions

  def __call__(self, pixels_observation: jnp.ndarray) -> jnp.ndarray:
    """Return the policy logits, conditioned on the `pixels_observation`."""
    embedding = GridWorldConvEncoder()(pixels_observation)
    policy_logits = LayerNormMLP(
        output_sizes=(256, 128, self._num_actions),
        activate_final=False)(embedding)
    return policy_logits  # [B, A]


class BCLearnerState(NamedTuple):
  """Container for holding the BC learner's state."""
  opt_state: optax.OptState
  num_unique_steps: int


class BCActorState(NamedTuple):
  """Container for holding the BC actor's state."""
  network_state: parts.State
  num_unique_steps: int


def get_config() -> ml_collections.ConfigDict:
  """Return the default config for the behavioural cloning agent."""
  config = ml_collections.ConfigDict()
  config.learning_rate = 1e-4
  return config


class BCAgent(parts.Agent):
  """A simple behavioural cloning agent."""

  def __init__(
      self,
      env: parts.Environment,
      *,
      config: parts.Config = get_config(),
  ) -> None:
    """Construct a behavioural cloning agent."""
    super().__init__(env=env, config=config)

    # Initialise the network used by the agent.
    self._network = hk.without_apply_rng(
        hk.transform(lambda x: BCNetwork(self._action_spec.num_values)(x)))
    # Initialise the optimizer used by the learner.
    self._optimiser = optax.adam(learning_rate=self._cfg.learning_rate)
    # Initialise the loss function used by the learner.
    self._loss_fn = BCLoss(network_fn=self._network.apply)

  def should_learn(
      self,
      learner_state: BCLearnerState,
      actor_state: BCActorState,
  ) -> bool:
    """Whether the agent is ready to call `learner_step`."""
    del learner_state, actor_state
    return True

  def initial_params(self, rng_key: parts.PRNGKey) -> hk.Params:
    """Return the agent's initial parameters."""
    dummy_observation = self._observation_spec['pixels'].generate_value()[None]
    return self._network.init(rng_key, dummy_observation)

  def initial_learner_state(
      self,
      rng_key: parts.PRNGKey,
      params: hk.Params,
  ) -> BCLearnerState:
    """Return the agent's initial learner state."""
    del rng_key
    opt_state = self._optimiser.init(params)
    num_unique_steps = 0
    return BCLearnerState(opt_state, num_unique_steps)

  def initial_actor_state(self, rng_key: parts.PRNGKey) -> BCActorState:
    """Return the agent's initial actor state."""
    del rng_key
    network_state = ()
    num_unique_steps = 0
    return BCActorState(network_state, num_unique_steps)

  @ft.partial(jax.jit, static_argnums=0)
  def actor_step(
      self,
      params: hk.Params,
      env_output: parts.EnvOutput,  # Unbatched.
      actor_state: BCActorState,
      rng_key: parts.PRNGKey,
      evaluation: bool,
  ) -> Tuple[parts.AgentOutput, BCActorState, parts.InfoDict]:
    """Perform an actor step.

    Args:
      params: The agent's parameters.
      env_output: The environment's **unbatched** output.
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
    policy_logits = self._network.apply( # [B, A]
        params, env_output.observation['pixels'][None])
    policy = distrax.Categorical(logits=policy_logits)
    greedy_action = jnp.argmax(policy_logits, axis=-1)  # [B]
    sample_action = policy.sample(seed=rng_key)  # [B]
    action = jax.lax.select(evaluation, greedy_action, sample_action)[0]  # []
    new_actor_state = actor_state._replace(
        num_unique_steps=actor_state.num_unique_steps + 1)
    return parts.AgentOutput(action=action), new_actor_state, dict()

  @ft.partial(jax.jit, static_argnums=0)
  def learner_step(
      self,
      params: hk.Params,
      *transitions: parts.Transition,  # [B, ...]
      learner_state: BCLearnerState,
      rng_key: parts.PRNGKey,
  ) -> Tuple[hk.Params, BCLearnerState, parts.InfoDict]:
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

    (loss, logging_dict), grads = jax.value_and_grad(
        self._loss_fn, has_aux=True)(params, transition)
    updates, new_opt_state = self._optimiser.update(
        grads, learner_state.opt_state)
    logging_dict['global_gradient_norm'] = optax.global_norm(updates)

    new_params = optax.apply_updates(params, updates)

    new_learner_state = learner_state._replace(
        opt_state=new_opt_state,
        num_unique_steps=learner_state.num_unique_steps + 1)

    return new_params, new_learner_state, dict(loss=loss, **logging_dict)
