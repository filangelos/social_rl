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
"""Simple inverse temporal difference learning agent."""

import functools as ft
from typing import NamedTuple, Optional, Sequence, Tuple

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import ml_collections
import optax

from social_rl import parts
from social_rl import tree_utils
from social_rl.losses import BCLoss, ITDLoss
from social_rl.networks import GridWorldConvEncoder, LayerNormMLP


class ITDLearnerState(NamedTuple):
  """Container for holding the ITD learner's state."""
  opt_state: optax.OptState
  num_unique_steps: int


class ITDActorState(NamedTuple):
  """Container for holding the ITD actor's state."""
  network_state: parts.State
  num_unique_steps: int


def get_config() -> ml_collections.ConfigDict:
  """Return the default config for the behavioural cloning agent."""
  config = ml_collections.ConfigDict()
  config.learning_rate = 1e-4
  config.num_cumulants = 7
  config.num_demonstrators = 1
  config.gamma = 0.9
  return config


@chex.dataclass(frozen=True)
class ITDNetworkOutput:
  """Container for holding the `ITDNetwork`'s output."""
  cumulants: chex.Array  # [B, num_cumulants, num_actions]
  successor_features: chex.Array  # [B, num_demonstrators, num_cumulants, num_actions]
  preference_vectors: chex.Array  # [num_demonstrators, num_cumulants]
  # Derived outputs.
  reward: chex.Array  # [B, num_demonstrators, num_actions]
  policy_params: chex.Array  # [B, num_demonstrators, num_actions]


class ITDNetwork(hk.Module):
  """A simple behavioural cloning neural network."""

  def __init__(
      self,
      num_actions: int,
      num_cumulants: int,
      num_demonstrators: int,
      name: Optional[str] = None,
  ) -> None:
    """Construct a simple multi-head neural network, with the following output
    heads: (i) cumulants head; (ii) successor features head and (iii)
    preference vectors.

    Args:
      num_actions: The number of available discrete actions.
      num_cumulants: The number of cumulants used.
      num_demonstrators: The number of distinct demonstrators.
    """
    super().__init__(name=name)
    self._num_actions = num_actions
    self._num_cumulants = num_cumulants
    self._num_demonstrators = num_demonstrators

  def __call__(self, pixels_observation: jnp.ndarray) -> jnp.ndarray:
    """Return the policy logits, conditioned on the `pixels_observation`."""
    # Torso network.
    embedding = GridWorldConvEncoder()(pixels_observation)
    embedding = LayerNormMLP(
        output_sizes=(256, 64), activate_final=True)(embedding)

    # Cumulants head.
    cumulants = hk.nets.MLP(
        output_sizes=(64, self._num_cumulants * self._num_actions),
        activate_final=False)(embedding)
    cumulants = hk.Reshape(
        output_shape=(self._num_cumulants, self._num_actions))(cumulants)

    # Successor features head.
    successor_features = hk.nets.MLP(
        output_sizes=(
            64,
            self._num_demonstrators * self._num_cumulants * self._num_actions),
        activate_final=False)(embedding)
    successor_features = hk.Reshape(
        output_shape=(
            self._num_demonstrators, self._num_cumulants,
            self._num_actions))(successor_features)

    # Preference vectors head.
    preference_vectors = hk.get_parameter(
        'preference_vectors',
        shape=(self._num_demonstrators, self._num_cumulants),
        init=hk.initializers.RandomNormal())

    # Derive the rewards.
    reward = jnp.einsum('nc,bca->bna', preference_vectors, cumulants)
    policy_params = jnp.einsum(
        'nc,bnca->bna', preference_vectors, successor_features)

    return ITDNetworkOutput(
        cumulants=cumulants,
        successor_features=successor_features,
        preference_vectors=preference_vectors,
        reward=reward,
        policy_params=policy_params)


class ITDAgent(parts.Agent):
  """A simple inverse temporal difference learning agent."""

  def __init__(
      self,
      env: parts.Environment,
      *,
      config: parts.Config = get_config(),
  ) -> None:
    """Construct an inverse temporal difference learning agent."""
    super().__init__(env=env, config=config)

    # Initialise the network used by the agent.
    self._network = hk.without_apply_rng(
        hk.transform(
            lambda x: ITDNetwork(
                num_actions=self._action_spec.num_values,
                num_cumulants=self._cfg.num_cumulants,
                num_demonstrators=self._cfg.num_demonstrators)(x)))
    # Initialise the optimizer used by the learner.
    self._optimiser = optax.adam(learning_rate=self._cfg.learning_rate)

  def should_learn(
      self,
      learner_state: ITDLearnerState,
      actor_state: ITDActorState,
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
  ) -> ITDLearnerState:
    """Return the agent's initial learner state."""
    del rng_key
    opt_state = self._optimiser.init(params)
    num_unique_steps = 0
    return ITDLearnerState(opt_state, num_unique_steps)

  def initial_actor_state(self, rng_key: parts.PRNGKey) -> ITDActorState:
    """Return the agent's initial actor state."""
    del rng_key
    network_state = ()
    num_unique_steps = 0
    return ITDActorState(network_state, num_unique_steps)

  @ft.partial(jax.jit, static_argnums=0)
  def actor_step(
      self,
      params: hk.Params,
      env_output: parts.EnvOutput,  # Unbatched.
      actor_state: ITDActorState,
      rng_key: parts.PRNGKey,
      evaluation: bool,
  ) -> Tuple[parts.AgentOutput, ITDActorState, parts.InfoDict]:
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
      learner_state: ITDLearnerState,
      rng_key: parts.PRNGKey,
  ) -> Tuple[hk.Params, ITDLearnerState, parts.InfoDict]:
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
    assert len(transitions) == self._cfg.num_demonstrators

    (loss, logging_dict), grads = jax.value_and_grad(
        self._loss_fn, has_aux=True)(params, transitions)
    updates, new_opt_state = self._optimiser.update(
        grads, learner_state.opt_state)
    logging_dict['global_gradient_norm'] = optax.global_norm(updates)

    new_params = optax.apply_updates(params, updates)

    new_learner_state = learner_state._replace(
        opt_state=new_opt_state,
        num_unique_steps=learner_state.num_unique_steps + 1)

    return new_params, new_learner_state, dict(loss=loss, **logging_dict)

  def _loss_fn(
      self,
      params: hk.Params,
      transitions: Sequence[parts.Transition],
  ) -> parts.LossOutput:
    """Return the ITD + BC loss, evaluated on network's `params` and real
    `transition`."""

    # Wrap the `self._network` to a `BCLoss`-friendly function.
    def network_fn(
        params: hk.Params,
        s_tm1: jnp.ndarray,
        demonstrator_index: int,
    ) -> jnp.ndarray:
      """Return the policy parameters, conditioned on `s_tm1`."""
      return self._network.apply(params,
                                 s_tm1).policy_params[:, demonstrator_index]

    # Build the loss functions.
    bc_loss_fns = {
        'bc_demo_{}'.format(n):
        BCLoss(network_fn=lambda p, s: network_fn(p, s, n))
        for n in range(self._cfg.num_demonstrators)
    }
    itd_loss_fns = {
        'itd_demo_{}'.format(n): ITDLoss(
            network_fn=self._network.apply,
            demonstrator_index=n,
            gamma=self._cfg.gamma,
            l1_loss_coef=0.0) for n in range(self._cfg.num_demonstrators)
    }

    # Apply the loss functions on the network `params` and `transition`.
    bc_loss_outputs = dict()
    for (label, bc_loss_fn), transition in zip(
        bc_loss_fns.items(),
        transitions,
    ):
      bc_loss_outputs[label] = bc_loss_fn(params, transition)
    itd_loss_outputs = dict()
    for (label, itd_loss_fn), transition in zip(
        itd_loss_fns.items(),
        transitions,
    ):
      itd_loss_outputs[label] = itd_loss_fn(params, transition)
    # Merge loss outputs.
    loss_output = tree_utils.merge_loss_outputs(
        **bc_loss_outputs, **itd_loss_outputs)

    return loss_output
