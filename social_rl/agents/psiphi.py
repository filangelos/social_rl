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
"""A simple PsiPhi learning[1] agent.

[1] https://arxiv.org/abs/2102.12560.
"""

import copy
import functools as ft
from typing import NamedTuple, Optional, Sequence, Tuple

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import ml_collections
import optax
import rlax

from social_rl import losses
from social_rl import parts
from social_rl import tree_utils
from social_rl.agents.itd import ITDNetworkOutput
from social_rl.networks import GridWorldConvEncoder, LayerNormMLP


class PsiPhiLearnerState(NamedTuple):
  """Container for holding the PsiPhi learner's state."""
  target_params: hk.Params
  opt_state: optax.OptState
  num_unique_steps: int


class PsiPhiActorState(NamedTuple):
  """Container for holding the PsiPhi actor's state."""
  network_state: parts.State
  num_unique_steps: int
  preference_vector: Optional[chex.Array] = None


def get_config() -> ml_collections.ConfigDict:
  """Return the default config for the PsiPhi agent."""
  config = ml_collections.ConfigDict()
  config.num_cumulants = 7
  config.num_demonstrators = 2
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


@chex.dataclass(frozen=True)
class PsiPhiNetworkOutput:
  """Container for holding the `PsiPhiNetwork`'s output."""
  cumulants: chex.Array  # [B, num_cumulants, num_actions]
  others_successor_features: chex.Array  # [B, num_demonstrators, num_cumulants, num_actions]
  ego_successor_features: chex.Array  # [B, num_cumulants, num_actions]
  others_preference_vectors: chex.Array  # [num_demonstrators, num_cumulants]
  ego_preference_vector: chex.Array  # [num_cumulants]
  # Derived outputs.
  others_policy_params: chex.Array  # [B, num_demonstrators, num_actions]
  others_rewards: chex.Array  # [B, num_demonstrators, num_actions]
  ego_reward: chex.Array  # [B, num_actions]
  ego_action_value: chex.Array  # [B, num_actions]


class PsiPhiNetwork(hk.Module):
  """A simple neural network for PhiPsi learners."""

  def __init__(
      self,
      num_actions: int,
      num_cumulants: int,
      num_demonstrators: int,
      name: Optional[str] = None,
  ) -> None:
    """Construct a simple multi-head neural network, with the following output
    heads: (i) cumulants head; (ii) others' successor features head; (iii)
    others' preference vectors; (iv) ego successor features and (v) ego
    preference vector.

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
    """Return the multiple outputs, conditioned on the `pixels_observation`."""
    # Tensor shape aliases.
    N = self._num_demonstrators
    A = self._num_actions
    C = self._num_cumulants

    # Torso network.
    embedding = GridWorldConvEncoder()(pixels_observation)
    embedding = LayerNormMLP(
        output_sizes=(256, 128), activate_final=True)(embedding)

    # Cumulants head.
    cumulants = hk.nets.MLP(
        output_sizes=(64, C * A), activate_final=False)(embedding)
    cumulants = jax.nn.tanh(cumulants)
    cumulants = hk.Reshape(output_shape=(C, A))(cumulants)

    # Others' successor features head.
    others_successor_features = hk.nets.MLP(
        output_sizes=(64, N * C * A), activate_final=False)(embedding)
    others_successor_features = hk.Reshape(
        output_shape=(N, C, A))(others_successor_features)

    # Ego successor features head.
    ego_successor_features = hk.nets.MLP(
        output_sizes=(64, C * A), activate_final=False)(embedding)
    ego_successor_features = hk.Reshape(
        output_shape=(C, A))(ego_successor_features)

    # Others' preference vectors head.
    others_preference_vectors = hk.get_parameter(
        'others_preference_vectors',
        shape=(N, C),
        init=hk.initializers.RandomNormal())

    # Ego preference vector head.
    ego_preference_vector = hk.get_parameter(
        'ego_preference_vector',
        shape=(C,),
        init=hk.initializers.RandomNormal())

    # Derive the rewards.
    others_rewards = jnp.einsum(
        'nc,bca->bna', others_preference_vectors, cumulants)
    ego_reward = jnp.einsum('c,bca->ba', ego_preference_vector, cumulants)

    # Derive the others' policy logits.
    others_policy_params = jnp.einsum(
        'nc,bnca->bna', others_preference_vectors, others_successor_features)

    # Derive the ego action value.
    ego_action_value = jnp.einsum(
        'c,bca->ba', ego_preference_vector, ego_successor_features)

    return PsiPhiNetworkOutput(
        cumulants=cumulants,
        others_successor_features=others_successor_features,
        ego_successor_features=ego_successor_features,
        others_preference_vectors=others_preference_vectors,
        ego_preference_vector=ego_preference_vector,
        others_policy_params=others_policy_params,
        others_rewards=others_rewards,
        ego_reward=ego_reward,
        ego_action_value=ego_action_value)


class PsiPhiAgent(parts.Agent):
  """A simple PsiPhi agent."""

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
        hk.transform(
            lambda x: PsiPhiNetwork(
                num_actions=self._action_spec.num_values,
                num_cumulants=self._cfg.num_cumulants,
                num_demonstrators=self._cfg.num_demonstrators)(x)))
    # Initialise the optimizer used by the learner.
    self._optimizer = optax.adam(learning_rate=self._cfg.learning_rate)

    # Epsilon linear schedule for data collection.
    self._train_epsilon = optax.linear_schedule(**self._cfg.train_epsilon)
    self._eval_epsilon = self._cfg.eval_epsilon

  def should_learn(
      self,
      learner_state: PsiPhiLearnerState,
      actor_state: PsiPhiActorState,
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
  ) -> PsiPhiLearnerState:
    """Return the agent's initial learner state."""
    del rng_key
    target_params = copy.deepcopy(params)
    opt_state = self._optimizer.init(params)
    return PsiPhiLearnerState(
        target_params=target_params, opt_state=opt_state, num_unique_steps=0)

  def initial_actor_state(self, rng_key: parts.PRNGKey) -> PsiPhiActorState:
    """Return the agent's initial actor state."""
    del rng_key
    network_state = ()
    num_unique_steps = 0
    # Fall back to the learned `ego_preference_vector`.
    preference_vector = None
    return PsiPhiActorState(network_state, num_unique_steps, preference_vector)

  @ft.partial(jax.jit, static_argnums=0)
  def actor_step(
      self,
      params: hk.Params,
      env_output: parts.EnvOutput,  # Unbatched.
      actor_state: PsiPhiActorState,
      rng_key: parts.PRNGKey,
      evaluation: bool,
  ) -> Tuple[parts.AgentOutput, PsiPhiActorState, parts.InfoDict]:
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
    network_output = self._network.apply( # [B]
        params, env_output.observation['pixels'][None])
    preferences = gpi_policy(network_output)
    # Choose between the `train` and `eval` epsilon parameters.
    epsilon = jax.lax.select(
        evaluation, self._eval_epsilon,
        self._train_epsilon(actor_state.num_unique_steps))
    # Act according to an epislon-greedy policy.
    policy = rlax.epsilon_greedy(epsilon)
    action = policy.sample(key=rng_key, preferences=preferences)  # [B]
    action = jnp.squeeze(action)  # []
    # Update the actor state.
    new_actor_state = actor_state._replace(
        num_unique_steps=actor_state.num_unique_steps + 1)
    return parts.AgentOutput(action=action), new_actor_state, dict(
        epsilon=epsilon)

  @ft.partial(jax.jit, static_argnums=0)
  def learner_step(
      self,
      params: hk.Params,
      *transitions: parts.Transition,  # [B, ...]
      learner_state: PsiPhiLearnerState,
      rng_key: parts.PRNGKey,
  ) -> Tuple[hk.Params, PsiPhiLearnerState, parts.InfoDict]:
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
    assert len(transitions) == 1 + self._cfg.num_demonstrators

    new_target_params = rlax.periodic_update(
        params,
        learner_state.target_params,
        steps=learner_state.num_unique_steps,
        update_period=self._cfg.update_target_every)

    (loss, logging_dict), grads = jax.value_and_grad(
        self._loss_fn,
        has_aux=True)(params, learner_state.target_params, transitions)
    updates, new_opt_state = self._optimizer.update(
        grads, learner_state.opt_state)
    logging_dict['global_gradient_norm'] = optax.global_norm(updates)

    new_params = optax.apply_updates(params, updates)

    new_learner_state = learner_state._replace(
        target_params=new_target_params,
        opt_state=new_opt_state,
        num_unique_steps=learner_state.num_unique_steps + 1)

    return new_params, new_learner_state, dict(loss=loss, **logging_dict)

  def _loss_fn(
      self,
      params: hk.Params,
      target_params: hk.Params,
      transitions: Sequence[parts.Transition],
  ) -> parts.LossOutput:
    """Return the DQN + ITD + BC loss, evaluated on network's `params` and real
    `transitions`."""

    # Wrap the `self._network` in a `DQNLoss`-friendly function.
    def dqn_network_fn(params: hk.Params, s_tm1: jnp.ndarray) -> jnp.ndarray:
      """Return the ego agent's action value."""
      return self._network.apply(params, s_tm1).ego_action_value

    # Wrap the `self._network` in a `RewardLoss`-friendly function.
    def reward_network_fn(params: hk.Params, s_tm1: jnp.ndarray) -> jnp.ndarray:
      """Return the ego agent's reward prediction."""
      return self._network.apply(params, s_tm1).ego_reward

    # Wrap the `self._network` to a `BCLoss`-friendly function.
    def bc_network_fn(
        params: hk.Params,
        s_tm1: jnp.ndarray,
        demonstrator_index: int,
    ) -> jnp.ndarray:
      """Return the policy parameters, conditioned on `s_tm1`."""
      policy_logits = self._network.apply(params, s_tm1).others_policy_params
      chex.assert_rank(policy_logits, 3)  # [B, N, A]
      return policy_logits[:, demonstrator_index]  # [B, A]

    # Wrap the `self._network` to a `ITDLoss`-friendly function.
    def itd_network_fn(
        params: hk.Params,
        s_tm1: jnp.ndarray,
    ) -> ITDNetworkOutput:
      """Return the `PsiPhiNetworkOutput` in a `ITDLoss`-friendly format."""
      psiphi_network_output = self._network.apply(params, s_tm1)
      return ITDNetworkOutput(
          cumulants=psiphi_network_output.cumulants,
          successor_features=psiphi_network_output.others_successor_features,
          preference_vectors=psiphi_network_output.others_preference_vectors,
          reward=psiphi_network_output.others_rewards,
          policy_params=psiphi_network_output.others_policy_params)

    dqn_loss_fn = losses.DQNLoss(
        network_fn=dqn_network_fn, gamma=self._cfg.gamma)
    reward_loss_fn = losses.RewardLoss(network_fn=reward_network_fn)
    # HACK(filangelos): `n=n` needs attention
    # https://stackoverflow.com/questions/47165783/what-does-i-i-mean-when-creating-a-lambda-in-python
    bc_loss_fns = {
        'bc_demo_{}'.format(n):
        losses.BCLoss(network_fn=lambda p, s, n=n: bc_network_fn(p, s, n))
        for n in range(self._cfg.num_demonstrators)
    }
    itd_loss_fns = {
        'itd_demo_{}'.format(n): losses.ITDLoss(
            network_fn=itd_network_fn,
            demonstrator_index=n,
            gamma=self._cfg.gamma,
            l1_loss_coef=0.0) for n in range(self._cfg.num_demonstrators)
    }

    # Split ego experience from others' experience.
    ego_transition, *others_transitions = transitions

    # Apply the loss functions on the network `params` and `transition`.
    dqn_loss_outputs = {
        'dqn': dqn_loss_fn(params, target_params, ego_transition)
    }
    reward_loss_output = {'reward': reward_loss_fn(params, ego_transition)}
    bc_loss_outputs = dict()
    for (label, bc_loss_fn), transition in zip(
        bc_loss_fns.items(),
        others_transitions,
    ):
      bc_loss_outputs[label] = bc_loss_fn(params, transition)
    itd_loss_outputs = dict()
    for (label, itd_loss_fn), transition in zip(
        itd_loss_fns.items(),
        others_transitions,
    ):
      itd_loss_outputs[label] = itd_loss_fn(params, transition)
    # Merge loss outputs.
    loss_output = tree_utils.merge_loss_outputs(
        **dqn_loss_outputs, **reward_loss_output, **bc_loss_outputs,
        **itd_loss_outputs)

    return loss_output


def gpi_policy(psiphi_network_output: PsiPhiNetworkOutput) -> jnp.ndarray:
  """Return the GPI policy's preferences."""

  # Stack all the successor features.
  successor_features = jnp.concatenate(  # (B, 1+N, C, A)
      [
          psiphi_network_output.ego_successor_features[:, None], # (B, 1, C, A)
          psiphi_network_output.others_successor_features # (B, N, C, A)
      ],
      axis=1)

  # Calculate the per-agent action values for the `ego_preference_vector` task.
  ego_task_energies = jnp.einsum( # (B, N, A)
      'c,bnca->bna', psiphi_network_output.ego_preference_vector,
      successor_features)

  # Max over policies.
  ego_task_preferences = jnp.max(ego_task_energies, axis=-2)  # (B, A)

  return ego_task_preferences
