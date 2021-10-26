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
"""Loss functions used by social RL agents."""

import chex
import distrax
import haiku as hk
import jax
import jax.numpy as jnp
import rlax

from social_rl import parts


class RewardLoss:
  """A `__call__`-able reward grounding loss."""

  def __init__(self, network_fn) -> None:
    """Construct a reward groundling loss object."""
    self._network_fn = network_fn

  def __call__(
      self, params: hk.Params,
      transition: parts.Transition) -> parts.LossOutput:
    """Return the mean squared error of the reward prediction to the
    `transition.r_t`, evaluated on network's `params`."""

    @jax.vmap
    def mse_loss_fn(
        pred: jnp.ndarray,  # (A,)
        target: jnp.ndarray,  # ()
        action: int,  # ()
    ) -> jnp.ndarray:
      """Return the mean squared error loss."""
      chex.assert_rank([pred, target], [1, 0])
      return (pred[action] - target)**2

    reward_prediction = self._network_fn(params, transition.s_tm1['pixels'])
    loss = mse_loss_fn(
        reward_prediction, transition.r_t, transition.a_tm1)  # [B]
    loss = jnp.mean(loss)  # []

    return parts.LossOutput(loss=loss, aux_data=dict())


class BCLoss:
  """A `__call__`-able behavioural cloning loss."""

  def __init__(self, network_fn) -> None:
    """Construct a behavioural cloning loss object."""
    self._network_fn = network_fn

  def __call__(
      self,
      params: hk.Params,
      transition: parts.Transition,
  ) -> parts.LossOutput:
    """Return the negative log-likelihood of the `transition`, evaluated on
    network's `params`."""

    @jax.vmap
    def nll_loss_fn(
        pred_logits: jnp.ndarray,
        target: jnp.ndarray,
    ) -> jnp.ndarray:
      """Return the negative log-likelihood of `target` under `pred_dist`."""
      chex.assert_rank([pred_logits, target], [1, 0])
      pred_dist = distrax.Categorical(logits=pred_logits)
      return -pred_dist.log_prob(target)  # []

    @jax.vmap
    def accuracy_fn(
        pred_logits: jnp.ndarray,
        target: jnp.ndarray,
    ) -> jnp.ndarray:
      """Return the prediction accuracy."""
      chex.assert_rank([pred_logits, target], [1, 0])
      greedy_prediction = jnp.argmax(pred_logits, axis=-1)  # []
      return jnp.float32(greedy_prediction == target)  # []

    policy_logits = self._network_fn(params, transition.s_tm1['pixels'])
    loss = nll_loss_fn(policy_logits, transition.a_tm1)  # [B]
    loss = jnp.mean(loss)  # []
    accuracy = accuracy_fn(policy_logits, transition.a_tm1)  # [B]
    accuracy = jnp.mean(accuracy)  # []

    return parts.LossOutput(loss=loss, aux_data=dict(accuracy=accuracy))


class ITDLoss:
  """A `__call__`-able inverse temporal difference learning loss."""

  def __init__(
      self,
      network_fn,
      *,
      demonstrator_index: int,
      gamma: float,
      l1_loss_coef: float,
      stop_target_gradients: bool = True,
  ) -> None:
    """Construct an inverse temporal difference learning loss object."""
    self._network_fn = network_fn
    self._demonstrator_index = demonstrator_index
    self._gamma = gamma
    self._l1_loss_coef = l1_loss_coef  # This is unused for now.
    self._stop_target_gradients = stop_target_gradients

  def __call__(
      self,
      params: hk.Params,
      transition: parts.Transition,
  ) -> parts.LossOutput:
    """Return the inverse temporal difference learning loss, evaluated on
    network's `params` and real `transition`."""

    @jax.vmap
    def itd_loss_fn(
        psi_tm1: jnp.ndarray,
        a_tm1: jnp.ndarray,
        phi_t: jnp.ndarray,
        discount_t: jnp.ndarray,
        psi_t: jnp.ndarray,
        a_t: jnp.ndarray,
    ) -> jnp.ndarray:
      """Return the inverse temporald difference learning loss."""
      batched_sarsa_loss = jax.vmap(
          rlax.sarsa, in_axes=(0, None, 0, None, 0, None, None))
      # Collect the φ(s_tm1, a_tm1).
      phi_t_a = phi_t[..., a_tm1]
      loss = batched_sarsa_loss( # [num_cumulants]
          psi_tm1, a_tm1, phi_t_a, discount_t, psi_t, a_t,
          self._stop_target_gradients)
      return jnp.mean(rlax.l2_loss(loss))  # []

    # Parse `transition`.
    s_tm1 = transition.s_tm1['pixels']
    a_tm1 = transition.a_tm1
    r_t = transition.r_t
    s_t = transition.s_t['pixels']
    discount_t = transition.discount_t * self._gamma
    a_t = transition.a_t

    # Calculate the successor features for time-step `t-1`.
    psi_tm1 = self._network_fn(params, s_tm1).successor_features  # [B, N, C, A]
    psi_tm1 = psi_tm1[:, self._demonstrator_index]  # [B, C, A]

    # Calculate the successor features and cumulants for time-step `t-1`.
    network_output_t = self._network_fn(params, s_t)
    psi_t = network_output_t.successor_features
    psi_t = psi_t[:, self._demonstrator_index]  # [B, C, A]
    phi_t = network_output_t.cumulants  # [B, C, A]

    # Calculate and aggregate the loss.
    loss = itd_loss_fn(psi_tm1, a_tm1, phi_t, discount_t, psi_t, a_t)  # [B]
    loss = jnp.mean(loss)  # []

    logging_dict = dict(
        # Model predictions.
        max_psi_tm1=jnp.max(psi_tm1),
        min_psi_tm1=jnp.min(psi_tm1),
        mean_psi_tm1=jnp.mean(psi_tm1),
        max_psi_t=jnp.max(psi_t),
        min_psi_t=jnp.min(psi_t),
        mean_psi_t=jnp.mean(psi_t),
        max_phi_t=jnp.max(phi_t),
        min_phi_t=jnp.min(phi_t),
        mean_phi_t=jnp.mean(phi_t),
        # Batch info.
        mean_discount=jnp.mean(discount_t))

    return parts.LossOutput(loss=loss, aux_data=logging_dict)


class DQNLoss:
  """A `__call__`able DQN loss for a given agent specification."""

  def __init__(self, network_fn, *, gamma: float) -> None:
    """Construct a `__call__`able DQN loss."""
    # Internalise arguments.
    self._network_fn = network_fn
    self._gamma = gamma

  def __call__(
      self,
      params: hk.Params,
      target_params: hk.Params,
      transition: parts.Transition,
  ) -> parts.LossOutput:
    """Return the DQN(λ) loss, evaluated on `params` and `rollout`."""

    # Parse `transition`, all arrays are of shape [B, ...].
    s_tm1 = transition.s_tm1['pixels']
    a_tm1 = transition.a_tm1
    r_t = transition.r_t
    s_t = transition.s_t['pixels']
    discount_t = transition.discount_t * self._gamma

    # Calculate the action values for time-step `t-1`.
    q_tm1 = self._network_fn(params, s_tm1)  # [B, A]

    # Calculate the action values for time-step `t`.
    q_t = self._network_fn(target_params, s_t)  # [B, A]

    # Vectorised operations in order to handle the batch dimension.
    batched_q_learning = jax.vmap(rlax.q_learning)

    # Calculate and aggregate the loss.
    batched_loss = batched_q_learning(q_tm1, a_tm1, r_t, discount_t, q_t)
    loss = jnp.mean(rlax.l2_loss(batched_loss))  # []

    logging_dict = dict(
        # Model predictions.
        max_q=jnp.max(q_tm1),
        min_q=jnp.min(q_tm1),
        mean_q=jnp.mean(q_tm1),
        median_q=jnp.median(q_tm1),
        # Batch info.
        max_r=jnp.max(r_t),
        min_r=jnp.min(r_t),
        mean_r=jnp.mean(r_t),
        mean_discount=jnp.mean(discount_t))

    return parts.LossOutput(loss=loss, aux_data=logging_dict)
