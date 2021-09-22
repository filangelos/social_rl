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
