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
"""Loss functions used by social RL agents."""

from typing import Callable

import chex
import distrax
import haiku as hk
import jax
import jax.numpy as jnp

from social_rl import parts


class BCLoss:
  """A `__call__`-able behavioural cloning loss."""

  def __init__(
      self,
      network_fn: Callable[[hk.Params, jnp.ndarray], jnp.ndarray],
  ) -> None:
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