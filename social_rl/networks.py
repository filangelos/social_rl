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
"""Neural network modules shared across the agents."""

from typing import Optional, Sequence

import chex
import haiku as hk
import jax
import jax.numpy as jnp

uniform_initializer = hk.initializers.UniformScaling(scale=0.333)


class GridWorldConvEncoder(hk.Module):
  """A simple, light-weight convolution encoder used for `GridWorld` tasks."""

  def __call__(self, pixels_observation: jnp.ndarray) -> jnp.ndarray:
    """Transform the RGB observation to a flat vector.

    Args:
      pixels_observation: The **batched** pixels observation, with shape
        [B, H, W, C].

    Returns:
      The transformed observation, with shape [B, D].
    """
    chex.assert_rank(pixels_observation, 4)

    # Apply convolutions.
    embedding = hk.Conv2D(
        output_channels=32, kernel_shape=(8, 8), stride=4)(pixels_observation)
    embedding = jax.nn.relu(embedding)
    embedding = hk.Conv2D(
        output_channels=64, kernel_shape=(4, 4), stride=2)(embedding)
    embedding = jax.nn.relu(embedding)
    embedding = hk.Conv2D(
        output_channels=64, kernel_shape=(3, 3), stride=1)(embedding)
    embedding = jax.nn.relu(embedding)

    # Flatten embedding to vector.
    embedding = hk.Flatten()(embedding)

    return embedding  # [B, D]


class LayerNormMLP(hk.Module):
  """Simple feedforward MLP torso with initial layer-norm.

  This module is an MLP which uses LayerNorm (with a tanh normalizer) on the
  first layer and non-linearities (elu) on all but the last remaining layers.

  Borrowed from dm-acme.
  """

  def __init__(
      self,
      output_sizes: Sequence[int],
      activate_final: bool = False,
      name: Optional[str] = None,
  ) -> None:
    """Construct the MLP.

    Args:
      layer_sizes: a sequence of ints specifying the size of each layer.
      activate_final: whether or not to use the activation function on the final
        layer of the neural network.
    """
    super().__init__(name=name)
    self._output_sizes = output_sizes
    self._activate_final = activate_final

  def __call__(self, embedding: jnp.ndarray) -> jnp.ndarray:
    """Forwards the policy network."""
    # Apply the first linear layer.
    embedding = hk.Linear(
        self._output_sizes[0], w_init=uniform_initializer)(embedding)
    # Apply the normalisation layer.
    embedding = hk.LayerNorm(
        axis=-1, create_scale=True, create_offset=True)(embedding)
    embedding = jax.nn.tanh(embedding)
    # Apply the MLP module.
    embedding = hk.nets.MLP(
        output_sizes=self._output_sizes[1:],
        w_init=uniform_initializer,
        activation=jax.nn.elu,
        activate_final=self._activate_final)(embedding)
    return embedding