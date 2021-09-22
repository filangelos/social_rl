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
"""Utility functions for nested objects."""

from typing import Any, Sequence

import numpy as np
import tree

NestedObject = Any

from social_rl.parts import LossOutput


def length(nested_array: NestedObject) -> int:
  """Return the length (i.e., first batch dimension) of `nested_array`."""
  return tree.flatten(nested_array)[0].shape[0]


def slice(nested_array: NestedObject, indices: Sequence[int]) -> NestedObject:
  """Return the sliced `nested_array`."""
  return tree.map_structure(lambda x: x[indices], nested_array)


def to_numpy(nested_array: NestedObject) -> NestedObject:
  """Return a `nested_array`, casted to NumPy."""
  return tree.map_structure(np.array, nested_array)


def stack(list_of_nested_arrays: Sequence[NestedObject]) -> NestedObject:
  """Return a `nested_array` by stacking the `list_of_nested_arrays`."""
  return tree.map_structure(lambda *x: np.stack(x), *list_of_nested_arrays)


def merge_loss_outputs(**loss_outputs) -> LossOutput:
  """Return a merged `LossOutput`.

  Sample usage:
  >>> from social_rl import parts
  >>> from social_rl.tree_utils import merge_loss_outputs
  >>> actor_loss_output = parts.LossOutput(loss=-1.3, aux_data=dict(stddev=0.0))
  >>> critic_loss_output = parts.LossOutput(
  ...     loss=2.1, aux_data=dict(q_max=4.0, q_min=2.7))
  >>> print(merge_loss_output(actor=actor_loss_output, critic=critic_loss_output))
  >>> # LossOutput(loss=0.8, aux_data={'actor/stddev': 0.0, 'actor/loss': -1.3,
  ... #     'critic/q_max': 4.0, 'critic/q_min': 2.7, 'critic/loss': 2.1})

  Args:
    **loss_outputs: Keyword arguments, whose **unique** keys are used as prefix
      for the `aux_data` property.

  Returns:
    The merged loss output, whose `loss` property is the sum of the `loss`es of
    the individual loss outputs and its `aux_data` is prefixed according to the
    provided keywords.
  """
  total_loss = 0.0
  aux_data = dict()
  for prefix, loss_output in loss_outputs.items():
    total_loss += loss_output.loss
    aux_data.update(
        {
            '{}/{}'.format(prefix, key): value
            for key, value in loss_output.aux_data.items()
        })
    aux_data['{}/loss'.format(prefix)] = loss_output.loss
  return LossOutput(loss=total_loss, aux_data=aux_data)
