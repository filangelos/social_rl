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
"""Replay buffers and utility functions."""

import glob
import os
from typing import Optional

import numpy as np
import tree
import tqdm

from social_rl import tree_utils
from social_rl.io_utils import load_rollout_from_disk
from social_rl.parts import AgentOutput, EnvOutput, ReplayBuffer, Transition


class DemonstrationsBuffer(ReplayBuffer):
  """Replays experience from near-expert demonstrators."""

  def __init__(
      self,
      data_dir: str,
      train_eval_split: float = 0.9,
      seed: Optional[int] = None,
  ) -> None:
    """Construct a demonstrations replay buffer."""
    assert 0 <= train_eval_split <= 1.0
    self._data_dir = data_dir
    self._train_eval_split = train_eval_split
    self._fnames = glob.glob(os.path.join(self._data_dir, '*.rollout'))
    assert len(self._fnames) > 0
    self._rng = np.random.RandomState(seed)

    # Load all the demonstrations on RAM.
    demos = list()
    for fname in tqdm.tqdm(self._fnames):
      demo = load_rollout_from_disk(fname)
      # Convert rollouts to SARSD tuples.
      demo = demo.to_transition()
      # Append to the buffer.
      demos.append(demo)
    # Stack lists to batched nested objects.
    demos: Transition = tree.map_structure(
        lambda *x: np.concatenate(x, axis=0), *demos)
    # De-`None` the `reward` and `discount` of the initial states.
    demos = demos._replace(
        r_t=np.nan_to_num(demos.r_t, nan=0.0).astype(np.float32),
        discount_t=np.nan_to_num(demos.discount_t, nan=1.0).astype(np.float32))
    # Split demonstrations to `train` and `eval` sets.
    num_demos = tree_utils.length(demos)
    train_indices = self._rng.choice(
        num_demos, size=int(num_demos * self._train_eval_split), replace=False)
    eval_indices = np.zeros(shape=num_demos, dtype=np.bool)
    eval_indices[~train_indices] = 1.0
    self._demos = dict(
        train=tree_utils.slice(demos, train_indices),
        eval=tree_utils.slice(demos, eval_indices))
    # Cache the size of each dataset.
    self._num_demos = dict(
        train=tree_utils.length(self._demos['train']),
        eval=tree_utils.length(self._demos['eval']))

  def num_samples(self, evaluation: bool = False) -> int:
    """Return the number of available elements."""
    return self._num_demos['eval' if evaluation else 'train']

  def sample(self, batch_size: int, evaluation: bool = False) -> Transition:
    """Return a batch if transitions."""
    assert self.can_sample(batch_size)

    indices = self._rng.choice(
        a=self.num_samples(evaluation), size=batch_size, replace=False)
    return tree.map_structure(
        lambda x: x[indices].copy(),
        self._demos['eval' if evaluation else 'train'])

  def can_sample(self, batch_size: int, evaluation: bool = False) -> Transition:
    """Return `True` if there is a sufficient number of transitions available."""
    return self.num_samples(evaluation) >= batch_size

  def add(
      self,
      env_output: EnvOutput,
      agent_output: Optional[AgentOutput] = None,
  ) -> None:
    """Raise not implemented error, since the demonstrations are loaded
    offline."""
    del env_output, agent_output
    raise NotImplementedError(self)
