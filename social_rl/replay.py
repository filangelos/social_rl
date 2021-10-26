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
"""Replay buffers and utility functions."""

import collections
import copy
import glob
import os
import random
from typing import Mapping, Optional, Sequence, Union

import numpy as np
import tree
import tqdm

from social_rl import parts
from social_rl import tree_utils
from social_rl.io_utils import load_rollout_from_disk


class ExperienceBuffer(parts.ReplayBuffer):
  """Replay ego experience."""

  def __init__(
      self,
      batch_size: int,
      capacity: int,
      seed: Optional[int] = None,
  ) -> None:
    """Construct a minimal replay buffer for onlin RL agents."""
    super().__init__(batch_size=batch_size)
    self._capacity = capacity
    random.seed(seed)
    self._buffer = collections.deque(maxlen=self._capacity)
    self._prev_env_output = None

  def num_samples(self, evaluation: bool = False) -> int:
    """Return the number of available elements."""
    assert not evaluation
    return len(self._buffer)

  def add(
      self,
      env_output: parts.EnvOutput,
      agent_output: Optional[parts.AgentOutput] = None,
  ) -> None:
    """Append the agent-environment interaction to the buffer."""

    if agent_output is not None:  # Enter this when not in first step.
      transition = parts.Transition(
          s_tm1=self._prev_env_output.observation,
          a_tm1=agent_output.action,
          r_t=env_output.reward,
          s_t=env_output.observation,
          discount_t=env_output.discount,
          a_t=np.array(-1.0, dtype=np.float32))  # Dummy a_t.
      transition = tree_utils.to_numpy(transition)
      transition = tree.map_structure(copy.deepcopy, transition)
      self._buffer.append(transition)

    # Cache environment output for next timestep.
    self._prev_env_output = env_output

  def sample(self, evaluation: bool = False) -> parts.Transition:
    """Return a batch if transitions."""
    assert self.can_sample(evaluation)

    unbatched_sample = random.sample(self._buffer, self._batch_size)
    return tree_utils.stack(unbatched_sample)


class DemonstrationsBuffer(parts.ReplayBuffer):
  """Replays experience from near-expert demonstrators."""

  def __init__(
      self,
      batch_size: int,
      data_dir: str,
      train_eval_split: float = 0.9,
      seed: Optional[int] = None,
  ) -> None:
    """Construct a demonstrations replay buffer."""
    super().__init__(batch_size=batch_size)
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
    demos: parts.Transition = tree.map_structure(
        lambda *x: np.concatenate(x, axis=0), *demos)
    # De-`None` the `reward` and `discount` of the initial states.
    demos = demos._replace(
        r_t=np.nan_to_num(demos.r_t, nan=0.0).astype(np.float32),
        discount_t=np.nan_to_num(demos.discount_t, nan=1.0).astype(np.float32))
    # Split demonstrations to `train` and `eval` sets.
    num_demos = tree_utils.length(demos)
    train_indices = self._rng.choice(
        num_demos, size=int(num_demos * self._train_eval_split), replace=False)
    eval_indices = np.ones(shape=num_demos, dtype=np.bool)
    eval_indices[train_indices] = 0.0
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

  def sample(self, evaluation: bool = False) -> parts.Transition:
    """Return a batch if transitions."""
    assert self.can_sample(evaluation)

    indices = self._rng.choice(
        a=self.num_samples(evaluation), size=self._batch_size, replace=False)
    return tree.map_structure(
        lambda x: x[indices].copy(),
        self._demos['eval' if evaluation else 'train'])

  def add(
      self,
      env_output: parts.EnvOutput,
      agent_output: Optional[parts.AgentOutput] = None,
  ) -> None:
    """Raise not implemented error, since the demonstrations are loaded
    offline."""
    del env_output, agent_output
    raise NotImplementedError(self)


class EgoOthersBuffer(parts.ReplayBuffer):
  """Replays both ego and others' experience."""

  def __init__(
      self,
      *,
      ego_buffer: ExperienceBuffer,
      others_buffers: Sequence[Union[ExperienceBuffer, DemonstrationsBuffer]],
  ) -> None:
    """Construct an experience replay buffer that samples from an `ego` buffer and """
    assert len(others_buffers) > 0
    self._ego_buffer = ego_buffer
    self._others_buffers = others_buffers

  def num_samples(
      self,
      evaluation: bool = False) -> Mapping[str, Union[int, Sequence[int]]]:
    """Return the number of available elements."""
    return dict(
        ego_buffer=self._ego_buffer.num_samples(evaluation),
        others_buffers=[
            rb.num_samples(evaluation) for rb in self._others_buffers
        ])

  def sample(
      self,
      evaluation: bool = False
  ) -> Sequence[Union[parts.Rollout, parts.Transition]]:
    """Return samples from all the buffers."""
    ego_sample = self._ego_buffer.sample(evaluation)
    others_samples = tuple(
        [rb.sample(evaluation) for rb in self._others_buffers])
    return (ego_sample,) + others_samples

  def can_sample(self, evaluation: bool = False) -> bool:
    """Return `True` if there is a sufficient number of transitions available."""
    return all(
        [
            rb.can_sample(evaluation)
            for rb in type(self._others_buffers)([self._ego_buffer]) +
            self._others_buffers
        ])

  def add(
      self,
      env_output: parts.EnvOutput,
      agent_output: Optional[parts.AgentOutput] = None,
  ) -> None:
    """Append the agent-environment interaction to the ego buffer."""
    self._ego_buffer.add(env_output, agent_output)
