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
"""Environment used in the experiments."""

from typing import Sequence
from typing import Optional
from typing import Tuple

import dm_env
from dm_env import specs
import numpy as np

Position = Tuple[int, int]


class GridWorld(dm_env.Environment):
  """A toy ridworld environment."""

  def __init__(
      self,
      goal_color: str = 'red',
      seed: Optional[int] = None,
  ) -> None:
    """Build the four rooms environment.

    Args:
      goal_color: One of {'red', 'green', 'blue'}.

    Raises:
      ValueError: If the `n` specified is too small, or if the goal is
        misplaced.
    """
    assert goal_color in ('red', 'green', 'blue')

    # Setup the static board.
    board = [
        '*********',
        '*       *',
        '*       *',
        '*   *   *',
        '*  ***  *',
        '*   *   *',
        '*       *',
        '*       *',
        '*********',
    ]
    board = np.asarray(
        [[{
            '*': 1,
            ' ': 0
        }[char] for char in line] for line in board]).astype(np.float32)

    actions = [np.array(x) for x in [(0, -1), (0, 1), (-1, 0), (1, 0), (0, 0)]]

    # This environment is tabular, so make a mapping from states <--- integers.
    n, m = board.shape
    self._cell_to_state = {}
    state_num = 0
    for i in range(n):
      for j in range(m):
        if board[i, j] == 0:
          self._cell_to_state[(i, j)] = state_num
          state_num += 1

    self._state_to_cell = {v: k for k, v in self._cell_to_state.items()}

    # Store the current state and miscellaneous.
    self._wind_prob = 0.0  # Make it explicit that the environment is deterministic.
    self._rng = np.random.RandomState(seed)
    self._goal_color = goal_color
    self._num_states = state_num
    self._num_actions = len(actions)
    self._actions = actions
    self._board = board
    self._reset_next_step = True
    self._timeout = 25
    self._episode_len = 0
    self._rng = np.random.RandomState(seed)
    self._reward_per_step = 0.0
    self._goal_reward = 1.0

    # Initialise the agent and goals randomly.
    self._init_random_positions()

  def reset(self) -> dm_env.TimeStep:
    """Resets the environment."""
    self._reset_next_step = False
    self._episode_len = 0

    # Initialise the agent and goals randomly.
    self._init_random_positions()

    return dm_env.restart(observation=self._observation())

  def step(self, action: int) -> dm_env.TimeStep:
    """Performs one step in the environment."""

    # Reset if terminal
    if self._reset_next_step:
      return self.reset()

    self._episode_len += 1

    offset = self._actions[action]

    # Attempts to move the agent.
    new_pos = self._agent_pos + offset
    if not self._board[new_pos[0], new_pos[1]]:
      self._agent_pos = new_pos

    # Generate the observation.
    observation = self._observation()

    # If we landed on the goal, terminate the episode and give reward.
    reward, discount = self._reward_and_discount()
    if discount == 0.0:
      self._reset_next_step = True
      return dm_env.termination(reward=reward, observation=observation)

    # Terminate the episode if we reached the timeout.
    if self._episode_len == self._timeout:
      self._reset_next_step = True
      return dm_env.truncation(reward=reward, observation=observation)

    # Otherwise the episode continues.
    return dm_env.transition(reward=reward, observation=observation)

  def observation_spec(self) -> specs.Array:
    pixels_spec = specs.BoundedArray(
        shape=self._board.shape + (3,),
        dtype=np.float32,
        minimum=0.0,
        maximum=1.0,
        name="pixels_observation")
    tabular_spec = specs.DiscreteArray(
        num_values=self._num_states, name="tabular_observation")
    return dict(pixels=pixels_spec, tabular=tabular_spec)

  def action_spec(self) -> specs.DiscreteArray:
    return specs.DiscreteArray(num_values=self._num_actions, name="action")

  def reward_spec(self) -> specs.BoundedArray:
    min_reward = min(self._reward_per_step, self._goal_reward)
    max_reward = max(self._reward_per_step, self._goal_reward)
    return specs.BoundedArray(
        shape=(),
        dtype=float,
        name="reward",
        minimum=min_reward,
        maximum=max_reward)

  def _pixels(self) -> np.ndarray:
    """Builds a [H, W, C] RGB pixel representation of the GridWorld state."""

    # 'Maze' in all channels (white).
    black_to_white_board = np.logical_not(self._board.astype(np.bool)).astype(
        np.float32)
    obs = np.expand_dims(black_to_white_board, axis=-1)
    obs = np.tile(obs, [1, 1, 3])

    # 'Goal' in blue channel.
    for color, goal_pos in self._goals_pos.items():
      goal_i, goal_j = goal_pos
      obs[goal_i, goal_j] = dict(
          red=(1.0, 0.0, 0.0), green=(0.0, 1.0, 0.0),
          blue=(0.0, 0.0, 1.0))[color]

    agent_i, agent_j = self._agent_pos
    # Yellow color when a goal is hit.
    if not (obs[agent_i, agent_j] == np.array([1.0, 1.0, 1.0])).all():
      obs[agent_i, agent_j] = (1.0, 1.0, 0.0)
    else:  # 'Agent' in gray color.
      obs[agent_i, agent_j] = 0.5

    return obs

  def _reward_and_discount(self):
    # Check if any goal was reached.
    reward = self._reward_per_step
    discount = 1.0
    for color, goal_pos in self._goals_pos.items():
      if (self._agent_pos == goal_pos).all():
        reward = self._goal_reward if color == self._goal_color else -self._goal_reward
        discount = 0.0
        break
    return reward, discount

  def _observation(self):
    return dict(
        pixels=self._pixels(),
        tabular=self._cell_to_state[tuple(self._agent_pos)])

  def _init_random_positions(self) -> None:
    """Place the agent, and the goals randomly on the board."""
    self._agent_pos = self._place_randomly(occupied_cells=())
    self._goals_pos = dict()
    self._goals_pos['red'] = self._place_randomly(
        occupied_cells=(self._agent_pos,))
    self._goals_pos['green'] = self._place_randomly(
        occupied_cells=(self._agent_pos, self._goals_pos['red']))
    self._goals_pos['blue'] = self._place_randomly(
        occupied_cells=(
            self._agent_pos, self._goals_pos['red'], self._goals_pos['green']))

  def _place_randomly(
      self,
      occupied_cells: Sequence[Position],
  ) -> Position:
    """Return a random, unoccupied cell position."""
    found = False
    occupied_states = [
        self._cell_to_state[(cell[0], cell[1])] for cell in occupied_cells
    ]
    while not found:
      random_state = self._rng.randint(self._num_states)
      found = random_state not in occupied_states
    return np.array(self._state_to_cell[random_state])
