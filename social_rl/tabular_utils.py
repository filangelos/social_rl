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
"""Utility functions for tabular expriments."""

import copy
from typing import Callable, NamedTuple, Optional, Tuple

import numpy as np
import tree

from social_rl.environments import GridWorld
from social_rl.parts import AgentOutput, EnvOutput, Rollout


class MDP(NamedTuple):
  """Container for holding the parametrisation of Markov decision process."""
  rewards: np.ndarray  # [S, A]
  transition_probs: np.ndarray  # [S, A, S]


# A function that converts any state function to a cell function.
StateToCellFn = Callable[[np.ndarray], np.ndarray]


def gridworld_env_to_tabular(env: GridWorld) -> Tuple[MDP, StateToCellFn]:
  """Convert a sample-able `env` to a tabular MDP."""
  assert env._wind_prob == 0.0
  assert not env._reset_next_step

  num_states = env._num_states
  num_actions = env._num_actions

  def step(s_tm1: int, a_tm1: int) -> Tuple[int, float]:
    """Perform a step in the environment, assuming that `env` is deterministic."""
    cloned_env = copy.deepcopy(env)
    cloned_env._timeout = int(1e6)
    # Set agent's position at `s_tm1`.
    pos_tm1 = cloned_env._state_to_cell[s_tm1]
    # If spawned on a goal, treat this as a draining state.
    spawned_on_goal = False
    for _, goal_pos in cloned_env._goals_pos.items():
      if (pos_tm1 == goal_pos).all():
        r_t = 0.0
        s_t = s_tm1
        spawned_on_goal = True
        break
    if not spawned_on_goal:
      cloned_env._agent_pos = cloned_env._state_to_cell[s_tm1]
      time_step = cloned_env.step(a_tm1)
      s_t = time_step.observation['tabular']
      r_t = time_step.reward
    return s_t, r_t

  # Containers for holding the MDP representation.
  rewards = np.zeros(shape=(num_states, num_actions), dtype=np.float32)
  transition_probs = np.zeros(
      shape=(num_states, num_actions, num_states), dtype=np.float32)

  for s_tm1 in range(num_states):
    for a_tm1 in range(num_actions):
      s_t, r_t = step(s_tm1, a_tm1)
      transition_probs[s_tm1, a_tm1, s_t] = 1.0
      rewards[s_tm1, a_tm1] = r_t

  # Make sure that there is no pointer leakage.
  state_to_cell = copy.deepcopy(env._state_to_cell)

  def state_to_cell_fn(state_fn: np.ndarray) -> np.ndarray:
    """A generic function that converts any state function to cell function."""
    cell_fn = np.zeros_like(env._board)
    for s in range(num_states):
      cell_fn[state_to_cell[s]] = state_fn[s]
    return cell_fn

  return MDP(rewards, transition_probs), state_to_cell_fn


def q_policy_iteration(
    mdp: MDP,
    agent_discount: float = 0.99,
    tolerance: float = 1e-4,
) -> np.ndarray:
  """Run policy iteration on a tabular MDP.

  Adapted from github.com/google-research/rl_metrics_aaai2021.

  Args:
    mdp: The tabular MDP.
    agent_discount: The agent's discount factor.
    tolerance: Evaluation stops when the value function change is less than
      the tolerance.

  Returns:
    Numpy array with Q^{*}.
  """
  rewards = mdp.rewards
  transition_probs = mdp.transition_probs
  num_states, num_actions = rewards.shape

  q_values = np.zeros(shape=(num_states, num_actions))
  # Random policy
  policy = np.ones(shape=(num_states, num_actions)) / num_actions
  policy_stable = False
  while not policy_stable:
    # Policy evaluation
    while True:
      delta = 0.
      for s in range(num_states):
        v = rewards[s, :] + agent_discount * np.matmul(
            transition_probs[s, :, :], np.sum(q_values * policy, axis=1))
        delta = max(delta, np.max(abs(v - q_values[s])))
        q_values[s] = v
      if delta < tolerance:
        break
    # Policy improvement
    policy_stable = True
    for s in range(num_states):
      old = policy[s].copy()
      greedy_actions = np.argwhere(q_values[s] == np.amax(q_values[s]))
      for a in range(num_actions):
        if a in greedy_actions:
          policy[s, a] = 1 / len(greedy_actions)
        else:
          policy[s, a] = 0
      if not np.array_equal(policy[s], old):
        policy_stable = False

  return q_values


def expert_rollout(
    env: GridWorld,
    epsilon: float,
    seed: Optional[int] = None,
) -> Rollout:
  """Return a near-expert `rollout` for environment `env` by converting the
  environment to a tabular MDP and using policy iteration for solving the
  tabular MDP.

  Args:
    env: The environemnt to solve.
    epsilon: The expert's sub-optimality, in [0.0, 1.0].
    seed: The random number generator's seed.

  Returns:
    An `epsilon`-optimal trajectory.
  """
  assert 0 <= epsilon <= 1
  assert env._reset_next_step
  rng = np.random.RandomState(seed)

  # Output container.
  rollout = list()

  # Reset environment and reset the buffer.
  env_output = env.reset()
  dummy_agent_output = AgentOutput(action=-1)
  rollout.append(Rollout(env_output, dummy_agent_output))

  # Convert environment to tabular MDP.
  mdp, _ = gridworld_env_to_tabular(env)
  q_star = q_policy_iteration(mdp, agent_discount=0.9, tolerance=1e-4)
  pi_star = np.argmax(q_star, axis=1)  # [S]

  def agent(env_output: EnvOutput) -> AgentOutput:
    """Sample action from the `epsilon`-optimal tabular policy."""
    if rng.uniform() < epsilon:
      action = rng.randint(env._num_actions)
    else:
      action = pi_star[env_output.observation['tabular']]
    return AgentOutput(action=action)

  # Run policy on environment and collect trajectory.
  while not env_output.last():
    agent_output = agent(env_output)
    env_output = env.step(agent_output.action)
    rollout.append(Rollout(env_output, agent_output))

  # Stack the list of objects.
  rollout = tree.map_structure(lambda *x: np.stack(x), *rollout)

  return rollout  # [T, ...]
