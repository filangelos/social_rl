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
"""Core interfaces and data structures."""

import abc
from typing import Any, Mapping, NamedTuple, Optional, Tuple, Union

import dm_env
import haiku as hk
import jax.numpy as jnp
import ml_collections
import numpy as np
import tree

# Type definitions and aliases.
Config = ml_collections.ConfigDict
Environment = dm_env.Environment
EnvOutput = dm_env.TimeStep
InfoDict = Mapping[str, Any]
PRNGKey = jnp.ndarray
State = Any


class AgentOutput(NamedTuple):
  """Container for holding the agent's/actor's output."""
  action: np.ndarray


class Transition(NamedTuple):
  """Container for holding a SARSD tuple."""
  s_tm1: np.ndarray
  a_tm1: np.ndarray
  r_t: np.ndarray
  s_t: np.ndarray
  discount_t: np.ndarray
  a_t: Optional[np.ndarray] = None


class Rollout(NamedTuple):
  """Container for holding an agent-environment rollout."""
  env_output: EnvOutput  # [T, ...]
  agent_output: AgentOutput  # [T, ...]

  def to_transition(self) -> Transition:
    """Return the parsed SARSD tuple, with shape [T-1, ...]."""
    s_tm1 = tree.map_structure(lambda o: o[:-2], self.env_output.observation)
    a_tm1 = self.agent_output.action[1:-1]
    r_t = self.env_output.reward[1:-1]
    s_t = tree.map_structure(lambda o: o[1:-1], self.env_output.observation)
    discount_t = self.env_output.discount[1:-1]
    a_t = self.agent_output.action[2:]
    return Transition(s_tm1, a_tm1, r_t, s_t, discount_t, a_t)


class LossOutput(NamedTuple):
  """Container for holding the output of a loss function."""
  loss: Any
  aux_data: Mapping[str, Any]


class Agent(abc.ABC):
  """Minimal interface for RL agents."""

  def __init__(self, env: Environment, *, config: Config) -> None:
    """Construct an agent."""
    self._observation_spec = env.observation_spec()
    self._action_spec = env.action_spec()
    assert isinstance(self._action_spec, dm_env.specs.DiscreteArray)
    self._cfg = config

  @abc.abstractmethod
  def should_learn(self, learner_state: State, actor_state: State) -> bool:
    """Whether the agent is ready to call `learner_step`."""

  @abc.abstractmethod
  def initial_params(self, rng_key: PRNGKey) -> hk.Params:
    """Return the agent's initial parameters."""

  @abc.abstractmethod
  def initial_learner_state(self, rng_key: PRNGKey, params: hk.Params) -> State:
    """Return the agent's initial learner state."""

  @abc.abstractmethod
  def initial_actor_state(self, rng_key: PRNGKey) -> State:
    """Return the agent's initial actor state."""

  @abc.abstractmethod
  def actor_step(
      self,
      params: hk.Params,
      env_output: EnvOutput,  # Unbatched.
      actor_state: State,
      rng_key: PRNGKey,
      evaluation: bool,
  ) -> Tuple[AgentOutput, State, InfoDict]:
    """Perform an actor step.

    Args:
      params: The agent's parameters.
      env_output: The environment's **unbatched** output.
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

  @abc.abstractmethod
  def learner_step(
      self,
      params: hk.Params,
      *transitions: Transition,  # [B, ...]
      learner_state: State,
      rng_key: PRNGKey,
  ) -> Tuple[hk.Params, State, InfoDict]:
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


class ReplayBuffer(abc.ABC):
  """Interface for a simple replay buffer."""

  def __init__(self, batch_size: int, *args, **kwargs) -> None:
    """Construct a replay buffer."""
    self._batch_size = batch_size

  @abc.abstractmethod
  def num_samples(self, evaluation: bool = False) -> int:
    """Return the number of available elements."""

  @abc.abstractmethod
  def sample(self) -> Union[Rollout, Transition]:
    """Return a batch of elements."""

  def can_sample(self, evaluation: bool = False) -> Transition:
    """Return `True` if there is a sufficient number of transitions available."""
    return self.num_samples(evaluation) >= self._batch_size

  @abc.abstractmethod
  def add(
      self,
      env_output: EnvOutput,
      agent_output: Optional[AgentOutput] = None,
  ) -> None:
    """Add single-step interaction to the buffer."""


class Loop(abc.ABC):
  """Interface for `run`-able loops."""

  @abc.abstractmethod
  def run(self, rng_key: PRNGKey, num_iterations: int, **kwargs) -> Any:
    """Executes the loop for `num_iterations`, an iteration may refer to, e.g.,
    steps in the environment, complete episodes, etc."""
