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
"""Configuration for the `collect_rollouts` experiment."""

import ml_collections


def get_config(goal_color: str) -> ml_collections.ConfigDict:
  """Return the configuration file."""
  config = ml_collections.ConfigDict()
  config.num_rollouts = 1_000
  config.epsilon = 0.05
  config.goal_color = goal_color
  config.seed = 42
  config.output_dir = 'data'

  return config