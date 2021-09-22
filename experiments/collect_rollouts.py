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
"""Scirpt for generating near-expert trajectories."""

import os

import tqdm
from absl import app
from absl import flags
from absl import logging
from ml_collections.config_flags import config_flags

from social_rl import io_utils
from social_rl.environments import GridWorld
from social_rl.tabular_utils import expert_rollout

logging.set_verbosity(logging.DEBUG)
FLAGS = flags.FLAGS
config_flags.DEFINE_config_file('config')


def main(argv):
  # Debugging purposes.
  logging.debug(argv)
  logging.debug(FLAGS)

  # Parses command line arguments.
  config = FLAGS.config
  output_dir = config.output_dir
  num_rollouts = config.num_rollouts
  epsilon = config.epsilon
  goal_color = config.goal_color
  seed = config.seed

  # Prepare the output directory.
  here = os.path.abspath(os.path.dirname(__file__))
  output_dir = os.path.join(
      here, os.pardir, output_dir, config.goal_color, str(config.epsilon))
  os.makedirs(output_dir, exist_ok=True)
  print(output_dir)

  def env_builder(seed: int) -> GridWorld:
    """Return a **randomly** initialised environment."""
    return GridWorld(goal_color=goal_color, seed=seed)

  for step in tqdm.trange(num_rollouts):
    rollout_seed = seed + step
    # Initialize new environment.
    env = env_builder(seed=rollout_seed)
    rollout = expert_rollout(env, epsilon=epsilon, seed=rollout_seed)
    io_utils.save_rollout_to_disk(rollout, output_dir)


if __name__ == "__main__":
  flags.mark_flag_as_required("config")
  app.run(main)
