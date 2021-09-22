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
"""Utility functions for plotting."""

import matplotlib.pyplot as plt

# Configure `matplotlib` style.
try:
  plt.style.use(
      'https://github.com/filangel/mplstyles/raw/main/matlab.mplstyle')
  COLORS = {
      'black': 'k',
      'white': 'w',
      'blue': 'C0',
      'orange': 'C1',
      'yellow': 'C2',
      'purple': 'C3',
      'green': 'C4',
      'light_blue': 'C5',
      'red': 'C6'
  }
except:
  print("Unable to load 'matlab.mplstyle' from github.com/filangelos.")
  COLORS = {
      'black': 'k',
      'white': 'w',
      'blue': 'b',
      'orange': '#ff7f0e',
      'yellow': 'y',
      'purple': 'm',
      'green': 'g',
      'light_blue': 'c',
      'red': 'r'
  }


def plot_learner_buffer_loop_stats(stats):
  """Visualise the `stats` returned from
  `social_rl.loops.LearnerBufferLoop.run`."""
  figax = list()
  first_key = list(stats.keys())[0]
  stats_keys = stats[first_key]['stats'].keys()
  for stat_key in stats_keys:
    fig, ax = plt.subplots()
    for type_key, logging_dict in stats.items():
      ax.plot(
          logging_dict['step'], logging_dict['stats'][stat_key], label=type_key)
    ax.legend()
    ax.set(xlabel='Learner Steps', ylabel=stat_key)
    figax.append((fig, ax))
  return figax


def plot_agent_environment_loop_stats(stats):
  """Visualise the `stats` returned from
  `social_rl.loops.AgentEnvironmentLoop.run`."""
  import pandas as pd
  fig, ax = plt.subplots()
  rolling_window = 50
  x_axis = pd.Series(stats['episode_metrics']['train']['step'])
  y_axis = pd.Series(stats['episode_metrics']['train']['stats']['returns'])
  x_axis = x_axis.rolling(rolling_window).mean()
  y_axis = y_axis.rolling(rolling_window).mean()
  ax.plot(x_axis, y_axis, label='train')
  ax.legend()
  ax.set(xlabel='Number of Steps', ylabel='Episode Returns', ylim=(-1.05, 1.05))
  try:
    from IPython.display import display
    display(fig)
    plt.close()
  except:
    pass