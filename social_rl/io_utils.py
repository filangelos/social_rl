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
"""Input/Output utility functions."""

import datetime
import os
import pickle
import uuid

import tree

from social_rl.parts import Rollout


def save_rollout_to_disk(rollout: Rollout, output_dir: str) -> str:
  """Store the `rollout` to disk and return its filename."""
  timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
  identifier = str(uuid.uuid4().hex)
  sequence_length = tree.flatten(rollout)[0].shape[0]
  fname = os.path.join(
      output_dir,
      '{}-{}-{}.rollout'.format(timestamp, identifier, sequence_length))
  with open(fname, 'wb') as f:
    pickle.dump(rollout, f)
  return fname


def load_rollout_from_disk(fname: str) -> Rollout:
  """Load and parse `rollout` from the `fname`."""
  with open(fname, 'rb') as f:
    rollout = pickle.load(f)
    assert isinstance(rollout, Rollout)
  return rollout
