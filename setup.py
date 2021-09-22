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

import os

from setuptools import find_packages
from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))

# Get the long description from the README.mf file
with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
  long_description = f.read()

# Get the core dependencies from `requirements.txt`.
with open(os.path.join(here, "requirements.txt"), encoding="utf-8") as f:
  requirements = [
      l.rstrip() for l in f if not (l.isspace() or l.startswith("#"))
  ]

setup(
    name="social_rl",
    version="0.0.1",
    description="PsiPhi-Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/filangelos/social_rl",
    author="Angelos Filos",
    author_email="filos.angel@gmail.com",
    license="Apache License, Version 2.0",
    packages=find_packages(),
    install_requires=requirements,
    tests_require=[],
    extras_require={},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)