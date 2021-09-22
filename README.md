# PsiPhi-Learning and Inverse Temporal Difference Learning

## Installation

```bash
export ENV_NAME=social_rl
conda create -n $ENV_NAME python=3.7 cudatoolkit=11.1 -c conda-forge -y
conda activate $ENV_NAME
pip install -e .
pip install --upgrade "jax[cuda111]" -f https://storage.googleapis.com/jax-releases/jax_releases.html
# Development tools.
pip install -r requirements-dev.txt
```
