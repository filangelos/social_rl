# Social Reinforcement Learning

## Installation

```bash
pip install -e .[dev]
```

## Notes

This repository is under **heavy** refactoring, please do not raise issues unless you catch any breaking bugs.
The list of changes that are under consideration are the following:

- [ ] Remove any `dm-reverb` dependencies.
- [ ] Host demonstrations on an Oxford server for guaranteeing reproducibility.
- [ ] Un-`.gitignore` the `experiments/run_psiphi.ipynb` after testing sensitivity to the `l1_loss_coef` parameter.
- [ ] Add `stop_successor_features_gradients` parameter to the `ITDLoss`.
- [ ] Consider trading scalability with readability for the `loss_output`s in the `_loss_fn`s.
- [ ] Remove any `markov` dependencies.