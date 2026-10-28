# cumbria_parts

Checkpoints written by `experiments/cumbria_experiment.py`, so an interrupted run can restart where
it stopped. The final results are combined in `results/cumbria_experiment.json`. You do not need
these files to read the results.

| File | Contents |
|---|---|
| `seed0.json`, `seed1.json`, `seed2.json` | Results for each VAE training seed: test-winter evaluation, comparisons with the training winters, realism metrics, chi, normal-score exceedance and Desmond hit counts. The Gaussian copula has no training randomness, so it is in `seed0.json` only. |
| `vae_seed0.pt`, `vae_seed1.pt`, `vae_seed2.pt` | The trained VAE dependence models, fitted on training winters only. `experiments/desmond_analogues.py` uses `vae_seed0.pt`. |
| `candidates_seed0.npz`, `candidates_seed1.npz`, `candidates_seed2.npz` | Generated normal-score fields that could reach Storm Desmond's size under any bootstrap refit of the margins. They are kept so that the Desmond bootstrap can re-evaluate the same draws exactly. |

Deleting a seed's files makes the experiment recompute that seed.
