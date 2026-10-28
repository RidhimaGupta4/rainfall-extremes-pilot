# src

The `rainpilot` package: the models and diagnostics used by the experiments and notebooks.

| Module | Contents |
|---|---|
| `rainpilot/data.py` | Loading the data, the interleaved split by winter with its leakage check, distances between cells, the Storm Desmond date |
| `rainpilot/margins.py` | Per-cell marginal distributions: empirical, or EVT (empirical below the 95th percentile, generalised Pareto above it, with one shape ξ shared by all cells and fitted by profile likelihood) |
| `rainpilot/generators.py` | The Gaussian copula, the VAE dependence model, and the generator that combines margins with a dependence model |
| `rainpilot/vae.py` | Convolutional VAE on normal-score fields, with early stopping on validation winters and a latent Gaussian mixture for sampling (ex-post density estimation) |
| `rainpilot/diagnostics.py` | Quantiles of area averages, streaming exact quantiles, chi(u) against distance, wet-area and bulk summaries, return periods |
| `rainpilot/experiment.py` | Shared machinery: fitting the 2 × 2 generators, paired sampling with common random numbers, log-ratio bias, the simulated true worlds |

Each experiment script adds `src/` to the Python path itself, so no installation is needed.
