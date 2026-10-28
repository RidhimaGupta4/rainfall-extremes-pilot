# rainpilot

The Python package behind the pilot. The experiments and notebooks import it from `src/`.

| Module | Main contents |
|---|---|
| `data.py` | `load()` reads the NetCDF file. `interleaved_split()` assigns whole winters to train / validation / test by start year (0 or 5 → test, 3 or 8 → validation). `Split.check()` raises an error if any day or winter is shared between sets. `cell_distances_km()` gives distances between cells. `DESMOND_DATE` is the held-out storm. |
| `margins.py` | `Margins("evt")`: empirical below each cell's 95th percentile and a generalised Pareto tail above it, with cell-specific scale and one shape ξ shared by all cells (profile likelihood). `Margins("empirical")` cannot exceed the training maximum. Both map rainfall ↔ normal scores, with a randomised probability integral transform for exact zeros. |
| `generators.py` | `GaussianCopula`; `VAEDependence`; `Generator`, which combines margins with a dependence model and samples in batches to keep memory low |
| `vae.py` | `ConvVAE`: convolutional VAE on 16 × 16 normal-score fields with a learned per-pixel decoder variance. `train_vae()` uses early stopping on validation winters, and seeds are independent. `fit_latent_density()` fits a Gaussian mixture to the training encodings, choosing its size on validation winters. |
| `diagnostics.py` | `tile_means` (area averages at 5–80 km); `QuantileAccumulator` (streaming exact quantiles); `chi_by_distance` (chi(u) in contiguous distance bands); `bulk_summaries` (wet-area fraction, correlation, mean map); `return_period_winters` |
| `experiment.py` | `fit_two_by_two` (the four generators with shared components); `paired_quantiles` (common random numbers across margin types); `log_ratio` and `log_ratio_abs` (bias against a reference); `TrueWorld` (Gaussian or Student-t truth for the simulation study) |

All fitting functions are given training data only; the leakage rules are enforced in the
experiment scripts and checked in `tests/`.
