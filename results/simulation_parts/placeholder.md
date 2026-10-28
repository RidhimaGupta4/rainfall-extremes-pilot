# simulation_parts

Checkpoints written by `experiments/simulation_study.py`, so the study can restart where it
stopped. The final results are combined in `results/simulation_study.json`. You do not need these
files to read the results.

| File | Contents |
|---|---|
| `gaussian_truth.json`, `t_truth.json`, `t_varxi_truth.json` | The known truth for each world: the true tail quantiles of single cells and area averages, the true chi(u) against distance, and the true GPD shape ξ (one shared value, or a value per cell in the varying-ξ world) |
| `gaussian_rep0.json` … `gaussian_rep2.json` | Replicates in the Gaussian world |
| `t_rep0.json` … `t_rep2.json` | Replicates in the Student-t world |
| `t_varxi_rep0.json` … `t_varxi_rep2.json` | Replicates in the Student-t world whose tail shape ξ varies from west to east, so the fitted shared-ξ EVT margins are misspecified |

Each replicate file holds the fitted ξ, the size of the VAE's latent mixture, and for each of the
four generators the bias against the truth (mean and mean absolute log ratio) and chi(u).

Deleting a file makes the study recompute that piece.
