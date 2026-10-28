# results

The saved outputs of the experiments. The notebooks read these files, so they run in seconds
without rerunning anything.

| File | Produced by | Contents |
|---|---|---|
| `cumbria_experiment.json` | `experiments/cumbria_experiment.py` | Real-data results: test-winter references and their bootstrap intervals; training- and validation-winter baselines; tail quantiles against test and training winters; calibration; realism metrics; chi; normal-score exceedance and its noise reference; Storm Desmond probabilities with Monte Carlo and tail-uncertainty intervals; bootstrap ξ |
| `desmond_analogues.json` | `experiments/desmond_analogues.py` | Desmond-sized days and most extreme days from each generator (400,000 draws each) |
| `simulation_study.json` | `experiments/simulation_study.py` | Simulation study: bias of each generator against the known truth in three true worlds, 3 replicates each |
| `cumbria_parts/` | `experiments/cumbria_experiment.py` | Checkpoints: per-seed results, the trained VAEs (`vae_seed*.pt`) and the screened candidate draws for the Desmond bootstrap |
| `simulation_parts/` | `experiments/simulation_study.py` | Checkpoints: the truth for each world and each finished replicate |

The checkpoints let an interrupted run restart where it stopped. Deleting a file in `*_parts/`
makes the experiment recompute that piece.
