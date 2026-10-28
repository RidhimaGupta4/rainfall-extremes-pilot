# experiments

The scripts that produce everything in `results/`. Each saves its progress, so it can be stopped and
restarted.

| Script | What it does | Time (2-core CPU) |
|---|---|---|
| `cumbria_experiment.py` | Fits the four generators (Gaussian copula or VAE, with empirical or EVT margins) on the training winters. Evaluates them on the held-out test winters, with training- and validation-winter baselines. Estimates the probability of a Storm Desmond-sized day, with block-bootstrap intervals over whole winters. | about 1.5 hours |
| `desmond_analogues.py` | Draws Desmond-sized days and the most extreme days from each generator for the figure in notebook 03. Run it after `cumbria_experiment.py`. | a few minutes |
| `simulation_study.py` | Controlled simulation study with a known truth in three worlds: Gaussian, Student-t, and Student-t with a tail shape that varies across the region | about 1 hour per world |

    python experiments/cumbria_experiment.py
    python experiments/desmond_analogues.py
    python experiments/simulation_study.py --reps 3 --worlds gaussian
    python experiments/simulation_study.py --reps 3 --worlds t
    python experiments/simulation_study.py --reps 3 --worlds t_varxi

**Leakage rule.** Every fitted quantity uses training winters only. Validation winters are used only
for VAE early stopping and choosing the size of the latent mixture. Test winters are used only for
evaluation.
