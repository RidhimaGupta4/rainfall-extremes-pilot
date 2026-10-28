# notebooks

Executed notebooks that walk through the pilot. Read them in order. Each opens on Google Colab
and reads the saved `results/`, so no experiment needs rerunning. Set `RUN = True` to rerun one.

| Notebook | Contents |
|---|---|
| `01_data_overview.ipynb` | The data, the pre-specified split, Storm Desmond, the trend in winter rainfall, spatial dependence |
| `02_simulation_study.ipynb` | The four generators against known tails in three true worlds, including one where the shared tail shape is wrong |
| `03_cumbria_generators.ipynb` | Real data: why the test winters cannot rank tail models, tail quantiles against test and training winters, realism, joint extremes, and how likely Storm Desmond was |

The notebooks are generated from `build_notebooks.py` in the repository root.
