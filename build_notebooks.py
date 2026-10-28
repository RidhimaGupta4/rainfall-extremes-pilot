"""Builds the project notebooks from source. Run: python build_notebooks.py"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).parent
META = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
        "language_info": {"name": "python"}}

SETUP = '''# Setup: works locally (from the repository root or notebooks/) and on Google Colab
import sys, os
from pathlib import Path
if "google.colab" in sys.modules and not Path("rainfall-extremes-pilot").exists():
    !git clone -q https://github.com/RidhimaGupta4/rainfall-extremes-pilot.git
    os.chdir("rainfall-extremes-pilot")
    !pip install -q -r requirements.txt
ROOT = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT / "src"))
DATA = ROOT / "data" / "cumbria_winter_rainfall_5km.nc"
RESULTS = Path(os.environ.get("RAINPILOT_RESULTS", ROOT / "results"))
FIGS = ROOT / "figures"; FIGS.mkdir(exist_ok=True)
import numpy as np, matplotlib.pyplot as plt
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10})'''


def md(s):
    return nbf.v4.new_markdown_cell(s.strip("\n"))


def code(s):
    return nbf.v4.new_code_cell(s.strip("\n"))


def write(name, cells):
    nb = nbf.v4.new_notebook(); nb.metadata = META; nb.cells = cells
    (ROOT / "notebooks").mkdir(exist_ok=True)
    nbf.write(nb, ROOT / "notebooks" / name)
    print("wrote", name)


# ------------------------------------------------------------------------------------------
nb1 = [
    md("""
# 01 · The data: 64 winters of daily rainfall over Cumbria

**Source:** HadUK-Grid v1.3.2.ceda, daily rainfall on a 5 km grid (Met Office, via CEDA; Open
Government Licence v3.0). The pilot uses an 80 km × 80 km box (16 × 16 cells) over the Lake
District, October–March only, for the winters of 1961/62 to 2024/25.

**Why Cumbria:** it contains the wettest terrain in England, and on 5 December 2015 Storm Desmond
brought the largest daily rainfall in this record. That day is **held out** and used as a stress
test in notebook 03.

This notebook only *describes* the data. The split and the 95% threshold of the EVT margins were
fixed before any model was fitted. Choices made later, after looking at results, are listed in the
README under "Choices made after seeing results".
"""),
    code(SETUP),
    code("""
from rainpilot.data import load, interleaved_split, cell_distances_km, DESMOND_DATE
from rainpilot.diagnostics import DIST_BINS, DIST_EDGES
x, t, winter, da = load(DATA)
sp = interleaved_split(winter)
print(f"{len(x)} days, {x.shape[1]} x {x.shape[2]} cells, {str(t[0])[:10]} to {str(t[-1])[:10]}")
print(f"winters: train {len(sp.train_winters)}, validation {len(sp.val_winters)}, test {len(sp.test_winters)}")
print("test winters:", sp.test_winters.tolist())
"""),
    md("""
## The pre-specified split

Whole winters are assigned to sets by their **start year**: years ending in 0 or 5 → **test**;
ending in 3 or 8 → **validation**; all others → **training**. This was fixed before any model was
fitted. Two reasons for interleaving rather than a chronological split:

* rainfall days within a winter are dependent, so whole winters are kept together (no leakage
  between neighbouring days);
* mean winter rainfall has risen over the record (below). A chronological split would confuse
  that trend with model error; interleaving spreads every set across 1961–2025.

Storm Desmond's winter (2015/16) falls in the test set under this rule; it was not placed there
by hand.
"""),
    code("""
d = int(np.flatnonzero([str(v)[:10] == DESMOND_DATE for v in t])[0])
X = x.reshape(len(x), -1)
print(f"Storm Desmond, {DESMOND_DATE}: box mean {x[d].mean():.1f} mm, wettest cell {x[d].max():.1f} mm")
for name, idx in [("train", sp.train), ("validation", sp.val), ("test", sp.test)]:
    print(f"  {name:10s}: largest box mean {X[idx].mean(1).max():5.1f} mm | largest cell {X[idx].max():5.1f} mm")
"""),
    md("""
Desmond is the most extreme day in the record, but only **modestly** beyond the training data:
its box mean is about 3% above the largest training day and its wettest cell about 4% above the
largest training cell. It is a test of mild extrapolation, not of an event far outside experience.
"""),
    code("""
yr = x.mean((1, 2))
W = np.unique(winter)
wmax = np.array([yr[winter == w].max() for w in W])
wmean = np.array([yr[winter == w].mean() for w in W])
role = np.where(np.isin(W, sp.test_winters), "test", np.where(np.isin(W, sp.val_winters), "validation", "train"))
from scipy import stats
print("Trend in winter mean (Kendall):", stats.kendalltau(W, wmean))
print("Trend in winter maximum (Kendall):", stats.kendalltau(W, wmax))

fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), gridspec_kw={"width_ratios": [1, 1, 1.6]})
lon_ext = [da.x.values[0] - 2500, da.x.values[-1] + 2500, da.y.values[0] - 2500, da.y.values[-1] + 2500]
ext = [v / 1000 for v in lon_ext]
im = axes[0].imshow(x.mean(0), origin="lower", extent=ext, cmap="Blues")
axes[0].set_title("A. Mean winter rainfall (mm/day)"); plt.colorbar(im, ax=axes[0], shrink=0.8)
im = axes[1].imshow(x[d], origin="lower", extent=ext, cmap="Blues")
axes[1].set_title(f"B. Storm Desmond, {DESMOND_DATE} (mm)"); plt.colorbar(im, ax=axes[1], shrink=0.8)
for a in axes[:2]:
    a.set_xlabel("Easting (km)"); a.set_ylabel("Northing (km)")
cols = {"train": "#9a9a9a", "validation": "#0072B2", "test": "#D55E00"}
for r in ["train", "validation", "test"]:
    m = role == r
    axes[2].scatter(W[m], wmax[m], s=22, color=cols[r], label=f"{r} winters", zorder=3)
axes[2].plot(W, wmax, color="#cccccc", lw=0.8, zorder=1)
axes[2].annotate("Storm Desmond, Dec 2015 (test)", (2015, wmax[W == 2015][0]), xytext=(1983, 86),
                 arrowprops=dict(arrowstyle="->", color="#555555"), fontsize=9)
axes[2].annotate("Jan 2005 storm (training)", (2004, wmax[W == 2004][0]), xytext=(1975, 74),
                 arrowprops=dict(arrowstyle="->", color="#555555"), fontsize=9)
axes[2].set_ylim(top=92)
axes[2].set_title("C. Wettest day of each winter (box mean, mm)")
axes[2].set_xlabel("Winter (start year)"); axes[2].legend(frameon=False, loc="upper left")
plt.tight_layout(); plt.savefig(FIGS / "data_overview.png", bbox_inches="tight"); plt.show()
"""),
    md("""
**Trend.** Mean winter rainfall has increased over 1961–2025, while the trend in winter maxima is
smaller and not significant at the 5% level. The Kendall test treats winters as independent, so
its p-value is approximate. The generators in this pilot are
**stationary**, so they describe the 1961–2025 climate as a whole. Conditioning on climate, the
fourth research question of the E5 project, is outside the pilot's scope.
"""),
    code("""
D = cell_distances_km()
C = np.corrcoef(np.log1p(X[sp.train]).T)
bins = DIST_BINS    # contiguous distance bands, labelled by their centre
corr_d = [C[(D > lo) & (D <= hi)].mean() for lo, hi in zip(DIST_EDGES[:-1], DIST_EDGES[1:])]
wet = (X[sp.train] > 1).mean(1)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
axes[0].plot(bins, corr_d, "o-", color="#0072B2"); axes[0].set_ylim(0, 1)
axes[0].set_xlabel("Distance between cells (km)"); axes[0].set_ylabel("Correlation of log(1 + rain)")
axes[0].set_title("Spatial dependence (training winters)")
axes[1].hist(wet, bins=40, color="#0072B2", alpha=0.8)
axes[1].set_xlabel("Share of cells with > 1 mm"); axes[1].set_ylabel("Days")
axes[1].set_title("Wet-area fraction (training winters)")
plt.tight_layout(); plt.savefig(FIGS / "dependence_and_wet_area.png", bbox_inches="tight"); plt.show()
same = winter[:-1] == winter[1:]                       # consecutive days within one winter
print(f"lag-1 autocorrelation of the box mean (within winters): {np.corrcoef(yr[:-1][same], yr[1:][same])[0, 1]:.2f}")
"""),
    md("""
**What this means for the models**

* Rainfall is strongly correlated across the box, still about 0.7 at 100 km, so a storm often
  covers much of the region. Area-averaged extremes therefore depend heavily on how well a
  generator captures **joint** extremes, not just single-cell ones.
* The wet-area fraction is spread across the whole range, from dry days to days when every cell
  is wet: the bulk of the distribution matters too.
* Consecutive days are correlated (about 0.3), so uncertainty is assessed by resampling **whole
  winters**, never single days.
"""),
]

# ------------------------------------------------------------------------------------------
NAMES = ["Gaussian copula + empirical margins", "Gaussian copula + EVT margins",
         "VAE + empirical margins", "VAE + EVT margins"]
COLOURS = '{"Gaussian copula + empirical margins": "#9ecae1", "Gaussian copula + EVT margins": "#0072B2", ' \
          '"VAE + empirical margins": "#fdae6b", "VAE + EVT margins": "#D55E00"}'

nb2 = [
    md("""
# 02 · Controlled simulation study: can each generator recover a tail we know?

The E5 project asks for "controlled simulation studies, in which tail properties are known".
This notebook is a small version of one.

**Three true worlds.** All use the correlation matrix and margins fitted to the real Cumbria
*training* winters:

* **Gaussian world**: the chance that two cells are both extreme vanishes at the highest levels
  (asymptotic independence).
* **Student-t world** (4 degrees of freedom): joint extremes stay likely however high the level
  (asymptotic dependence), closer to how large storms behave.
* **Student-t world with varying tail shape**: as the Student-t world, but the true GPD shape ξ
  varies from west to east across the box (from the fitted value − 0.1 to + 0.1). The EVT margins
  assume one shared ξ, so here they are **misspecified**.

**Read the first two worlds as best cases.** Their true margins have exactly the form of the
fitted EVT margins (GPD above the 95th percentile, one shared ξ), and the Gaussian world *is* the
Gaussian copula + EVT generator. So in the Gaussian world the copula is correctly specified by
construction. The third world removes the first of these advantages.

**Each replicate** draws a synthetic training set (7,107 days) and validation set (2,370 days),
the sizes of the real ones, fits the four generators exactly as for the real data, and compares
their quantiles with the **true** quantiles. Levels go well beyond the synthetic training data:
one day in 1,000, one in 10,000, and one in 100,000 (the training set has about 7,000 days).

**Metrics:** log(model quantile / true quantile), averaged over cells or areas (0 = unbiased,
−0.1 ≈ 10% too low), and the mean *absolute* log ratio, in which errors of opposite sign in
different cells cannot cancel. Synthetic days are independent, unlike real days.

To rerun the study (about 1 hour per world on a 2-core CPU), set `RUN = True`.
"""),
    code(SETUP),
    code("""
import json, subprocess
RUN = False
if RUN:
    for w in ["gaussian", "t", "t_varxi"]:
        subprocess.run([sys.executable, str(ROOT / "experiments/simulation_study.py"), "--reps", "3",
                        "--worlds", w], check=True)
res = json.loads((RESULTS / "simulation_study.json").read_text())
NAMES = """ + repr(NAMES) + """
COL = """ + COLOURS + """
WNAME = {"gaussian": "Gaussian", "t": "Student-t", "t_varxi": "Student-t, varying xi"}
WORLDS = [w for w in WNAME if w in res["worlds"]]
PS = res["settings"]["ps"]; LBL = {str(p): f"1 in {round(1/(1-p)):,}" for p in PS}
for w in WORLDS:
    tx = np.array(res["worlds"][w]["truth_xi"])
    print(f"{WNAME[w]:22s} true xi {tx.min():.4f} to {tx.max():.4f} | fitted shared xi per replicate:",
          [round(r["fitted_xi"], 4) for r in res["worlds"][w]["reps"]])
"""),
    md("""
## Bias in tail quantiles, single cells and the whole box

Each entry is the mean over 3 replicates, with the range over replicates in brackets. With three
replicates this shows the spread between datasets; it is not a confidence interval.
"""),
    code("""
import pandas as pd
rows = []
for w in WORLDS:
    wd = res["worlds"][w]
    for name in NAMES:
        for tile, tname in [("1", "single cell"), ("16", "80 km box")]:
            for p in PS:
                v = np.array([r["models"][name]["log_ratio"][tile][str(p)] for r in wd["reps"]])
                a = np.array([r["models"][name]["log_ratio_abs"][tile][str(p)] for r in wd["reps"]])
                rows.append({"world": WNAME[w], "model": name, "scale": tname, "level": LBL[str(p)],
                             "bias": f"{v.mean():+.2f} ({v.min():+.2f} to {v.max():+.2f})",
                             "mean |log ratio|": f"{a.mean():.2f}"})
tab = pd.DataFrame(rows)
tab.pivot_table(index=["world", "scale", "model"], columns="level", values="bias", aggfunc="first",
                sort=False)[[LBL[str(p)] for p in PS]]
"""),
    code("""
# Single cells: mean absolute log ratio (cell errors of opposite sign cannot cancel)
tab[tab.scale == "single cell"].pivot_table(index=["world", "model"], columns="level",
    values="mean |log ratio|", aggfunc="first", sort=False)[[LBL[str(p)] for p in PS]]
"""),
    code("""
fig, axes = plt.subplots(2, len(WORLDS), figsize=(4.6 * len(WORLDS), 7), sharey="row", squeeze=False)
xs = np.arange(len(PS))
for i, (tile, tname) in enumerate([("1", "Single cell"), ("16", "Whole 80 km box")]):
    for j, w in enumerate(WORLDS):
        ax = axes[i, j]
        for k, name in enumerate(NAMES):
            v = np.array([[r["models"][name]["log_ratio"][tile][str(p)] for p in PS] for r in res["worlds"][w]["reps"]])
            off = (k - 1.5) * 0.08
            ax.errorbar(xs + off, v.mean(0), yerr=[v.mean(0) - v.min(0), v.max(0) - v.mean(0)], fmt="o-",
                        color=COL[name], label=name, capsize=3, lw=1.5)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(xs); ax.set_xticklabels([LBL[str(p)] + " days" for p in PS], fontsize=8)
        ax.set_title(f"{tname}: {WNAME[w]} world", fontsize=10)
        if j == 0: ax.set_ylabel("log(model / true quantile)")
axes[0, 0].legend(frameon=False, fontsize=7.5, loc="lower left")
fig.text(0.5, -0.01, "Points: mean of 3 replicates; bars: range over replicates", ha="center", fontsize=9)
plt.tight_layout(); plt.savefig(FIGS / "simulation_tail_bias.png", bbox_inches="tight"); plt.show()
"""),
    md("""
## Joint extremes: chi(u) against distance

chi(u) is the probability that a cell is extreme (above its u-quantile), given that another cell
at that distance is extreme. It is the most direct measure of whether a generator gets the
**spatial extent** of extremes right. Cell pairs are grouped into contiguous distance bands,
labelled by the band centre.
"""),
    code("""
from rainpilot.diagnostics import DIST_BINS as bins
us = res["settings"]["us"]
fig, axes = plt.subplots(1, len(WORLDS), figsize=(4.6 * len(WORLDS), 3.8), sharey=True, squeeze=False)
u = str(us[-1])
for ax, w in zip(axes[0], WORLDS):
    ax.plot(bins, res["worlds"][w]["chi_true"][u], "k-", lw=2.5, label="truth")
    for name in NAMES:
        if "empirical" in name:
            continue   # dependence is shared with the EVT version; chi depends only on ranks
        v = np.array([r["models"][name]["chi"][u] for r in res["worlds"][w]["reps"]])
        ax.plot(bins, v.mean(0), "o-", color=COL[name], label=name.split(" + ")[0] + " (mean of 3)")
        ax.fill_between(bins, v.min(0), v.max(0), color=COL[name], alpha=0.15)
    ax.set_title(f"chi(u = {u}): {WNAME[w]} world", fontsize=10)
    ax.set_xlabel("Distance (km)"); ax.set_ylim(0, 1)
axes[0, 0].set_ylabel("chi(u)"); axes[0, 0].legend(frameon=False, fontsize=8)
plt.tight_layout(); plt.savefig(FIGS / "simulation_chi.png", bbox_inches="tight"); plt.show()
for w in WORLDS:
    i100 = list(bins).index(100)
    print(f"{WNAME[w]:22s} chi({u}) at 100 km: truth {res['worlds'][w]['chi_true'][u][i100]:.3f}; " + "; ".join(
        f"{n.split(' + ')[0]} " + ", ".join(f"{r['models'][n]['chi'][u][i100]:.3f}" for r in res['worlds'][w]['reps'])
        for n in ["Gaussian copula + EVT margins", "VAE + EVT margins"]))
"""),
    md("""
## Reading the results

The interpretation below is written from the run stored in `results/`. If you rerun the study,
check that it still holds. "Bias" is the mean log ratio over cells (or over the box). Ranges are
over the 3 replicates.

**1. Empirical margins fail predictably beyond the training data.** At one day in 100,000, their
single-cell quantiles are too low by about 30% in the Gaussian world (log ratio −0.36), 40% in the
Student-t world (−0.50) and 28% in the varying-ξ world (−0.33), whatever the dependence model. The training set
holds about 7,000 days, so these levels are out of reach.

**2. When the EVT margins are correctly specified, they remove most of that bias on average, but
the tail shape is poorly determined.** With the Gaussian copula, the single-cell bias at one day in
100,000 averages −0.03 (Gaussian world) and −0.12 (Student-t world). The spread between replicates
is wide: in the Gaussian world, −0.30 to +0.24. In all six of these replicates the bias is ordered
by the fitted shared ξ, which ranged from −0.0075 to 0.138 against a true value of 0.0775. About
7,000 strongly dependent days do not pin ξ down, even when it is pooled over 256 cells. These two
worlds are best cases: the truth has exactly the form the EVT margins assume.

**3. When the shared-ξ assumption is wrong, errors appear at the level of single cells.** In the
varying-ξ world (true ξ from −0.02 in the west to 0.18 in the east):
* **Average bias is modest:** −0.08 at one day in 100,000.
* **Single cells are badly off,** with a mean absolute log ratio of 0.22 over the 3 replicates.
  One shared ξ is too heavy in the west and too light in the east, and these cell errors cancel
  in the average.
* **Even an accurate fitted ξ leaves this error.** In the two replicates whose fitted ξ was 0.0825,
  close to the true mean of 0.0775, the mean absolute error was still 0.17–0.18. In the third,
  whose ξ was badly estimated (−0.0075), it was 0.30. In the Student-t world,
  replicates with a well-estimated ξ (0.052 and 0.072) gave only 0.03–0.04.

Averaged over the 3 replicates, EVT margins still beat empirical margins here (mean absolute
error 0.22 against 0.33), but a shared ξ is a real limitation for regional maps of tail risk.

**4. Joint extremes matter for area totals.** In both Student-t worlds the Gaussian copula
underestimates chi at u = 0.999 (0.18–0.24 against a true 0.44 at 100 km). Its bias for the 80 km
box is larger than for single cells (Student-t world: −0.19 against −0.12 at one day in 100,000).
This is consistent with its missing joint dependence. The study does not separate that from the
other causes.

**5. The VAE is inconsistent.**
* **Between replicates:** its chi at u = 0.999 and 100 km varies widely:
  * Student-t: 0.16–0.42 (truth 0.44).
  * Gaussian: 0.12–0.30 (truth 0.20, where the Gaussian copula gets 0.19–0.24).
  * Varying ξ: 0.17–0.24 (truth 0.44).
* **Its tail quantiles are too low on average** even at one day in 1,000 (though not in every
  replicate: Gaussian-world replicate 0 is too high):
  * single cells: −0.10, −0.09 and −0.11 in the three worlds;
  * the box: −0.11 to −0.14.
* **EVT margins help it only partly:** −0.20 against −0.36 (Gaussian world) and −0.19 against −0.50
  (Student-t world) at one day in 100,000.

On real data (notebook 03) the VAE produces too few extreme normal scores, which is consistent
with this, but the simulation does not test that explanation directly.

**Caveats.**
* Only three replicates per world, so the ranges are rough.
* Synthetic days are independent.
* The truth for the 80 km box at one day in 100,000 is itself estimated from 4 million simulated
  days (about 40 exceedances), which adds an error of roughly 0.01–0.02 on the log scale, shared by
  all generators.
"""),
]

# ------------------------------------------------------------------------------------------
nb3 = [
    md("""
# 03 · Real rainfall: four generators judged on held-out winters, and Storm Desmond

**The question.** Can a generator that produces realistic-looking rainfall fields also assign the
right probabilities to the most extreme days, including one more severe than any it was trained on?

**Design: a 2 × 2 comparison.** Every generator maps rainfall to normal scores with per-cell
margins, models the spatial dependence of those scores, and maps samples back to millimetres.

| | empirical margins (cannot exceed the training maximum) | EVT margins (GPD tail above the 95th percentile) |
|---|---|---|
| **Gaussian copula** | simplest model | classical spatial statistics |
| **VAE** (convolutional, latent Gaussian mixture) | "learn the data distribution" | hybrid: neural dependence, EVT tails |

Comparing rows isolates the dependence model; comparing columns isolates the margins.

**Leakage rules.** Margins, normal scores, the copula and the VAE use **training** winters only.
**Validation** winters are used only for VAE early stopping and choosing the latent mixture size.
**Test** winters, including Storm Desmond's 2015/16, are used only here, after all fitting.

**Baselines.** Twelve test winters are a small sample of the climate. So every comparison with the
test winters is also made for the **training winters themselves**: the score of a model that
reproduced its training data perfectly. This baseline is a reference point, not a lower bound:
a model can match the test winters better than its own training data does, by chance.

To rerun the experiment (about 1.5 hours on a 2-core CPU), set `RUN = True`.
"""),
    code(SETUP),
    code("""
import json, subprocess, pandas as pd
from scipy import stats
RUN = False
if RUN:
    subprocess.run([sys.executable, str(ROOT / "experiments/cumbria_experiment.py")], check=True)
res = json.loads((RESULTS / "cumbria_experiment.json").read_text())
NAMES = """ + repr(NAMES) + """
COL = """ + COLOURS + """
EVT = ["Gaussian copula + EVT margins", "VAE + EVT margins"]
ref, fl, ds = res["reference"], res["floors"], res["desmond"]
print("days train / val / test:", res["split"]["n_days"])
print(f"fitted shared GPD shape xi = {res['fitted']['xi']:.3f}")
print("VAE runs:", [(v["seed"], v["best_epoch"], v["latent_components"]) for v in res["vae"]], "(seed, best epoch, mixture size)")
def vals(name, f):
    return np.array([f(r) for r in res["models"][name]])
def fmt(v, d=3, sign=False):
    f = f"{{:{'+' if sign else ''}.{d}f}}"
    return f.format(v.mean()) + ("" if len(v) == 1 else f" ({f.format(v.min())} to {f.format(v.max())})")
"""),
    md("""
## 1 · The baseline: how different are the test winters from the training winters?

If a model reproduced the training winters exactly, how would its quantiles compare with the test
winters? The bracket is a 95% block-bootstrap interval over test winters: the range of scores such
a perfect copy could get from test-winter sampling alone.
"""),
    code("""
TL = {"1": "single cell", "2": "10 km", "4": "20 km", "8": "40 km", "16": "80 km box"}
rows = []
for k, kn in TL.items():
    r = {"scale": kn}
    for p in ["0.99", "0.999"]:
        lo, hi = fl["log_ratio_train_vs_test_ci"][k][p]
        r[f"train vs test, 1 in {round(1/(1-float(p)))}"] = f"{fl['log_ratio_train_vs_test'][k][p]:+.3f} ({lo:+.2f} to {hi:+.2f})"
        r[f"val vs test, 1 in {round(1/(1-float(p)))}"] = f"{fl['log_ratio_val_vs_test'][k][p]:+.3f}"
    rows.append(r)
pd.DataFrame(rows).set_index("scale")
"""),
    md("""
## 2 · Tail quantiles, against the test winters and against the training winters

Below, the 1-in-100-day and 1-in-1,000-day quantiles of rainfall averaged over areas from 5 km to
80 km across. The **dashed line** is the baseline (the training winters themselves) and the **grey
band** its test-sampling interval. A model on the dashed line reproduces its training climate;
the distance between the dashed line and zero is a property of the test winters, not of any model.
"""),
    code("""
tiles = list(TL); km = [5, 10, 20, 40, 80]
fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True)
for ax, p in zip(axes, ["0.99", "0.999"]):
    f0 = np.array([fl["log_ratio_train_vs_test"][k][p] for k in tiles])
    ci = np.array([fl["log_ratio_train_vs_test_ci"][k][p] for k in tiles])
    ax.fill_between(km, ci[:, 0], ci[:, 1], color="#e3e3e3", label="training winters: test-sampling 95% interval")
    ax.plot(km, f0, "k--", lw=1.5, label="training winters (baseline)")
    for name in NAMES:
        v = np.array([[m["log_ratio_vs_test"][k][p] for k in tiles] for m in res["models"][name]])
        ax.plot(km, v.mean(0), "o-", color=COL[name], label=name + ("" if len(v) == 1 else " (mean of 3)"))
    ax.axhline(0, color="black", lw=0.8); ax.set_xscale("log", base=2); ax.set_xticks(km); ax.set_xticklabels(km)
    ax.set_xlabel("Width of averaging area (km)")
    ax.set_title(f"1-in-{round(1/(1-float(p))):,}-day quantile")
axes[0].set_ylabel("log(model / test-winter quantile)"); axes[0].legend(frameon=False, fontsize=7)
plt.tight_layout(); plt.savefig(FIGS / "areal_quantiles.png", bbox_inches="tight"); plt.show()
"""),
    code("""
# EVT generators: bias against the test winters and against the training winters, and the
# difference between the two generators (which does not depend on the reference at all).
rows = []
for k in ["1", "4", "16"]:
    for p in ["0.99", "0.999"]:
        r = {"scale": TL[k], "level": f"1 in {round(1/(1-float(p))):,}"}
        for name in EVT:
            short = name.split(" + ")[0]
            r[f"{short} vs test"] = fmt(vals(name, lambda m: m["log_ratio_vs_test"][k][p]), 3, True)
            r[f"{short} vs train"] = fmt(vals(name, lambda m: m["log_ratio_vs_train"][k][p]), 3, True)
        g = vals(EVT[0], lambda m: m["log_ratio_vs_test"][k][p])[0]
        r["copula minus VAE"] = fmt(g - vals(EVT[1], lambda m: m["log_ratio_vs_test"][k][p]), 3, True)
        rows.append(r)
quant_tab = pd.DataFrame(rows).set_index(["scale", "level"])
quant_tab
"""),
    md("""
**Calibration at one day in 100.** The share of test days above each generator's per-cell
1-in-100-day level, against the same share for the training winters' own level (the baseline). The
interval is a block bootstrap over test winters.
"""),
    code("""
lo, hi = fl["test_exceedance_of_train_quantile_ci"]["0.99"]
byw = np.array(fl["test_exceedance_by_winter"]["0.99"])
print(f"baseline (training winters' own 1-in-100 level): {100*fl['test_exceedance_of_train_quantile']['0.99']:.2f}% "
      f"(95% interval {100*lo:.2f}% to {100*hi:.2f}%; single test winters range {100*byw.min():.2f}% to {100*byw.max():.2f}%)")
for name in NAMES:
    v = 100 * vals(name, lambda m: m["test_exceedance_of_model_quantile"]["0.99"])
    print(f"  {name:38s} {fmt(v, 2)}%")
"""),
    md("""
## 3 · Realism: ordinary rainfall and joint extremes, against the baseline

Errors against the test winters for the spatial correlation of log(1 + rainfall), the wet-area
fraction (quantiles of the share of cells above 1 mm), the mean rainfall map (relative error), and
chi(u) at u = 0.95 and 0.98. The baseline rows give the same errors for the training and validation
winters themselves: the error that comes from the test winters being a different sample. A model
can fall below it by chance.
"""),
    code("""
MET = [("corr. error", "corr_log_abs_error", None), ("wet-area error", "wet_fraction_abs_error", None),
       ("mean-map error", "cell_mean_rel_error", None), ("chi(0.95) error", "chi_abs_error", "0.95"),
       ("chi(0.98) error", "chi_abs_error", "0.98")]
def get(m, key, u):
    return m[key][u] if u else m[key]
rows = []
for name in NAMES:
    rows.append({"": name, **{lab: fmt(vals(name, lambda m: get(m, key, u)), 3) for lab, key, u in MET}})
for which in ["train", "val"]:
    b = fl[f"bulk_{which}_vs_test"]
    rows.append({"": f"baseline: {'training' if which == 'train' else 'validation'} winters",
                 **{lab: f"{(fl[f'chi_abs_error_{which}_vs_test'][u] if u else b[key]):.3f}" for lab, key, u in MET}})
realism_tab = pd.DataFrame(rows).set_index("")
realism_tab
"""),
    code("""
fig, ax = plt.subplots(figsize=(7.5, 3.6))
labs = [m[0] for m in MET]
for j, (lab, key, u) in enumerate(MET):
    f_tr = fl["chi_abs_error_train_vs_test"][u] if u else fl["bulk_train_vs_test"][key]
    ax.plot([f_tr], [j], "k|", ms=18, mew=2, label="baseline: training winters" if j == 0 else None)
    for i, name in enumerate(EVT):
        v = vals(name, lambda m: get(m, key, u))
        ax.errorbar(v.mean(), j + (0.15 if i else -0.15), xerr=[[v.mean() - v.min()], [v.max() - v.mean()]],
                    fmt="o", color=COL[name], capsize=3, label=name if j == 0 else None)
ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs); ax.invert_yaxis()
ax.set_xlabel("Error against the test winters (lower is better)"); ax.set_xlim(left=0)
ax.legend(frameon=False, fontsize=8, loc="lower right")
ax.set_title("Realism metrics, with the training-winter baseline", fontsize=10)
plt.tight_layout(); plt.savefig(FIGS / "realism_metrics.png", bbox_inches="tight"); plt.show()
"""),
    md("""
## 4 · Joint extremes in space: chi(u) against distance

Cell pairs are grouped into contiguous distance bands, labelled by the band centre. The grey band
is a pointwise 95% block-bootstrap interval over test winters (10 bands, so a single excursion is
not strong evidence by itself). The dashed line is chi for the training winters.
"""),
    code("""
from rainpilot.diagnostics import DIST_BINS as bins
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
for ax, u in zip(axes, ["0.95", "0.98"]):
    lo, hi = np.array(ref["chi_ci"][u])
    ax.fill_between(bins, lo, hi, color="#dddddd", label="test winters, pointwise 95% interval")
    ax.plot(bins, ref["chi"][u], "k-", lw=2, label="test winters")
    ax.plot(bins, fl["chi_train"][u], "k--", lw=1.2, label="training winters")
    for name in EVT:
        v = np.array([m["chi"][u] for m in res["models"][name]])
        ax.plot(bins, v.mean(0), "o-", color=COL[name], label=name.split(" + ")[0])
    ax.set_title(f"chi(u = {u})"); ax.set_xlabel("Distance (km)"); ax.set_ylim(0, 1)
axes[0].set_ylabel("chi(u)"); axes[0].legend(frameon=False, fontsize=8)
plt.tight_layout(); plt.savefig(FIGS / "chi_cumbria.png", bbox_inches="tight"); plt.show()
for name in EVT:
    for u in ["0.95", "0.98"]:
        v = np.array([m["chi"][u] for m in res["models"][name]]).mean(0)
        lo, hi = np.array(ref["chi_ci"][u])
        print(f"{name:32s} u={u}: above test interval in {int((v > hi).sum())}/10 bands, below in {int((v < lo).sum())}/10")
"""),
    md("""
## 5 · Does the dependence model produce extreme normal scores often enough?

EVT margins can only extrapolate if the dependence model produces extreme normal scores. The
ratio below is the share of generated normal scores above the 1-in-100, 1-in-1,000 and
1-in-10,000 levels, divided by its target (1 = correct). Generated scores are strongly dependent
across cells, so Monte Carlo noise is much larger than for independent draws. The noise reference
is the range over 20 independent samples from the Gaussian copula, whose scores are exactly normal.
"""),
    code("""
nz = res["normal_score_noise"]
for p in ["0.99", "0.999", "0.9999"]:
    lo, hi = nz["range"][p]
    print(f"1 in {round(1/(1-float(p))):>6,}: noise reference {lo:.2f} to {hi:.2f} | " + " | ".join(
        f"{n.split(' + ')[0]}: " + ", ".join(f"{v:.2f}" for v in vals(n, lambda m: m['normal_score_exceedance_ratio'][p]))
        for n in EVT))
"""),
    md("""
## 6 · Stress test: how likely was Storm Desmond?

For each generator: the daily probability of a day at least as extreme as 5 December 2015, as
(a) a box mean of 83.5 mm or more, and (b) any cell with 254.8 mm or more, converted to an average
number of winters between such days (182.25 days per October–March winter, ignoring clustering).
None of the generators saw Desmond. The training data's largest values were 81.3 mm (box mean,
January 2005) and 245.2 mm (one cell).

**Two kinds of uncertainty are shown.**
* *Monte Carlo*: an exact Poisson interval for the number of hits.
* *Tail (margin) uncertainty*: the EVT margins are refitted on 100 block-bootstrap resamples of
  the training winters and applied to the **same** generated normal scores (2 million for the
  copula, 6 million pooled over three VAE seeds). The dependence models are held fixed, so these
  intervals do not include uncertainty in the copula's correlations or in VAE training.

Empirical margins cannot produce a wettest cell above the training maximum (245.2 mm), so their
"never" for the cell is guaranteed by construction, not an empirical finding.
"""),
    code("""
D = 182.25
def rp(h, n):
    return np.inf if h == 0 else n / (h * D)
def poisson_rp(h, n):
    lo = stats.chi2.ppf(0.025, 2 * h) / 2 if h > 0 else 0.0
    hi = stats.chi2.ppf(0.975, 2 * h + 2) / 2
    return n / (hi * D), (np.inf if lo == 0 else n / (lo * D))
def s_rp(v):
    return "never" if np.isinf(v) else f"{v:.0f}"
rows = []
for name in NAMES:
    d = [m["desmond"] for m in res["models"][name]]
    n = sum(x["n"] for x in d)
    r = {"model": name, "generated days": f"{n:,}"}
    for kind in ["box", "cell"]:
        h = sum(x[f"hits_{kind}"] for x in d)
        a, b = poisson_rp(h, n)
        r[f"{kind}: hits"] = h
        r[f"{kind}: return period (winters)"] = s_rp(rp(h, n))
        r[f"{kind}: Monte Carlo 95%"] = f"{s_rp(a)} to {s_rp(b)}"
        bt = res["bootstrap_train"]["models"].get(name)
        if bt:
            ps = np.array([x[f"p_{kind}"] for x in bt])
            p_lo, p_hi = np.percentile(ps, [2.5, 97.5])
            inv = lambda q: np.inf if q == 0 else 1 / (q * D)     # return period falls as p rises
            r[f"{kind}: tail-uncertainty 95%"] = f"{s_rp(inv(p_hi))} to {s_rp(inv(p_lo))}"
        else:
            r[f"{kind}: tail-uncertainty 95%"] = "n/a"
        if len(d) > 1:
            r[f"{kind}: per VAE seed"] = ", ".join(s_rp(rp(x[f"hits_{kind}"], x["n"])) for x in d)
    rows.append(r)
desmond_tab = pd.DataFrame(rows).set_index("model").T
desmond_tab
"""),
    code("""
xi_b = np.array(res["bootstrap_train"]["xi"])
print(f"shared xi: {res['fitted']['xi']:.3f}; bootstrap 95% interval over training winters: "
      f"{np.percentile(xi_b, 2.5):.3f} to {np.percentile(xi_b, 97.5):.3f}")
for name in EVT:
    h = np.array([x["hits_cell"] for x in res["bootstrap_train"]["models"][name]])
    print(f"{name}: hits (cell) per bootstrap replicate, median {np.median(h):.0f}, range {h.min()} to {h.max()}")
"""),
    md("""
### Storm structure at Desmond's severity, and at each generator's own extreme

**Top row:** for a like-for-like comparison of *structure*, each generator's day whose box mean
is closest to Desmond's 83.5 mm, from 400,000 draws (seed 0). **Bottom row:** each generator's
single most extreme box-mean day from the same 400,000 draws. That row shows how each generator
extrapolates, not a typical event. Fields come from `experiments/desmond_analogues.py`. One
colour scale is used throughout. These are single samples, so the figure is illustrative only.
"""),
    code("""
an = json.loads((RESULTS / "desmond_analogues.json").read_text())
obs = np.array(ds["field"])
vmax = max(obs.max(), max(np.max(v["most_extreme"]) for v in an["models"].values()))
fig, axes = plt.subplots(2, 5, figsize=(16, 6.6))
for row in range(2):
    ax = axes[row, 0]
    if row == 0:
        im = ax.imshow(obs, origin="lower", cmap="Blues", vmin=0, vmax=vmax)
        ax.set_title(f"Observed, 5 Dec 2015\\nbox mean {obs.mean():.0f} mm, max {obs.max():.0f} mm", fontsize=9)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, f"Most extreme day\\nof {an['n_draws']:,} draws\\n(not a typical event)",
                ha="center", va="center", fontsize=10)
    for ax, name in zip(axes[row, 1:], NAMES):
        f = np.array(an["models"][name]["analogue" if row == 0 else "most_extreme"])
        ax.imshow(f, origin="lower", cmap="Blues", vmin=0, vmax=vmax)
        ax.set_title(f"{name}\\nbox mean {f.mean():.0f} mm, max {f.max():.0f} mm", fontsize=8)
for ax in axes.ravel():
    ax.set_xticks([]); ax.set_yticks([])
fig.colorbar(im, ax=axes, shrink=0.6, label="mm")
plt.savefig(FIGS / "desmond_vs_generated.png", bbox_inches="tight"); plt.show()
"""),
    md("""
Two simple checks on the analogues, against the 20 wettest training days (box mean): the
correlation of each field with the observed Desmond field, and a roughness index (mean absolute
difference between neighbouring cells, divided by the field mean). If the analogues fall inside
the range of ordinary large training days, the figure does not distinguish the generators.
"""),
    code("""
from rainpilot.data import load, interleaved_split
x_all, t_all, w_all, _ = load(DATA)
xtr = x_all[interleaved_split(w_all).train]
def rough(f):
    f = np.asarray(f)
    return (np.abs(np.diff(f, axis=0)).mean() + np.abs(np.diff(f, axis=1)).mean()) / 2 / f.mean()
def corr(f):
    return np.corrcoef(np.ravel(f), obs.ravel())[0, 1]
top = xtr[np.argsort(xtr.mean((1, 2)))[-20:]]
rt, ct = np.array([rough(f) for f in top]), np.array([corr(f) for f in top])
print(f"observed Desmond: roughness {rough(obs):.3f}")
print(f"20 wettest training days: roughness {rt.min():.3f} to {rt.max():.3f} (median {np.median(rt):.3f}); "
      f"correlation with Desmond {ct.min():.2f} to {ct.max():.2f} (median {np.median(ct):.2f})")
for name in NAMES:
    f = an["models"][name]["analogue"]
    print(f"  analogue, {name:38s} roughness {rough(f):.3f}, correlation with Desmond {corr(f):.2f}")
"""),
    md("""
## Reading the results

The interpretation below is written from the run stored in `results/`. If you rerun the
experiment, check that it still holds.

**1. The test winters are wetter than the training winters, by about as much as the models differ.**
Against the 12 test winters, the training winters' own quantiles are about 6% lower for single
cells (log ratio −0.058 at one day in 100) and about 9% lower for the 80 km box (−0.096). A model that copied
its training data perfectly would therefore look "too low" by that amount. The validation winters
differ from the test winters by a similar amount (−0.08 to −0.18). From test-winter sampling alone,
such a perfect copy could score anywhere from about −0.17 to +0.07 (single cells, one day in 100)
or −0.30 to +0.17 (one day in 1,000). Every generator lies inside these ranges, so **the test
winters cannot rank the generators' tails.** Judged against the test winters alone, the Gaussian
copula would look better at every scale. That conclusion would be an artefact of the test sample.

**2. Against the training winters, the answer depends on scale.** This compares each generator
with the data it was fitted to (EVT margins; VAE values are the range over 3 seeds):
* **Single cells:** the Gaussian copula matches its training climate (−0.002 at one day in 100,
  +0.015 at one in 1,000). The VAE is 2–5% too low (−0.02 to −0.05) and 3–8% too low (−0.03 to −0.08).
* **80 km box:** the Gaussian copula is **too high**, by about 6% (+0.055) and 9% (+0.086). The VAE is
  close (−0.004 to +0.013, and −0.05 to +0.01).

So the copula is better for single cells and the VAE better for area averages. The difference
between the two generators does not depend on any reference: at one day in 100 the copula's box
quantile exceeds the VAE's by 4–6% (+0.042 to +0.059 on the log scale, depending on the VAE seed).

**3. Calibration at one day in 100 does not separate the generators either.** Test days exceed the
training winters' own per-cell 1-in-100 level on 1.26% of days (95% interval 0.78–1.81%; single
test winters range from 0.26% to 3.4%). The Gaussian copula gives 1.26% and the VAE 1.35–1.46%:
all inside the test-sampling interval.

**4. Realism.** Against the baseline (training winters versus test winters):
* **Spatial correlation:** the VAE is closer (error 0.009–0.012 against 0.018; baseline 0.006).
* **Joint extremes, chi(0.98):** the VAE is much closer (0.019–0.036 against 0.080; baseline 0.014).
  The Gaussian copula **overstates** joint extremes: its chi(0.98) lies above the test winters'
  pointwise 95% interval in all 10 distance bands, and the training winters' chi is close to the test
  winters' (error 0.014), so this is a property of the model, not of the test sample. It is
  consistent with the copula overstating area-averaged quantiles in point 2. At u = 0.95 the two
  generators are indistinguishable (0.019 against 0.009–0.028; baseline 0.021).
* **At 5 km the VAE is too weak:** its chi lies below the test interval in the 5 km band for every
  seed, at both levels. This is consistent with its independent per-pixel decoder noise, which
  makes neighbouring cells less alike; that explanation is not tested here.
* **Wet-area fraction and mean rainfall map:** both generators are close to the baseline (wet area 0.023
  against 0.020–0.022, baseline 0.019; mean map 0.019 against 0.020–0.024, baseline 0.020).

So the VAE's advantage in realism is specific: spatial correlation and high-level joint extremes.

**5. The VAE produces too few extreme normal scores.** Its rate of normal scores above the
1-in-1,000 level is 0.59–0.74 times the target over 3 seeds, against a Monte Carlo noise range of
0.92–1.17 for the Gaussian copula, whose scores are exactly normal. The shortfall is also outside
the noise at one in 100 (0.85–0.92 against 0.95–1.02) and one in 10,000 (0.31–0.69 against 0.79–1.31).
The GPD tail is therefore reached too rarely, which is consistent with the VAE's low single-cell
quantiles (point 2). The size of the shortfall depends on the training run. The first full run of
this experiment had three further VAEs whose seeds were not fully independent (see the README).
Across all six trained VAEs, the ratio at one in 1,000 ranges from 0.47 to 0.89, and at one in 100
from 0.79 to 0.92. Why the shortfall does not show up in the box average is not tested here.

**6. Storm Desmond.**
* **Wettest cell (≥ 255 mm).** Generators with empirical margins never produce it. This is
  guaranteed by construction: the largest training value is 245.2 mm. With EVT margins, such a
  day recurs on average every **64 winters** for the Gaussian copula (Monte Carlo 95% interval
  55–75; tail-uncertainty interval 25–282) and every **201 winters** for the VAE (3 seeds pooled;
  Monte Carlo 172–235; tail uncertainty 57–1,244; single seeds 129, 196 and 477).
* **Box mean (≥ 83.5 mm).** This is only 2.7% above the largest training day, so **empirical
  margins can produce it** by combining cells: every 83 winters for the Gaussian copula and 383 for
  the VAE. With EVT margins: 65 winters (tail uncertainty 34–175) and 222 (115–806; single seeds
  114, 392 and 457).
* **What this shows.** Desmond is the wettest day in 64 winters, which is consistent with a return
  period of about 64 winters but does not rule out 200: a 1-in-200-winter day appears within 64
  winters with probability about 27%. One event cannot separate these generators. The useful
  result is the **size of the uncertainty**:
  * the tail-uncertainty intervals span a factor of 5 to 22;
  * within them, most of the variation comes from the shared shape ξ (bootstrap 95% interval 0.018
    to 0.122; ξ explains 67–79% of the variance of the log probability across replicates);
  * those intervals hold the dependence model fixed, and VAE training randomness is at least as
    large. Across six trained VAEs (these three plus three from the first full run) the box-mean
    return period ranges from 42 to 457 winters, and the pooled VAE estimate was 91 winters in
    the first run against 222 here. The VAE's Desmond probabilities should be read as uncertain
    by about an order of magnitude from training randomness alone.

**7. Structure at equal severity (figure above).** This is illustrative only: it shows one sample
per generator.
* Both Gaussian-copula analogues concentrate rain over the western fells, as on 5 December 2015.
* The VAE analogues are flatter. The VAE + EVT analogue peaks in the south-east, and its
  correlation with the observed field (0.40) is at the bottom of the range of the 20 wettest
  training days (0.37 to 0.91).
* The roughness of every analogue (0.130–0.160) lies within the range of those training days
  (0.115–0.184).
* At their most extreme days, all four generators place the heaviest rain in the west.
"""),
]


if __name__ == "__main__":
    write("01_data_overview.ipynb", nb1)
    write("02_simulation_study.ipynb", nb2)
    write("03_cumbria_generators.ipynb", nb3)
