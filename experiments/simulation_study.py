"""Controlled simulation study: the four generators against KNOWN tail properties.

Three true worlds use the correlation matrix and the margins fitted to the real Cumbria
TRAINING winters:
  * "gaussian": joint extremes become independent at the highest levels (chi(u) -> 0);
  * "t" (Student-t, 4 df): joint extremes stay dependent (chi(u) -> positive limit);
  * "t_varxi": as "t", but the true GPD shape varies across cells (west-east, from the fitted
    xi - 0.1 to xi + 0.1), so the fitted EVT margins (one shared xi) are MISSPECIFIED.
In the first two worlds the truth's margins have exactly the form of the fitted EVT margins
(GPD above the 95% threshold, one shared xi), and the "gaussian" world is exactly the
Gaussian-copula + EVT generator: those are best cases for the EVT margins and for the copula.
In each replicate, a synthetic training set and validation set of the same sizes as the real
ones are drawn from the true world; the four generators are fitted to them exactly as for the
real data; and their quantiles are compared with the true quantiles, including levels well
beyond anything in the synthetic training data (1 in 10,000 and 1 in 100,000 days).

Run:  python experiments/simulation_study.py --reps 3 --worlds gaussian
      python experiments/simulation_study.py --reps 3 --worlds t
      python experiments/simulation_study.py --reps 3 --worlds t_varxi
Each replicate is saved as it finishes (results/simulation_parts/), so an interrupted run
can be restarted and will skip what is already done.
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rainpilot.data import interleaved_split, load                       # noqa: E402
from rainpilot.diagnostics import TILES                                  # noqa: E402
from rainpilot.experiment import (TrueWorld, fit_two_by_two, log_ratio,  # noqa: E402
                                  log_ratio_abs, model_chi, model_quantiles, paired_quantiles)
from rainpilot.margins import Margins                                    # noqa: E402

PS = (0.999, 1 - 1e-4, 1 - 1e-5)
US = (0.95, 0.99, 0.999)
WORLDS = {"gaussian": ("shared", "gaussian", 0), "t": ("shared", "t", 1), "t_varxi": ("varying", "t", 2)}


def varying_xi(m, spread=0.1):
    """Copy of fitted EVT margins whose GPD shape varies linearly from west to east."""
    tm = copy.deepcopy(m)
    east = np.tile(np.arange(16), 16) / 15.0              # 0 (west) .. 1 (east); fields are [north, east]
    tm.xi = m.xi + spread * (2 * east - 1)                # never exactly 0 on this grid
    return tm


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--data", default=str(ROOT / "data/cumbria_winter_rainfall_5km.nc"))
    a.add_argument("--reps", type=int, default=2)
    a.add_argument("--n-truth", type=int, default=4_000_000)
    a.add_argument("--n-model", type=int, default=2_000_000)
    a.add_argument("--n-chi", type=int, default=200_000)
    a.add_argument("--vae-epochs", type=int, default=400)
    a.add_argument("--out", default=str(ROOT / "results/simulation_study.json"))
    a.add_argument("--parts", default=str(ROOT / "results/simulation_parts"))
    a.add_argument("--worlds", nargs="+", default=list(WORLDS), choices=list(WORLDS))
    args = a.parse_args()

    x, _, winter, _ = load(args.data)
    sp = interleaved_split(winter)
    xt = x[sp.train].reshape(len(sp.train), -1)
    fitted = Margins("evt").fit(xt)                           # basis of the truth's margins
    z = fitted.to_normal(xt, np.random.default_rng(0))
    corr = np.corrcoef(z.T)
    n_train, n_val = len(sp.train), len(sp.val)

    parts = Path(args.parts)
    parts.mkdir(parents=True, exist_ok=True)
    settings = {k: v for k, v in vars(args).items() if k != "worlds"} | {
        "ps": PS, "us": US, "tiles": TILES, "n_train": n_train, "n_val": n_val, "fitted_xi": fitted.xi}
    for world in args.worlds:
        kind, cop, wid = WORLDS[world]
        truth_margins = fitted if kind == "shared" else varying_xi(fitted)
        tw = TrueWorld(truth_margins, corr, copula=cop)
        truth_file = parts / f"{world}_truth.json"
        if truth_file.exists():                              # resume: truth already computed
            tj = json.loads(truth_file.read_text())
            q_true = {int(k): {float(p): np.array(v) for p, v in d.items()} for k, d in tj["q"].items()}
            chi_true = {float(u): np.array(v) for u, v in tj["chi"].items()}
            print(f"[{world}] truth loaded from {truth_file.name}", flush=True)
        else:
            t0 = time.time()
            q_true = model_quantiles(tw, args.n_truth, PS, seed=999)
            # Single cells: the true quantiles are known exactly from the true margins.
            q_true[1] = {p: truth_margins.ppf(np.full((1, 256), p))[0] for p in PS}
            chi_true = model_chi(tw, args.n_chi, US, seed=998)
            truth_file.write_text(json.dumps({"truth_xi": np.atleast_1d(truth_margins.xi).tolist(),
                "q": {str(k): {str(p): v.tolist() for p, v in d.items()} for k, d in q_true.items()},
                "chi": {str(u): v.tolist() for u, v in chi_true.items()}}))
            print(f"[{world}] truth computed in {time.time() - t0:.0f}s", flush=True)
        for r in range(args.reps):
            rep_file = parts / f"{world}_rep{r}.json"
            if rep_file.exists():                            # resume: replicate already done
                print(f"[{world}] rep {r} already done, skipping", flush=True)
                continue
            rng = np.random.default_rng(1000 * (r + 1) + wid)
            x_tr, x_va = tw.sample(n_train, rng), tw.sample(n_val, rng)
            t0 = time.time()
            gens, m_evt = fit_two_by_two(x_tr, x_va, seed=r, vae_kw={"epochs": args.vae_epochs})
            rep = {"seed": r, "fitted_xi": m_evt.xi,
                   "train_max_cell": float(x_tr.max()),
                   "vae_latent_components": int(gens[("evt", "vae")].dep.model.latent_k),
                   "models": {}}
            for dep in ["gaussian", "vae"]:
                # common random numbers: the same normal-score fields for both margin types
                g_emp, g_evt = gens[("empirical", dep)], gens[("evt", dep)]
                qs = paired_quantiles(g_evt.dep, {g_emp.name: g_emp.margins, g_evt.name: g_evt.margins},
                                      args.n_model, PS, seed=50 + r)
                cm = model_chi(g_evt, args.n_chi, US, seed=60 + r)   # rank-based: same for both margins
                for g in (g_emp, g_evt):
                    rep["models"][g.name] = {
                        "log_ratio": {str(k): {str(p): v for p, v in d.items()}
                                      for k, d in log_ratio(qs[g.name], q_true, PS).items()},
                        "log_ratio_abs": {str(k): {str(p): v for p, v in d.items()}
                                          for k, d in log_ratio_abs(qs[g.name], q_true, PS).items()},
                        "chi": {str(u): cm[u].tolist() for u in US},
                    }
            rep_file.write_text(json.dumps(rep))             # checkpoint after every replicate
            print(f"[{world}] rep {r} done in {time.time() - t0:.0f}s (fitted xi {m_evt.xi:.3f}, "
                  f"truth mean xi {np.mean(truth_margins.xi):.3f})", flush=True)
    merge(parts, Path(args.out), settings)


def merge(parts, out, settings):
    """Combine all finished parts into one results file."""
    results = {"settings": settings, "worlds": {}}
    for world in WORLDS:
        tf = parts / f"{world}_truth.json"
        reps = sorted(parts.glob(f"{world}_rep*.json"), key=lambda f: int(f.stem.split("rep")[1]))
        if tf.exists() and reps:
            results["worlds"][world] = {"chi_true": json.loads(tf.read_text())["chi"],
                                        "truth_xi": json.loads(tf.read_text())["truth_xi"],
                                        "reps": [json.loads(f.read_text()) for f in reps]}
    results["settings"]["worlds"] = list(results["worlds"])          # what this file contains
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1))
    print("saved", out, {w: len(v["reps"]) for w, v in results["worlds"].items()})


if __name__ == "__main__":
    main()
