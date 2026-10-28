"""The four generators on real HadUK-Grid winter rainfall over Cumbria, judged on held-out winters.

Leakage rules (enforced by code, not convention):
  * margins, normal scores, the Gaussian copula and the VAE are fitted on TRAINING winters only;
  * VALIDATION winters are used only for VAE early stopping and choosing the latent mixture size;
  * TEST winters (start years ending in 0 or 5, including Storm Desmond's 2015/16) are used only
    in the evaluation below, after everything has been fitted.

Reference floors (what a model that reproduced the TRAINING data exactly would score):
  every test-winter comparison is also computed for the training winters themselves (and for the
  validation winters), so that differences between the test winters and the training climate are
  not mistaken for model error. Model quantiles are also compared with the training winters.

Uncertainty:
  * test-winter sampling: block bootstrap over whole test winters;
  * fitted tails: block bootstrap over whole training winters, refitting the EVT margins each
    time while holding the dependence models (copula, trained VAEs) fixed. These intervals
    therefore cover margin (tail) uncertainty only, for both generators. Desmond probabilities in
    the bootstrap use the same draws as the point estimates (2 million per fitted generator; all
    three VAE seeds pooled), evaluated exactly via a provably conservative screen (see
    `Envelope`), so the intervals are not dominated by Monte Carlo noise.

Run:  python experiments/cumbria_experiment.py
Results are saved as each seed and every 10 bootstrap replicates finish
(results/cumbria_parts/), so an interrupted run can be restarted and skips finished work.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rainpilot.data import DESMOND_DATE, interleaved_split, load               # noqa: E402
from rainpilot.diagnostics import (TILES, bulk_summaries, chi_by_distance,      # noqa: E402
                                   empirical_quantiles, return_period_winters)
from rainpilot.experiment import fit_two_by_two, paired_quantiles                 # noqa: E402
from rainpilot.generators import GaussianCopula                                   # noqa: E402
from rainpilot.margins import Margins                                            # noqa: E402

PS = (0.99, 0.999)
US = (0.95, 0.98)
Z_LEVELS = (0.99, 0.999, 0.9999)


def tolist(d):
    if isinstance(d, dict):
        return {str(k): tolist(v) for k, v in d.items()}
    if isinstance(d, (list, tuple)):
        return [tolist(v) for v in d]
    if isinstance(d, np.ndarray):
        return d.tolist()
    if isinstance(d, (np.floating, np.integer)):
        return d.item()
    return d


def mean_log_ratio(q_a, q_b):
    """{k: {p: mean over tiles of log(q_a / q_b)}}."""
    return {k: {p: float(np.mean(np.log(q_a[k][p] / q_b[k][p]))) for p in PS} for k in TILES}


def bulk_errors(b, ref):
    return {"wet_fraction_abs_error": float(np.mean(np.abs(b["wet_fraction_quantiles"]
                                                           - ref["wet_fraction_quantiles"]))),
            "corr_log_abs_error": float(np.mean(np.abs(b["corr_log"] - ref["corr_log"]))),
            "cell_mean_rel_error": float(np.mean(np.abs(b["cell_mean"] / ref["cell_mean"] - 1)))}


class Envelope:
    """Upper bound, over a set of margins, of the map normal score -> mm, cell by cell.

    For each margin and cell, U(g) is an upper bound on sup{ppf(t) : t <= g}. Then for any
    normal score z, the value of U at the next grid point g >= z bounds ppf(z) for every margin
    added. The quantile functions are non-decreasing except for one small downward step in the
    EVT margins at p = 1 - zeta, where the GPD tail (which starts at u) takes over from the
    empirical body (whose value there can slightly exceed u). So for grid points at or above
    that step, U also includes the body's value at the step, and U is made non-decreasing along
    the grid. A draw that cannot reach the Desmond thresholds under this bound cannot reach them
    under any of the margins, so only the remaining 'candidate' draws need to be evaluated
    exactly for each bootstrap replicate."""

    def __init__(self, lo=-8.5, hi=8.5, n=4001, ncell=256):
        self.grid = np.linspace(lo, hi, n)
        self.U = np.full((n, ncell), -np.inf)
        self._p = np.tile(stats.norm.cdf(self.grid)[:, None], (1, ncell))

    def add(self, m):
        u_m = m.ppf(self._p)
        if m.tail == "evt":
            p_step = 1 - m.zeta                                   # where the GPD tail takes over
            left = m.ppf(p_step[None, :])[0]                      # body value at the step
            at_or_above = self.grid[:, None] >= stats.norm.ppf(p_step)[None, :]
            u_m = np.where(at_or_above, np.maximum(u_m, left), u_m)
        u_m = np.maximum.accumulate(u_m, axis=0)                  # sup over t <= g
        self.U = np.maximum(self.U, u_m)

    def bound(self, z):
        i = np.searchsorted(self.grid, z, side="left")          # grid[i] >= z
        beyond = i >= len(self.grid)                            # above the grid: no bound
        ub = self.U[np.minimum(i, len(self.grid) - 1), np.arange(z.shape[1])]
        return np.where(beyond, np.inf, ub)

    def candidates(self, z, box_thr, cell_thr):
        ub = self.bound(z)
        return (ub.mean(1) >= box_thr) | (ub.max(1) >= cell_thr)


def run_seed(s, args, x_tr, x_va, xflat_te, refs, box_thr, cell_thr, env, parts):
    """Fit the 2 x 2 generators with seed s and evaluate them on the test winters."""
    t0 = time.time()
    gens, m_evt = fit_two_by_two(x_tr, x_va, seed=s, vae_kw={"epochs": args.vae_epochs})
    vdep = gens[("evt", "vae")].dep
    part = {"vae": {"seed": s, "best_epoch": int(vdep.best_epoch), "latent_components": int(vdep.model.latent_k),
                    "latent_scores": vdep.model.latent_scores}, "records": {}, "candidates": {}}
    if s == 0:
        part["fitted"] = {"xi": m_evt.xi, "threshold_q": m_evt.q,
                          "sigma_map": m_evt.sigma.reshape(16, 16), "u_map": m_evt.u.reshape(16, 16)}
    torch.save(vdep, parts / f"vae_seed{s}.pt")
    for dep in ["gaussian", "vae"]:
        if dep == "gaussian" and s > 0:
            continue                                    # the Gaussian copula has no training randomness
        g_emp, g_evt = gens[("empirical", dep)], gens[("evt", dep)]
        margins = {g_emp.name: g_emp.margins, g_evt.name: g_evt.margins}
        hits = {n: [0, 0] for n in margins}
        cand = []

        def count(name, f):                            # Desmond-sized days, same draws as quantiles
            hits[name][0] += int((f.mean((1, 2)) >= box_thr).sum())
            hits[name][1] += int((f.max((1, 2)) >= cell_thr).sum())

        def keep(z):                                   # draws that could be Desmond-sized under
            c = env.candidates(z, box_thr, cell_thr)   # ANY bootstrap margin (kept for later)
            cand.append(z[c].copy())                   # original dtype: exact re-evaluation

        qs = paired_quantiles(g_evt.dep, margins, args.n_model, PS, seed=100 + s, extra=count, on_z=keep)
        part["candidates"][dep] = np.concatenate(cand)
        z = g_evt.dep.sample_normal(args.n_chi, np.random.default_rng(200 + s))
        # Does the dependence model produce extreme normal scores at the right rate? Target: 1 - p.
        z_tail = {p: float((z > stats.norm.ppf(p)).mean() / (1 - p)) for p in Z_LEVELS}
        for g in (g_emp, g_evt):
            qm = qs[g.name]
            xs = g.margins.from_normal(z).reshape(-1, 16, 16)
            cm = chi_by_distance(xs, US)               # rank-based: identical for both margins
            bulk = bulk_summaries(xs)
            pb, pc = hits[g.name][0] / args.n_model, hits[g.name][1] / args.n_model
            rec = {
                "seed": s,
                "log_ratio_vs_test": mean_log_ratio(qm, refs["q_test"]),
                "log_ratio_vs_train": mean_log_ratio(qm, refs["q_train"]),
                "quantile_tile_mean": {k: {p: float(qm[k][p].mean()) for p in PS} for k in TILES},
                "test_exceedance_of_model_quantile": {p: float((xflat_te > qm[1][p]).mean()) for p in PS},
                "normal_score_exceedance_ratio": z_tail,
                "chi": cm,
                "chi_abs_error": {u: float(np.mean(np.abs(cm[u] - refs["chi_test"][u]))) for u in US},
                "chi_abs_error_vs_train": {u: float(np.mean(np.abs(cm[u] - refs["chi_train"][u]))) for u in US},
                "wet_fraction_quantiles": bulk["wet_fraction_quantiles"],
                **bulk_errors(bulk, refs["bulk_test"]),
                "vs_train": bulk_errors(bulk, refs["bulk_train"]),
                "desmond": {"p_box": pb, "p_cell": pc, "n": args.n_model,
                            "hits_box": hits[g.name][0], "hits_cell": hits[g.name][1],
                            "return_period_winters_box": return_period_winters(pb),
                            "return_period_winters_cell": return_period_winters(pc)},
                "example_extreme_fields": xs[np.argsort(xs.mean((1, 2)))[-3:]].tolist(),
            }
            part["records"][g.name] = tolist(rec)
    print(f"seed {s} fitted and evaluated ({time.time() - t0:.0f}s); candidates kept: "
          f"{ {k: len(v) for k, v in part['candidates'].items()} }", flush=True)
    return part


def fit_boot_margins(b, sp, x_tr, trw):
    rng = np.random.default_rng([7, b])             # per-replicate stream: resumable
    ws = rng.choice(sp.train_winters, size=len(sp.train_winters), replace=True)
    idx = np.concatenate([np.flatnonzero(trw == w) for w in ws])
    m = Margins("evt").fit(x_tr[idx].reshape(len(idx), -1))
    m.pos_sorted = None                             # only ppf is needed: keep memory small
    m.P = m.P.astype(np.float32)
    return m


def desmond_hits(m, z, box_thr, cell_thr):
    if len(z) == 0:
        return 0, 0
    f = m.from_normal(z).reshape(len(z), 16, 16)      # same operations as the point estimate
    return int((f.mean((1, 2)) >= box_thr).sum()), int((f.max((1, 2)) >= cell_thr).sum())


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--data", default=str(ROOT / "data/cumbria_winter_rainfall_5km.nc"))
    a.add_argument("--seeds", type=int, default=3)
    a.add_argument("--n-model", type=int, default=2_000_000)
    a.add_argument("--n-chi", type=int, default=100_000)
    a.add_argument("--boot-test", type=int, default=500)
    a.add_argument("--boot-train", type=int, default=100)
    a.add_argument("--z-noise-reps", type=int, default=20)
    a.add_argument("--vae-epochs", type=int, default=400)
    a.add_argument("--out", default=str(ROOT / "results/cumbria_experiment.json"))
    a.add_argument("--parts", default=str(ROOT / "results/cumbria_parts"))
    args = a.parse_args()
    t_start = time.time()

    x, t, winter, _ = load(args.data)
    sp = interleaved_split(winter)
    x_tr, x_va, x_te = x[sp.train], x[sp.val], x[sp.test]
    d_idx = int(np.flatnonzero([str(v)[:10] == DESMOND_DATE for v in t])[0])
    assert d_idx in set(sp.test), "Storm Desmond must be in the test set"
    desmond = x[d_idx]
    box_thr, cell_thr = float(desmond.mean()), float(desmond.max())
    out = {"settings": tolist(vars(args)) | {"ps": PS, "us": US, "tiles": TILES},
           "split": {"train_winters": sp.train_winters.tolist(), "val_winters": sp.val_winters.tolist(),
                     "test_winters": sp.test_winters.tolist(),
                     "n_days": [len(sp.train), len(sp.val), len(sp.test)]},
           "desmond": {"date": DESMOND_DATE, "box_mean_mm": box_thr, "max_cell_mm": cell_thr,
                       "field": desmond.tolist(),
                       "train_max_box_mean": float(x_tr.mean((1, 2)).max()),
                       "train_max_cell": float(x_tr.max())}}

    # ---------------- references: TEST winters (with block bootstrap), and the floors
    refs = {"q_test": empirical_quantiles(x_te, PS), "q_train": empirical_quantiles(x_tr, PS),
            "q_val": empirical_quantiles(x_va, PS),
            "chi_test": chi_by_distance(x_te, US), "chi_train": chi_by_distance(x_tr, US),
            "chi_val": chi_by_distance(x_va, US),
            "bulk_test": bulk_summaries(x_te), "bulk_train": bulk_summaries(x_tr),
            "bulk_val": bulk_summaries(x_va)}
    xflat_te, xflat_tr = x_te.reshape(len(x_te), -1), x_tr.reshape(len(x_tr), -1)
    q_cell_train = {p: np.quantile(xflat_tr, p, axis=0) for p in PS}
    rng = np.random.default_rng(2024)
    tw = winter[sp.test]
    test_w = sp.test_winters
    boot_q = {k: {p: [] for p in PS} for k in TILES}
    boot_floor = {k: {p: [] for p in PS} for k in TILES}     # log(train / resampled test)
    boot_exc = {p: [] for p in PS}                           # test days above TRAINING per-cell quantile
    boot_chi = {u: [] for u in US}
    for b in range(args.boot_test):
        ws = rng.choice(test_w, size=len(test_w), replace=True)
        idx = np.concatenate([np.flatnonzero(tw == w) for w in ws])
        xb = x_te[idx]
        qb = empirical_quantiles(xb, PS)
        for k in TILES:
            for p in PS:
                boot_q[k][p].append(qb[k][p].mean())       # tile-averaged quantile
                boot_floor[k][p].append(np.mean(np.log(refs["q_train"][k][p] / qb[k][p])))
        for p in PS:
            boot_exc[p].append(float((xb.reshape(len(xb), -1) > q_cell_train[p]).mean()))
        if b < 200:
            cb = chi_by_distance(xb, US)
            for u in US:
                boot_chi[u].append(cb[u])
    pct = lambda v: np.percentile(np.asarray(v), [2.5, 97.5], axis=0)
    out["reference"] = {
        "quantile_tile_mean": {k: {p: float(refs["q_test"][k][p].mean()) for p in PS} for k in TILES},
        "quantile_tile_mean_ci": {k: {p: pct(boot_q[k][p]) for p in PS} for k in TILES},
        "quantile_maps": {p: refs["q_test"][1][p].reshape(16, 16) for p in PS},
        "chi": refs["chi_test"], "chi_ci": {u: pct(boot_chi[u]) for u in US},
        "wet_fraction_quantiles": refs["bulk_test"]["wet_fraction_quantiles"],
        "cell_mean": refs["bulk_test"]["cell_mean"].reshape(16, 16),
    }
    out["floors"] = {
        "note": "what a model reproducing the TRAINING (or validation) winters exactly would score "
                "against the test winters; intervals: block bootstrap over test winters",
        "log_ratio_train_vs_test": mean_log_ratio(refs["q_train"], refs["q_test"]),
        "log_ratio_train_vs_test_ci": {k: {p: pct(boot_floor[k][p]) for p in PS} for k in TILES},
        "log_ratio_val_vs_test": mean_log_ratio(refs["q_val"], refs["q_test"]),
        "test_exceedance_of_train_quantile": {p: float((xflat_te > q_cell_train[p]).mean()) for p in PS},
        "test_exceedance_of_train_quantile_ci": {p: pct(boot_exc[p]) for p in PS},
        "test_exceedance_by_winter": {p: [float((xflat_te[tw == w] > q_cell_train[p]).mean()) for w in test_w]
                                      for p in PS},
        "chi_train": refs["chi_train"], "chi_val": refs["chi_val"],
        "chi_abs_error_train_vs_test": {u: float(np.mean(np.abs(refs["chi_train"][u] - refs["chi_test"][u])))
                                        for u in US},
        "chi_abs_error_val_vs_test": {u: float(np.mean(np.abs(refs["chi_val"][u] - refs["chi_test"][u])))
                                      for u in US},
        "bulk_train_vs_test": bulk_errors(refs["bulk_train"], refs["bulk_test"]),
        "bulk_val_vs_test": bulk_errors(refs["bulk_val"], refs["bulk_test"]),
    }
    print(f"reference and floors done ({time.time() - t_start:.0f}s)", flush=True)

    # ---------------- bootstrap margins over TRAINING winters (refitted EVT margins) + envelope
    trw = winter[sp.train]
    xt_flat = x_tr.reshape(len(x_tr), -1)
    m_evt0, m_emp0 = Margins("evt").fit(xt_flat), Margins("empirical").fit(xt_flat)
    env = Envelope()
    env.add(m_evt0); env.add(m_emp0)
    boot_m = []
    for b in range(args.boot_train):
        boot_m.append(fit_boot_margins(b, sp, x_tr, trw))
        env.add(boot_m[-1])
        if (b + 1) % 10 == 0:
            print(f"  bootstrap margins {b + 1}/{args.boot_train} ({time.time() - t_start:.0f}s)", flush=True)

    # ---------------- fit the 2 x 2 generators (one fit per seed) and evaluate on TEST winters
    out["models"] = {}
    parts = Path(args.parts)
    parts.mkdir(parents=True, exist_ok=True)
    cands = {"gaussian": [], "vae": []}
    for s in range(args.seeds):
        seed_file = parts / f"seed{s}.json"
        cand_file = parts / f"candidates_seed{s}.npz"
        if seed_file.exists() and cand_file.exists():       # resume: this seed is already done
            part = json.loads(seed_file.read_text())
            part["candidates"] = dict(np.load(cand_file))
            print(f"seed {s} loaded from {seed_file.name}", flush=True)
        else:
            part = run_seed(s, args, x_tr, x_va, xflat_te, refs, box_thr, cell_thr, env, parts)
            np.savez(cand_file, **part["candidates"])
            seed_file.write_text(json.dumps(tolist({k: v for k, v in part.items() if k != "candidates"})))
        if "fitted" in part:
            out["fitted"] = part["fitted"]
        out.setdefault("vae", []).append(part["vae"])
        for name, rec in part["records"].items():
            out["models"].setdefault(name, []).append(rec)
        for dep, z in part["candidates"].items():
            cands[dep].append(z)
    zc = {dep: np.concatenate(v) if v else np.zeros((0, 256)) for dep, v in cands.items()}

    # Consistency check: the candidates reproduce the point-estimate hit counts exactly.
    for dep, name in [("gaussian", "Gaussian copula"), ("vae", "VAE")]:
        for mk, m in [("EVT", m_evt0), ("empirical", m_emp0)]:
            hb, hc = desmond_hits(m, zc[dep], box_thr, cell_thr)
            recs = out["models"][f"{name} + {mk} margins"]
            assert hb == sum(r["desmond"]["hits_box"] for r in recs), (dep, mk, "box")
            assert hc == sum(r["desmond"]["hits_cell"] for r in recs), (dep, mk, "cell")
    print("candidate screen reproduces the point-estimate hit counts exactly", flush=True)

    # ---------------- Desmond probabilities under each bootstrap margin (same draws, exact)
    n_draws = {"gaussian": args.n_model, "vae": args.n_model * args.seeds}
    boot = {"Gaussian copula + EVT margins": [], "VAE + EVT margins": []}
    for m in boot_m:
        for dep, name in [("gaussian", "Gaussian copula + EVT margins"), ("vae", "VAE + EVT margins")]:
            hb, hc = desmond_hits(m, zc[dep], box_thr, cell_thr)
            boot[name].append({"hits_box": hb, "hits_cell": hc, "n": n_draws[dep],
                               "p_box": hb / n_draws[dep], "p_cell": hc / n_draws[dep]})
    out["bootstrap_train"] = {
        "note": "EVT margins refitted on resampled training winters; dependence models fixed; "
                "Gaussian copula: 2e6 draws; VAE: the 3 seeds' 2e6 draws each, pooled",
        "xi": [m.xi for m in boot_m], "models": boot,
        "n_candidates": {k: len(v) for k, v in zc.items()}}

    # ---------------- Monte Carlo noise reference for the normal-score exceedance ratio
    gc = GaussianCopula().fit(m_evt0.to_normal(xt_flat, np.random.default_rng(0)).reshape(-1, 16, 16))
    zr = {p: [] for p in Z_LEVELS}
    for r in range(args.z_noise_reps):
        z = gc.sample_normal(args.n_chi, np.random.default_rng([300, r]))
        for p in Z_LEVELS:
            zr[p].append(float((z > stats.norm.ppf(p)).mean() / (1 - p)))
    out["normal_score_noise"] = {
        "note": f"ratio for a Gaussian copula (exactly normal margins) in {args.z_noise_reps} "
                f"independent samples of {args.n_chi} fields: pure Monte Carlo noise",
        "ratios": zr, "range": {p: [min(v), max(v)] for p, v in zr.items()}}
    out["runtime_s"] = time.time() - t_start
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(tolist(out)))
    print(f"saved {args.out} ({out['runtime_s'] / 60:.0f} min)")


if __name__ == "__main__":
    main()
