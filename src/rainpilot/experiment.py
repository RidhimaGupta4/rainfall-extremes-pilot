"""Shared machinery for the simulation study and the Cumbria experiment."""
import numpy as np
from scipy import stats

from .diagnostics import TILES, chi_by_distance, stream_quantiles
from .generators import SHAPE, GaussianCopula, Generator, VAEDependence
from .margins import Margins

MODELS = [("empirical", "gaussian"), ("evt", "gaussian"), ("empirical", "vae"), ("evt", "vae")]


def fit_two_by_two(x_train, x_val, seed=0, q=0.95, vae_kw=None):
    """Fit the four generators so that they share components exactly:
    both margin types are fitted on the same training days; the dependence models are fitted
    once, on normal scores from the EVT margins, and reused for both margin types.
    Below the threshold the two margin types are identical, so the normal scores differ only
    for the top 5% of values in each cell."""
    rng = np.random.default_rng(seed)
    xt = x_train.reshape(len(x_train), -1)
    xv = x_val.reshape(len(x_val), -1)
    m_evt = Margins("evt", q).fit(xt)
    m_emp = Margins("empirical", q).fit(xt)
    zt = m_evt.to_normal(xt, rng).reshape(-1, *SHAPE)
    zv = m_evt.to_normal(xv, rng).reshape(-1, *SHAPE)
    deps = {"gaussian": GaussianCopula().fit(zt),
            "vae": VAEDependence(seed, **(vae_kw or {})).fit(zt, zv)}
    gens = {}
    for marg, dep in MODELS:
        g = Generator(marg, dep, seed=seed, q=q)
        g.margins = m_evt if marg == "evt" else m_emp
        g.dep = deps[dep]
        gens[(marg, dep)] = g
    return gens, m_evt


def model_quantiles(gen, n, ps, seed, tiles=TILES, batch=25_000):
    rng = np.random.default_rng(seed)
    return stream_quantiles(gen.sample_batches(n, rng, batch), ps, tiles)


def paired_quantiles(dep, margins_by_name, n, ps, seed, tiles=TILES, batch=25_000, extra=None,
                     on_z=None):
    """Common random numbers: draw normal-score fields from ONE dependence model and map each
    batch through every margin type. Returns {name: quantiles}. `extra(name, fields)` is an
    optional callback per batch (e.g. counting Desmond-sized days); `on_z(z)` receives each
    batch of normal scores [b, 256]."""
    from .diagnostics import QuantileAccumulator
    rng = np.random.default_rng(seed)
    accs = {name: QuantileAccumulator(ps, tiles) for name in margins_by_name}
    for k in range(0, n, batch):
        z = dep.sample_normal(min(batch, n - k), rng)
        if on_z is not None:
            on_z(z)
        for name, m in margins_by_name.items():
            fields = m.from_normal(z).reshape(len(z), *SHAPE)
            accs[name].add(fields)
            if extra is not None:
                extra(name, fields)
    return {name: a.result() for name, a in accs.items()}


def model_chi(gen, n, us, seed):
    return chi_by_distance(gen.sample(n, np.random.default_rng(seed)), us)


def log_ratio(q_model, q_ref, ps, tiles=TILES):
    """Mean over tiles of log(model / reference) quantile, per tile size and level.
    0 = unbiased; -0.1 = about 10% too low."""
    return {k: {p: float(np.mean(np.log(np.maximum(q_model[k][p], 1e-9) / q_ref[k][p])))
                for p in ps} for k in tiles}


def log_ratio_abs(q_model, q_ref, ps, tiles=TILES):
    """Mean over tiles of |log(model / reference)|: errors of opposite sign do not cancel."""
    return {k: {p: float(np.mean(np.abs(np.log(np.maximum(q_model[k][p], 1e-9) / q_ref[k][p]))))
                for p in ps} for k in tiles}


# ---------------------------------------------------------------- "true worlds" for simulation
class TrueWorld:
    """Known data-generating process: given margins + a Gaussian or Student-t copula."""

    def __init__(self, margins, corr, copula="t", df=4):
        self.margins, self.copula, self.df = margins, copula, df
        self.L = np.linalg.cholesky(corr + 1e-8 * np.eye(len(corr)))

    def sample(self, n, rng):
        if n > 25_000:                                   # keep temporary arrays small
            return np.concatenate(list(self.sample_batches(n, rng)))
        z = rng.standard_normal((n, self.L.shape[0])) @ self.L.T
        if self.copula == "t":
            w = rng.chisquare(self.df, size=(n, 1)) / self.df
            u = stats.t.cdf(z / np.sqrt(w), self.df)
        else:
            u = stats.norm.cdf(z)
        return self.margins.ppf(u).reshape(n, *SHAPE)

    def sample_batches(self, n, rng, batch=25_000):
        for k in range(0, n, batch):
            yield self.sample(min(batch, n - k), rng)
