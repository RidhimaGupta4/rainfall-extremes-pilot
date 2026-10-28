"""Per-cell marginal distributions, fitted on TRAINING days only.

Two variants, which differ only above a high threshold:
* "evt": empirical below the threshold u_s (the q-quantile of cell s), generalised Pareto
  (GPD) above it, with cell-specific scale sigma_s and one shape xi shared by all cells.
  This can extrapolate beyond the largest training value.
* "empirical": empirical throughout, so it can never exceed the largest training value.
  This is what a generator implicitly does when it simply learns the data distribution.

Both map rainfall (mm) <-> uniform <-> standard normal scores. Ties at exactly 0 mm are
broken with a randomised probability integral transform.
"""
import numpy as np
from scipy import stats

XI_GRID = np.round(np.arange(-0.2475, 0.6, 0.005), 4)   # offset so xi = 0 exactly is never used


def _gpd_nll(y, n, sigma, xi):
    """Negative log-likelihood of GPD(sigma, xi) for excesses y (padded with NaN), per cell."""
    z = 1.0 + xi * y / sigma[:, None]
    bad = np.nansum(np.where(np.isnan(y), 0, z <= 0), axis=1) > 0
    if abs(xi) < 1e-8:
        nll = n * np.log(sigma) + np.nansum(y, axis=1) / sigma
    else:
        with np.errstate(invalid="ignore", divide="ignore"):
            nll = n * np.log(sigma) + (1.0 + 1.0 / xi) * np.nansum(np.log(z), axis=1)
    return np.where(bad, np.inf, nll)


def _profile_sigma(y, n, xi, iters=80):
    """For fixed xi, minimise each cell's GPD NLL over log(sigma) by vectorised golden section."""
    scale = np.nanmean(y, axis=1)
    lo, hi = np.log(scale) - 4.0, np.log(scale) + 4.0
    g = (np.sqrt(5) - 1) / 2
    a, b = hi - g * (hi - lo), lo + g * (hi - lo)
    fa, fb = _gpd_nll(y, n, np.exp(a), xi), _gpd_nll(y, n, np.exp(b), xi)
    for _ in range(iters):
        left = fa < fb
        hi = np.where(left, b, hi); lo = np.where(left, lo, a)
        b_new = np.where(left, a, lo + g * (hi - lo)); a_new = np.where(left, hi - g * (hi - lo), b)
        a, b = a_new, b_new
        fa, fb = _gpd_nll(y, n, np.exp(a), xi), _gpd_nll(y, n, np.exp(b), xi)
    s = np.exp((lo + hi) / 2)
    return s, _gpd_nll(y, n, s, xi)


def fit_gpd_shared_xi(excess_lists):
    """MLE of per-cell sigma_s and a single shared xi (profile likelihood over a xi grid).
    Treats cells as independent for estimation: the point estimate is fine, but likelihood-
    based standard errors would be wrong, so uncertainty is assessed by block bootstrap."""
    n = np.array([len(e) for e in excess_lists], dtype=float)
    y = np.full((len(excess_lists), int(n.max())), np.nan)
    for i, e in enumerate(excess_lists):
        y[i, :len(e)] = e
    totals = []
    for xi in XI_GRID:
        totals.append(_profile_sigma(y, n, xi)[1].sum())
    k = int(np.argmin(totals))
    xi = XI_GRID[k]
    sigma, _ = _profile_sigma(y, n, xi)
    return sigma, float(xi), np.array(totals)


class Margins:
    def __init__(self, tail="evt", q=0.95):
        assert tail in ("evt", "empirical")
        self.tail, self.q = tail, q

    def fit(self, x_train):
        """x_train: [n_days, n_cells] rainfall in mm, TRAINING days only."""
        x = np.asarray(x_train, dtype=float)
        self.n, self.ncell = x.shape
        self.p0 = (x == 0).mean(0)                                  # P(exactly 0 mm)
        self.pos_sorted = [np.sort(c[c > 0]) for c in x.T]          # positive values
        self.u = np.quantile(x, self.q, axis=0)                     # thresholds
        self.zeta = (x > self.u).mean(0)                            # P(X > u)
        self.max_train = x.max(0)
        # Padded matrix of sorted positive values, for the vectorised quantile function
        self.m_pos = np.array([len(p) for p in self.pos_sorted])
        M = self.m_pos.max()
        self.P = np.vstack([np.pad(p, (0, M - len(p)), mode="edge") for p in self.pos_sorted])
        if self.tail == "evt":
            exc = [c[c > u] - u for c, u in zip(x.T, self.u)]
            self.sigma, self.xi, self.xi_profile = fit_gpd_shared_xi(exc)
        return self

    # ---- empirical part (positive values), linear interpolation between order statistics
    def _G(self, s, v):
        p = self.pos_sorted[s]; m = len(p)
        pp = np.arange(1, m + 1) / (m + 1)
        return np.interp(v, p, pp, left=0.5 / (m + 1), right=m / (m + 1))

    def _Ginv(self, s, g):
        p = self.pos_sorted[s]; m = len(p)
        pp = np.arange(1, m + 1) / (m + 1)
        return np.interp(g, pp, p, left=p[0], right=p[-1])

    def cdf(self, x, rng=None):
        """Rainfall [n, n_cells] -> uniform (0,1). Exact zeros get U(0, p0) if rng is given."""
        x = np.asarray(x, dtype=float)
        out = np.empty_like(x)
        for s in range(self.ncell):
            v = x[:, s]
            body = self.p0[s] + (1 - self.p0[s]) * self._G(s, v)
            if self.tail == "evt":
                above = v > self.u[s]
                with np.errstate(invalid="ignore"):
                    tail = 1 - self.zeta[s] * np.power(
                        np.maximum(1 + self.xi * (v - self.u[s]) / self.sigma[s], 1e-300), -1 / self.xi)
                body = np.where(above, tail, body)
            zero = v == 0
            if rng is not None:
                body = np.where(zero, rng.uniform(0, self.p0[s], size=v.shape), body)
            else:
                body = np.where(zero, self.p0[s] / 2, body)
            out[:, s] = body
        return np.clip(out, 1e-12, 1 - 1e-12)

    def ppf(self, p):
        """Uniform (0,1) [n, n_cells] -> rainfall in mm. Vectorised over cells: the plotting
        positions i/(m+1) are evenly spaced, so linear interpolation between order statistics
        reduces to index arithmetic (identical to np.interp, see tests)."""
        p = np.asarray(p, dtype=float)
        g = np.clip((p - self.p0) / (1 - self.p0), 0, 1)
        m = self.m_pos
        pos = np.clip(g * (m + 1) - 1, 0, m - 1)
        i = np.minimum(np.floor(pos).astype(np.int64), m - 2)
        f = pos - i
        cols = np.arange(self.ncell)
        lo, hi = self.P[cols, i], self.P[cols, i + 1]
        v = np.where(p <= self.p0, 0.0, lo + f * (hi - lo))
        if self.tail == "evt":
            above = p > 1 - self.zeta
            with np.errstate(invalid="ignore", divide="ignore"):
                tail = self.u + self.sigma / self.xi * (
                    np.power(self.zeta / np.maximum(1 - p, 1e-300), self.xi) - 1)
            v = np.where(above, tail, v)
        return v

    def ppf_reference(self, p):
        """Slow per-cell version of ppf using np.interp, kept for testing."""
        p = np.asarray(p, dtype=float)
        out = np.empty_like(p)
        for s in range(self.ncell):
            u = p[:, s]
            g = np.clip((u - self.p0[s]) / (1 - self.p0[s]), 0, 1)
            v = np.where(u <= self.p0[s], 0.0, self._Ginv(s, g))
            if self.tail == "evt":
                above = u > 1 - self.zeta[s]
                with np.errstate(invalid="ignore", divide="ignore"):
                    tail = self.u[s] + self.sigma[s] / self.xi * (
                        np.power(self.zeta[s] / np.maximum(1 - u, 1e-300), self.xi) - 1)
                v = np.where(above, tail, v)
            out[:, s] = v
        return out

    def to_normal(self, x, rng=None):
        return stats.norm.ppf(self.cdf(x, rng))

    def from_normal(self, z):
        return self.ppf(stats.norm.cdf(z))
