"""Diagnostics, each targeted at a question in the E5 project advert.

* marginal tail quantiles per cell              -> local intensities
* quantiles of rainfall averaged over areas     -> "rainfall accumulated over ... catchment sizes"
* chi(u) against distance                       -> extremal dependence / spatial extent
* wet-area fraction and bulk correlation        -> "dry regions and ordinary rainfall", visual realism
* probability of a Desmond-sized day            -> a withheld severe event as a stress test

All functions take fields in mm, shape [n, 16, 16]. Rank-based quantities (chi) are computed
within each dataset, so they do not depend on the margins.
"""
import numpy as np

from .data import cell_distances_km

TILES = (1, 2, 4, 8, 16)
DIST_BINS = np.array([5, 10, 20, 30, 40, 50, 60, 70, 80, 100])        # band labels (km)
DIST_EDGES = np.array([2.5, 7.5, 15, 25, 35, 45, 55, 65, 75, 90, 110])  # contiguous bands
WET_MM = 1.0


def tile_means(x, k):
    """Average over non-overlapping k x k tiles: [n, 16, 16] -> [n, (16/k)^2]."""
    n = len(x)
    return x.reshape(n, 16 // k, k, 16 // k, k).mean((2, 4)).reshape(n, -1)


class QuantileAccumulator:
    """Upper quantiles (levels >= min(ps)) of every cell and tile size, fed batch by batch.

    Each batch is trimmed straight away to its largest values (3x the expected number above
    the lowest level requested, plus 30). The result is exact provided no batch had all of its
    kept values above the final quantile; this is checked in result(), which raises otherwise."""

    def __init__(self, ps, tiles=TILES):
        self.ps, self.tiles, self.worst, self.n = tuple(ps), tuple(tiles), 1 - min(ps), 0
        self.kept = {k: [] for k in tiles}
        self.kept_min = {k: [] for k in tiles}

    def add(self, b):
        b = np.asarray(b, dtype=np.float64)
        self.n += len(b)
        m = min(len(b), int(np.ceil(3 * self.worst * len(b))) + 30)
        for k in self.tiles:
            v = tile_means(b, k)
            top = v if m >= len(v) else np.partition(v, len(v) - m, axis=0)[-m:].copy()   # copy: a view would keep the whole batch alive
            self.kept[k].append(top)
            self.kept_min[k].append(top.min(0) if m < len(v) else np.full(v.shape[1], -np.inf))
        return self

    def result(self):
        out = {}
        for k in self.tiles:
            v = -np.sort(-np.vstack(self.kept[k]), axis=0)
            out[k] = {}
            for p in self.ps:
                r = int(np.floor(self.n * (1 - p)))     # values strictly above the quantile
                if r >= len(v):
                    raise ValueError("not enough upper values kept")
                out[k][p] = v[r]
            q_low = out[k][min(self.ps)]
            if np.any(np.vstack(self.kept_min[k]) > q_low):
                raise ValueError("a batch was trimmed too aggressively; increase the margin")
        return out


def stream_quantiles(batches, ps, tiles=TILES):
    """Quantiles at levels ps for every cell and every tile size, from batches of fields."""
    acc = QuantileAccumulator(ps, tiles)
    for b in batches:
        acc.add(b)
    return acc.result()


def empirical_quantiles(x, ps, tiles=TILES):
    return {k: {p: np.quantile(tile_means(x, k), p, axis=0) for p in ps} for k in tiles}


def chi_by_distance(x, us=(0.95, 0.98), dist=None):
    """chi_ij(u) = P(U_i > u, U_j > u) / (1 - u) with U from within-dataset ranks,
    averaged over cell pairs in contiguous distance bands (edges DIST_EDGES, labels DIST_BINS)."""
    n = len(x)
    f = x.reshape(n, -1)
    ranks = np.argsort(np.argsort(f, axis=0, kind="stable"), axis=0, kind="stable")
    uu = (ranks + 1) / (n + 1)
    dist = cell_distances_km() if dist is None else dist
    out = {}
    for u in us:
        e = (uu > u).astype(np.float32)
        joint = (e.T @ e) / n / (1 - u)
        out[u] = np.array([joint[(dist > lo) & (dist <= hi)].mean()
                           for lo, hi in zip(DIST_EDGES[:-1], DIST_EDGES[1:])])
    return out


def bulk_summaries(x):
    n = len(x)
    f = x.reshape(n, -1)
    wet = (f > WET_MM).mean(1)
    return {
        "wet_fraction_quantiles": np.quantile(wet, [0.1, 0.25, 0.5, 0.75, 0.9]),
        "dry_day_share": float((wet == 0).mean()),
        "corr_log": np.corrcoef(np.log1p(f).T),
        "cell_mean": f.mean(0),
    }


def exceedance_prob(batches, box_threshold, cell_threshold):
    """P(box-mean >= box_threshold) and P(max cell >= cell_threshold), from batches."""
    n = hb = hc = 0
    for b in batches:
        n += len(b)
        hb += int((b.mean((1, 2)) >= box_threshold).sum())
        hc += int((b.max((1, 2)) >= cell_threshold).sum())
    return hb / n, hc / n, n, hb, hc


DAYS_PER_WINTER = 182.25


def return_period_winters(p_daily):
    """Average number of winters between such days, ignoring day-to-day clustering."""
    return np.inf if p_daily == 0 else 1.0 / (p_daily * DAYS_PER_WINTER)
