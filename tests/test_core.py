import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rainpilot.data import Split, interleaved_split            # noqa: E402
from rainpilot.diagnostics import chi_by_distance, stream_quantiles, tile_means  # noqa: E402
from rainpilot.margins import Margins                          # noqa: E402

WINTERS = np.repeat(np.arange(1961, 2025), 182)


def test_split_is_disjoint_by_winter_and_complete():
    sp = interleaved_split(WINTERS)
    assert set(WINTERS[sp.test]) == {w for w in range(1961, 2025) if w % 5 == 0}
    assert 2015 in set(WINTERS[sp.test])                        # Storm Desmond winter is held out
    assert len(sp.train) + len(sp.val) + len(sp.test) == len(WINTERS)


def test_split_check_catches_leakage():
    sp = interleaved_split(WINTERS)
    leaky = Split(np.append(sp.train, sp.test[:1]), sp.val, sp.test,
                  sp.train_winters, sp.val_winters, sp.test_winters)
    with pytest.raises(AssertionError):
        leaky.check(WINTERS)


@pytest.mark.parametrize("tail", ["evt", "empirical"])
def test_margins_round_trip(tail):
    rng = np.random.default_rng(0)
    x = rng.gamma(0.7, 6.0, size=(4000, 3)) * (rng.random((4000, 3)) > 0.1)   # with exact zeros
    m = Margins(tail).fit(x)
    pos = x > 0
    assert np.allclose(m.ppf(m.cdf(x))[pos], x[pos], atol=1e-8)


def test_gpd_tail_recovers_known_shape():
    rng = np.random.default_rng(1)
    xi_true = 0.15
    x = stats.genpareto.rvs(xi_true, scale=8.0, size=(20000, 4), random_state=rng)
    m = Margins("evt", q=0.9).fit(x)
    assert abs(m.xi - xi_true) < 0.05
    p = 1 - 1e-4                                    # beyond what the empirical margin can reach
    truth = stats.genpareto.ppf(p, xi_true, scale=8.0)
    est = m.ppf(np.full((1, 4), p))[0]
    assert np.all(np.abs(np.log(est / truth)) < 0.25)
    emp = Margins("empirical").fit(x).ppf(np.full((1, 4), p))[0]
    assert np.all(emp <= x.max(0) + 1e-9)           # empirical margins cannot extrapolate


def test_stream_quantiles_match_exact():
    rng = np.random.default_rng(2)
    data = rng.gamma(0.8, 5.0, size=(60000, 16, 16))
    batches = (data[k:k + 7000] for k in range(0, len(data), 7000))
    sq = stream_quantiles(batches, ps=(0.99, 0.999), tiles=(1, 4, 16))
    for k in (1, 4, 16):
        v = tile_means(data, k)
        for p in (0.99, 0.999):
            r = int(np.floor(len(v) * (1 - p)))
            exact = -np.sort(-v, axis=0)[r]
            assert np.allclose(sq[k][p], exact)


def test_chi_known_cases():
    rng = np.random.default_rng(3)
    indep = rng.standard_normal((40000, 16, 16))
    chi = chi_by_distance(indep, us=(0.95,))[0.95]
    assert np.all(np.abs(chi[1:] - 0.05) < 0.02)    # independent: chi(u) = 1 - u
    common = np.repeat(rng.standard_normal((40000, 1, 1)), 16, 1).repeat(16, 2)
    chi = chi_by_distance(common + 1e-9 * rng.standard_normal(common.shape), us=(0.95,))[0.95]
    assert np.all(chi > 0.95)                       # perfectly dependent: chi(u) = 1


@pytest.mark.parametrize("tail", ["evt", "empirical"])
def test_vectorised_ppf_matches_reference(tail):
    rng = np.random.default_rng(4)
    x = rng.gamma(0.7, 6.0, size=(3000, 5)) * (rng.random((3000, 5)) > 0.2)
    x[:, 2] = rng.gamma(0.7, 6.0, size=3000)            # one cell without zeros
    m = Margins(tail).fit(x)
    u = np.vstack([rng.random((20000, 5)), np.full((1, 5), 1e-9), np.full((1, 5), 1 - 1e-9),
                   np.tile(m.p0, (1, 1)), np.tile(1 - m.zeta, (1, 1))])
    assert np.allclose(m.ppf(u), m.ppf_reference(u), rtol=1e-10, atol=1e-10)


def test_envelope_bounds_every_margin_including_the_tail_step():
    """The screen used for the Desmond bootstrap must never be below any margin's value."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    from cumbria_experiment import Envelope
    rng = np.random.default_rng(3)
    x = rng.gamma(0.6, 8.0, size=(4000, 6)) * (rng.uniform(size=(4000, 6)) > 0.3)
    margins = [Margins("evt").fit(x[rng.integers(0, 4000, 4000)]) for _ in range(4)]
    margins.append(Margins("empirical").fit(x))
    env = Envelope(ncell=6)
    for m in margins:
        env.add(m)
    # dense draws everywhere, plus draws just below and above each margin's tail step
    z = [rng.normal(size=(20000, 6)) * 1.5]
    for m in margins[:-1]:
        zs = stats.norm.ppf(1 - m.zeta)
        z.append(zs + rng.uniform(-1e-3, 1e-3, size=(2000, 6)))
    z = np.concatenate(z)
    b = env.bound(z)
    for m in margins:
        assert np.all(b >= m.from_normal(z) - 1e-12)
