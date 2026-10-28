"""Loading the Cumbria rainfall fields and the pre-specified, leakage-free winter split."""
from dataclasses import dataclass

import numpy as np
import xarray as xr

CELL_KM = 5.0
DESMOND_DATE = "2015-12-05"


def load(path):
    """Return (fields [n_days, 16, 16] in mm, times, winter label per day, DataArray)."""
    da = xr.open_dataarray(path).load()
    x = da.values.astype(np.float64)
    if np.isnan(x).any() or (x < 0).any():
        raise ValueError("Data contain NaN or negative values; check the download.")
    t = da.time.values
    year, month = da.time.dt.year.values, da.time.dt.month.values
    winter = np.where(month >= 10, year, year - 1)   # Dec 2015 and Jan 2016 -> winter 2015
    return x, t, winter, da


@dataclass(frozen=True)
class Split:
    """Day indices for train / validation / test, built from whole winters only."""
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    train_winters: np.ndarray
    val_winters: np.ndarray
    test_winters: np.ndarray

    def check(self, winter):
        """Fail loudly if any day or winter is shared between sets."""
        for a, b in [(self.train, self.val), (self.train, self.test), (self.val, self.test)]:
            assert len(np.intersect1d(a, b)) == 0, "day indices overlap between sets"
        sets = [set(winter[self.train]), set(winter[self.val]), set(winter[self.test])]
        assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]), \
            "a winter is split across sets"
        assert len(self.train) + len(self.val) + len(self.test) == len(winter)
        return self


def interleaved_split(winter):
    """Pre-specified split by winter start year (fixed before any model was fitted):
    test = years ending in 0 or 5, validation = years ending in 3 or 8, train = the rest.
    Interleaving spreads every set over the whole 1961-2024 period, so a long-term trend
    is not confused with model error. Winter 2015 (Storm Desmond) falls in the test set."""
    test_w = np.unique(winter[winter % 5 == 0])
    val_w = np.unique(winter[winter % 5 == 3])
    train_w = np.setdiff1d(np.unique(winter), np.concatenate([test_w, val_w]))
    idx = lambda ws: np.flatnonzero(np.isin(winter, ws))
    return Split(idx(train_w), idx(val_w), idx(test_w), train_w, val_w, test_w).check(winter)


def cell_distances_km(ny=16, nx=16):
    """Pairwise distances (km) between cell centres, shape [ny*nx, ny*nx]."""
    iy, ix = np.divmod(np.arange(ny * nx), nx)
    return CELL_KM * np.hypot(iy[:, None] - iy[None, :], ix[:, None] - ix[None, :])
