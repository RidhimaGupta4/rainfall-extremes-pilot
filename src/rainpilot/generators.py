"""The 2 x 2 design: {Gaussian copula, VAE} dependence x {empirical, EVT} margins.

Every generator works the same way:
  1. margins are fitted on training days, and training/validation fields are mapped to
     normal scores with those margins;
  2. a dependence model is fitted to the training normal-score fields;
  3. sampling draws normal-score fields from the dependence model and maps them back to
     millimetres with the margins.
So comparing columns isolates the effect of the margins, and comparing rows isolates the
effect of the dependence model.
"""
import numpy as np
import torch

from .margins import Margins
from .vae import train_vae

SHAPE = (16, 16)


class GaussianCopula:
    def fit(self, z_train):
        z = z_train.reshape(len(z_train), -1)
        self.corr = np.corrcoef(z.T)
        self.L = np.linalg.cholesky(self.corr + 1e-8 * np.eye(len(self.corr)))
        return self

    def sample_normal(self, n, rng, batch=25_000):
        out = []
        for k in range(0, n, batch):
            m = min(batch, n - k)
            out.append(rng.standard_normal((m, self.L.shape[0])) @ self.L.T)
        return np.concatenate(out)


class VAEDependence:
    def __init__(self, seed=0, **kw):
        self.seed, self.kw = seed, kw

    def fit(self, z_train, z_val):
        self.model, self.history, self.best_epoch = train_vae(z_train, z_val, seed=self.seed, **self.kw)
        self.model.fit_latent_density(z_train, z_val, seed=self.seed)
        return self

    def sample_normal(self, n, rng):
        g = torch.Generator().manual_seed(int(rng.integers(2**31)))
        return self.model.sample(n, generator=g).reshape(n, -1)


class Generator:
    """margins: 'evt' or 'empirical'; dependence: 'gaussian' or 'vae'."""

    def __init__(self, margins="evt", dependence="gaussian", seed=0, q=0.95, vae_kw=None):
        self.margins_kind, self.dep_kind, self.seed = margins, dependence, seed
        self.q, self.vae_kw = q, vae_kw or {}

    @property
    def name(self):
        return f"{'Gaussian copula' if self.dep_kind == 'gaussian' else 'VAE'} + " \
               f"{'EVT' if self.margins_kind == 'evt' else 'empirical'} margins"

    def fit(self, x_train, x_val, margins=None, z_train=None, z_val=None, dependence=None):
        """x_*: [n, 16, 16] mm. Pre-fitted margins / normal scores / dependence can be passed in,
        so the 2 x 2 cells share exactly the same components where they should."""
        rng = np.random.default_rng(self.seed)
        xt, xv = x_train.reshape(len(x_train), -1), x_val.reshape(len(x_val), -1)
        self.margins = margins or Margins(self.margins_kind, self.q).fit(xt)
        if z_train is None:
            z_train = self.margins.to_normal(xt, rng).reshape(-1, *SHAPE)
            z_val = self.margins.to_normal(xv, rng).reshape(-1, *SHAPE)
        if dependence is not None:
            self.dep = dependence
        elif self.dep_kind == "gaussian":
            self.dep = GaussianCopula().fit(z_train)
        else:
            self.dep = VAEDependence(self.seed, **self.vae_kw).fit(z_train, z_val)
        return self

    def sample(self, n, rng):
        """n fields in mm, shape [n, 16, 16] (float32 to save memory)."""
        if n > 25_000:
            return np.concatenate(list(self.sample_batches(n, rng)))
        z = self.dep.sample_normal(n, rng)
        return self.margins.from_normal(z).reshape(n, *SHAPE).astype(np.float32)

    def sample_batches(self, n, rng, batch=25_000):
        for k in range(0, n, batch):
            yield self.sample(min(batch, n - k), rng)
