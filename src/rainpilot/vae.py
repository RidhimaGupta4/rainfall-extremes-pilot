"""A small convolutional VAE for 16 x 16 fields of normal scores.

Trained on TRAINING winters; early stopping uses VALIDATION winters only.
Decoder: per-pixel Gaussian with a learned per-pixel log-variance, so samples are
decoder mean + noise. That noise is independent between pixels, which is one of the
modelling choices the pilot puts to the test.
"""
import copy

import numpy as np
import torch
from torch import nn


class ConvVAE(nn.Module):
    def __init__(self, latent=16, width=32):
        super().__init__()
        w = width
        self.enc = nn.Sequential(
            nn.Conv2d(1, w, 3, padding=1), nn.GELU(),
            nn.Conv2d(w, 2 * w, 4, stride=2, padding=1), nn.GELU(),        # 8 x 8
            nn.Conv2d(2 * w, 2 * w, 4, stride=2, padding=1), nn.GELU(),    # 4 x 4
            nn.Flatten(), nn.Linear(2 * w * 16, 2 * latent))
        self.dec_fc = nn.Sequential(nn.Linear(latent, 2 * w * 16), nn.GELU())
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(2 * w, 2 * w, 4, stride=2, padding=1), nn.GELU(),   # 8 x 8
            nn.ConvTranspose2d(2 * w, w, 4, stride=2, padding=1), nn.GELU(),       # 16 x 16
            nn.Conv2d(w, 1, 3, padding=1))
        self.logvar_x = nn.Parameter(torch.full((1, 16, 16), -1.0))
        self.latent, self.w = latent, w

    def encode(self, x):
        mu, logvar = self.enc(x).chunk(2, dim=1)
        return mu, logvar

    def decode(self, z):
        h = self.dec_fc(z).view(-1, 2 * self.w, 4, 4)
        return self.dec(h)

    def neg_elbo(self, x):
        """Mean over the batch of -ELBO (Gaussian likelihood, standard normal prior), in nats."""
        mu, logvar = self.encode(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        xm = self.decode(z)
        lv = self.logvar_x.clamp(-8, 4)
        nll = 0.5 * (((x - xm) ** 2) / lv.exp() + lv + np.log(2 * np.pi)).sum((1, 2, 3))
        kl = 0.5 * (mu ** 2 + logvar.exp() - 1 - logvar).sum(1)
        return (nll + kl).mean()

    @torch.no_grad()
    def encode_means(self, x, batch=10_000):
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
        return np.concatenate([self.encode(x[k:k + batch])[0].numpy() for k in range(0, len(x), batch)])

    def fit_latent_density(self, z_train, z_val, components=(1, 2, 4, 8, 16, 32), seed=0):
        """Ex-post density estimation (Ghosh et al., 2020): sampling from the N(0, I) prior gives
        under-dispersed fields because the training encodings do not fill the prior ("prior holes").
        Instead, fit a Gaussian mixture to the encoder means of the TRAINING days and choose the
        number of components by log-likelihood of the VALIDATION days' encodings."""
        from sklearn.mixture import GaussianMixture
        et, ev = self.encode_means(z_train), self.encode_means(z_val)
        scores = {}
        for k in components:
            gm = GaussianMixture(k, covariance_type="full", reg_covar=1e-4, random_state=seed,
                                 n_init=2, max_iter=500).fit(et)
            scores[k] = (gm.score(ev), gm)
        self.latent_k = max(scores, key=lambda k: scores[k][0])
        self.latent_gmm = scores[self.latent_k][1]
        self.latent_scores = {k: v[0] for k, v in scores.items()}
        return self

    @torch.no_grad()
    def sample(self, n, generator=None, batch=5_000, prior="gmm"):
        """prior='gmm' (default, needs fit_latent_density) or 'normal' (the N(0, I) prior)."""
        out = []
        sd = (0.5 * self.logvar_x.clamp(-8, 4)).exp()
        seed = int(torch.randint(0, 2**31 - 1, (1,), generator=generator))
        rng = np.random.default_rng(seed)
        for k in range(0, n, batch):
            m = min(batch, n - k)
            if prior == "gmm":
                comp = rng.choice(self.latent_k, size=m, p=self.latent_gmm.weights_)
                z = np.empty((m, self.latent))
                for c in range(self.latent_k):
                    idx = comp == c
                    if idx.any():
                        cov = self.latent_gmm.covariances_[c]
                        z[idx] = rng.multivariate_normal(self.latent_gmm.means_[c], (cov + cov.T) / 2,
                                                         idx.sum(), method="eigh")
                z = torch.tensor(z, dtype=torch.float32)
            else:
                z = torch.randn(m, self.latent, generator=generator)
            xm = self.decode(z)
            out.append((xm + sd * torch.randn(xm.shape, generator=generator)).squeeze(1).numpy())
        return np.concatenate(out)


def train_vae(z_train, z_val, seed=0, latent=16, epochs=400, patience=25, lr=1e-3, batch=128,
              verbose=False):
    """z_*: normal-score fields [n, 16, 16]. Returns the model at its best VALIDATION -ELBO,
    plus the training history."""
    torch.manual_seed(seed); np.random.seed(seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    xt = torch.tensor(z_train, dtype=torch.float32).unsqueeze(1)
    xv = torch.tensor(z_val, dtype=torch.float32).unsqueeze(1)
    model = ConvVAE(latent=latent)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    best, best_state, wait, hist = np.inf, None, 0, []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(xt), generator=g)
        tot = 0.0
        for k in range(0, len(xt), batch):
            xb = xt[perm[k:k + batch]]
            loss = model.neg_elbo(xb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(xb)
        model.eval()
        # Fixed noise for a stable validation score, drawn inside fork_rng so that it does NOT
        # reset the global random state used for training (seeds stay independent).
        with torch.no_grad(), torch.random.fork_rng():
            torch.manual_seed(10_000 + ep)
            v = np.mean([model.neg_elbo(xv).item() for _ in range(3)])
        hist.append((tot / len(xt), v))
        if v < best - 1e-3:
            best, best_state, wait = v, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break
        if verbose and ep % 20 == 0:
            print(f"  epoch {ep:3d}  train {hist[-1][0]:.2f}  val {v:.2f}")
    model.load_state_dict(best_state)
    model.eval()
    return model, np.array(hist), len(hist) - wait - 1
