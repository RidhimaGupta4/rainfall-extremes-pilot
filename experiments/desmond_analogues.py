"""Desmond-sized days from each generator, for a like-for-like visual comparison of storm structure.

For each of the four generators (seed 0, the same fitted components as in cumbria_experiment.py),
draw days and keep the one whose box mean is closest to Storm Desmond's 83.5 mm. Also report each
generator's most extreme box-mean day out of the same draws. Uses training winters only for
fitting; the Desmond field itself is test data and is used only as the comparison target.

Run after cumbria_experiment.py (it reuses the saved seed-0 VAE):
    python experiments/desmond_analogues.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rainpilot.data import DESMOND_DATE, interleaved_split, load      # noqa: E402
from rainpilot.generators import GaussianCopula                       # noqa: E402
from rainpilot.margins import Margins                                 # noqa: E402

N = 400_000
BATCH = 25_000


def main():
    x, t, winter, _ = load(ROOT / "data/cumbria_winter_rainfall_5km.nc")
    sp = interleaved_split(winter)
    xt = x[sp.train].reshape(len(sp.train), -1)
    d = int(np.flatnonzero([str(v)[:10] == DESMOND_DATE for v in t])[0])
    target = float(x[d].mean())

    m_evt, m_emp = Margins("evt").fit(xt), Margins("empirical").fit(xt)
    gc = GaussianCopula().fit(m_evt.to_normal(xt, np.random.default_rng(0)))
    vae = torch.load(ROOT / "results/cumbria_parts/vae_seed0.pt", weights_only=False)
    out = {"target_box_mean": target, "n_draws": N, "models": {}}
    for dep_name, dep in [("Gaussian copula", gc), ("VAE", vae)]:
        rng = np.random.default_rng(42)
        best = {k: (np.inf, None) for k in ("empirical", "EVT")}
        top = {k: (-np.inf, None) for k in ("empirical", "EVT")}
        for k0 in range(0, N, BATCH):
            z = dep.sample_normal(min(BATCH, N - k0), rng)
            for mk, m in (("empirical", m_emp), ("EVT", m_evt)):
                f = m.from_normal(z).reshape(-1, 16, 16)
                bm = f.mean((1, 2))
                i = int(np.argmin(np.abs(bm - target)))
                if abs(bm[i] - target) < best[mk][0]:
                    best[mk] = (abs(bm[i] - target), f[i])
                j = int(np.argmax(bm))
                if bm[j] > top[mk][0]:
                    top[mk] = (bm[j], f[j])
        for mk in ("empirical", "EVT"):
            name = f"{dep_name} + {mk} margins"
            out["models"][name] = {"analogue": best[mk][1].tolist(), "most_extreme": top[mk][1].tolist()}
            print(f"{name:36s} analogue box mean {best[mk][1].mean():.1f} (max cell {best[mk][1].max():.0f}) | "
                  f"most extreme of {N:,}: box mean {top[mk][1].mean():.0f}, max cell {top[mk][1].max():.0f}")
    (ROOT / "results/desmond_analogues.json").write_text(json.dumps(out))
    print("saved results/desmond_analogues.json")


if __name__ == "__main__":
    main()
