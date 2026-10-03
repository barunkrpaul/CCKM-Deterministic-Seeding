"""Quantile-radius runs and the O(n) radial-mode diagnostic on the twenty
extension datasets (inputs to Table 9 of the paper).

Writes ext_quantile.csv and ext_diagnostic.json in the current directory.
Run from the repository root after ./fetch_data.sh:
    python tools/run_extension_extras.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
from cckm_experiments import project2d, cckm_seeds  # noqa: E402
from cckm_extension import load_extension, lloyd_ext  # noqa: E402


def radial_modes(Z, k):
    """Same rule as cckm_experiments.exp_diagnostic."""
    rho = np.linalg.norm(Z - Z.mean(0), axis=1)
    hist, _ = np.histogram(rho, bins=max(20, 3 * k))
    sm = np.convolve(hist, np.ones(3) / 3, mode="same")
    return int(sum(1 for i in range(1, len(sm) - 1)
                   if sm[i] > sm[i - 1] and sm[i] >= sm[i + 1]
                   and sm[i] > 0.05 * sm.max()))


rows, diag = [], {}
for name, (X, y, k) in load_extension().items():
    Z, evr = project2d(X)
    s, _ = cckm_seeds(Z, k, 4, "quantile")
    r = lloyd_ext(Z, y, s)
    rows.append(dict(dataset=name, n=len(X), d=X.shape[1], k=k, EVR2=round(evr, 3),
                     SSE=round(r["SSE"], 4),
                     ARI=None if np.isnan(r["ARI"]) else round(r["ARI"], 4),
                     NMI=None if np.isnan(r["NMI"]) else round(r["NMI"], 4),
                     SH=round(r["SH"], 4), IT=int(r["IT"])))
    diag[name] = dict(k=k, modes=radial_modes(Z, k))
    print(f"{name}: quantile SSE={r['SSE']:.1f}, modes={diag[name]['modes']}", flush=True)
pd.DataFrame(rows).to_csv("ext_quantile.csv", index=False)
json.dump(diag, open("ext_diagnostic.json", "w"), indent=1)
print("written ext_quantile.csv, ext_diagnostic.json")
