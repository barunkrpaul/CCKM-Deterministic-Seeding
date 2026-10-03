"""ECKM baseline on the ten core datasets (unified protocol).

Uses the ECKM seeding of cckm_extension.eckm_seeds_timed (ported from the
published ECKM code, Biswas, Giri and Roy, ESWA 2023) and the same Lloyd
settings as cckm_experiments.py. Writes results_eckm.csv in the current
directory. Run from the repository root:  python tools/run_eckm_core.py
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.getcwd())
from cckm_experiments import load_all, project2d, lloyd  # noqa: E402
from cckm_extension import eckm_seeds_timed  # noqa: E402

rows = []
for name, (X, y, k) in load_all().items():
    Z, _ = project2d(X)
    seeds, tsec = eckm_seeds_timed(Z, k)
    r = lloyd(Z, y, seeds)
    rows.append(dict(dataset=name, method="ECKM", seed=-1,
                     seed_time_s=round(tsec, 4), **r))
    print(f"ECKM {name}: SSE={r['SSE']:.2f} IT={r['IT']} seed_time={tsec:.3f}s",
          flush=True)
pd.DataFrame(rows).to_csv("results_eckm.csv", index=False)
print("written results_eckm.csv")
