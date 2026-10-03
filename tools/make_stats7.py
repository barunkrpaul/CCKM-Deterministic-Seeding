"""Seven-method statistics on the ten core datasets (Table 5 of the paper)
and the pairwise Wilcoxon signed-rank tests against CCKM with Holm
correction over the six comparisons (Section 6.2).

Inputs : results_main.csv (cckm_experiments.py), results_eckm.csv
         (tools/run_eckm_core.py), both in the current directory.
Outputs: results_stats7.json, results_wilcoxon7.json
Run from the repository root:  python tools/make_stats7.py
"""
import json

import pandas as pd
from scipy import stats

ORDER = ["CCKM", "Maxmin", "ECKM", "KM++", "KM++greedy", "BF-refine", "KM-random"]
METRICS = [("SSE", "lower"), ("ARI", "higher"), ("NMI", "higher"),
           ("SH", "higher"), ("IT", "lower")]

main = pd.read_csv("results_main.csv")
eckm = pd.read_csv("results_eckm.csv")
df = pd.concat([main, eckm], ignore_index=True)
# Round to 1e-6 so that numerically identical results (e.g. several methods
# reaching the same optimum) receive tied ranks instead of float-noise ranks.
means = df.groupby(["dataset", "method"])[[m for m, _ in METRICS]].mean().round(6)

out, wil = {}, {}
for metric, better in METRICS:
    M = means[metric].unstack()[ORDER]
    R = M.rank(axis=1, ascending=(better == "lower"))
    chi, p = stats.friedmanchisquare(*[M[c].values for c in ORDER])
    out[metric] = dict(avg_ranks=R.mean().round(2).to_dict(),
                       friedman_chi2=round(float(chi), 3),
                       friedman_p=float(f"{p:.4g}"))
    raw = {}
    for other in ORDER[1:]:
        try:
            raw[other] = float(stats.wilcoxon(M["CCKM"], M[other]).pvalue)
        except ValueError:          # all differences zero
            raw[other] = 1.0
    holm, run = {}, 0.0
    for i, (name, p) in enumerate(sorted(raw.items(), key=lambda t: t[1])):
        run = max(run, min(1.0, p * (len(raw) - i)))
        holm[name] = round(run, 4)
    wil[metric] = dict(raw_p={k: round(v, 4) for k, v in raw.items()}, holm_p=holm)

json.dump(out, open("results_stats7.json", "w"), indent=1)
json.dump(wil, open("results_wilcoxon7.json", "w"), indent=1)
print(json.dumps(wil, indent=1))
