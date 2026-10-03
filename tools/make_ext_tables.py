"""Summary tables for the thirty-dataset evaluation (Tables 9 and 10).

Inputs (current directory): results_main.csv, results_eckm.csv,
ext_small.csv, ext_large.csv, ext_eckm.csv, ext_quantile.csv,
ext_diagnostic.json.
Outputs: ext_means_all.csv, ext_stats30.json, ext_table.csv
Run from the repository root:  python tools/make_ext_tables.py
"""
import json

import pandas as pd
from scipy import stats

COLS = ["SSE", "ARI", "NMI", "SH", "IT"]
ORDER = ["CCKM", "Maxmin", "ECKM", "KM++", "KM++greedy", "BF-refine", "KM-random"]
METRICS = [("SSE", "lower"), ("ARI", "higher"), ("NMI", "higher"),
           ("SH", "higher"), ("IT", "lower")]

runs = pd.concat([pd.read_csv(f)[["dataset", "method", "seed"] + COLS]
                  for f in ("results_main.csv", "ext_small.csv", "ext_large.csv")])
means = runs.groupby(["dataset", "method"])[COLS].mean()
means.to_csv("ext_means_all.csv")

eckm = pd.concat([pd.read_csv("results_eckm.csv"), pd.read_csv("ext_eckm.csv")])
allm = pd.concat([means, eckm.set_index(["dataset", "method"])[COLS]])

out = {"stats": {}}
for metric, better in METRICS:
    # round to 1e-6 so numerically identical results get tied ranks
    M = allm[metric].unstack()[ORDER].dropna().round(6)
    R = M.rank(axis=1, ascending=(better == "lower"))
    chi, p = stats.friedmanchisquare(*[M[c].values for c in ORDER])
    out["stats"][metric] = dict(N=len(M), avg_ranks=R.mean().round(2).to_dict(),
                                chi2=round(float(chi), 2), p=float(f"{p:.3g}"))

# win/loss: P(a single k-means++ run, vanilla or greedy, beats CCKM on SSE)
ext = pd.concat([pd.read_csv("ext_small.csv"), pd.read_csv("ext_large.csv")])
wl = {}
for name, sub in ext.groupby("dataset"):
    c = sub[sub.method == "CCKM"].SSE.iloc[0]
    pool = sub[sub.method.isin(["KM++", "KM++greedy"])].SSE.values
    wl[name] = round(float((pool < c - 1e-9).mean()), 2)
out["winloss_ext"] = dict(sorted(wl.items()))

# iterations: CCKM vs each method, Wilcoxon + Holm over six comparisons
IT = allm["IT"].unstack()[ORDER].round(6)
out["it_wins"] = int((IT["CCKM"] < IT["KM-random"]).sum())
raw = {o: float(stats.wilcoxon(IT["CCKM"], IT[o]).pvalue) for o in ORDER[1:]}
# Holm step: p_(i) * (m - i), as in cckm_experiments.exp_stats
holm = {name: round(min(p * (len(raw) - i), 1.0), 4)
        for i, (name, p) in enumerate(sorted(raw.items(), key=lambda t: t[1]))}
out["it_holm"] = holm
json.dump(out, open("ext_stats30.json", "w"), indent=1)

# Table 9
q = pd.read_csv("ext_quantile.csv").set_index("dataset")
diag = json.load(open("ext_diagnostic.json"))
rows = []
for name in q.index:
    cm = means.loc[(name, "CCKM")]
    rows.append(dict(dataset=name, n=int(q.loc[name, "n"]), d_=int(q.loc[name, "d"]),
                     k=int(q.loc[name, "k"]), EVR2=round(float(q.loc[name, "EVR2"]), 3),
                     SSE_eq=round(cm.SSE, 1), SSE_qt=round(q.loc[name, "SSE"], 1),
                     IT_eq=cm.IT, IT_qt=int(q.loc[name, "IT"]),
                     best_SSE=round(means.loc[name].SSE.min(), 1),  # six non-ECKM methods
                     P_beat=wl[name],
                     modes_k=round(diag[name]["modes"] / diag[name]["k"], 2)))
pd.DataFrame(rows).to_csv("ext_table.csv", index=False)
print("written ext_means_all.csv, ext_stats30.json, ext_table.csv")
