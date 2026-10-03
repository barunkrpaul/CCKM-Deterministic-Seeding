"""
cckm_experiments.py — Unified-protocol evaluation of polar-grid seeding.

Implements CCKM(m, radii) [equal-width or quantile] and compares against:
  KM-random, KM++ (vanilla), KM++greedy (scikit-learn), Maxmin
  (farthest-first from the point farthest from the global centroid),
  and Bradley-Fayyad refinement (J=10 subsamples).

Experiments:
  main      : 10 benchmarks x 6 methods, 50 seeds for randomized methods
  stats     : Friedman + Wilcoxon-Holm on the main results
  winloss   : P(single k-means++ run beats CCKM) per dataset
  diagnostic: O(n) radial-mode count per dataset
  ablation  : (E4) m in {4,8,16} x {equal, quantile}
  noise     : (E3) 1/2/5% uniform contamination; seed displacement + ARI
  scaling   : (E2) seeding wall-clock vs n on synthetic Gaussian mixtures

All randomized methods use fixed seeds 0..49. Deterministic methods run once.
Protocol: z-score -> PCA(2) -> seeding -> Lloyd (max_iter=300, tol=1e-4, n_init=1).
"""
import json, time, warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, kmeans_plusplus
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             silhouette_score)

warnings.filterwarnings("ignore")
RNGSEEDS = list(range(50))

# --------------------------------------------------------------- data loading
def read_arff(path):
    X, y, on = [], [], False
    for line in open(path, errors="ignore"):
        line = line.strip()
        if not line or line.startswith("%"):
            continue
        if line.upper().startswith("@DATA"):
            on = True; continue
        if on:
            parts = line.split(",")
            X.append([float(v) for v in parts[:-1]])
            y.append(parts[-1].strip())
    X = np.array(X)
    _, y = np.unique(np.array(y), return_inverse=True)
    return X, y

def load_all():
    d = {}
    iris = load_iris(); d["Iris"] = (iris.data, iris.target, 3)
    S = np.loadtxt("data/seeds.csv", delimiter=",")
    d["Seeds"] = (S[:, :7], S[:, 7].astype(int) - 1, 3)
    X, y = read_arff("data/ecoli.arff");       d["Ecoli"] = (X, y, 8)
    X, y = read_arff("data/tetra.arff");       d["Tetra"] = (X, y, 4)
    X, y = read_arff("data/chainlink.arff");   d["Chainlink"] = (X, y, 2)
    X, y = read_arff("data/flame.arff");       d["Flame"] = (X, y, 2)
    X, y = read_arff("data/pathbased.arff");   d["Pathbased"] = (X, y, 3)
    X, y = read_arff("data/aggregation.arff"); d["Aggregation"] = (X, y, 7)
    X, y = read_arff("data/R15.arff");         d["R15"] = (X, y, 15)
    X, y = read_arff("data/D31.arff");         d["D31"] = (X, y, 31)
    return d

def project2d(X):
    """z-score then PCA(2) for ALL datasets, including d=2 (a rotation).
    Sign convention: scikit-learn's deterministic svd_flip. Stated explicitly
    because CCKM's densest-sector selection is sign-sensitive on some data."""
    Z = StandardScaler().fit_transform(X)
    p = PCA(n_components=2, svd_solver="full", random_state=0)
    Z2 = p.fit_transform(Z)
    evr = float(p.explained_variance_ratio_.sum())
    return Z2, evr

# ------------------------------------------------------------------- seeders
def cckm_seeds(Z, k, m=4, radii="equal"):
    """Polar-grid seeding. Returns (seeds, n_repairs)."""
    c = Z.mean(axis=0)
    P = Z - c
    rho = np.linalg.norm(P, axis=1)
    theta = np.mod(np.arctan2(P[:, 1], P[:, 0]), 2 * np.pi)
    if radii == "quantile":
        edges = np.quantile(rho, np.linspace(0, 1, k + 1))
    else:
        edges = rho.min() + (rho.max() - rho.min()) * np.arange(k + 1) / k
    a = np.clip(np.searchsorted(edges[1:-1], rho, side="right"), 0, k - 1)
    q = np.minimum((theta * m / (2 * np.pi)).astype(int), m - 1)
    seeds, repairs = [None] * k, 0
    for i in range(k):
        idx = np.where(a == i)[0]
        if len(idx) == 0:
            continue
        counts = np.bincount(q[idx], minlength=m)
        j = int(np.argmax(counts))               # ties -> smallest index
        seeds[i] = Z[idx[q[idx] == j]].mean(axis=0)
    # repairs: empty annuli then coincident seeds -> farthest-first steps
    S = [s for s in seeds if s is not None]
    for i in range(k):
        if seeds[i] is None:
            D = np.min(np.linalg.norm(Z[:, None, :] - np.asarray(S)[None], axis=2), axis=1)
            seeds[i] = Z[int(np.argmax(D))]; S.append(seeds[i]); repairs += 1
    seeds = np.array(seeds)
    for i in range(1, k):
        if np.min(np.linalg.norm(seeds[:i] - seeds[i], axis=1)) < 1e-12:
            D = np.min(np.linalg.norm(Z[:, None, :] - seeds[None], axis=2), axis=1)
            seeds[i] = Z[int(np.argmax(D))]; repairs += 1
    return seeds, repairs

def maxmin_seeds(Z, k):
    c = Z.mean(axis=0)
    idx = [int(np.argmax(np.linalg.norm(Z - c, axis=1)))]
    D = np.linalg.norm(Z - Z[idx[0]], axis=1)
    for _ in range(k - 1):
        idx.append(int(np.argmax(D)))
        D = np.minimum(D, np.linalg.norm(Z - Z[idx[-1]], axis=1))
    return Z[idx]

def kmpp_vanilla(Z, k, seed):
    rng = np.random.default_rng(seed)
    seeds = [Z[rng.integers(len(Z))]]
    D2 = np.linalg.norm(Z - seeds[0], axis=1) ** 2
    for _ in range(k - 1):
        p = D2 / D2.sum()
        seeds.append(Z[rng.choice(len(Z), p=p)])
        D2 = np.minimum(D2, np.linalg.norm(Z - seeds[-1], axis=1) ** 2)
    return np.array(seeds)

def bf_refine(Z, k, seed, J=10, frac=0.1):
    """Bradley-Fayyad: cluster J subsample solutions, refine over pooled seeds."""
    rng = np.random.default_rng(seed)
    n = len(Z); sub = max(k, int(frac * n))
    CMs = []
    for j in range(J):
        idx = rng.choice(n, size=sub, replace=False)
        km = KMeans(k, init="k-means++", n_init=1, max_iter=300, tol=1e-4,
                    random_state=int(rng.integers(1 << 31))).fit(Z[idx])
        CMs.append(km.cluster_centers_)
    pool = np.vstack(CMs)
    best, bestJ = None, np.inf
    for CM in CMs:
        km = KMeans(k, init=CM, n_init=1, max_iter=300, tol=1e-4).fit(pool)
        if km.inertia_ < bestJ:
            bestJ, best = km.inertia_, km.cluster_centers_
    return best

def lloyd(Z, y, seeds):
    km = KMeans(len(seeds), init=seeds, n_init=1, max_iter=300, tol=1e-4).fit(Z)
    lab = km.labels_
    sh = silhouette_score(Z, lab) if len(np.unique(lab)) > 1 else np.nan
    return dict(SSE=km.inertia_, ARI=adjusted_rand_score(y, lab),
                NMI=normalized_mutual_info_score(y, lab), SH=sh,
                IT=km.n_iter_)

def run_method(Z, y, k, method, seed=None):
    if method == "CCKM":
        s, _ = cckm_seeds(Z, k); return lloyd(Z, y, s)
    if method.startswith("CCKM("):                    # e.g. CCKM(8,quantile)
        m, rr = method[5:-1].split(","); s, _ = cckm_seeds(Z, k, int(m), rr)
        return lloyd(Z, y, s)
    if method == "Maxmin":
        return lloyd(Z, y, maxmin_seeds(Z, k))
    if method == "KM++":
        return lloyd(Z, y, kmpp_vanilla(Z, k, seed))
    if method == "KM++greedy":
        s, _ = kmeans_plusplus(Z, k, random_state=seed); return lloyd(Z, y, s)
    if method == "BF-refine":
        return lloyd(Z, y, bf_refine(Z, k, seed))
    if method == "KM-random":
        rng = np.random.default_rng(seed)
        return lloyd(Z, y, Z[rng.choice(len(Z), k, replace=False)])
    raise ValueError(method)

# ------------------------------------------------------------ E: experiments
def exp_main(data):
    det = ["CCKM", "Maxmin"]
    rnd = ["KM++", "KM++greedy", "BF-refine", "KM-random"]
    rows = []
    for name, (X, y, k) in data.items():
        Z, evr = project2d(X)
        for meth in det:
            r = run_method(Z, y, k, meth)
            rows.append(dict(dataset=name, method=meth, seed=-1, EVR2=evr, **r))
        for meth in rnd:
            for s in RNGSEEDS:
                r = run_method(Z, y, k, meth, s)
                rows.append(dict(dataset=name, method=meth, seed=s, EVR2=evr, **r))
        print(f"  main done: {name}")
    return pd.DataFrame(rows)

def summarize(df):
    g = df.groupby(["dataset", "method"])[["SSE", "ARI", "NMI", "SH", "IT"]]
    return g.mean().round(4), g.std().round(4)

def exp_stats(df):
    means, _ = summarize(df)
    order = ["CCKM", "Maxmin", "KM++", "KM++greedy", "BF-refine", "KM-random"]
    out = {}
    for metric, better in [("SSE", "lower"), ("ARI", "higher"), ("NMI", "higher"),
                           ("SH", "higher"), ("IT", "lower")]:
        M = means[metric].unstack()[order]
        R = M.rank(axis=1, ascending=(better == "lower"))
        chi, p = stats.friedmanchisquare(*[M[c].values for c in order])
        out[metric] = dict(avg_ranks=R.mean().round(2).to_dict(),
                           friedman_chi2=round(chi, 3), friedman_p=float(f"{p:.2e}"))
    # Wilcoxon CCKM vs KM-random on IT (Holm over 5 pairwise vs CCKM)
    it = means["IT"].unstack()
    ps = {}
    for other in order[1:]:
        try:
            _, p = stats.wilcoxon(it["CCKM"], it[other])
        except ValueError:
            p = 1.0
        ps[other] = p
    holm = {}
    for i, (kk, p) in enumerate(sorted(ps.items(), key=lambda t: t[1])):
        holm[kk] = round(min(p * (len(ps) - i), 1.0), 4)
    out["IT_wilcoxon_holm_vs_CCKM"] = holm
    return out

def exp_winloss(df):
    out = {}
    for name, sub in df.groupby("dataset"):
        cckm = sub[sub.method == "CCKM"].SSE.iloc[0]
        pool = sub[sub.method.isin(["KM++", "KM++greedy"])].SSE.values
        out[name] = round(float((pool < cckm - 1e-9).mean()), 2)
    return out

def exp_diagnostic(data):
    """O(n) radial-mode count: KDE-free histogram mode counting."""
    out = {}
    for name, (X, y, k) in data.items():
        Z, _ = project2d(X)
        rho = np.linalg.norm(Z - Z.mean(0), axis=1)
        hist, _ = np.histogram(rho, bins=max(20, 3 * k))
        sm = np.convolve(hist, np.ones(3) / 3, mode="same")
        modes = sum(1 for i in range(1, len(sm) - 1)
                    if sm[i] > sm[i - 1] and sm[i] >= sm[i + 1] and sm[i] > 0.05 * sm.max())
        out[name] = dict(k=k, radial_modes=int(modes))
    return out

def exp_ablation(data):
    rows = []
    for name, (X, y, k) in data.items():
        Z, _ = project2d(X)
        for m in [4, 8, 16]:
            for rr in ["equal", "quantile"]:
                s, rep = cckm_seeds(Z, k, m, rr)
                r = lloyd(Z, y, s)
                rows.append(dict(dataset=name, m=m, radii=rr, repairs=rep, **r))
        print(f"  ablation done: {name}")
    return pd.DataFrame(rows)

def exp_noise(data):
    rows = []
    for name, (X, y, k) in data.items():
        Z, _ = project2d(X)
        lo, hi = Z.min(0), Z.max(0)
        span = hi - lo
        clean = {"Maxmin": maxmin_seeds(Z, k),
                 "CCKM(equal)": cckm_seeds(Z, k, 4, "equal")[0],
                 "CCKM(quantile)": cckm_seeds(Z, k, 4, "quantile")[0]}
        for frac in [0.01, 0.02, 0.05]:
            for s in range(10):                     # 10 noise draws
                rng = np.random.default_rng(1000 + s)
                nn = max(1, int(frac * len(Z)))
                noise = rng.uniform(lo - 0.25 * span, hi + 0.25 * span, size=(nn, 2))
                Zc = np.vstack([Z, noise])
                yc = np.concatenate([y, np.full(nn, -1)])
                for mname, cseed in clean.items():
                    if mname == "Maxmin":
                        sd = maxmin_seeds(Zc, k)
                    else:
                        rr = "equal" if "equal" in mname else "quantile"
                        sd = cckm_seeds(Zc, k, 4, rr)[0]
                    # matched seed displacement (greedy nearest matching)
                    disp = float(np.mean(np.min(
                        np.linalg.norm(cseed[:, None] - sd[None], axis=2), axis=1)))
                    km = KMeans(k, init=sd, n_init=1, max_iter=300, tol=1e-4).fit(Zc)
                    mask = yc >= 0
                    rows.append(dict(dataset=name, method=mname, frac=frac, draw=s,
                                     seed_disp=disp,
                                     ARI=adjusted_rand_score(y, km.labels_[mask]),
                                     SSE_clean_pts=float(np.sum(np.min(
                                         np.linalg.norm(Z[:, None] - km.cluster_centers_[None],
                                                        axis=2) ** 2, axis=1)))))
        print(f"  noise done: {name}")
    return pd.DataFrame(rows)

def exp_scaling():
    rows = []
    for n in [10_000, 100_000, 1_000_000]:
        rng = np.random.default_rng(0)
        k = 10
        mus = rng.uniform(-10, 10, size=(k, 2))
        Z = (mus[rng.integers(0, k, n)] + rng.standard_normal((n, 2))).astype(np.float64)
        y = np.zeros(n, int)
        for meth, fn in [
            ("CCKM(equal)",    lambda: cckm_seeds(Z, k, 4, "equal")),
            ("CCKM(quantile)", lambda: cckm_seeds(Z, k, 4, "quantile")),
            ("Maxmin",         lambda: maxmin_seeds(Z, k)),
            ("KM++greedy",     lambda: kmeans_plusplus(Z, k, random_state=0)),
            ("KM++vanilla",    lambda: kmpp_vanilla(Z, k, 0)),
        ]:
            ts = []
            for _ in range(3):
                t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
            rows.append(dict(n=n, method=meth, seed_time_s=round(float(np.median(ts)), 4)))
        print(f"  scaling done: n={n}")
    return pd.DataFrame(rows)

# --------------------------------------------------------------------- main
if __name__ == "__main__":
    data = load_all()
    for name, (X, y, k) in data.items():
        print(f"{name}: n={len(X)}, d={X.shape[1]}, k={k}, classes={len(np.unique(y))}")

    print("\n== MAIN =="); df = exp_main(data); df.to_csv("results_main.csv", index=False)
    means, stds = summarize(df)
    means.to_csv("results_means.csv"); stds.to_csv("results_stds.csv")

    print("\n== STATS =="); st = exp_stats(df)
    json.dump(st, open("results_stats.json", "w"), indent=1)

    print("\n== WIN/LOSS =="); wl = exp_winloss(df)
    json.dump(wl, open("results_winloss.json", "w"), indent=1)

    print("\n== DIAGNOSTIC =="); dg = exp_diagnostic(data)
    json.dump(dg, open("results_diagnostic.json", "w"), indent=1)

    print("\n== ABLATION (E4) =="); ab = exp_ablation(data)
    ab.to_csv("results_ablation.csv", index=False)

    print("\n== NOISE (E3) =="); nz = exp_noise(data)
    nz.to_csv("results_noise.csv", index=False)

    print("\n== SCALING (E2, synthetic) =="); sc = exp_scaling()
    sc.to_csv("results_scaling.csv", index=False)

    print("\nAll results written.")
