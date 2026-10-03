"""
cckm_extension.py — E1: extended evaluation on 20 additional datasets.

Adds to the core ten: Fränti S1–S4, FCPS (jain, spiral, compound, atom,
hepta, lsun, target, twodiamonds, wingnut, engytime), Birch-rg1/rg2
(n=100k, unlabeled -> SSE/IT/SH only), MNIST (real, 60k x 784, via the
fgnt/mnist mirror), and sklearn's real Digits, Wine, Breast-Cancer.

Silhouette is subsampled (5000 points, seed 0) when n > 20000.
ECKM is attempted on every dataset with a 600 s budget; DNF is recorded.
"""
import gzip, struct, time, json, warnings
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits, load_wine, load_breast_cancer
from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             silhouette_score)
from cckm_experiments import (read_arff, project2d, cckm_seeds, maxmin_seeds,
                              kmpp_vanilla, bf_refine, RNGSEEDS)
from sklearn.cluster import kmeans_plusplus
warnings.filterwarnings("ignore")

# ------------------------------------------------------------------ loaders
def read_arff_auto(path):
    """ARFF reader that auto-detects a class attribute (nominal, named
    class) as the last column; returns (X, y) with y=None if absent."""
    X, on, has_class = [], False, False
    for line in open(path, errors="ignore"):
        line = line.strip()
        if not line or line.startswith("%"): continue
        u = line.upper()
        if u.startswith("@ATTRIBUTE") and "CLASS" in u and "{" in line:
            has_class = True
        if u.startswith("@DATA"): on = True; continue
        if on: X.append(line.split(","))
    if has_class:
        y = np.unique([r[-1].strip() for r in X], return_inverse=True)[1]
        X = np.array([[float(v) for v in r[:-1]] for r in X])
        return X, y
    return np.array([[float(v) for v in r] for r in X]), None

def read_arff_unlabeled(path):
    return read_arff_auto(path)

def load_mnist():
    with gzip.open("data/mnist-images.gz") as f:
        _, n, r, c = struct.unpack(">IIII", f.read(16))
        X = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, r * c).astype(np.float64)
    with gzip.open("data/mnist-labels.gz") as f:
        f.read(8); y = np.frombuffer(f.read(), dtype=np.uint8).astype(int)
    return X, y

def load_extension():
    d = {}
    for i in (1, 2, 3, 4):
        X, y = read_arff_auto(f"data/s-set{i}.arff"); d[f"S{i}"] = (X, y, 15)
    fcps = dict(jain=2, spiral=3, compound=6, atom=2, hepta=7, lsun=3,
                target=6, twodiamonds=2, wingnut=2, engytime=2)
    for name, k in fcps.items():
        X, y = read_arff_auto(f"data/{name}.arff"); d[name.capitalize()] = (X, y, k)
    for tag in ("rg1", "rg2"):
        X, _ = read_arff_auto(f"data/birch-{tag}.arff")
        d[f"Birch-{tag}"] = (X, None, 100)
    X, y = load_mnist(); d["MNIST"] = (X, y, 10)
    dg = load_digits(); d["Digits"] = (dg.data, dg.target, 10)
    wn = load_wine(); d["Wine"] = (wn.data, wn.target, 3)
    bc = load_breast_cancer(); d["BreastCancer"] = (bc.data, bc.target, 2)
    return d

# ------------------------------------------------------------------- lloyd
def lloyd_ext(Z, y, seeds):
    km = KMeans(len(seeds), init=seeds, n_init=1, max_iter=300, tol=1e-4).fit(Z)
    lab = km.labels_
    n = len(Z)
    sh = (silhouette_score(Z, lab, sample_size=5000, random_state=0)
          if n > 20000 else silhouette_score(Z, lab)) if len(np.unique(lab)) > 1 else np.nan
    ari = adjusted_rand_score(y, lab) if y is not None else np.nan
    nmi = normalized_mutual_info_score(y, lab) if y is not None else np.nan
    return dict(SSE=km.inertia_, ARI=ari, NMI=nmi, SH=sh, IT=km.n_iter_)

def run_method(Z, y, k, method, seed=None):
    if method == "CCKM":
        s, _ = cckm_seeds(Z, k); return lloyd_ext(Z, y, s)
    if method == "Maxmin":
        return lloyd_ext(Z, y, maxmin_seeds(Z, k))
    if method == "KM++":
        return lloyd_ext(Z, y, kmpp_vanilla(Z, k, seed))
    if method == "KM++greedy":
        s, _ = kmeans_plusplus(Z, k, random_state=seed); return lloyd_ext(Z, y, s)
    if method == "BF-refine":
        return lloyd_ext(Z, y, bf_refine(Z, k, seed))
    if method == "KM-random":
        rng = np.random.default_rng(seed)
        return lloyd_ext(Z, y, Z[rng.choice(len(Z), k, replace=False)])
    raise ValueError(method)

def eckm_seeds_timed(P, k, budget=600, tol=1e-6):
    """Faithful ECKM seeding with a wall-clock budget. The interior-hull
    test is vectorized via matplotlib.path (identical result to the
    per-point shapely test); all other steps follow the published code."""
    from scipy.spatial import Voronoi, ConvexHull, distance
    from matplotlib.path import Path
    t0 = time.perf_counter()
    hull = ConvexHull(P); vor = Voronoi(P)
    if time.perf_counter() - t0 > budget: return None, time.perf_counter() - t0
    poly = Path(hull.points[hull.vertices])
    mask = poly.contains_points(vor.vertices)
    vor_in = vor.vertices[mask]
    circles = []
    for i, c in enumerate(vor_in):
        if i % 256 == 0 and time.perf_counter() - t0 > budget:
            return None, time.perf_counter() - t0
        circles.append((c, float(np.min(distance.cdist([c], P)))))
    circles.sort(key=lambda x: x[1], reverse=True)
    circum = []
    for j, (center, r) in enumerate(circles):
        if j % 256 == 0 and time.perf_counter() - t0 > budget:
            return None, time.perf_counter() - t0
        d = distance.cdist([center], P)[0]
        for p in P[np.abs(d - r) < tol]: circum.append(tuple(p))
    final = np.array(list(dict.fromkeys(circum)))
    return final[:k], time.perf_counter() - t0

# --------------------------------------------------------------------- main
if __name__ == "__main__":
    import sys
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    data = load_extension()

    small = [d for d, (X, y, k) in data.items() if len(X) <= 20000]
    large = [d for d, (X, y, k) in data.items() if len(X) > 20000]

    if stage in ("small", "all"):
        sel = sys.argv[2].split(",") if len(sys.argv) > 2 else small
        rows = []
        for name in [s for s in small if s in sel]:
            X, y, k = data[name]
            Z, evr = project2d(X)
            for meth in ["CCKM", "Maxmin"]:
                rows.append(dict(dataset=name, method=meth, seed=-1, EVR2=evr,
                                 n=len(X), k=k, **run_method(Z, y, k, meth)))
            for meth in ["KM++", "KM++greedy", "BF-refine", "KM-random"]:
                for s in RNGSEEDS:
                    rows.append(dict(dataset=name, method=meth, seed=s, EVR2=evr,
                                     n=len(X), k=k, **run_method(Z, y, k, meth, s)))
            print(f"small done: {name} (n={len(X)}, k={k}, EVR2={evr:.3f})", flush=True)
            import os
            hdr = not os.path.exists("ext_small.csv")
            pd.DataFrame(rows).to_csv("ext_small.csv", index=False, mode="a", header=hdr)
            rows = []

    if stage in ("large", "all"):
        sel = sys.argv[2].split(",") if len(sys.argv) > 2 else large
        rows = []
        for name in [s for s in large if s in sel]:
            X, y, k = data[name]
            Z, evr = project2d(X)
            for meth in ["CCKM", "Maxmin"]:
                rows.append(dict(dataset=name, method=meth, seed=-1, EVR2=evr,
                                 n=len(X), k=k, **run_method(Z, y, k, meth)))
                print(f"  {name} {meth} done", flush=True)
            for meth in ["KM++", "KM++greedy", "BF-refine", "KM-random"]:
                for s in RNGSEEDS:
                    rows.append(dict(dataset=name, method=meth, seed=s, EVR2=evr,
                                     n=len(X), k=k, **run_method(Z, y, k, meth, s)))
                print(f"  {name} {meth} x50 done", flush=True)
        import os
        hdr = not os.path.exists("ext_large.csv")
        pd.DataFrame(rows).to_csv("ext_large.csv", index=False, mode="a", header=hdr)

    if stage in ("eckm", "all"):
        sel = sys.argv[2].split(",") if len(sys.argv) > 2 else list(data)
        rows = []
        for name, (X, y, k) in [(n, v) for n, v in data.items() if n in sel]:
            Z, evr = project2d(X)
            seeds, tsec = eckm_seeds_timed(Z, k)
            if seeds is None or len(seeds) < k:
                rows.append(dict(dataset=name, method="ECKM", seed=-1, n=len(X),
                                 k=k, seed_time_s=round(tsec, 2), DNF=True))
                print(f"ECKM {name}: DNF after {tsec:.1f}s", flush=True)
            else:
                r = lloyd_ext(Z, y, seeds)
                rows.append(dict(dataset=name, method="ECKM", seed=-1, n=len(X),
                                 k=k, seed_time_s=round(tsec, 3), DNF=False, **r))
                print(f"ECKM {name}: SSE={r['SSE']:.2f} IT={r['IT']} "
                      f"seedtime={tsec:.2f}s", flush=True)
        import os
        hdr = not os.path.exists("ext_eckm.csv")
        pd.DataFrame(rows).to_csv("ext_eckm.csv", index=False, mode="a", header=hdr)

    if stage in ("ksweep", "all"):
        # k-sweep seeding time on Birch-rg1 (n=100k)
        X, _, _ = data["Birch-rg1"]
        Z, _ = project2d(X)
        rows = []
        for k in [2, 10, 25, 50, 100]:
            for meth, fn in [("CCKM(equal)", lambda: cckm_seeds(Z, k, 4, "equal")),
                             ("CCKM(quantile)", lambda: cckm_seeds(Z, k, 4, "quantile")),
                             ("Maxmin", lambda: maxmin_seeds(Z, k)),
                             ("KM++greedy", lambda: kmeans_plusplus(Z, k, random_state=0)),
                             ("KM++vanilla", lambda: kmpp_vanilla(Z, k, 0))]:
                ts = []
                for _ in range(3):
                    t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
                rows.append(dict(k=k, method=meth,
                                 seed_time_s=round(float(np.median(ts)), 4)))
            print(f"ksweep k={k} done", flush=True)
        pd.DataFrame(rows).to_csv("ext_ksweep.csv", index=False)

    print("stage complete:", stage)
