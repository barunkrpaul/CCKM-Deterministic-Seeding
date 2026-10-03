"""Generate figures for the TKDE CCKM manuscript.

Fig 1: CCKM pipeline on a synthetic radially separated dataset (algorithm actually run).
Fig 2: (a) radial-collision failure illustration; (b) real D31 benchmark with CCKM annuli
       (skipped automatically if D31.arff is not present).
Figs 3-4: Critical-difference diagrams built from the reported average ranks
          (Table "Average ranks over the ten datasets" of the manuscript;
          seven methods, N = 10, Nemenyi CD = 2.85 at alpha = 0.05).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from sklearn.cluster import KMeans

rng = np.random.default_rng(7)
plt.rcParams.update({"font.size": 8, "font.family": "serif",
                     "axes.linewidth": 0.6, "figure.dpi": 200})

# ---------------------------------------------------------------- CCKM seeding
def cckm_seeds(Z, k, m=4, quantile=False):
    c = Z.mean(axis=0)
    P = Z - c
    rho = np.linalg.norm(P, axis=1)
    theta = np.mod(np.arctan2(P[:, 1], P[:, 0]), 2 * np.pi)
    if quantile:
        edges = np.quantile(rho, np.linspace(0, 1, k + 1))
    else:
        dmin, dmax = rho.min(), rho.max()
        edges = dmin + (dmax - dmin) * np.arange(k + 1) / k
    a = np.clip(np.searchsorted(edges, rho, side="right") - 1, 0, k - 1)
    q = np.minimum((theta / (2 * np.pi / m)).astype(int), m - 1)
    seeds, used = [], []
    for i in range(k):
        idx = np.where(a == i)[0]
        if len(idx) == 0:
            seeds.append(None); continue
        counts = np.bincount(q[idx], minlength=m)
        j = int(np.argmax(counts))
        cell = idx[q[idx] == j]
        seeds.append(Z[cell].mean(axis=0))
    # farthest-first repair for empty annuli
    S = [s for s in seeds if s is not None]
    for i in range(k):
        if seeds[i] is None:
            d = np.min(np.linalg.norm(Z[:, None, :] - np.array(S)[None, :, :], axis=2), axis=1)
            seeds[i] = Z[int(np.argmax(d))]
            S.append(seeds[i])
    return np.array(seeds), c, edges

# ------------------------------------------------------------------- Figure 1
def synth_radial(n=420):
    """Three clusters at distinct radii around the origin (radially separated)."""
    centers = np.array([[0.7, 0.4], [-1.9, 1.4], [2.6, -2.1]])
    X = np.vstack([ce + 0.32 * rng.standard_normal((n // 3, 2)) for ce in centers])
    y = np.repeat(np.arange(3), n // 3)
    return X, y

X, y = synth_radial()
Z = (X - X.mean(0)) / X.std(0)          # z-score; already 2-D so PCA is identity up to rotation
k = 3
seeds, c, edges = cckm_seeds(Z, k)
km = KMeans(n_clusters=k, init=seeds, n_init=1, max_iter=300, tol=1e-4).fit(Z)

fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.35))
ax = axes[0]
ax.scatter(Z[:, 0], Z[:, 1], s=4, c="0.55")
ax.plot(*c, "k*", ms=10)
ax.set_title("(a) PCA plane, global centroid $c$", fontsize=8)

ax = axes[1]
ax.scatter(Z[:, 0], Z[:, 1], s=4, c="0.7")
for r in edges:
    ax.add_patch(Circle(c, r, fill=False, ls="--", lw=0.7, ec="tab:purple"))
L = edges[-1] * 1.02
for ang in [0, np.pi / 2, np.pi, 3 * np.pi / 2]:
    ax.plot([c[0], c[0] + L * np.cos(ang)], [c[1], c[1] + L * np.sin(ang)],
            color="0.4", lw=0.5)
ax.scatter(seeds[:, 0], seeds[:, 1], marker="X", s=70, c="crimson",
           edgecolors="k", linewidths=0.5, zorder=5, label="CCKM seeds")
ax.legend(fontsize=6, loc="lower right")
ax.set_title("(b) $k{+}1$ circles, sectors, seeds", fontsize=8)

ax = axes[2]
ax.scatter(Z[:, 0], Z[:, 1], s=4, c=km.labels_, cmap="viridis")
ax.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], marker="X",
           s=70, c="crimson", edgecolors="k", linewidths=0.5, zorder=5)
ax.set_title(f"(c) Final clustering ({km.n_iter_} iterations)", fontsize=8)
for ax in axes:
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout()
fig.savefig("figures/fig1_pipeline.pdf", bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------------- Figure 2
# (a) radial collision: two clusters at the same radius, same sector
ca, cb = np.array([1.6, 0.9]), np.array([0.9, 1.6])   # equal radius, both in Q1
Xa = ca + 0.16 * rng.standard_normal((120, 2))
Xb = cb + 0.16 * rng.standard_normal((120, 2))
Xn = 0.35 * rng.standard_normal((120, 2))             # a third cluster at the centre
Xc = np.vstack([Xa, Xb, Xn])
Zc = (Xc - Xc.mean(0)) / Xc.std(0)
seeds2, c2, edges2 = cckm_seeds(Zc, 3)

fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
ax = axes[0]
ax.scatter(Zc[:, 0], Zc[:, 1], s=4, c="0.55")
for r in edges2:
    ax.add_patch(Circle(c2, r, fill=False, ls="--", lw=0.7, ec="tab:purple"))
L = edges2[-1] * 1.02
for ang in [0, np.pi / 2, np.pi, 3 * np.pi / 2]:
    ax.plot([c2[0], c2[0] + L * np.cos(ang)], [c2[1], c2[1] + L * np.sin(ang)],
            color="0.4", lw=0.5)
ax.scatter(seeds2[:, 0], seeds2[:, 1], marker="X", s=70, c="crimson",
           edgecolors="k", linewidths=0.5, zorder=5)
ax.set_title("(a) Radial collision: two clusters,\none cell, one seed (Theorem 4)", fontsize=8)
ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])

# (b) real D31 with every fourth CCKM annulus
D31 = "data/D31.arff"
if os.path.exists(D31):
    data = []
    with open(D31) as f:
        on = False
        for line in f:
            line = line.strip()
            if line.upper() == "@DATA":
                on = True; continue
            if on and line:
                x_, y_, cl = line.split(",")
                data.append((float(x_), float(y_), int(cl)))
    D = np.array(data)
    Zd = (D[:, :2] - D[:, :2].mean(0)) / D[:, :2].std(0)
    lab = D[:, 2].astype(int)
    _, cd, edgesd = cckm_seeds(Zd, 31)
    ax = axes[1]
    ax.scatter(Zd[:, 0], Zd[:, 1], s=2, c=lab, cmap="tab20")
    for r in edgesd[::4]:
        ax.add_patch(Circle(cd, r, fill=False, ls="--", lw=0.6, ec="k", alpha=0.6))
    ax.set_title("(b) D31: many clusters share each annulus", fontsize=8)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig("figures/fig2_failure.pdf", bbox_inches="tight")
else:
    print("data/D31.arff not found (run ./fetch_data.sh); skipping Figure 2 (existing PDF kept)")
plt.close(fig)

# --------------------------------------------------------- Figures 3-4: CD diagrams
def cd_diagram(ranks, cd, fname, title):
    methods = sorted(ranks, key=ranks.get)
    kk = len(methods)
    lo, hi = 1, kk
    fig, ax = plt.subplots(figsize=(5.4, 1.9))
    ax.set_xlim(lo - 0.2, hi + 0.2); ax.set_ylim(0, 3.4)
    ax.plot([lo, hi], [3.0, 3.0], "k-", lw=1)
    for t in range(lo, hi + 1):
        ax.plot([t, t], [2.93, 3.07], "k-", lw=1)
        ax.text(t, 3.16, str(t), ha="center", fontsize=7)
    # CD bar
    ax.plot([lo, lo + cd], [3.32, 3.32], "-", color="crimson", lw=2)
    ax.text(lo + cd / 2, 3.42, f"CD = {cd}", ha="center", fontsize=7, color="crimson")
    half = int(np.ceil(kk / 2))
    for i, mname in enumerate(methods):
        r = ranks[mname]
        if i < half:      # left column, hang below axis on the left
            yh = 2.55 - i * 0.52
            ax.plot([r, r], [3.0, yh], "k-", lw=0.7)
            ax.plot([r, lo - 0.15], [yh, yh], "k-", lw=0.7)
            ax.text(lo - 0.2, yh, f"{mname} ({r:.2f})", ha="right", va="center", fontsize=7)
        else:
            j = i - half
            yh = 2.55 - j * 0.52
            ax.plot([r, r], [3.0, yh], "k-", lw=0.7)
            ax.plot([r, hi + 0.15], [yh, yh], "k-", lw=0.7)
            ax.text(hi + 0.2, yh, f"{mname} ({r:.2f})", ha="left", va="center", fontsize=7)
    ax.set_title(title, fontsize=8)
    ax.axis("off")
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)

# Average ranks from the ten-dataset ranks table of the manuscript
# (seven methods including ECKM; Nemenyi CD = 2.85 at alpha = 0.05, N = 10).
def cd_axis(ax, ranks, cd, title):
    methods = sorted(ranks, key=ranks.get)
    kk = len(methods)
    lo, hi = 1, kk
    ax.set_xlim(lo - 0.2, hi + 0.2); ax.set_ylim(0, 3.6)
    ax.plot([lo, hi], [3.0, 3.0], "k-", lw=1)
    for t in range(lo, hi + 1):
        ax.plot([t, t], [2.93, 3.07], "k-", lw=1)
        ax.text(t, 3.14, str(t), ha="center", fontsize=7)
    ax.plot([lo, lo + cd], [3.36, 3.36], "-", color="crimson", lw=2)
    ax.text(lo + cd / 2, 3.44, f"CD = {cd}", ha="center", fontsize=7, color="crimson")
    half = int(np.ceil(kk / 2))
    for i, mname in enumerate(methods):
        r = ranks[mname]
        if i < half:
            yh = 2.55 - i * 0.52
            ax.plot([r, r], [3.0, yh], "k-", lw=0.7)
            ax.plot([r, lo - 0.15], [yh, yh], "k-", lw=0.7)
            ax.text(lo - 0.2, yh, f"{mname} ({r:.2f})", ha="right", va="center", fontsize=7)
        else:
            j = i - half
            yh = 2.55 - j * 0.52
            ax.plot([r, r], [3.0, yh], "k-", lw=0.7)
            ax.plot([r, hi + 0.15], [yh, yh], "k-", lw=0.7)
            ax.text(hi + 0.2, yh, f"{mname} ({r:.2f})", ha="left", va="center", fontsize=7)
    ax.set_title(title, fontsize=8)
    ax.axis("off")

# Average ranks of the seven methods on the ten core datasets, read from
# results/results_stats7.json (written by tools/make_stats7.py).
import json
_st = json.load(open("results/results_stats7.json"))
SSE_RANKS = _st["SSE"]["avg_ranks"]
IT_RANKS = _st["IT"]["avg_ranks"]

# standalone versions (kept for the repository)
cd_diagram(SSE_RANKS, 2.85, "figures/fig3_cd_sse.pdf",
           r"Critical-difference diagram, SSE (Nemenyi, $\alpha$=0.05)")
cd_diagram(IT_RANKS, 2.85, "figures/fig4_cd_it.pdf",
           r"Critical-difference diagram, IT (Nemenyi, $\alpha$=0.05)")

# combined two-panel version used in the manuscript
fig, axes = plt.subplots(2, 1, figsize=(5.4, 3.7))
cd_axis(axes[0], SSE_RANKS, 2.85, "(a) SSE")
cd_axis(axes[1], IT_RANKS, 2.85, "(b) Lloyd iterations")
fig.tight_layout(h_pad=1.0)
fig.savefig("figures/fig34_cd.pdf", bbox_inches="tight")
plt.close(fig)

print("figures done")
