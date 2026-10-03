# CCKM: Polar-Grid Seeding for k-Means

Code, data scripts and raw results for the paper

> B. Kr. Paul and K. Giri, *Polar-Grid Seeding for k-Means: A Deterministic
> Initializer Family with a Provable Operating Envelope*, submitted to IEEE
> Transactions on Knowledge and Data Engineering.

The base method was introduced in

> B. Kr. Paul and K. Giri, *CCKM: A Modified k-means Clustering using
> Concentric Circles*, Research Square preprint, 2026.
> https://doi.org/10.21203/rs.3.rs-10665044/v1

## Method in brief

The data are z-scored and projected onto the first two principal components.
The projected plane is divided into k radial bands about the global centroid
(equal-width or quantile radii), and each band into m angular sectors. The
seed of each band is the mean of its most populated sector. Empty bands and
coincident seeds are repaired by farthest-first steps. The base member,
CCKM(4, equal), has no parameter other than k and seeds in O(n) time. The
seeds are then passed to standard Lloyd iterations.

The implementation is `cckm_seeds()` in `cckm_experiments.py`.

## Repository layout

```
cckm_experiments.py     core ten-dataset study (main, stats, win/loss,
                        diagnostic, ablation, contamination, scaling)
cckm_extension.py       thirty-dataset extension, ECKM baseline, k-sweep
tools/
  run_eckm_core.py      ECKM on the ten core datasets
  make_stats7.py        seven-method ranks, Friedman and Wilcoxon-Holm tests
  run_extension_extras.py  quantile radii and radial-mode diagnostic (extension)
  make_ext_tables.py    summary tables for the thirty-dataset suite
  make_figs.py          Figs. 1, 2 and the critical-difference diagrams
fetch_data.sh           downloads all benchmark files into data/
results/                the result files used in the paper (see below)
figures/                figures used in the paper
```

## Installation

Python 3.12 was used. Install the pinned versions with

```
pip install -r requirements.txt
```

## Reproducing the results

All commands are run from the repository root. Each script writes its
output files into the current directory, so the files in `results/` can be
compared with a fresh run.

```
./fetch_data.sh                              # about 17 MB, into data/
python cckm_experiments.py                   # core suite, about 15 min
python tools/run_eckm_core.py                # ECKM, core suite
python tools/make_stats7.py                  # Table 5 and the Wilcoxon tests
python cckm_extension.py small               # extension, n <= 20,000
python cckm_extension.py large               # MNIST, Birch (slow)
python cckm_extension.py eckm                # ECKM, extension (Birch about 200 s each)
python cckm_extension.py ksweep              # seeding time vs k on Birch-rg1
python tools/run_extension_extras.py         # quantile radii, diagnostic
python tools/make_ext_tables.py              # Tables 9 and 10
python tools/make_figs.py                    # figures/
```

The extension stages append to their CSV files, so delete `ext_small.csv`,
`ext_large.csv` and `ext_eckm.csv` before a full re-run.

## Which file gives which table

| Result in the paper | File in `results/` | Produced by |
|---|---|---|
| Dataset table (n, d, k, EVR2) | `results_main.csv` (column EVR2) | `cckm_experiments.py` |
| Full results, ten datasets (two tables) | `results_means.csv`, `results_stds.csv`, `results_eckm.csv` | `cckm_experiments.py`, `tools/run_eckm_core.py` |
| Average ranks and Friedman tests, ten datasets | `results_stats7.json` | `tools/make_stats7.py` |
| Wilcoxon-Holm tests against CCKM | `results_wilcoxon7.json` | `tools/make_stats7.py` |
| Win/loss probabilities and radial-mode diagnostic | `results_winloss.json`, `results_diagnostic.json` | `cckm_experiments.py` |
| Ablation, equal vs quantile radii, m = 4, 8, 16 | `results_ablation.csv` | `cckm_experiments.py` |
| Seeding time vs k on Birch-rg1 | `ext_ksweep.csv` | `cckm_extension.py ksweep` |
| Seeding time vs n (synthetic) | `results_scaling.csv` | `cckm_experiments.py` |
| Contamination study | `results_noise.csv` | `cckm_experiments.py` |
| Extension datasets and per-dataset outcomes | `ext_table.csv`, `ext_quantile.csv`, `ext_diagnostic.json` | `tools/run_extension_extras.py`, `tools/make_ext_tables.py` |
| Average ranks and tests, thirty datasets | `ext_stats30.json`, `ext_means_all.csv` | `tools/make_ext_tables.py` |
| Raw per-run results | `results_main.csv`, `ext_small.csv`, `ext_large.csv`, `ext_eckm.csv`, `results_noise.csv` | as above |

`results_stats.json` is the six-method version (without ECKM) written by
`cckm_experiments.py`; the paper reports the seven-method version in
`results_stats7.json`.

## Notes

* **Determinism.** CCKM, Maxmin and ECKM run once. Randomized methods use
  seeds 0-49. PCA uses scikit-learn's `svd_flip` sign convention, which is
  part of the determinism of the method; flipping a component sign can change
  the densest-sector choice.
* **Ties in rank statistics.** Average ranks are computed after rounding
  the per-dataset means to 1e-6, so methods that reach the same solution
  (e.g. on Pathbased) receive tied ranks.
* **Timings.** The seeding-time tables report wall-clock times and vary with the
  machine and load. The ordering of the methods is stable across runs; the
  absolute values are not.
* **ECKM.** The ECKM baseline follows the published implementation of
  Biswas, Giri and Roy (Expert Systems with Applications, 2023),
  https://github.com/kinsukgiri/ECKM. Only the interior-hull test is
  vectorized; its output is unchanged.
* **Data sources.** Artificial and FCPS sets and Ecoli come from
  https://github.com/deric/clustering-benchmark, Seeds from
  https://github.com/jbrownlee/Datasets, MNIST from
  https://github.com/fgnt/mnist, and Iris, Wine, Digits and Breast-Cancer
  from scikit-learn. Each file was checked against its published instance
  count. The datasets are not redistributed here.

Environment used for the paper: Python 3.12, NumPy 2.4.4, SciPy 1.17.1,
pandas 3.0.2, scikit-learn 1.8.0, x86-64 Linux (Ubuntu 24.04).

## Citation

If you use this code, please cite the preprint above (see `CITATION.cff`).

## License

MIT, see `LICENSE`.
