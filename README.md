# geotypes — shape catalogues of physical response signals

[![CI](https://img.shields.io/github/actions/workflow/status/fsantibanezleal/CAOS_GeoTypes/ci.yml?branch=main&label=CI)](https://github.com/fsantibanezleal/CAOS_GeoTypes/actions)
[![License](https://img.shields.io/github/license/fsantibanezleal/CAOS_GeoTypes)](LICENSE)
[![Version](https://img.shields.io/github/v/tag/fsantibanezleal/CAOS_GeoTypes?label=version&sort=semver)](https://github.com/fsantibanezleal/CAOS_GeoTypes/tags)

`geotypes` builds a **catalogue of behaviour types** from response signals whose *shape* reflects
the underlying physical system — pressure transients of fractured reservoirs, hydrogeology pumping
tests, thermal response tests — and assigns new signals to that catalogue **with statistical
guarantees**. It packages the methodology of Kamel Targhi et al. (2026, Computational Geosciences,
DOI [10.1007/s10596-026-10459-w](https://doi.org/10.1007/s10596-026-10459-w)) as a reusable,
permissively-licensed library, and adds a conformal-prediction assignment layer on top.

> **Install name vs import name.** The PyPI distribution is **`caos-geotypes`**
> (`pip install caos-geotypes`); the import is `import geotypes`. The bare distribution name
> `geotypes` is taken by an unrelated geospatial-utilities package — do not co-install it with this
> library: both would claim the `geotypes` top-level module (ADR-0061 distribution-rename route).

**Why this package exists** (mid-2026 gap): scikit-learn-extra (k-medoids) is unmaintained, the
fast Rust `kmedoids` is GPL-3, tslearn/aeon drag numba/native dependencies that do not run in
Pyodide. The `geotypes` **core is pure numpy/scipy** — it runs unchanged offline and in the
browser (Pyodide) — with optional accelerated/attribution extras.

## The pipeline

```
raw (t, p) signals
   │  preprocess: log-resample → Bourdet derivative → p'' → normalize
   ▼
curves on a common log grid
   │  distance: Sakoe-Chiba banded DTW (numpy DP; dtaidistance backend offline)
   ▼
pairwise distance matrix
   │  cluster: PAM k-medoids (BUILD+SWAP, multi-init, seeded) + silhouette K-selection
   ▼
Catalogue (medoid curves + labels + provenance; exact JSON round-trip)
   │  assign: nearest-medoid  +  split-conformal p-values / prediction sets / OOD flag
   │  attribute (extra): RF + TreeSHAP + permutation cross-check + correlation pruning
   ▼
which behaviours exist · which one is this signal · what controls each behaviour
```

## Install

```bash
pip install geotypes            # pure core (numpy + scipy)
pip install geotypes[fast]      # + dtaidistance (C-fast offline DTW matrices)
pip install geotypes[attr]      # + scikit-learn + shap (attribution layer)
```

## Quickstart

```python
import numpy as np
from geotypes import (
    generate_warren_root_ensemble, prepare_curves, dtw_matrix,
    pam_kmedoids, select_k, build_catalogue, ConformalAssigner, nearest_medoid,
)
from geotypes.preprocess import prepare_curves

# 1. an ensemble of dual-porosity pressure responses (or bring your own curves)
ens = generate_warren_root_ensemble(200, seed=0)
t_list = [ens["tD"]] * 200
p_list = list(ens["curves"])

# 2. preprocess: common log grid + Bourdet derivative + z-score
t_grid, X = prepare_curves(t_list, p_list, n_points=96, derivative_order=1)

# 3. cluster into GeoTypes
D = dtw_matrix(X, window=10)                 # dtaidistance if installed, else numpy
diag = select_k(D, range(2, 9))              # silhouette + elbow diagnostics
res = pam_kmedoids(D, k=diag["best_k"], seed=0)

# 4. the persistent catalogue artifact
cat = build_catalogue(X, t_grid, D, k=res.k, dtw_window=10, result=res)
cat.to_json("catalogue.json")

# 5. assign a NEW curve — with conformal guarantees
assigner = ConformalAssigner(cat).fit(X_calibration, labels_calibration)
out = assigner.predict(new_curve, alpha=0.1)
out.point_prediction      # nearest medoid
out.prediction_set        # GeoTypes consistent with the curve at 90% coverage
out.out_of_catalogue      # honest "this shape is not in the catalogue" flag
```

## Modules

| Module | What | Deps |
|---|---|---|
| `geotypes.preprocess` | log resampling, Bourdet derivative (log-window L), second derivative p'', normalization, `prepare_curves` | numpy |
| `geotypes.distance` | banded DTW (numpy DP), pairwise matrices (optional dtaidistance backend, parity-tested), live `distances_to_references` | numpy (+fast) |
| `geotypes.cluster` | PAM k-medoids on precomputed distances, silhouette-from-distances, `select_k` | numpy |
| `geotypes.catalogue` | the `Catalogue` artifact: medoids + labels + preprocessing + provenance, exact JSON round-trip | numpy |
| `geotypes.assign` | nearest-medoid + `ConformalAssigner` (class-conditional split-conformal: p-values, prediction sets, OOD) | numpy |
| `geotypes.attribute` | Spearman correlation pruning, RF with accuracy gate, TreeSHAP + permutation importance cross-check | [attr] |
| `geotypes.synthetic` | Warren-Root dual-porosity + homogeneous radial generators (Gaver-Stehfest inversion), seeded ensembles | numpy+scipy |

## Guarantees & honesty

- **Determinism:** every stochastic step is seeded (`numpy.random.default_rng`); same inputs + seed
  → identical catalogue, bit-for-bit.
- **Conformal validity:** under exchangeability, prediction sets cover the true class with
  probability ≥ 1−α per class (class-conditional calibration). The empty set is reported as an
  **out-of-catalogue** flag, never silently replaced by the nearest medoid. Mind the finite-sample
  floor: the minimum achievable p-value is 1/(n_c+1), so α below that cannot produce empty sets.
- **Attribution gate:** if the Random Forest cannot predict the labels from the descriptors
  (held-out accuracy below the gate), SHAP/permutation importances are withheld — noise is not
  reported as insight. SHAP and permutation rankings are cross-checked (Spearman agreement).
- **Physics validation:** the synthetic generators are tested against closed-form limits
  (homogeneous late-time `0.5(ln tD + 0.80907)`, derivative plateau 0.5, Warren-Root → homogeneous
  as ω→1, valley-depth monotonicity in ω).

## Docs

- [docs/theory.md](docs/theory.md) — the science: PTA derivatives, DTW, PAM, conformal prediction, attribution pitfalls (with references)
- [docs/quickstart.md](docs/quickstart.md) — worked end-to-end example
- [docs/design.md](docs/design.md) — package design, licensing rationale, Pyodide lane, API stability

## Development

```bash
py -3.12 -m venv .venv && .venv/Scripts/pip install -e .[dev,attr,fast]
.venv/Scripts/python -m pytest    # 32 tests: physics limits, DTW properties/parity, PAM recovery,
                                  # conformal coverage + OOD, JSON round-trips, SHAP sanity
```

Versioning: CAOS `X.XX.XXX` display convention (CHANGELOG + git tags); PEP 440 in `pyproject.toml`.
Consumed by [CAOS_RES_FlowDNA](https://github.com/fsantibanezleal/CAOS_RES_FlowDNA) (FlowDNA).
