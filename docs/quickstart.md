# Quickstart — from raw curves to a guaranteed assignment

A complete, runnable walk-through (also exercised by the test suite). Scenario: you have an
ensemble of pressure-transient responses (here synthetic Warren-Root; in FlowDNA they come from
DFN flow simulation) and want (1) a catalogue of behaviour types, (2) to classify new curves with
confidence statements, (3) to know which parameters control each behaviour.

## 0. Install

```bash
pip install pygeotypes[fast,attr]   # offline; the pure core alone suffices in Pyodide
```

## 1. Generate (or load) the ensemble

```python
import numpy as np
from pygeotypes import generate_warren_root_ensemble

ens = generate_warren_root_ensemble(
    300, omega_range=(0.01, 0.5), lam_range=(1e-8, 1e-4), noise_sd=0.01, seed=0
)
t_list = [ens["tD"]] * 300
p_list = list(ens["curves"])
params = ens["params"]              # [{'omega':…, 'lam':…, 'skin':…}, …]
```

Bring-your-own-data: any list of `(t, p)` arrays with strictly increasing positive `t` works; the
preprocessing resamples everything onto the common log grid (intersection of time ranges).

## 2. Preprocess to shape space

```python
from pygeotypes import log_resample, bourdet_derivative  # building blocks
from pygeotypes.preprocess import prepare_curves          # the one-call version

t_grid, X = prepare_curves(
    t_list, p_list,
    n_points=96,
    derivative_order=2,   # p'' — offset-free (Freites et al. 2023); use 1 to see the classic valley
    L=0.2,                # Bourdet smoothing window (log cycles)
    norm="zscore",
)
```

## 3. Distance matrix + choose K + cluster

```python
from pygeotypes import dtw_matrix, select_k, pam_kmedoids

D = dtw_matrix(X, window=10)                    # Sakoe-Chiba half-width in grid cells
diag = select_k(D, range(2, 9), seed=0)         # {'k', 'cost', 'silhouette', 'best_k'}
res = pam_kmedoids(D, k=diag["best_k"], n_init=10, seed=0)
print(res.k, res.silhouette, res.medoid_indices)
```

## 4. Persist the catalogue (the artifact both lanes share)

```python
from pygeotypes import build_catalogue, Catalogue

cat = build_catalogue(
    X, t_grid, D, k=res.k, dtw_window=10,
    preprocessing={"derivative_order": 2, "L": 0.2, "norm": "zscore", "n_points": 96},
    provenance={"source": "warren-root synthetic v1", "date": "2026-07-03"},
    result=res,
)
cat.to_json("geotype_catalogue.json")
# later / elsewhere (including the browser):
cat = Catalogue.from_json("geotype_catalogue.json")
```

## 5. Conformal assignment of new curves

Split your data honestly: the catalogue was built on one part; calibrate on curves it never saw.

```python
from pygeotypes import ConformalAssigner, nearest_medoid

# X_cal, labels_cal: preprocessed calibration curves + their GeoType labels
assigner = ConformalAssigner(cat).fit(X_cal, labels_cal)
assigner.to_json("calibration.json")            # bake for the live lane

out = assigner.predict(x_new, alpha=0.1)
print(out.point_prediction)   # nearest medoid
print(out.prediction_set)     # e.g. [1] tight, or [1, 3] ambiguous — both honest
print(out.out_of_catalogue)   # True → this shape is NOT in the catalogue at 90% confidence
```

Calibration-size rule: empty sets (OOD) are only reachable when `n_c ≥ 1/alpha − 1` per class
(the conformal p-value floor is `1/(n_c+1)`).

## 6. Attribution — what controls each behaviour

```python
import numpy as np
from pygeotypes.attribute import attribute_geotypes

feats = np.array([[p["omega"], p["lam"], p["skin"]] for p in params])
report = attribute_geotypes(
    np.log10(feats + 1e-30),                # descriptors on sensible scales
    res.labels, ["log_omega", "log_lam", "skin"],
    accuracy_gate=0.7, seed=0,
)
report["gate"]                # accuracy + passed — if not passed, importances are withheld
report["shap_mean_abs"]       # per GeoType: which descriptor drives membership
report["rank_agreement_spearman"]  # SHAP vs permutation cross-check
```

For real fracture networks, replace (ω, λ, skin) with DFN descriptors (P32 intensity, length-law
exponent, orientation κ, aperture stats, percolation parameter, backbone fraction) — that layer
lives in the consuming product (FlowDNA), not in this package.

## 7. In the browser (Pyodide sketch)

```python
# inside a Pyodide worker: micropip.install("pygeotypes") or ship the wheel
from pygeotypes import Catalogue, ConformalAssigner
from pygeotypes.preprocess import log_resample, second_log_derivative, normalize

cat = Catalogue.from_json(catalogue_json_text)          # baked offline
assigner = ConformalAssigner.from_dict(cal_dict, cat)   # baked offline
# preprocess the user's curve EXACTLY as cat.preprocessing says, then:
result = assigner.predict(x_user, alpha=0.1)
```
