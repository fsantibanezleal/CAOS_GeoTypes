# Design — package shape, licensing rationale, lanes, stability

## Design goals (in priority order)

1. **Pure-numpy core** — every guarantee-carrying path (preprocess → distance → assign) must run in
   Pyodide unchanged, so a browser app can classify a user's curve live against a baked catalogue.
2. **Determinism** — seeded everything; a catalogue is a pure function of (curves, params, seed).
3. **Honesty primitives built-in** — OOD flag, attribution accuracy gate, ranking cross-checks and
   finite-sample caveats are part of the API, not left to the caller's discipline.
4. **Permissive license** (Apache-2.0) with no copyleft in the dependency core.

## Licensing rationale (why not the existing tools)

| Candidate | Problem (as of 2026-07) |
|---|---|
| scikit-learn-extra `KMedoids` | unmaintained since 2023; sklearn-version friction |
| `kmedoids` (Rust FasterPAM) | GPL-3 — viral for a permissive library |
| tslearn | BSD but numba/joblib heavy; no Pyodide; DTW k-means focus |
| aeon / sktime | numba required; heavy; no Pyodide |
| dtaidistance | Apache-2 ✔ — used, but only as an **optional offline backend** (C ext, no Pyodide) |

PAM on a precomputed matrix is small enough to own (~130 lines incl. diagnostics) and unlocks the
browser lane; `dtaidistance` accelerates the offline O(n²) matrix when installed, with CI parity
tests between backends.

## The two lanes

| Lane | What runs | Deps |
|---|---|---|
| **Offline** (pipeline, `.venv`) | full pairwise DTW matrix (dtaidistance), PAM over K range, catalogue build, calibration, RF+SHAP attribution → all baked to JSON | `pygeotypes[fast,attr]` |
| **Live** (browser, Pyodide) | preprocess one curve → K DTW distances to medoids (numpy DP) → conformal p-values from baked calibration scores | `pygeotypes` core only |

The `Catalogue` + `ConformalAssigner` JSON round-trips are the contract between the lanes: the
offline side bakes them, the live side loads them. Nothing recomputes clustering in the browser.

## API stability

Public API = what `pygeotypes.__init__` exports. `0.x`: breaking changes allowed with a CHANGELOG
entry; the JSON artifacts carry `schema_version` so consumers can gate. The package is consumed by
FlowDNA (CAOS_RES_FlowDNA `flowdnalab`), whose domain layer (GeoDFN ensembles, open-DARTS
simulation, fracture descriptors, contracts) deliberately stays OUT of this package: `pygeotypes` is
signal-shape machinery, not reservoir engineering.

## Performance envelope (measured on the dev workstation)

- numpy DTW: ~1 ms per pair at n=96, w=10 → full 200×200 matrix ≈ 20 s pure numpy; dtaidistance
  does the same in well under a second. Live lane needs K (≈4–8) distances → interactive.
- PAM SWAP is O(k(n−k)²) per iteration: fine to n ≈ 2–3k curves offline; beyond that, subsample to
  build the catalogue and assign the rest (the catalogue quality saturates long before that).

## Testing philosophy

Physics limits (closed forms), metric properties (identity/symmetry/band monotonicity), backend
parity, recovery-from-known-truth (three-regime ensembles), statistical guarantees (empirical
conformal coverage with slack, OOD reachability), artifact round-trips, and attribution sanity
(informative feature wins; noise labels are gated out). 32 tests, all seeded, no network.
