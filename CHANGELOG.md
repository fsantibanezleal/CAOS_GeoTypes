# Changelog

All notable changes to `geotypes`. Display format `X.XX.XXX` (CAOS convention); PEP 440 mirror in
`pyproject.toml`. Tag every release.

## [0.01.001] — 2026-07-03

### Changed
- PyPI distribution name set to **`caos-geotypes`** (the bare `geotypes` is taken on PyPI by an
  unrelated geospatial package; ADR-0061 route: rename the distribution, keep `import geotypes`).
  Co-install collision documented in the README.

### Added
- `publish-pypi.yml` (Trusted Publishing / OIDC) — publishing awaits the pending-publisher
  registration and the explicit opt-in to publish (repo stays private until then).

## [0.01.000] — 2026-07-03

### Added
- Initial release of the full core, pure numpy/scipy (Pyodide-safe):
  - `preprocess`: log resampling, Bourdet derivative (log-window L), second logarithmic
    derivative p'', z-score/max normalization, `prepare_curves` ensemble pipeline.
  - `distance`: Sakoe-Chiba banded DTW (numpy DP), pairwise matrices with optional
    `dtaidistance` backend (parity-tested), live `distances_to_references`.
  - `cluster`: PAM k-medoids (greedy BUILD + SWAP, multi-restart, seeded) on precomputed
    distances; silhouette-from-distances; `select_k` diagnostics.
  - `catalogue`: the persistent `Catalogue` artifact (medoid curves, labels, preprocessing +
    provenance metadata, `schema_version`), exact JSON round-trip.
  - `assign`: nearest-medoid + `ConformalAssigner` — class-conditional split-conformal p-values,
    prediction sets, out-of-catalogue flag; JSON persistence of calibration scores.
  - `attribute` (extra `[attr]`): Spearman correlation pruning, Random-Forest with held-out
    accuracy gate, TreeSHAP per-class importances + permutation-importance cross-check.
  - `synthetic`: Warren-Root dual-porosity + homogeneous radial generators via Gaver-Stehfest
    inversion (wellbore storage + skin supported), seeded ensemble generation.
- 32 tests: physics limits, metric properties, backend parity, clustering recovery, conformal
  coverage + OOD reachability, artifact round-trips, attribution sanity + gate.
- Docs: `docs/theory.md` (referenced), `docs/design.md`, `docs/quickstart.md`.
