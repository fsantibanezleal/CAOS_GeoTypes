# Theory — why each stage is what it is

This is the scientific backbone of `geotypes`. Each section states the method, why it was chosen,
its assumptions and failure modes, with references. (Deep dossiers with the full literature trail
live in the private management vault; this file is the self-contained repo summary.)

## 1. The shape carriers: pressure transients and their log derivatives

A drawdown/buildup test measures wellbore pressure vs time. The information about the reservoir's
structure is in the *shape* of the log-log diagnostic pair (Δp, p') where **p' = dΔp/d ln t** is the
Bourdet derivative (Bourdet, Ayoub & Pirard 1989, SPE Formation Evaluation 4(2):293–302,
DOI 10.2118/12777-PA). Flow regimes map to derivative signatures:

| Regime | p' signature |
|---|---|
| wellbore storage | unit slope (early) |
| radial (infinite acting) | plateau at 0.5 (dimensionless) |
| linear (fracture/channel) | slope 1/2 |
| bilinear (finite-conductivity fracture) | slope 1/4 |
| dual porosity (Warren-Root) | **valley** between two 0.5 plateaus |
| closed boundary | late unit slope |

The derivative is computed with the **Bourdet window** (weighted two-sided secants at ≥ L/2
log-cycles, L≈0.2) rather than naive differencing, because measured pressure is noisy and the
derivative amplifies noise.

**Why cluster the second derivative p''.** DTW cannot compensate a vertical offset between two
curves; the same behaviour at different absolute pressure levels would look different. The second
logarithmic derivative removes level and trend, leaving regime *transitions* — Freites, Corbett &
Geiger (2023, Transport in Porous Media, DOI 10.1007/s11242-023-01929-1) established p''-based DTW
clustering for exactly this reason, and Kamel Targhi et al. (2026) inherit it.

## 2. DTW with a Sakoe-Chiba band

Two responses with the same regime sequence but shifted transition times (e.g. the dual-porosity
valley arriving a decade later because λ is smaller) are the *same behaviour* at different scale.
Euclidean distance punishes the shift sample-by-sample; **Dynamic Time Warping** aligns the shapes
first (Sakoe & Chiba 1978, IEEE TASSP 26(1):43–49). The **band** (half-width w) bounds how far the
alignment may stray from the diagonal:

- w too small → DTW degenerates to Euclidean, timing noise contaminates clusters;
- w too large → early-time behaviour can match late-time behaviour (physically absurd) and all
  curves look alike.

On ~100-point log grids, w ≈ 10–20 cells is the working range (Freites et al. used w=20 on n≈100).
Local cost is the squared difference; the reported distance is the square root of the optimal
accumulated cost (the dtaidistance/tslearn convention), so distances have signal units. The pure
numpy DP is O(n·w) per pair; full offline matrices can delegate to `dtaidistance` (Apache-2, C
core) — the two backends are parity-tested in CI.

## 3. PAM k-medoids on the DTW matrix

K-means requires a mean, and the arithmetic mean of time series under DTW is ill-defined (the
barycenter problem); **k-medoids** only needs the distance matrix and returns *actual observed
curves* as cluster prototypes — physically meaningful representatives you can plot and interpret.
PAM (Partitioning Around Medoids; Kaufman & Rousseeuw 1990, *Finding Groups in Data*) is the exact
greedy version: BUILD picks medoids that maximally reduce total cost; SWAP exchanges medoid ↔
non-medoid while any exchange lowers cost. Multi-restart (seeded) guards local minima; the run is
deterministic for a given seed.

**K selection**: mean silhouette computed *from the DTW matrix* (not from a Euclidean embedding),
maximized over a K range, with the cost-vs-K elbow as the visual cross-check. Silhouette values
around 0.3–0.5 are typical and acceptable for transient-response ensembles (the 2026 paper reports
0.37–0.46) — behaviour space is a continuum; the catalogue quantizes it.

## 4. Conformal GeoType assignment (the novel layer)

Nearest-medoid assignment answers "which GeoType is closest?" but not "how sure are we?" — and it
*always* answers, even for a curve unlike anything in the catalogue. Split-conformal prediction
(Vovk, Gammerman & Shafer 2005, *Algorithmic Learning in a Random World*; Angelopoulos & Bates
2023, Found. Trends ML, DOI 10.1561/2200000101) fixes both with finite-sample guarantees:

- **Nonconformity score**: DTW distance to the class medoid.
- **Calibration**: held-out curves per class (class-conditional / Mondrian), giving per-class
  validity under exchangeability — important because GeoType populations are imbalanced.
- **Output**: p-value per class; the **prediction set** {g : p_g > α} covers the true class with
  probability ≥ 1−α; an **empty set is the out-of-catalogue flag** — the honest "this response is
  not in the catalogue" a characterisation workflow must surface.
- **Finite-sample floor**: min p-value = 1/(n_c+1); α below that floor cannot yield empty sets, so
  calibration sizes must satisfy n_c ≥ 1/α − 1.

Everything is pure numpy, so the guarantee-carrying assignment runs *live in the browser* against
a baked catalogue + calibration scores.

## 5. Attribution: RF + SHAP, hardened

Given GeoType labels and fracture-network descriptors (intensity P10/P21/P32, power-law length
exponent, orientation dispersion κ, aperture statistics, percolation parameter, backbone fraction),
a Random-Forest classifier + **TreeSHAP** (Lundberg et al. 2020, Nature Machine Intelligence 2:56–67,
DOI 10.1038/s42256-019-0138-9) ranks which descriptors control which behaviour. Known pitfalls,
guarded explicitly:

1. **Correlated descriptors** split credit arbitrarily → Spearman pruning (|ρ| ≥ 0.9) *before*
   fitting; dropped→kept mapping is reported, not hidden.
2. **Uninformative labels** produce confident-looking noise → a held-out **accuracy gate**; below
   it, importances are withheld.
3. **Method idiosyncrasy** → permutation importance computed alongside; the Spearman agreement of
   the two rankings is part of the artifact.

## 6. The analytical generators (classical rung + live lane)

Warren & Root (1963, SPE Journal 3(3):245–255, DOI 10.2118/426-PA) dual-porosity solution with
pseudo-steady interporosity flow, in Laplace space
`pwD = K0(√(s·f(s)))/s`, `f(s) = (ω(1−ω)s + λ)/((1−ω)s + λ)`; wellbore storage + skin per Agarwal,
Al-Hussainy & Ramey (1970, SPE Journal 10(3):279–290, DOI 10.2118/2466-PA). Numerical inversion by
Gaver-Stehfest (Stehfest 1970, CACM 13(1):47–49, DOI 10.1145/361953.361969), N=12 — the standard
well-testing choice, accurate for these smooth monotone transforms. Tests pin the known limits:
late-time homogeneous `pwD ≈ 0.5(ln tD + 0.80907)`, derivative plateau 0.5, ω→1 reduction,
valley-depth monotonicity in ω, valley timing in 1/λ.

These curves are (a) the classical PTA rung of any model ladder built on this package, (b) the
synthetic ensemble source for tests/demos, and (c) fast enough to run interactively in Pyodide.

## References

- Bourdet D., Ayoub J.A., Pirard Y.M. (1989). *Use of pressure derivative in well-test
  interpretation.* SPE Formation Evaluation 4(2). DOI 10.2118/12777-PA
- Warren J.E., Root P.J. (1963). *The behavior of naturally fractured reservoirs.* SPE J. 3(3). DOI 10.2118/426-PA
- Agarwal R.G., Al-Hussainy R., Ramey H.J. (1970). *An investigation of wellbore storage and skin
  effect in unsteady liquid flow.* SPE J. 10(3). DOI 10.2118/2466-PA
- Stehfest H. (1970). *Algorithm 368: numerical inversion of Laplace transforms.* CACM 13(1). DOI 10.1145/361953.361969
- Sakoe H., Chiba S. (1978). *Dynamic programming algorithm optimization for spoken word
  recognition.* IEEE TASSP 26(1). DOI 10.1109/TASSP.1978.1163055
- Kaufman L., Rousseeuw P.J. (1990). *Finding Groups in Data.* Wiley. DOI 10.1002/9780470316801
- Freites A., Corbett P.W.M., Geiger S. (2023). *Semi-supervised sequential clustering of pressure
  transient responses.* Transport in Porous Media. DOI 10.1007/s11242-023-01929-1
- Kamel Targhi E., Rongier G., Bruna P.-O., Daniilidis A., Geiger S. (2026). *Unsupervised learning
  for geologically consistent fluid flow analysis in fractured reservoirs.* Computational
  Geosciences 30:57. DOI 10.1007/s10596-026-10459-w
- Vovk V., Gammerman A., Shafer G. (2005). *Algorithmic Learning in a Random World.* Springer. DOI 10.1007/b106715
- Angelopoulos A.N., Bates S. (2023). *Conformal prediction: a gentle introduction.* Found. Trends
  ML 16(4). DOI 10.1561/2200000101
- Lundberg S.M. et al. (2020). *From local explanations to global understanding with explainable AI
  for trees.* Nature Machine Intelligence 2. DOI 10.1038/s42256-019-0138-9
