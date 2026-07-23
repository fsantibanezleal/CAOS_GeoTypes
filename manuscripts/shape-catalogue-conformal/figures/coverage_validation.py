#!/usr/bin/env python3
"""Reproducible empirical-coverage validation of the pygeotypes conformal assignment layer.

Builds a GeoType catalogue from a Warren-Root dual-porosity pressure-transient ensemble, fits the
class-conditional split-conformal assigner on a held-out calibration split, and measures on a fresh test split:

  (1) empirical marginal coverage vs the 1 - alpha target, across a grid of alpha (the conformal calibration
      curve), plus the mean prediction-set size, and
  (2) the out-of-catalogue empty-set rate on shapes alien to the catalogue (high-frequency oscillations),
      which a characterisation tool must flag rather than force to the nearest medoid.

Ground truth is the self-consistent nearest-medoid label under the same DTW metric (the honest definition for an
unsupervised catalogue, mirroring tests/test_catalogue_assign.py). Also exports the catalogue medoid curves for
the catalogue figure. Writes ../data/coverage_validation.json and ../data/catalogue_medoids.json.

Run:  python coverage_validation.py
Deps: pygeotypes[fast], numpy.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pygeotypes import (ConformalAssigner, build_catalogue, dtw_matrix,
                        generate_warren_root_ensemble, nearest_medoid, pam_kmedoids, select_k)
from pygeotypes.preprocess import prepare_curves

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
DATA.mkdir(exist_ok=True)

N_POINTS = 96
DTW_BAND = 10
ALPHAS = [0.05, 0.10, 0.15, 0.20, 0.30]


def _prep(n, seed):
    # wide omega + lambda span -> a shape spectrum with two well-separated modes (early- and late-transition
    # dual porosity), so the split-conformal calibration is clean and the medoid prototypes are smooth.
    ens = generate_warren_root_ensemble(n, seed=seed, omega_range=(0.02, 0.6), lam_range=(1e-7, 1e-4))
    t_list = [ens["tD"]] * n
    p_list = list(ens["curves"])
    t_grid, X = prepare_curves(t_list, p_list, n_points=N_POINTS, derivative_order=1)
    return t_grid, X


def main():
    # 1. catalogue from a training ensemble; K by silhouette
    t_grid, Xtr = _prep(160, seed=0)
    D = dtw_matrix(Xtr, window=DTW_BAND)
    diag = select_k(D, range(2, 8))
    K = diag["best_k"]
    res = pam_kmedoids(D, k=K, seed=0)
    cat = build_catalogue(Xtr, t_grid, D, k=res.k, dtw_window=DTW_BAND, result=res)

    # 2. calibration + test splits from fresh seeds (exchangeable)
    _, Xcal = _prep(320, seed=1)
    _, Xte = _prep(400, seed=2)
    cal_labels = np.array([nearest_medoid(x, cat)[0] for x in Xcal])
    assigner = ConformalAssigner(cat).fit(Xcal, cal_labels)
    class_counts = {int(g): int((cal_labels == g).sum()) for g in range(K)}

    # 3. empirical coverage vs alpha (marginal + per class), + mean set size
    truth_te = np.array([nearest_medoid(x, cat)[0] for x in Xte])
    rows = []
    for a in ALPHAS:
        covered, set_sizes = 0, []
        per_class = {g: [0, 0] for g in range(K)}
        for x, tg in zip(Xte, truth_te):
            r = assigner.predict(x, alpha=a)
            hit = int(tg in r.prediction_set)
            covered += hit
            set_sizes.append(len(r.prediction_set))
            per_class[tg][0] += hit
            per_class[tg][1] += 1
        rows.append({"alpha": a, "target": round(1 - a, 3),
                     "empirical_coverage": round(covered / len(Xte), 3),
                     "mean_set_size": round(float(np.mean(set_sizes)), 2),
                     "per_class_coverage": {int(g): round(c[0] / c[1], 3) for g, c in per_class.items() if c[1]}})
        print(f"alpha={a} target={1-a:.2f} coverage={covered/len(Xte):.3f} mean|set|={np.mean(set_sizes):.2f}")

    # 4. out-of-catalogue: alien high-frequency shapes must yield empty sets; sweep alpha
    rng = np.random.default_rng(7)
    n_ood = 120
    alien = []
    for _ in range(n_ood):
        freq = 25 + 40 * rng.random()
        w = np.sin(np.linspace(0, freq * np.pi, N_POINTS)) * (2 + rng.random())
        alien.append((w - w.mean()) / (w.std() + 1e-9))
    ood_rows = []
    for a in ALPHAS:
        flagged = sum(assigner.predict(w, alpha=a).out_of_catalogue for w in alien)
        ood_rows.append({"alpha": a, "empty_set_rate": round(flagged / n_ood, 3)})
        print(f"OOD alpha={a}: empty-set rate {flagged/n_ood:.3f}")

    out = {
        "meta": {"n_points": N_POINTS, "dtw_band": DTW_BAND, "K": int(K),
                 "n_train": 160, "n_cal": 320, "n_test": 400, "noise_sd": 0.0,
                 "silhouette_best_k": int(K), "class_counts_cal": class_counts,
                 "note": "pygeotypes class-conditional split-conformal empirical coverage + OOD"},
        "coverage": rows,
        "ood": {"alien_shapes": n_ood, "by_alpha": ood_rows},
    }
    (DATA / "coverage_validation.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    # export medoid curves for the catalogue figure
    medoids = {"t_grid": np.asarray(t_grid).tolist(),
               "medoids": cat.medoid_curves.tolist(),
               "counts": [int(c) for c in cat.counts()], "K": int(K)}
    (DATA / "catalogue_medoids.json").write_text(json.dumps(medoids), encoding="utf-8")
    print(f"wrote coverage_validation.json + catalogue_medoids.json  K={K} OOD={flagged}/{n_ood}")


if __name__ == "__main__":
    main()
