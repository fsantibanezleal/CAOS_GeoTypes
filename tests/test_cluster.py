import numpy as np
import pytest

from geotypes.cluster import pam_kmedoids, select_k, silhouette_from_distances
from geotypes.distance import dtw_matrix
from geotypes.preprocess import prepare_curves
from geotypes.synthetic import warren_root_pd


def _three_regime_ensemble(n_per: int = 8, seed: int = 0):
    """Three well-separated dual-porosity regimes + light noise.

    Separation is driven by valley TIMING (λ decades apart) + depth (ω); the tight DTW band
    (window=4 on 48 points) keeps timing discriminative instead of warping it away.
    """
    rng = np.random.default_rng(seed)
    tD = np.logspace(2, 10, 200)
    regimes = [
        {"omega": 0.35, "lam": 1e-4},   # shallow, early valley
        {"omega": 0.03, "lam": 1e-6},   # deep, mid valley
        {"omega": 0.01, "lam": 5e-9},   # deep, late valley
    ]
    t_list, p_list, truth = [], [], []
    for g, reg in enumerate(regimes):
        for _ in range(n_per):
            omega = reg["omega"] * float(np.exp(rng.normal(0, 0.05)))
            lam = reg["lam"] * float(np.exp(rng.normal(0, 0.08)))
            y = warren_root_pd(tD, omega=min(omega, 0.99), lam=lam)
            t_list.append(tD)
            p_list.append(y * np.exp(rng.normal(0, 0.005, size=y.shape)))
            truth.append(g)
    tg, X = prepare_curves(t_list, p_list, n_points=48, derivative_order=1, norm="zscore")
    return tg, X, np.array(truth)


def _purity(labels, truth):
    total = 0
    for c in np.unique(labels):
        vals, counts = np.unique(truth[labels == c], return_counts=True)
        total += counts.max()
    return total / labels.size


def test_pam_recovers_three_regimes():
    _, X, truth = _three_regime_ensemble()
    D = dtw_matrix(X, window=4, backend="numpy")
    res = pam_kmedoids(D, k=3, n_init=5, seed=0)
    assert _purity(res.labels, truth) >= 0.9
    assert res.silhouette > 0.2
    assert res.medoid_indices.size == 3
    assert res.per_curve_distance.shape == (X.shape[0],)


def test_pam_deterministic_for_seed():
    _, X, _ = _three_regime_ensemble()
    D = dtw_matrix(X, window=4, backend="numpy")
    r1 = pam_kmedoids(D, k=3, n_init=5, seed=7)
    r2 = pam_kmedoids(D, k=3, n_init=5, seed=7)
    assert np.array_equal(r1.medoid_indices, r2.medoid_indices)
    assert np.array_equal(r1.labels, r2.labels)
    assert r1.cost == r2.cost


def test_pam_input_validation():
    D = np.zeros((4, 4))
    with pytest.raises(ValueError):
        pam_kmedoids(D, k=0)
    with pytest.raises(ValueError):
        pam_kmedoids(D, k=5)
    with pytest.raises(ValueError):
        pam_kmedoids(np.zeros((3, 4)), k=2)


def test_silhouette_perfect_separation():
    # two tight blobs, far apart -> silhouette near 1
    D = np.array(
        [
            [0.0, 0.1, 5.0, 5.0],
            [0.1, 0.0, 5.0, 5.0],
            [5.0, 5.0, 0.0, 0.1],
            [5.0, 5.0, 0.1, 0.0],
        ]
    )
    s = silhouette_from_distances(D, np.array([0, 0, 1, 1]))
    assert s > 0.95


def test_select_k_prefers_true_k():
    _, X, _ = _three_regime_ensemble()
    D = dtw_matrix(X, window=4, backend="numpy")
    out = select_k(D, k_range=range(2, 6), n_init=3, seed=0)
    assert out["best_k"] == 3
    assert len(out["k"]) == len(out["silhouette"]) == len(out["cost"]) == 4
