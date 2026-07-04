import numpy as np

from pygeotypes.preprocess import bourdet_derivative
from pygeotypes.synthetic import (
    generate_warren_root_ensemble,
    homogeneous_pd,
    stehfest_weights,
    warren_root_pd,
)


def test_stehfest_weights_sum_property():
    # Stehfest weights of even N sum to zero (classic sanity check)
    for N in (8, 12, 16):
        V = stehfest_weights(N)
        assert abs(V.sum()) < 1e-6 * np.abs(V).max()


def test_homogeneous_late_time_matches_log_approximation():
    tD = np.logspace(5, 8, 40)
    pd_num = homogeneous_pd(tD)
    pd_ref = 0.5 * (np.log(tD) + 0.80907)
    assert np.max(np.abs(pd_num - pd_ref) / pd_ref) < 1e-3


def test_homogeneous_derivative_plateaus_at_half():
    tD = np.logspace(4, 9, 120)
    dp = bourdet_derivative(tD, homogeneous_pd(tD))
    late = dp[80:110]
    assert np.allclose(late, 0.5, atol=5e-3)


def test_warren_root_reduces_to_homogeneous_at_omega_one():
    tD = np.logspace(3, 8, 50)
    assert np.allclose(warren_root_pd(tD, omega=1.0, lam=1e-6), homogeneous_pd(tD), rtol=1e-8)


def test_warren_root_derivative_valley():
    tD = np.logspace(2, 10, 160)
    dp = bourdet_derivative(tD, warren_root_pd(tD, omega=0.02, lam=1e-7))
    # the dual-porosity signature: derivative dips well below the 0.5 radial plateau, then returns
    assert dp.min() < 0.3
    assert abs(dp[-10:].mean() - 0.5) < 0.05


def test_smaller_omega_gives_deeper_valley():
    tD = np.logspace(2, 10, 160)
    d_shallow = bourdet_derivative(tD, warren_root_pd(tD, omega=0.2, lam=1e-7)).min()
    d_deep = bourdet_derivative(tD, warren_root_pd(tD, omega=0.01, lam=1e-7)).min()
    assert d_deep < d_shallow


def test_ensemble_deterministic_and_shaped():
    a = generate_warren_root_ensemble(5, seed=42)
    b = generate_warren_root_ensemble(5, seed=42)
    assert np.array_equal(a["curves"], b["curves"])
    assert a["curves"].shape == (5, a["tD"].size)
    assert len(a["params"]) == 5
    c = generate_warren_root_ensemble(5, seed=43)
    assert not np.array_equal(a["curves"], c["curves"])
