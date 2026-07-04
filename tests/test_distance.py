import numpy as np
import pytest

from pygeotypes.distance import distances_to_references, dtw_banded, dtw_matrix


def test_identity_and_symmetry():
    rng = np.random.default_rng(0)
    x = rng.normal(size=50)
    y = rng.normal(size=50)
    assert dtw_banded(x, x) == 0.0
    assert dtw_banded(x, y) == pytest.approx(dtw_banded(y, x))
    assert dtw_banded(x, y) > 0


def test_dtw_leq_euclidean():
    rng = np.random.default_rng(1)
    x = rng.normal(size=40)
    y = rng.normal(size=40)
    assert dtw_banded(x, y) <= np.linalg.norm(x - y) + 1e-12


def test_band_absorbs_small_shift():
    t = np.linspace(0, 6 * np.pi, 80)
    x = np.sin(t)
    y = np.roll(x, 4)  # small phase shift
    d_banded = dtw_banded(x, y, window=8)
    d_locked = dtw_banded(x, y, window=0)  # diagonal-locked == Euclidean path
    assert d_banded < d_locked


def test_tighter_band_never_decreases_distance():
    rng = np.random.default_rng(2)
    x = rng.normal(size=60).cumsum()
    y = rng.normal(size=60).cumsum()
    d_wide = dtw_banded(x, y, window=30)
    d_mid = dtw_banded(x, y, window=10)
    d_tight = dtw_banded(x, y, window=2)
    assert d_wide <= d_mid + 1e-12 <= d_tight + 2e-12


def test_matrix_numpy_backend_consistent_with_scalar():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(5, 30))
    D = dtw_matrix(X, window=5, backend="numpy")
    assert D.shape == (5, 5)
    assert np.allclose(D, D.T)
    assert np.allclose(np.diag(D), 0)
    assert D[1, 2] == pytest.approx(dtw_banded(X[1], X[2], window=5))


def test_matrix_dtaidistance_parity_if_available():
    pytest.importorskip("dtaidistance")
    rng = np.random.default_rng(4)
    X = rng.normal(size=(6, 40))
    Dn = dtw_matrix(X, window=6, backend="numpy")
    Dd = dtw_matrix(X, window=6, backend="dtaidistance")
    assert np.allclose(Dn, Dd, rtol=1e-9, atol=1e-9)


def test_distances_to_references():
    rng = np.random.default_rng(5)
    refs = rng.normal(size=(3, 25))
    d = distances_to_references(refs[1], refs, window=5)
    assert d.shape == (3,)
    assert d[1] == 0.0
