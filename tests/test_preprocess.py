import numpy as np
import pytest

from pygeotypes.preprocess import (
    bourdet_derivative,
    log_resample,
    normalize,
    prepare_curves,
    second_log_derivative,
)


def test_bourdet_derivative_of_log_line_is_constant():
    t = np.logspace(0, 5, 200)
    m, c = 1.7, 3.0
    dp = bourdet_derivative(t, m * np.log(t) + c)
    assert np.allclose(dp, m, atol=1e-6)


def test_second_derivative_removes_offset():
    t = np.logspace(0, 6, 150)
    p = np.log(t) ** 1.5
    d2a = second_log_derivative(t, p)
    d2b = second_log_derivative(t, p + 100.0)  # vertical offset
    assert np.allclose(d2a, d2b, atol=1e-9)


def test_log_resample_grid_and_values():
    t = np.logspace(0, 3, 500)
    y = np.log10(t)  # linear in log10 t -> exact under our interpolation
    tg, yg = log_resample(t, y, n_points=64)
    assert tg.size == 64
    assert np.allclose(np.diff(np.log10(tg)), np.diff(np.log10(tg))[0])
    assert np.allclose(yg, np.log10(tg), atol=1e-12)


def test_log_resample_rejects_bad_input():
    with pytest.raises(ValueError):
        log_resample(np.array([0.0, 1.0, 2.0]), np.zeros(3))
    with pytest.raises(ValueError):
        log_resample(np.array([2.0, 1.0, 3.0]), np.zeros(3))


def test_normalize_zscore_and_max_and_flat():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    z = normalize(y, "zscore")
    assert abs(z.mean()) < 1e-12 and abs(z.std() - 1) < 1e-12
    m = normalize(y, "max")
    assert m.max() == 1.0
    flat = normalize(np.ones(5), "zscore")
    assert np.allclose(flat, 0.0)
    with pytest.raises(ValueError):
        normalize(y, "nope")


def test_prepare_curves_common_grid_and_shape():
    t1 = np.logspace(0, 5, 300)
    t2 = np.logspace(0.5, 5.5, 280)
    p1 = np.log(t1)
    p2 = 2 * np.log(t2)
    tg, X = prepare_curves([t1, t2], [p1, p2], n_points=48, derivative_order=1, norm="zscore")
    assert X.shape == (2, 48)
    # common grid = intersection of ranges
    assert tg[0] >= t2[0] - 1e-9 and tg[-1] <= t1[-1] + 1e-9
