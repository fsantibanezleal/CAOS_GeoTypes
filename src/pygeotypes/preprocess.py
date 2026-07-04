"""Curve preprocessing for shape-based analysis of response signals.

The shape carriers in well testing are the log-log diagnostic curves: the pressure change Δp(t) and
its Bourdet derivative p' = dΔp/d ln t. Clustering works best on the *second* logarithmic
derivative p'' = dp'/d ln t, which removes the vertical offset that DTW cannot compensate
(Freites, Corbett & Geiger 2023, Transp. Porous Media, DOI 10.1007/s11242-023-01929-1).

All functions are pure numpy and operate on 1-D arrays; ensembles are handled row-wise.
"""

from __future__ import annotations

import numpy as np

__all__ = ["log_resample", "bourdet_derivative", "second_log_derivative", "normalize", "prepare_curves"]


def log_resample(
    t: np.ndarray,
    y: np.ndarray,
    n_points: int = 128,
    t_min: float | None = None,
    t_max: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample a signal onto a uniform grid in log10(t).

    DTW compares samples index-by-index along the warping path, so every curve of an ensemble must
    live on a common, log-uniform time grid before distances are computed.

    Parameters
    ----------
    t, y : arrays of equal length; t strictly positive and increasing.
    n_points : size of the target grid.
    t_min, t_max : grid limits (default: the data limits).

    Returns
    -------
    (t_grid, y_grid) : the log-uniform grid and the interpolated signal (linear in log10 t).
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if t.ndim != 1 or t.shape != y.shape:
        raise ValueError("t and y must be 1-D arrays of equal length")
    if np.any(t <= 0):
        raise ValueError("t must be strictly positive for a log grid")
    if np.any(np.diff(t) <= 0):
        raise ValueError("t must be strictly increasing")
    lo = np.log10(t_min if t_min is not None else t[0])
    hi = np.log10(t_max if t_max is not None else t[-1])
    if not hi > lo:
        raise ValueError("t_max must exceed t_min")
    lg = np.linspace(lo, hi, n_points)
    yg = np.interp(lg, np.log10(t), y)
    return 10.0**lg, yg


def bourdet_derivative(t: np.ndarray, p: np.ndarray, L: float = 0.2) -> np.ndarray:
    """Bourdet logarithmic derivative dp/d ln t with a smoothing window of L log-cycles.

    For each sample i the derivative is the weighted average of the left and right secant slopes,
    taken at the first neighbours at least L/2 log-cycles away (Bourdet, Ayoub & Pirard 1989,
    SPE Formation Evaluation 4(2):293-302, DOI 10.2118/12777-PA):

        p'_i = (Δp_L/ΔX_L · ΔX_R + Δp_R/ΔX_R · ΔX_L) / (ΔX_L + ΔX_R),   X = ln t

    End points fall back to the one-sided slope. L≈0.1–0.3 is standard practice; L=0.2 default.
    """
    t = np.asarray(t, dtype=float)
    p = np.asarray(p, dtype=float)
    if np.any(t <= 0):
        raise ValueError("t must be strictly positive")
    x = np.log(t)
    n = x.size
    half = (L / 2.0) * np.log(10.0)  # L is in log10 cycles; X is natural log
    dp = np.empty(n, dtype=float)
    for i in range(n):
        # left neighbour at least `half` away (else the farthest available = index 0)
        jl = i - 1
        while jl > 0 and (x[i] - x[jl]) < half:
            jl -= 1
        # right neighbour at least `half` away
        jr = i + 1
        while jr < n - 1 and (x[jr] - x[i]) < half:
            jr += 1
        if i == 0:
            dp[i] = (p[jr] - p[i]) / (x[jr] - x[i])
        elif i == n - 1:
            dp[i] = (p[i] - p[jl]) / (x[i] - x[jl])
        else:
            dxl = x[i] - x[jl]
            dxr = x[jr] - x[i]
            sl = (p[i] - p[jl]) / dxl
            sr = (p[jr] - p[i]) / dxr
            dp[i] = (sl * dxr + sr * dxl) / (dxl + dxr)
    return dp


def second_log_derivative(t: np.ndarray, p: np.ndarray, L: float = 0.2) -> np.ndarray:
    """Second logarithmic derivative p'' = d(dp/d ln t)/d ln t (two Bourdet passes).

    p'' is offset-free: two responses with the same flow-regime sequence but different absolute
    pressure levels produce the same p'' shape, which is exactly what shape clustering needs.
    """
    return bourdet_derivative(t, bourdet_derivative(t, p, L=L), L=L)


def normalize(y: np.ndarray, method: str = "zscore") -> np.ndarray:
    """Normalize a curve (or each row of a 2-D ensemble) for shape comparison.

    method: 'zscore' (zero mean, unit variance; degenerate flat curves map to zeros),
    'max' (divide by max |y|), or 'none'.
    """
    y = np.asarray(y, dtype=float)
    if method == "none":
        return y.copy()
    if y.ndim == 1:
        y2 = y[None, :]
    else:
        y2 = y
    if method == "zscore":
        mu = y2.mean(axis=1, keepdims=True)
        sd = y2.std(axis=1, keepdims=True)
        out = np.where(sd > 0, (y2 - mu) / np.where(sd == 0, 1.0, sd), 0.0)
    elif method == "max":
        mx = np.abs(y2).max(axis=1, keepdims=True)
        out = y2 / np.where(mx == 0, 1.0, mx)
    else:
        raise ValueError(f"unknown normalization method: {method!r}")
    return out[0] if y.ndim == 1 else out


def prepare_curves(
    t_list: list[np.ndarray],
    p_list: list[np.ndarray],
    n_points: int = 128,
    derivative_order: int = 2,
    L: float = 0.2,
    norm: str = "zscore",
    t_min: float | None = None,
    t_max: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Full preprocessing of an ensemble: resample → Bourdet derivative(s) → normalize.

    derivative_order: 0 = pressure itself, 1 = Bourdet derivative, 2 = second derivative (default,
    per Freites et al. 2023). Returns (t_grid, X) with X shaped (n_curves, n_points).

    The common grid is the *intersection* of all curves' time ranges unless t_min/t_max are given,
    so no curve is extrapolated.
    """
    if len(t_list) != len(p_list) or len(t_list) == 0:
        raise ValueError("t_list and p_list must be non-empty and of equal length")
    lo = t_min if t_min is not None else max(float(np.min(t)) for t in t_list)
    hi = t_max if t_max is not None else min(float(np.max(t)) for t in t_list)
    if not hi > lo:
        raise ValueError("curves have no common time range")
    rows = []
    t_grid = None
    for t, p in zip(t_list, p_list):
        tg, yg = log_resample(t, p, n_points=n_points, t_min=lo, t_max=hi)
        t_grid = tg
        if derivative_order == 1:
            yg = bourdet_derivative(tg, yg, L=L)
        elif derivative_order == 2:
            yg = second_log_derivative(tg, yg, L=L)
        elif derivative_order != 0:
            raise ValueError("derivative_order must be 0, 1 or 2")
        rows.append(yg)
    X = normalize(np.vstack(rows), method=norm)
    return t_grid, X
