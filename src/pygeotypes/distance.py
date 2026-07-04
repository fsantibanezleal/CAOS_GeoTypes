"""Dynamic Time Warping distances with a Sakoe-Chiba band.

The core is a pure-numpy O(n·w) dynamic program (Pyodide-safe; fast enough for the live lane's
one-curve-vs-K-medoids use). Full pairwise matrices over large offline ensembles can delegate to
the C-accelerated `dtaidistance` package when installed (extra: `pygeotypes[fast]`).

Conventions: local cost = squared difference; the returned distance is the square root of the
accumulated cost along the optimal path (the same convention as dtaidistance / tslearn Euclidean
DTW), so `dtw(x, x) == 0` and units match the signal's.
"""

from __future__ import annotations

import numpy as np

__all__ = ["dtw_banded", "dtw_matrix", "distances_to_references"]


def dtw_banded(x: np.ndarray, y: np.ndarray, window: int | None = None) -> float:
    """DTW distance between two 1-D sequences with a Sakoe-Chiba band of half-width `window`.

    window=None means unconstrained. A band of ~10-20% of the sequence length is standard for
    diagnostic curves on a common log grid (Freites et al. 2023 used w=20 on n≈100 grids): it
    permits regime-timing shifts of up to that many log-grid cells while forbidding degenerate
    warpings that match early-time behaviour to late-time behaviour.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n, m = x.size, y.size
    if n == 0 or m == 0:
        raise ValueError("empty sequence")
    w = max(window if window is not None else max(n, m), abs(n - m))
    INF = np.inf
    prev = np.full(m + 1, INF)
    prev[0] = 0.0
    for i in range(1, n + 1):
        cur = np.full(m + 1, INF)
        j_lo = max(1, i - w)
        j_hi = min(m, i + w)
        xi = x[i - 1]
        for j in range(j_lo, j_hi + 1):
            c = (xi - y[j - 1]) ** 2
            best = prev[j]
            if prev[j - 1] < best:
                best = prev[j - 1]
            if cur[j - 1] < best:
                best = cur[j - 1]
            cur[j] = c + best
        prev = cur
    return float(np.sqrt(prev[m]))


def dtw_matrix(X: np.ndarray, window: int | None = None, backend: str = "auto") -> np.ndarray:
    """Symmetric pairwise DTW distance matrix over the rows of X (n_curves × n_points).

    backend: 'auto' (dtaidistance if importable, else numpy), 'numpy', or 'dtaidistance'.
    Both backends honour the same Sakoe-Chiba half-width and cost convention, so results are
    interchangeable (CI asserts parity).
    """
    X = np.ascontiguousarray(np.asarray(X, dtype=float))
    if X.ndim != 2:
        raise ValueError("X must be 2-D (n_curves, n_points)")
    if backend not in ("auto", "numpy", "dtaidistance"):
        raise ValueError(f"unknown backend: {backend!r}")
    use_dtai = backend == "dtaidistance"
    if backend == "auto":
        try:
            import dtaidistance  # noqa: F401
            use_dtai = True
        except ImportError:
            use_dtai = False
    if use_dtai:
        from dtaidistance import dtw as _dtw

        # dtaidistance's `window` is the full band width in its C code semantics; passing the
        # half-width+1 keeps parity with our DP (verified by tests/test_distance.py).
        D = _dtw.distance_matrix_fast(
            [row for row in X], window=None if window is None else window + 1
        )
        D = np.asarray(D, dtype=float)
        D[np.isinf(D)] = 0.0
        return np.maximum(D, D.T)
    n = X.shape[0]
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = dtw_banded(X[i], X[j], window=window)
    return D


def distances_to_references(x: np.ndarray, refs: np.ndarray, window: int | None = None) -> np.ndarray:
    """DTW distance from one curve to each reference curve (rows of `refs`).

    This is the live-lane primitive: classify-my-curve computes K distances (K = number of
    medoids), never a full matrix — cheap enough for the browser.
    """
    x = np.asarray(x, dtype=float)
    refs = np.asarray(refs, dtype=float)
    if refs.ndim == 1:
        refs = refs[None, :]
    return np.array([dtw_banded(x, r, window=window) for r in refs], dtype=float)
