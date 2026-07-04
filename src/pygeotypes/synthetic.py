"""Analytical pressure-transient generators (Warren-Root dual-porosity, homogeneous radial).

These solutions serve three roles: (1) the *classical* rung of the model ladder, (2) fast synthetic
ensembles for tests and demos, (3) the live browser lane (pure numpy + scipy.special — Pyodide-safe;
no compiled Laplace-inversion dependency).

Physics (dimensionless, line-source producing well, infinite-acting reservoir):

- Homogeneous radial flow, Laplace space:      pwD(s) = K0(sqrt(s)) / s
- Warren & Root (1963) dual porosity (pseudo-steady-state interporosity flow), Laplace space:
      pwD(s) = K0( sqrt( s·f(s) ) ) / s,   f(s) = ( ω(1−ω)s + λ ) / ( (1−ω)s + λ )
  ω = storativity ratio (fracture / total), λ = interporosity flow coefficient.
  Late time the derivative returns to the radial 0.5 plateau; ω,λ shape the classic valley.
- Optional wellbore storage CD and skin S (Agarwal et al. 1970):
      pwD_wbs(s) = ( s·pwD(s) + S ) / ( s·( 1 + CD·s·( s·pwD(s) + S ) ) )

Numerical inversion: the Gaver-Stehfest algorithm (Stehfest 1970, CACM 13(1):47-49) with even N
(default 12) — the standard choice in well testing; accurate for these smooth monotone solutions.
"""

from __future__ import annotations

from math import factorial

import numpy as np
from scipy.special import k0

__all__ = [
    "stehfest_weights",
    "stehfest_invert",
    "homogeneous_pd",
    "warren_root_pd",
    "generate_warren_root_ensemble",
]


def stehfest_weights(N: int = 12) -> np.ndarray:
    """Gaver-Stehfest weights V_i, i=1..N (N must be even). Cached per N by numpy immutability."""
    if N % 2 != 0 or N < 2:
        raise ValueError("Stehfest N must be a positive even integer")
    V = np.zeros(N)
    half = N // 2
    for i in range(1, N + 1):
        s = 0.0
        for k in range((i + 1) // 2, min(i, half) + 1):
            s += (
                k**half
                * factorial(2 * k)
                / (
                    factorial(half - k)
                    * factorial(k)
                    * factorial(k - 1)
                    * factorial(i - k)
                    * factorial(2 * k - i)
                )
            )
        V[i - 1] = (-1) ** (i + half) * s
    return V


def stehfest_invert(F, t: np.ndarray, N: int = 12) -> np.ndarray:
    """Invert a Laplace-space function F(s) at times t (vectorized over t)."""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    if np.any(t <= 0):
        raise ValueError("t must be strictly positive")
    V = stehfest_weights(N)
    ln2_t = np.log(2.0) / t                       # (nt,)
    out = np.zeros_like(t)
    for i in range(1, N + 1):
        out += V[i - 1] * F(i * ln2_t)
    return out * ln2_t


def _with_wellbore(F_pd, CD: float, S: float):
    """Wrap a Laplace-space pwD(s) with wellbore storage + skin (Agarwal et al. 1970)."""

    def F(s):
        spd = s * F_pd(s) + S
        return spd / (s * (1.0 + CD * s * spd))

    return F


def homogeneous_pd(tD: np.ndarray, CD: float = 0.0, S: float = 0.0, N: int = 12) -> np.ndarray:
    """Dimensionless wellbore pressure for homogeneous radial flow (line source).

    Late time pwD ≈ 0.5(ln tD + 0.80907); its Bourdet derivative plateaus at 0.5.
    """

    def F_pd(s):
        return k0(np.sqrt(s)) / s

    F = _with_wellbore(F_pd, CD, S) if (CD > 0 or S != 0) else F_pd
    return stehfest_invert(F, tD, N=N)


def warren_root_pd(
    tD: np.ndarray,
    omega: float = 0.05,
    lam: float = 1e-6,
    CD: float = 0.0,
    S: float = 0.0,
    N: int = 12,
) -> np.ndarray:
    """Warren-Root dual-porosity dimensionless wellbore pressure (pseudo-steady interporosity).

    omega in (0, 1]: storativity ratio; lam > 0: interporosity flow coefficient. omega=1 reduces
    to the homogeneous solution. The Bourdet derivative shows the characteristic valley between
    two 0.5 plateaus; valley depth grows as omega shrinks, valley time scales with 1/lam.
    """
    if not 0.0 < omega <= 1.0:
        raise ValueError("omega must be in (0, 1]")
    if lam <= 0:
        raise ValueError("lam must be positive")

    def f(s):
        return (omega * (1.0 - omega) * s + lam) / ((1.0 - omega) * s + lam)

    def F_pd(s):
        return k0(np.sqrt(s * f(s))) / s

    F = _with_wellbore(F_pd, CD, S) if (CD > 0 or S != 0) else F_pd
    return stehfest_invert(F, tD, N=N)


def generate_warren_root_ensemble(
    n_curves: int,
    tD: np.ndarray | None = None,
    omega_range: tuple[float, float] = (0.01, 0.5),
    lam_range: tuple[float, float] = (1e-8, 1e-4),
    skin_range: tuple[float, float] = (0.0, 0.0),
    noise_sd: float = 0.0,
    seed: int | None = 0,
) -> dict:
    """Seeded synthetic ensemble of dual-porosity responses (log-uniform parameter sampling).

    Returns {'tD', 'curves' (n, nt), 'params' (list of dicts)}. With noise_sd > 0, multiplicative
    log-normal noise is applied (measurement-like), keeping curves positive.
    """
    rng = np.random.default_rng(seed)
    if tD is None:
        tD = np.logspace(2, 9, 128)
    tD = np.asarray(tD, dtype=float)
    curves = np.empty((n_curves, tD.size))
    params: list[dict] = []
    for i in range(n_curves):
        omega = float(10 ** rng.uniform(np.log10(omega_range[0]), np.log10(omega_range[1])))
        lam = float(10 ** rng.uniform(np.log10(lam_range[0]), np.log10(lam_range[1])))
        S = float(rng.uniform(*skin_range))
        y = warren_root_pd(tD, omega=omega, lam=lam, S=S)
        if noise_sd > 0:
            y = y * np.exp(rng.normal(0.0, noise_sd, size=y.shape))
        curves[i] = y
        params.append({"omega": omega, "lam": lam, "skin": S})
    return {"tD": tD, "curves": curves, "params": params}
