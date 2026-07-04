"""PAM k-medoids clustering on a precomputed distance matrix, plus K-selection diagnostics.

Why in-house: as of mid-2026 there is no maintained, permissively-licensed, Pyodide-friendly PAM —
scikit-learn-extra is unmaintained (last release 2023), the fast Rust `kmedoids` package is GPL-3,
and aeon/tslearn drag numba/native deps. PAM on a precomputed matrix is ~100 lines of numpy; for
ensemble sizes up to a few thousand curves the O(k(n-k)²) SWAP step is fine offline.

Algorithm: classic PAM (Kaufman & Rousseeuw 1990) — greedy BUILD initialization, then SWAP until
no single medoid↔non-medoid exchange lowers total cost. Multi-restart keeps the best of `n_init`
runs (BUILD is deterministic; restarts randomize via sampled initial medoid sets), all seeded.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["KMedoidsResult", "pam_kmedoids", "silhouette_from_distances", "select_k"]


@dataclass
class KMedoidsResult:
    """Result of a PAM run: medoid row-indices, per-curve labels (0..k-1) and diagnostics."""

    k: int
    medoid_indices: np.ndarray          # (k,) indices into the ensemble
    labels: np.ndarray                  # (n,) cluster id per curve
    cost: float                         # total sum of distances to assigned medoids
    silhouette: float                   # mean silhouette over all curves
    n_iter: int                         # SWAP iterations of the winning restart
    seed: int | None = None
    per_curve_distance: np.ndarray = field(default=None, repr=False)  # (n,) distance to own medoid

    def to_dict(self) -> dict:
        return {
            "k": int(self.k),
            "medoid_indices": self.medoid_indices.tolist(),
            "labels": self.labels.tolist(),
            "cost": float(self.cost),
            "silhouette": float(self.silhouette),
            "n_iter": int(self.n_iter),
            "seed": self.seed,
        }


def _assign(D: np.ndarray, medoids: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    sub = D[:, medoids]                       # (n, k)
    labels = np.argmin(sub, axis=1)
    dist = sub[np.arange(D.shape[0]), labels]
    return labels, dist, float(dist.sum())


def _build(D: np.ndarray, k: int) -> np.ndarray:
    """Greedy BUILD: first medoid minimizes total distance; each next maximizes cost reduction."""
    n = D.shape[0]
    medoids = [int(np.argmin(D.sum(axis=0)))]
    best_dist = D[:, medoids[0]].copy()
    while len(medoids) < k:
        gains = np.empty(n)
        for c in range(n):
            if c in medoids:
                gains[c] = -np.inf
                continue
            gains[c] = np.maximum(best_dist - D[:, c], 0.0).sum()
        nxt = int(np.argmax(gains))
        medoids.append(nxt)
        best_dist = np.minimum(best_dist, D[:, nxt])
    return np.array(sorted(medoids), dtype=int)


def _swap(D: np.ndarray, medoids: np.ndarray, max_iter: int) -> tuple[np.ndarray, int]:
    n = D.shape[0]
    medoids = medoids.copy()
    _, _, cost = _assign(D, medoids)
    it = 0
    improved = True
    while improved and it < max_iter:
        improved = False
        it += 1
        non_medoids = np.setdiff1d(np.arange(n), medoids, assume_unique=False)
        for mi in range(len(medoids)):
            trial = medoids.copy()
            for h in non_medoids:
                trial[mi] = h
                _, _, c = _assign(D, trial)
                if c + 1e-12 < cost:
                    cost = c
                    medoids = trial.copy()
                    improved = True
            # refresh in case medoids changed
            trial = medoids.copy()
    return np.array(sorted(medoids), dtype=int), it


def pam_kmedoids(
    D: np.ndarray,
    k: int,
    n_init: int = 10,
    max_iter: int = 100,
    seed: int | None = 0,
) -> KMedoidsResult:
    """PAM k-medoids on a precomputed symmetric distance matrix D (n×n).

    Restart 0 uses the deterministic greedy BUILD; restarts 1..n_init-1 start from random medoid
    sets drawn with the seeded generator. The lowest-cost run wins. Fully deterministic for a
    given (D, k, n_init, seed).
    """
    D = np.asarray(D, dtype=float)
    n = D.shape[0]
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("D must be a square distance matrix")
    if not (1 <= k <= n):
        raise ValueError(f"k must be in [1, {n}]")
    if np.any(D < -1e-12):
        raise ValueError("distances must be non-negative")
    rng = np.random.default_rng(seed)
    best: tuple[float, np.ndarray, int] | None = None
    for r in range(max(1, n_init)):
        init = _build(D, k) if r == 0 else np.sort(rng.choice(n, size=k, replace=False))
        med, it = _swap(D, np.asarray(init, dtype=int), max_iter)
        _, _, cost = _assign(D, med)
        if best is None or cost < best[0]:
            best = (cost, med, it)
    cost, medoids, n_iter = best
    labels, dist, _ = _assign(D, medoids)
    sil = silhouette_from_distances(D, labels) if k > 1 else 0.0
    return KMedoidsResult(
        k=k,
        medoid_indices=medoids,
        labels=labels,
        cost=cost,
        silhouette=sil,
        n_iter=n_iter,
        seed=seed,
        per_curve_distance=dist,
    )


def silhouette_from_distances(D: np.ndarray, labels: np.ndarray) -> float:
    """Mean silhouette coefficient computed directly from a distance matrix.

    s_i = (b_i − a_i)/max(a_i, b_i) with a_i the mean intra-cluster distance and b_i the smallest
    mean distance to another cluster. Singleton clusters contribute s_i = 0 (sklearn convention).
    """
    D = np.asarray(D, dtype=float)
    labels = np.asarray(labels)
    n = D.shape[0]
    uniq = np.unique(labels)
    if uniq.size < 2:
        return 0.0
    s = np.zeros(n)
    for i in range(n):
        same = labels == labels[i]
        n_same = same.sum()
        if n_same <= 1:
            s[i] = 0.0
            continue
        a = D[i, same].sum() / (n_same - 1)
        b = np.inf
        for c in uniq:
            if c == labels[i]:
                continue
            mask = labels == c
            b = min(b, D[i, mask].mean())
        s[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
    return float(s.mean())


def select_k(
    D: np.ndarray,
    k_range: range | list[int] = range(2, 9),
    n_init: int = 10,
    seed: int | None = 0,
) -> dict:
    """Run PAM across a K range and return per-K diagnostics for elbow + silhouette selection.

    Returns {'k': [...], 'cost': [...], 'silhouette': [...], 'best_k': int} where best_k maximizes
    the silhouette (the paper's criterion; the cost column supports a visual elbow check).
    """
    ks, costs, sils = [], [], []
    for k in k_range:
        res = pam_kmedoids(D, k, n_init=n_init, seed=seed)
        ks.append(int(k))
        costs.append(res.cost)
        sils.append(res.silhouette)
    best_k = int(ks[int(np.argmax(sils))])
    return {"k": ks, "cost": costs, "silhouette": sils, "best_k": best_k}
