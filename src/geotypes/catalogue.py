"""The GeoType catalogue: the persistent artifact a clustering run produces.

A catalogue is everything a *consumer* (web app, conformal assigner, report) needs, decoupled from
the ensemble that produced it: the common time grid, the K medoid curves, per-curve labels and
distances, preprocessing metadata (so a new curve can be preprocessed identically), and the run's
provenance (seed, DTW band, silhouette). JSON round-trip is exact for the float64 arrays.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

from geotypes.cluster import KMedoidsResult, pam_kmedoids

__all__ = ["Catalogue", "build_catalogue"]

_SCHEMA_VERSION = 1


@dataclass
class Catalogue:
    """A catalogue of GeoTypes (behaviour classes) for one preprocessed ensemble."""

    k: int
    t_grid: np.ndarray                    # (n_points,) common log-uniform time grid
    medoid_curves: np.ndarray             # (k, n_points) the K representative curves
    medoid_indices: np.ndarray            # (k,) row indices into the source ensemble
    labels: np.ndarray                    # (n,) GeoType id per source curve
    per_curve_distance: np.ndarray        # (n,) DTW distance to own medoid
    silhouette: float
    dtw_window: int | None                # Sakoe-Chiba half-width used everywhere
    preprocessing: dict = field(default_factory=dict)   # derivative_order, L, norm, n_points...
    provenance: dict = field(default_factory=dict)      # seed, n_init, source dataset id, dates
    names: list[str] | None = None        # optional human names per GeoType

    # ---------- persistence ----------
    def to_dict(self) -> dict:
        d = asdict(self)
        d["schema_version"] = _SCHEMA_VERSION
        for key in ("t_grid", "medoid_curves", "medoid_indices", "labels", "per_curve_distance"):
            d[key] = np.asarray(getattr(self, key)).tolist()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Catalogue":
        d = dict(d)
        d.pop("schema_version", None)
        d["t_grid"] = np.asarray(d["t_grid"], dtype=float)
        d["medoid_curves"] = np.asarray(d["medoid_curves"], dtype=float)
        d["medoid_indices"] = np.asarray(d["medoid_indices"], dtype=int)
        d["labels"] = np.asarray(d["labels"], dtype=int)
        d["per_curve_distance"] = np.asarray(d["per_curve_distance"], dtype=float)
        return cls(**d)

    def to_json(self, path: str | Path | None = None, indent: int = 0) -> str:
        s = json.dumps(self.to_dict(), indent=indent or None)
        if path is not None:
            Path(path).write_text(s, encoding="utf-8")
        return s

    @classmethod
    def from_json(cls, source: str | Path) -> "Catalogue":
        p = Path(source)
        text = p.read_text(encoding="utf-8") if p.exists() else str(source)
        return cls.from_dict(json.loads(text))

    # ---------- convenience ----------
    def counts(self) -> np.ndarray:
        """Curves per GeoType (k,)."""
        return np.bincount(self.labels, minlength=self.k)

    def name_of(self, geotype: int) -> str:
        if self.names and 0 <= geotype < len(self.names):
            return self.names[geotype]
        return f"GT{geotype}"


def build_catalogue(
    X: np.ndarray,
    t_grid: np.ndarray,
    D: np.ndarray,
    k: int,
    dtw_window: int | None = None,
    preprocessing: dict | None = None,
    provenance: dict | None = None,
    n_init: int = 10,
    seed: int | None = 0,
    result: KMedoidsResult | None = None,
) -> Catalogue:
    """Cluster a preprocessed ensemble and package the result as a Catalogue.

    X: (n, n_points) preprocessed curves; D: the pairwise DTW matrix over X (same window as
    `dtw_window`). Pass `result` to reuse an existing PAM run instead of re-clustering.
    """
    X = np.asarray(X, dtype=float)
    res = result if result is not None else pam_kmedoids(D, k, n_init=n_init, seed=seed)
    return Catalogue(
        k=res.k,
        t_grid=np.asarray(t_grid, dtype=float),
        medoid_curves=X[res.medoid_indices],
        medoid_indices=res.medoid_indices,
        labels=res.labels,
        per_curve_distance=res.per_curve_distance,
        silhouette=res.silhouette,
        dtw_window=dtw_window,
        preprocessing=preprocessing or {},
        provenance={**(provenance or {}), "seed": seed, "n_init": n_init},
    )
