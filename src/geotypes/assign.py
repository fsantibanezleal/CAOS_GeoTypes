"""Assignment of new curves to a GeoType catalogue — plain and conformal.

`nearest_medoid` is the paper's implicit rule (closest medoid in DTW). The *conformal* assigner is
the novel layer: split-conformal prediction with a class-conditional (Mondrian) calibration turns
the catalogue into a decision tool with finite-sample guarantees:

- for a new curve it returns a p-value per GeoType and the **prediction set**
  {g : p_g > alpha}; under exchangeability the set covers the true GeoType with
  probability ≥ 1 − alpha *per class* (Vovk et al. 2005; Angelopoulos & Bates 2023).
- an **empty set** is an honest out-of-catalogue flag: the curve's shape is not consistent with
  ANY behaviour class at the requested confidence — exactly the case a characterisation workflow
  must surface instead of silently forcing the nearest medoid.

Nonconformity score: DTW distance to the class medoid (pure numpy; live-lane safe).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from geotypes.catalogue import Catalogue
from geotypes.distance import distances_to_references

__all__ = ["nearest_medoid", "ConformalAssigner", "ConformalResult"]


def nearest_medoid(curve: np.ndarray, catalogue: Catalogue) -> tuple[int, np.ndarray]:
    """Nearest-medoid assignment. Returns (geotype, distances_to_each_medoid)."""
    d = distances_to_references(curve, catalogue.medoid_curves, window=catalogue.dtw_window)
    return int(np.argmin(d)), d


@dataclass
class ConformalResult:
    """Outcome of a conformal assignment for one curve."""

    point_prediction: int                 # nearest medoid (argmin distance)
    p_values: np.ndarray                  # (k,) conformal p-value per GeoType
    prediction_set: list[int]             # GeoTypes with p > alpha
    alpha: float
    distances: np.ndarray                 # (k,) DTW distance to each medoid
    out_of_catalogue: bool                # prediction set empty at this alpha

    def to_dict(self) -> dict:
        return {
            "point_prediction": int(self.point_prediction),
            "p_values": self.p_values.tolist(),
            "prediction_set": [int(g) for g in self.prediction_set],
            "alpha": float(self.alpha),
            "distances": self.distances.tolist(),
            "out_of_catalogue": bool(self.out_of_catalogue),
        }


@dataclass
class ConformalAssigner:
    """Split-conformal GeoType assignment with class-conditional calibration.

    Fit on *calibration* curves that were NOT used to build the catalogue (split-conformal). For
    each class g the calibration scores are the DTW distances of the class-g calibration curves to
    medoid g. The p-value of a new curve x for class g is the usual conformal rank:

        p_g(x) = ( #{ s in S_g : s >= d(x, medoid_g) } + 1 ) / ( |S_g| + 1 )

    Class-conditional calibration (a Mondrian taxonomy over labels) gives per-class validity, which
    matters because GeoType populations are imbalanced in real ensembles.
    """

    catalogue: Catalogue
    calibration_scores: dict[int, np.ndarray] = field(default_factory=dict)

    def fit(self, curves: np.ndarray, labels: np.ndarray) -> "ConformalAssigner":
        curves = np.asarray(curves, dtype=float)
        labels = np.asarray(labels, dtype=int)
        if curves.ndim != 2 or curves.shape[0] != labels.shape[0]:
            raise ValueError("curves must be (n, n_points) with one label per row")
        n_pts = self.catalogue.medoid_curves.shape[1]
        if curves.shape[1] != n_pts:
            raise ValueError(f"curves must be preprocessed onto the catalogue grid ({n_pts} points)")
        self.calibration_scores = {}
        for g in range(self.catalogue.k):
            rows = curves[labels == g]
            if rows.shape[0] == 0:
                self.calibration_scores[g] = np.array([], dtype=float)
                continue
            med = self.catalogue.medoid_curves[g]
            self.calibration_scores[g] = np.sort(
                distances_to_references_batch(rows, med, self.catalogue.dtw_window)
            )
        return self

    def predict(self, curve: np.ndarray, alpha: float = 0.1) -> ConformalResult:
        if not self.calibration_scores:
            raise RuntimeError("ConformalAssigner is not fitted (call .fit first)")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        point, d = nearest_medoid(curve, self.catalogue)
        k = self.catalogue.k
        p = np.zeros(k)
        for g in range(k):
            s = self.calibration_scores.get(g, np.array([]))
            if s.size == 0:
                p[g] = 0.0  # no calibration evidence for this class -> never in the set
                continue
            # rank among calibration scores (>= convention)
            n_ge = s.size - np.searchsorted(s, d[g], side="left")
            p[g] = (n_ge + 1) / (s.size + 1)
        pred_set = [g for g in range(k) if p[g] > alpha]
        return ConformalResult(
            point_prediction=point,
            p_values=p,
            prediction_set=pred_set,
            alpha=alpha,
            distances=d,
            out_of_catalogue=len(pred_set) == 0,
        )

    # ---------- persistence (bake calibration into the web artifact) ----------
    def to_dict(self) -> dict:
        return {
            "calibration_scores": {str(g): s.tolist() for g, s in self.calibration_scores.items()},
        }

    def to_json(self, path: str | Path | None = None) -> str:
        s = json.dumps(self.to_dict())
        if path is not None:
            Path(path).write_text(s, encoding="utf-8")
        return s

    @classmethod
    def from_dict(cls, d: dict, catalogue: Catalogue) -> "ConformalAssigner":
        obj = cls(catalogue=catalogue)
        obj.calibration_scores = {
            int(g): np.asarray(v, dtype=float) for g, v in d["calibration_scores"].items()
        }
        return obj


def distances_to_references_batch(rows: np.ndarray, ref: np.ndarray, window: int | None) -> np.ndarray:
    """DTW distance of each row to a single reference curve."""
    return np.array([distances_to_references(r, ref, window=window)[0] for r in rows], dtype=float)
