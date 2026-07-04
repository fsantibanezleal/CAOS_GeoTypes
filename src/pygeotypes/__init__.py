"""pygeotypes — shape catalogues of physical response signals.

Build a catalogue of behaviour types (GeoTypes) from response signals whose *shape* reflects the
underlying physical system (pressure transients, pumping tests, thermal response tests, ...), and
assign new signals to it with statistical guarantees.

Pipeline (all pure numpy/scipy in the core, Pyodide-safe):

    preprocess  -> log resampling, Bourdet derivative, second derivative, normalization
    distance    -> Sakoe-Chiba banded DTW (numpy DP) + pairwise matrices (optional dtaidistance)
    cluster     -> PAM k-medoids on a precomputed distance matrix + silhouette / K-selection
    catalogue   -> the persistent GeoType catalogue artifact (JSON round-trip)
    assign      -> nearest-medoid + split-conformal assignment (p-values, prediction sets, OOD flag)
    attribute   -> RF + SHAP + permutation importance (optional extra: pygeotypes[attr])
    synthetic   -> Warren-Root dual-porosity / homogeneous radial generators (Stehfest inversion)

Display version follows the CAOS X.XX.XXX convention (see CHANGELOG.md).
"""

from pygeotypes.catalogue import Catalogue, build_catalogue
from pygeotypes.cluster import KMedoidsResult, pam_kmedoids, select_k, silhouette_from_distances
from pygeotypes.distance import dtw_banded, dtw_matrix
from pygeotypes.preprocess import (
    bourdet_derivative,
    log_resample,
    normalize,
    second_log_derivative,
)
from pygeotypes.assign import ConformalAssigner, nearest_medoid
from pygeotypes.synthetic import (
    homogeneous_pd,
    warren_root_pd,
    generate_warren_root_ensemble,
)

__version__ = "0.1.2"          # PEP 440; display form 0.01.002 (CHANGELOG)
__display_version__ = "0.01.000"

__all__ = [
    "Catalogue",
    "build_catalogue",
    "KMedoidsResult",
    "pam_kmedoids",
    "select_k",
    "silhouette_from_distances",
    "dtw_banded",
    "dtw_matrix",
    "bourdet_derivative",
    "log_resample",
    "normalize",
    "second_log_derivative",
    "ConformalAssigner",
    "nearest_medoid",
    "homogeneous_pd",
    "warren_root_pd",
    "generate_warren_root_ensemble",
    "__version__",
    "__display_version__",
]
