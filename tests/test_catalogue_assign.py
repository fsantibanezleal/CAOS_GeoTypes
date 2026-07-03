import numpy as np

from geotypes.assign import ConformalAssigner, nearest_medoid
from geotypes.catalogue import Catalogue, build_catalogue
from geotypes.cluster import pam_kmedoids
from geotypes.distance import dtw_matrix
from geotypes.preprocess import prepare_curves
from geotypes.synthetic import warren_root_pd

WINDOW = 6
N_POINTS = 48


def _ensemble(n_per: int, seed: int):
    rng = np.random.default_rng(seed)
    tD = np.logspace(2, 10, 200)
    regimes = [
        {"omega": 0.35, "lam": 1e-4},
        {"omega": 0.03, "lam": 1e-6},
        {"omega": 0.01, "lam": 5e-9},
    ]
    t_list, p_list, truth = [], [], []
    for g, reg in enumerate(regimes):
        for _ in range(n_per):
            omega = reg["omega"] * float(np.exp(rng.normal(0, 0.10)))
            lam = reg["lam"] * float(np.exp(rng.normal(0, 0.15)))
            y = warren_root_pd(tD, omega=min(omega, 0.99), lam=lam)
            t_list.append(tD)
            p_list.append(y * np.exp(rng.normal(0, 0.01, size=y.shape)))
            truth.append(g)
    tg, X = prepare_curves(t_list, p_list, n_points=N_POINTS, derivative_order=1, norm="zscore")
    return tg, X, np.array(truth)


def _catalogue(seed=0):
    tg, X, truth = _ensemble(8, seed)
    D = dtw_matrix(X, window=WINDOW, backend="numpy")
    res = pam_kmedoids(D, k=3, n_init=5, seed=0)
    cat = build_catalogue(
        X, tg, D, k=3, dtw_window=WINDOW,
        preprocessing={"derivative_order": 1, "n_points": N_POINTS, "norm": "zscore"},
        provenance={"source": "unit-test"},
        result=res,
    )
    return cat, X, res.labels, truth


def test_catalogue_json_roundtrip(tmp_path):
    cat, _, _, _ = _catalogue()
    path = tmp_path / "cat.json"
    cat.to_json(path)
    back = Catalogue.from_json(path)
    assert back.k == cat.k
    assert np.allclose(back.medoid_curves, cat.medoid_curves)
    assert np.array_equal(back.labels, cat.labels)
    assert back.dtw_window == cat.dtw_window
    assert back.preprocessing == cat.preprocessing
    assert back.counts().sum() == cat.labels.size


def test_nearest_medoid_assigns_medoid_to_itself():
    cat, X, labels, _ = _catalogue()
    for gi, mi in enumerate(cat.medoid_indices):
        g, d = nearest_medoid(X[mi], cat)
        assert g == gi
        assert d[gi] == 0.0


def test_conformal_coverage_and_ood():
    cat, X_train, labels_train, _ = _catalogue(seed=0)
    # calibration + test from fresh seeds (exchangeable with training regimes).
    # NOTE: class-conditional conformal needs n_c >= 1/alpha - 1 calibration samples per class
    # for an empty set to be reachable (min p-value = 1/(n_c+1)); 12 per class with alpha=0.2.
    _, X_cal, truth_cal = _ensemble(12, seed=1)
    _, X_test, truth_test = _ensemble(8, seed=2)
    # map regime truth -> catalogue GeoType ids via medoid assignment of regime prototypes
    assigner = ConformalAssigner(catalogue=cat)
    # calibrate with catalogue-consistent labels (nearest medoid of each calibration curve's regime)
    cal_labels = np.array([nearest_medoid(x, cat)[0] for x in X_cal])
    assigner.fit(X_cal, cal_labels)

    alpha = 0.2
    covered = 0
    for x in X_test:
        res = assigner.predict(x, alpha=alpha)
        true_g = nearest_medoid(x, cat)[0]  # consistent ground truth under the same metric
        if true_g in res.prediction_set:
            covered += 1
        assert res.p_values.shape == (3,)
        assert 0 <= res.point_prediction < 3
    coverage = covered / X_test.shape[0]
    assert coverage >= 1 - alpha - 0.1  # finite-sample slack

    # a shape alien to the catalogue (high-frequency oscillation) must be flagged OOD
    weird = np.sin(np.linspace(0, 60 * np.pi, N_POINTS)) * 3
    res = assigner.predict(weird, alpha=0.2)
    assert res.out_of_catalogue


def test_conformal_persistence_roundtrip(tmp_path):
    cat, X, labels, _ = _catalogue()
    _, X_cal, _ = _ensemble(6, seed=3)
    cal_labels = np.array([nearest_medoid(x, cat)[0] for x in X_cal])
    a = ConformalAssigner(catalogue=cat).fit(X_cal, cal_labels)
    d = a.to_dict()
    b = ConformalAssigner.from_dict(d, catalogue=cat)
    x = X_cal[0]
    ra, rb = a.predict(x, alpha=0.15), b.predict(x, alpha=0.15)
    assert np.allclose(ra.p_values, rb.p_values)
    assert ra.prediction_set == rb.prediction_set
