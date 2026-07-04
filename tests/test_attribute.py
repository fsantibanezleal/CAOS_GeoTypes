import numpy as np
import pytest

from pygeotypes.attribute import prune_correlated

sklearn = pytest.importorskip("sklearn")
shap = pytest.importorskip("shap")

from pygeotypes.attribute import attribute_geotypes  # noqa: E402


def test_prune_correlated_drops_duplicates():
    rng = np.random.default_rng(0)
    a = rng.normal(size=200)
    b = a * 3.0 + 1e-9 * rng.normal(size=200)   # monotone copy of a
    c = rng.normal(size=200)
    X = np.column_stack([a, b, c])
    Xp, kept, dropped = prune_correlated(X, ["a", "b", "c"], threshold=0.95)
    assert kept == ["a", "c"]
    assert Xp.shape == (200, 2)
    assert dropped[0][0] == "b" and dropped[0][1] == "a"


def test_attribution_finds_informative_feature():
    rng = np.random.default_rng(1)
    n = 600
    informative = rng.normal(size=n)
    noise1 = rng.normal(size=n)
    noise2 = rng.normal(size=n)
    labels = (informative > 0).astype(int)
    X = np.column_stack([informative, noise1, noise2])
    out = attribute_geotypes(X, labels, ["inf", "n1", "n2"], accuracy_gate=0.8, seed=0)
    assert out["gate"]["passed"]
    # global SHAP: informative feature dominates
    total = {f: 0.0 for f in out["kept_features"]}
    for cls_imp in out["shap_mean_abs"].values():
        for f, v in cls_imp.items():
            total[f] += v
    assert max(total, key=total.get) == "inf"
    pi = out["permutation_importance"]
    assert max(pi, key=lambda f: pi[f]["mean"]) == "inf"


def test_attribution_gate_blocks_noise_labels():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(300, 4))
    labels = rng.integers(0, 2, size=300)  # labels independent of X
    out = attribute_geotypes(X, labels, ["f0", "f1", "f2", "f3"], accuracy_gate=0.75, seed=0)
    assert not out["gate"]["passed"]
    assert out["shap_mean_abs"] is None
    assert out["permutation_importance"] is None
