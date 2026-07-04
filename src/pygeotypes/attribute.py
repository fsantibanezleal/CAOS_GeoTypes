"""Attribution: which descriptors control which GeoType (offline lane; extra `pygeotypes[attr]`).

Standard recipe (the paper's, hardened against its known pitfalls):

1. **Correlation pruning** first — fracture-network descriptors are correlated by construction
   (e.g. P21/P32/intensity), and both impurity importances and SHAP split credit arbitrarily
   between correlated features. Pruning to a representative subset makes attributions readable.
2. **Random-Forest classifier** on the GeoType labels, with a held-out **accuracy gate**: if the
   forest cannot predict the labels from the descriptors, its attributions are noise and the run
   is flagged instead of reported.
3. **TreeSHAP** global importances (mean |SHAP| per class) cross-checked against **permutation
   importance** — agreement between the two is reported; disagreement is a red flag, not a result.

Imports of sklearn/shap are deferred so the pure-numpy core stays importable without the extra.
"""

from __future__ import annotations

import numpy as np

__all__ = ["prune_correlated", "attribute_geotypes"]


def prune_correlated(
    X: np.ndarray, names: list[str], threshold: float = 0.9
) -> tuple[np.ndarray, list[str], list[tuple[str, str, float]]]:
    """Greedy Spearman-correlation pruning: keep the first feature of each correlated group.

    Returns (X_pruned, kept_names, dropped) where dropped lists (dropped_name, kept_name, rho).
    Spearman (rank) correlation is used because descriptor relations are monotone, not linear.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != len(names):
        raise ValueError("X must be (n_samples, n_features) matching names")
    # rank-transform columns, then Pearson on ranks == Spearman
    ranks = np.argsort(np.argsort(X, axis=0), axis=0).astype(float)
    R = np.corrcoef(ranks, rowvar=False)
    keep: list[int] = []
    dropped: list[tuple[str, str, float]] = []
    for j in range(X.shape[1]):
        hit = None
        for i in keep:
            if abs(R[i, j]) >= threshold:
                hit = (names[j], names[i], float(R[i, j]))
                break
        if hit is None:
            keep.append(j)
        else:
            dropped.append(hit)
    return X[:, keep], [names[j] for j in keep], dropped


def attribute_geotypes(
    X: np.ndarray,
    labels: np.ndarray,
    names: list[str],
    accuracy_gate: float = 0.7,
    prune_threshold: float = 0.9,
    n_estimators: int = 500,
    test_size: float = 0.25,
    seed: int = 0,
) -> dict:
    """RF + SHAP + permutation attribution of GeoType labels to descriptors.

    Returns a JSON-ready dict:
      {
        'gate': {'accuracy', 'passed', 'threshold', 'n_test'},
        'kept_features', 'dropped_features',
        'shap_mean_abs': {class -> {feature -> value}},       # TreeSHAP, interventional
        'permutation_importance': {feature -> {'mean','std'}},
        'rank_agreement_spearman': float,                     # SHAP vs permutation ranking
        'seed': int,
      }
    Raises ImportError with a clear message if the [attr] extra is missing.
    """
    try:
        import shap
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.inspection import permutation_importance
        from sklearn.model_selection import train_test_split
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "attribution requires the [attr] extra: pip install pygeotypes[attr]"
        ) from e

    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels, dtype=int)
    Xp, kept, dropped = prune_correlated(X, list(names), threshold=prune_threshold)

    X_tr, X_te, y_tr, y_te = train_test_split(
        Xp, labels, test_size=test_size, random_state=seed, stratify=labels
    )
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=seed, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    acc = float(rf.score(X_te, y_te))
    passed = acc >= accuracy_gate

    out: dict = {
        "gate": {"accuracy": acc, "passed": passed, "threshold": accuracy_gate, "n_test": int(y_te.size)},
        "kept_features": kept,
        "dropped_features": [{"dropped": a, "kept_as": b, "rho": r} for a, b, r in dropped],
        "seed": seed,
    }
    if not passed:
        out["shap_mean_abs"] = None
        out["permutation_importance"] = None
        out["rank_agreement_spearman"] = None
        return out

    # TreeSHAP (interventional perturbation on the test split)
    explainer = shap.TreeExplainer(rf)
    sv = explainer.shap_values(X_te)
    # normalize shape across shap versions: list of (n, f) per class OR (n, f, k)
    if isinstance(sv, list):
        per_class = [np.abs(np.asarray(s)).mean(axis=0) for s in sv]
    else:
        arr = np.asarray(sv)
        per_class = [np.abs(arr[:, :, c]).mean(axis=0) for c in range(arr.shape[2])]
    classes = [int(c) for c in rf.classes_]
    out["shap_mean_abs"] = {
        str(c): {f: float(v) for f, v in zip(kept, imp)} for c, imp in zip(classes, per_class)
    }

    # permutation importance cross-check
    pi = permutation_importance(rf, X_te, y_te, n_repeats=20, random_state=seed, n_jobs=-1)
    out["permutation_importance"] = {
        f: {"mean": float(m), "std": float(s)}
        for f, m, s in zip(kept, pi.importances_mean, pi.importances_std)
    }

    # rank agreement (Spearman between global SHAP and permutation rankings)
    shap_global = np.sum(per_class, axis=0)
    a = np.argsort(np.argsort(shap_global))
    b = np.argsort(np.argsort(pi.importances_mean))
    if a.size >= 2 and a.std() > 0 and b.std() > 0:
        out["rank_agreement_spearman"] = float(np.corrcoef(a, b)[0, 1])
    else:
        out["rank_agreement_spearman"] = None
    return out
