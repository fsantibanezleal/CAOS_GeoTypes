#!/usr/bin/env python3
"""Regenerate the two data figures for the pygeotypes shape-catalogue software note, from the COMMITTED
validation artifacts (produced by coverage_validation.py). No network, no recompute of the DEM/DTW.

  fig-catalogue.pdf  - the GeoType catalogue: the K medoid curves (behaviour prototypes) on the log grid.
  fig-coverage.pdf   - (a) empirical conformal coverage vs the 1 - alpha target (the calibration curve);
                       (b) mean prediction-set size and out-of-catalogue empty-set rate vs alpha.

The hand-authored fig-pipeline.svg is converted to PDF separately via svglib.

Run:  python make_figs.py
Deps: matplotlib, numpy.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"

INK = "#1a1a2e"
C1 = "#1b6ca8"
C2 = "#e07a3f"
C3 = "#3fa34d"
GRID = "#d8d8e0"

plt.rcParams.update({
    "font.family": "serif", "font.size": 9.4, "axes.edgecolor": INK,
    "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.linewidth": 0.8, "figure.dpi": 200,
})


def fig_catalogue():
    d = json.loads((DATA / "catalogue_medoids.json").read_text(encoding="utf-8"))
    t = np.asarray(d["t_grid"])
    meds = np.asarray(d["medoids"])
    counts = d["counts"]
    K = d["K"]
    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    cols = [C1, C2, C3, "#7d5ba6", "#b23a48"]
    for g in range(K):
        ax.plot(t, meds[g], color=cols[g % len(cols)], linewidth=1.9,
                label=f"GeoType {g} (n={counts[g]})")
    ax.set_xscale("log")
    ax.set_xlabel("dimensionless time (log grid)")
    ax.set_ylabel("normalized Bourdet derivative")
    ax.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=7.8, frameon=True, facecolor="white", edgecolor=GRID, loc="best")
    ax.set_title("catalogue medoids", fontsize=9.4)
    fig.tight_layout()
    fig.savefig(HERE / "fig-catalogue.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_coverage():
    d = json.loads((DATA / "coverage_validation.json").read_text(encoding="utf-8"))
    cov = d["coverage"]
    targets = [r["target"] for r in cov]
    emp = [r["empirical_coverage"] for r in cov]
    alphas = [r["alpha"] for r in cov]
    sizes = [r["mean_set_size"] for r in cov]
    classes = sorted({int(g) for r in cov for g in r["per_class_coverage"]})
    per_class = {g: [r["per_class_coverage"].get(str(g), r["per_class_coverage"].get(g)) for r in cov]
                 for g in classes}
    ood = {r["alpha"]: r["empty_set_rate"] for r in d["ood"]["by_alpha"]}
    ood_all = all(v >= 0.999 for v in ood.values())

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.0, 3.0))

    # (a) coverage calibration: marginal + per-class vs the target diagonal
    axa.plot([0.6, 1.0], [0.6, 1.0], color="#999", linewidth=1.0, linestyle="--", label="target $1-\\alpha$")
    ccols = [C1, C2, C3, "#7d5ba6"]
    for i, g in enumerate(classes):
        axa.plot(targets, per_class[g], "o:", color=ccols[i % len(ccols)], linewidth=1.1, markersize=3.4,
                 alpha=0.85, label=f"GeoType {g}")
    axa.plot(targets, emp, "o-", color=INK, linewidth=1.8, markersize=5, label="marginal")
    for tt, ee in zip(targets, emp):
        axa.annotate(f"{ee:.2f}", (tt, ee), textcoords="offset points", xytext=(4, -10), fontsize=7.0, color=INK)
    axa.set_xlabel("target coverage $1-\\alpha$")
    axa.set_ylabel("empirical coverage")
    axa.set_xlim(0.63, 0.99)
    axa.set_ylim(0.6, 1.0)
    axa.set_title("(a) conformal calibration", fontsize=9.2)
    axa.grid(True, color=GRID, linewidth=0.7)
    axa.set_axisbelow(True)
    axa.legend(fontsize=7.0, frameon=True, facecolor="white", edgecolor=GRID, loc="upper left", ncol=2)
    for s in ("top", "right"):
        axa.spines[s].set_visible(False)

    # (b) informativeness: mean prediction-set size vs alpha; OOD noted (perfect)
    axb.plot(alphas, sizes, "s-", color=C2, linewidth=1.7, markersize=5, label="mean prediction-set size")
    axb.axhline(1.0, color=GRID, linewidth=0.8, linestyle=":")
    axb.set_xlabel("miscoverage level $\\alpha$")
    axb.set_ylabel("mean prediction-set size")
    axb.set_ylim(0.6, 1.15)
    axb.set_title("(b) informativeness", fontsize=9.2)
    note = "alien shapes: 100% flagged\nout-of-catalogue (all $\\alpha$)" if ood_all else \
           f"OOD empty-set rate {min(ood.values()):.2f}-{max(ood.values()):.2f}"
    axb.text(0.97, 0.95, note, transform=axb.transAxes, ha="right", va="top", fontsize=7.8,
             color=C3, style="italic",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C3, lw=0.8))
    axb.grid(True, color=GRID, linewidth=0.7)
    axb.set_axisbelow(True)
    axb.legend(fontsize=7.4, frameon=True, facecolor="white", edgecolor=GRID, loc="lower left")
    for s in ("top", "right"):
        axb.spines[s].set_visible(False)

    fig.tight_layout()
    fig.savefig(HERE / "fig-coverage.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    fig_catalogue()
    fig_coverage()
    print("wrote fig-catalogue.pdf, fig-coverage.pdf")


if __name__ == "__main__":
    main()
