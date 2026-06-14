#!/usr/bin/env python
"""
Regenerate the three result figures from saved predictions / result CSVs:
  plot1_gap.png    generalization gap (internal vs cross-cancer, TransMIL)
  plot2_forest.png forest plot of UITT vs baseline with CIs + DeLong p
  plot3_roc.png    ROC curves, baseline vs UITT (TransMIL, STAD)

Expects:
  results/cross_cancer_stad.csv      (from eval_cross_cancer.py)
  results/stad_predictions.csv       per-slide probs (optional, for ROC)

Palette: navy #1E3A8A (baseline), orange #C2410C (UITT).
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

NAVY = "#1E3A8A"
ORANGE = "#C2410C"


def plot_gap(internal_base, internal_uitt, cross_base, cross_uitt, out):
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(2)
    w = 0.35
    ax.bar(x - w/2, [internal_base, cross_base], w, color=NAVY, label="Baseline")
    ax.bar(x + w/2, [internal_uitt, cross_uitt], w, color=ORANGE, label="UITT")
    for xi, (b, u) in zip(x, [(internal_base, internal_uitt), (cross_base, cross_uitt)]):
        ax.text(xi - w/2, b + 0.005, f"{b:.3f}", ha="center", fontsize=9)
        ax.text(xi + w/2, u + 0.005, f"{u:.3f}", ha="center", fontsize=9)
        ax.text(xi, max(b, u) + 0.03, f"+{u-b:.3f}", ha="center",
                fontsize=11, fontweight="bold", color=NAVY)
    ax.set_xticks(x)
    ax.set_xticklabels(["Internal\n(TCGA-COAD)", "Cross-cancer\n(TCGA-STAD)"])
    ax.set_ylabel("MSI AUC")
    ax.set_ylim(0.5, 1.0)
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"  saved {out}")


def plot_forest(df, out):
    fig, ax = plt.subplots(figsize=(7, 4))
    aggs = df["aggregator"].tolist()
    y = np.arange(len(aggs))[::-1]
    for yi, (_, r) in zip(y, df.iterrows()):
        ax.plot([r.ci_lo, r.ci_hi], [yi, yi], color=ORANGE, lw=2, zorder=1)
        ax.scatter(r.uitt, yi, color=ORANGE, s=60, zorder=3)
        ax.scatter(r.baseline, yi, color=NAVY, marker="D", s=60, zorder=3)
        ax.text(r.ci_hi + 0.01, yi, f"p={r.delong_p:g}",
                va="center", fontsize=9, fontweight="bold", color=NAVY)
    ax.axvline(0.5, ls="--", color="grey", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(aggs)
    ax.set_xlabel("MSI AUC (cross-cancer, TCGA-STAD)")
    ax.scatter([], [], color=ORANGE, s=60, label="UITT (95% CI)")
    ax.scatter([], [], color=NAVY, marker="D", s=60, label="Baseline")
    ax.legend(loc="lower left", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"  saved {out}")


def plot_roc(labels, base, uitt, out):
    fig, ax = plt.subplots(figsize=(5.2, 5))
    for probs, color, name in [(base, NAVY, "Baseline"), (uitt, ORANGE, "UITT")]:
        fpr, tpr, _ = roc_curve(labels, probs)
        auc = roc_auc_score(labels, probs)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"  saved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--assets_dir", default="assets")
    args = ap.parse_args()
    os.makedirs(args.assets_dir, exist_ok=True)

    df = pd.read_csv(os.path.join(args.results_dir, "cross_cancer_stad.csv"))
    tr = df[df.aggregator == "TransMIL"].iloc[0]

    # Gap plot uses the paper's internal TransMIL numbers; adjust if you recompute.
    plot_gap(internal_base=0.9398, internal_uitt=0.9571,
             cross_base=tr.baseline, cross_uitt=tr.uitt,
             out=os.path.join(args.assets_dir, "plot1_gap.png"))
    plot_forest(df, out=os.path.join(args.assets_dir, "plot2_forest.png"))

    pred_path = os.path.join(args.results_dir, "stad_predictions.csv")
    if os.path.exists(pred_path):
        p = pd.read_csv(pred_path)
        plot_roc(p["label"].values,
                 p["prob_baseline_TransMIL"].values,
                 p["prob_bio_prior_TransMIL"].values,
                 out=os.path.join(args.assets_dir, "plot3_roc.png"))
    else:
        print("  (skip ROC: results/stad_predictions.csv not found)")


if __name__ == "__main__":
    main()
