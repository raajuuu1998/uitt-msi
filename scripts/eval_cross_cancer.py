#!/usr/bin/env python
"""
Zero-shot cross-cancer evaluation on TCGA-STAD using COAD-trained models.

Loads the 5 fold models, averages per-patient probabilities (ensemble),
reports AUC for baseline and UITT, with bootstrap 95% CIs and the paired
DeLong test. Reproduces Table 3 of the paper.

Example
-------
python scripts/eval_cross_cancer.py \
    --coad_root data/TCGA_COAD \
    --stad_root data/TCGA_STAD \
    --results_dir results \
    --aggregators ABMIL CLAM_SB TransMIL
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from uitt import (
    get_model, build_immune_classifier, compute_uitt_10dim,
)
from uitt.stats import bootstrap_ci, delong_test

FM_DIM = 1536  # UNI2-h


def load_cohort(stad_root):
    feat_dir = f"{stad_root}/embeddings/uni2h"
    df = pd.read_csv(f"{stad_root}/labels_tcga_stad.csv")
    df = df.drop_duplicates(subset="patient_id", keep="first").reset_index(drop=True)
    df["label"] = (df["msi_label"] == "MSI-H").astype(int)
    available = set(f.stem for f in Path(feat_dir).glob("*.pt"))
    df = df[df.patient_id.isin(available)].reset_index(drop=True)
    return feat_dir, df["patient_id"].tolist(), df["label"].tolist()


def cache(feat_dir, pids, immune):
    feat_cache, uitt_cache = {}, {}
    km, ic, tg = immune
    for pid in pids:
        d = torch.load(f"{feat_dir}/{pid}.pt", map_location="cpu")
        feats = d["features"].float()
        feat_cache[pid] = feats
        uitt_cache[pid] = compute_uitt_10dim(feats, d["coords"].float(), d["slide_dims"], km, ic, tg)
    return feat_cache, uitt_cache


def ensemble_probs(agg, in_dim, prefix, results_dir, feat_cache, uitt_cache, pids, device, folds=5):
    fold_probs = []
    for fold in range(folds):
        tag = f"{prefix + '_' if prefix else ''}UNI2h_{agg}_fold{fold}.pt"
        mp = os.path.join(results_dir, tag)
        if not os.path.exists(mp):
            print(f"   missing {mp}")
            continue
        m = get_model(agg, in_dim).to(device)
        m.load_state_dict(torch.load(mp, map_location=device))
        m.eval()
        probs = []
        with torch.no_grad():
            for pid in pids:
                f = feat_cache[pid]
                if uitt_cache is not None:
                    f = torch.cat([f, uitt_cache[pid]], dim=1)
                probs.append(torch.sigmoid(m(f.to(device))).item())
        fold_probs.append(probs)
        del m
        torch.cuda.empty_cache()
    return np.mean(fold_probs, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coad_root", default="data/TCGA_COAD")
    ap.add_argument("--stad_root", default="data/TCGA_STAD")
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--aggregators", nargs="+", default=["ABMIL", "CLAM_SB", "TransMIL"])
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # immune classifier is fit on COAD (training cohort) and reused unchanged
    immune = build_immune_classifier(f"{args.coad_root}/embeddings/uni2h", seed=args.seed)

    feat_dir, pids, labels = load_cohort(args.stad_root)
    print(f"STAD: {len(pids)} patients | MSI-H {sum(labels)} | MSS {len(labels)-sum(labels)}")
    feat_cache, uitt_cache = cache(feat_dir, pids, immune)

    rows = []
    for agg in args.aggregators:
        base = ensemble_probs(agg, FM_DIM, "", args.results_dir,
                              feat_cache, None, pids, device)
        uitt = ensemble_probs(agg, FM_DIM + 10, "bioprior", args.results_dir,
                             feat_cache, uitt_cache, pids, device)
        auc_b, _, _ = bootstrap_ci(labels, base, args.n_boot, seed=args.seed)
        auc_u, lo, hi = bootstrap_ci(labels, uitt, args.n_boot, seed=args.seed)
        _, _, p = delong_test(labels, base, uitt)
        print(f"  {agg:10s} base {auc_b:.4f} -> UITT {auc_u:.4f} "
              f"(95% CI {lo:.3f}-{hi:.3f}), DeLong p={p:.4f}")
        rows.append({"aggregator": agg, "baseline": round(auc_b, 4),
                     "uitt": round(auc_u, 4), "ci_lo": round(lo, 3),
                     "ci_hi": round(hi, 3), "delong_p": round(p, 4)})

    out = os.path.join(args.results_dir, "cross_cancer_stad.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
