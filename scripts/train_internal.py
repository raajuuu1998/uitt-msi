#!/usr/bin/env python
"""
Train MIL aggregators on internal TCGA-COAD with 5-fold stratified CV,
optionally injecting the UITT prior.

Examples
--------
# Baseline (frozen embedding only), all aggregators, UNI2-h:
python scripts/train_internal.py --fm UNI2h --config baseline

# UITT (embedding + 10 descriptors), TransMIL only:
python scripts/train_internal.py --fm UNI2h --config uitt --aggregators TransMIL

Outputs per-fold checkpoints to <results_dir>/{prefix}_{fm}_{agg}_fold{k}.pt
and a CSV summary of mean +/- std AUC.

Reproduces the internal numbers in Table 2 of the paper.
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from uitt import (
    get_model, MILDataset, mil_collate,
    build_immune_classifier, compute_uitt_10dim,
    sinusoidal_encoding, peripheral_encoding,
)
from uitt.engine import train_one_epoch, evaluate

CONFIG_DIMS = {"baseline": 0, "uitt": 10, "peripheral": 1, "sinpe": 12}
CONFIG_PREFIX = {"baseline": "", "uitt": "bioprior", "peripheral": "peripheral", "sinpe": "sinpe"}


def precompute_extra(config, pids, feat_dir, immune=None, n_freqs=3):
    """Build the pid -> extra-feature dict for a given prior config."""
    if config == "baseline":
        return None
    km, ic, tg = immune if immune is not None else (None, None, None)
    extra = {}
    for pid in pids:
        fp = f"{feat_dir}/{pid}.pt"
        if not os.path.exists(fp):
            continue
        d = torch.load(fp, map_location="cpu")
        feats, coords, dims = d["features"].float(), d["coords"].float(), d["slide_dims"]
        if config == "uitt":
            extra[pid] = compute_uitt_10dim(feats, coords, dims, km, ic, tg)
        elif config == "peripheral":
            extra[pid] = peripheral_encoding(coords, dims)
        elif config == "sinpe":
            extra[pid] = sinusoidal_encoding(coords, dims, n_freqs=n_freqs)
    return extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/TCGA_COAD")
    ap.add_argument("--fm", default="UNI2h", choices=["UNI2h", "CONCH"])
    ap.add_argument("--config", default="baseline", choices=list(CONFIG_DIMS))
    ap.add_argument("--aggregators", nargs="+",
                    default=["ABMIL", "CLAM_SB", "TransMIL"])
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.results_dir, exist_ok=True)

    feat_dir = f"{args.data_root}/embeddings/{args.fm.lower()}"
    csv_path = f"{args.data_root}/labels_tcga_coad.csv"
    fm_dim = 1536 if args.fm == "UNI2h" else 512
    in_dim = fm_dim + CONFIG_DIMS[args.config]
    prefix = CONFIG_PREFIX[args.config]

    df = pd.read_csv(csv_path)
    df["label"] = (df["msi_label"] == "MSI-H").astype(int)
    available = set(f.stem for f in Path(feat_dir).glob("*.pt"))
    df = df[df.patient_id.isin(available)].reset_index(drop=True)
    pids = df["patient_id"].values
    labels = df["label"].values
    print(f"{args.fm} | {len(df)} patients | MSI-H {labels.sum()} | MSS {(labels==0).sum()}")

    immune = build_immune_classifier(feat_dir, seed=args.seed) if args.config == "uitt" else None
    extra = precompute_extra(args.config, pids, feat_dir, immune)

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    rows = []
    for agg in args.aggregators:
        fold_aucs = []
        for fold, (tr, va) in enumerate(skf.split(pids, labels)):
            tr_ds = MILDataset(pids[tr], labels[tr], feat_dir, extra)
            va_ds = MILDataset(pids[va], labels[va], feat_dir, extra)
            tr_dl = DataLoader(tr_ds, batch_size=1, shuffle=True, collate_fn=mil_collate)
            va_dl = DataLoader(va_ds, batch_size=1, shuffle=False, collate_fn=mil_collate)

            model = get_model(agg, in_dim).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            best = 0.0
            tag = f"{prefix + '_' if prefix else ''}{args.fm}_{agg}_fold{fold}.pt"
            for _ in range(args.epochs):
                train_one_epoch(model, tr_dl, opt, device)
                auc, _, _ = evaluate(model, va_dl, device)
                if auc > best:
                    best = auc
                    torch.save(model.state_dict(), os.path.join(args.results_dir, tag))
            fold_aucs.append(best)
            print(f"  {args.fm} | {agg} | {args.config} | fold {fold} | AUC {best:.4f}")
        mu, sd = float(np.mean(fold_aucs)), float(np.std(fold_aucs))
        print(f"  == {args.fm} | {agg} | {args.config} | {mu:.4f} +/- {sd:.4f}")
        rows.append({"fm": args.fm, "aggregator": agg, "config": args.config,
                     "mean_auc": round(mu, 4), "std_auc": round(sd, 4)})

    out = os.path.join(args.results_dir, f"internal_{args.fm}_{args.config}.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
