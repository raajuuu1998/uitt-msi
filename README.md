<div align="center">

# Beyond Appearance: Universal Immune-Tumor Topology (UITT)

### Biologically-Grounded Spatial Priors for Zero-Shot Cross-Cancer MSI Prediction

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-under%20review-orange.svg)]()

</div>

---

## Overview

Foundation-model (FM) features for histopathology entangle conserved immune biology with tissue-specific appearance, so a model trained to predict microsatellite instability (MSI) on one cancer type generalizes poorly to another. **UITT** is a set of ten tissue-agnostic spatial descriptors, computed from frozen FM embeddings and tile coordinates **without annotation**, that make the conserved immune architecture of MSI-high tumors explicit. Appended to tile embeddings before multiple-instance-learning (MIL) aggregation, UITT requires **no target-domain data** and transfers where appearance-based features do not.

<p align="center">
  <img src="assets/pipeline_msi.png" width="95%" alt="UITT pipeline">
  <br>
  <em>The UITT pipeline. A frozen FM embeds each tile; immune-like tiles are identified in feature space without supervision; ten spatial descriptors are computed per tile and appended to the embedding before MIL aggregation. Because they encode immune biology rather than appearance, they transfer across cancer types.</em>
</p>

**Highlights**

- **Cross-cancer transfer** — train on colorectal (TCGA-COAD), evaluate zero-shot on gastric (TCGA-STAD): no retraining, no target labels.
- **Statistically significant** — every cross-cancer improvement is significant under a paired DeLong test, with 95% bootstrap confidence intervals.
- **Benefit grows with shift** — neutral on near-saturated internal data, increasingly helpful as distribution shift increases.
- **Lightweight & backbone-agnostic** — no architectural change, no added aggregator parameters, a few seconds of CPU per slide on any frozen backbone.

> **Status.** This repository accompanies a paper currently **under review**. Code, precomputed embeddings, and results are released for reproducibility.

---

## Table of Contents

- [The UITT transform](#the-uitt-transform)
- [Results](#results)
- [Installation](#installation)
- [Data](#data)
- [Reproducing the paper](#reproducing-the-paper)
- [What's in the code](#whats-in-the-code)
- [Repository structure](#repository-structure)
- [Notes and scope](#notes-and-scope)
- [License & citation](#license--citation)

---

## The UITT transform

UITT first identifies immune-like tiles **without supervision** (K-means on FM embeddings; the tightest clusters correspond to morphologically regular lymphocytes), then computes ten spatial descriptors per tile across four biologically-motivated groups:

| Group | Biological basis | Descriptors | Dim |
|-------|------------------|-------------|:---:|
| **G1** | Tertiary lymphoid structures | membership, distance, size | 3 |
| **G2** | Peritumoral margin reaction (Crohn's-like) | margin distance, peri-immune band | 2 |
| **G3** | Multi-scale TIL density | density at _k_ = 10, 30, 100 | 3 |
| **G4** | Immune-tumor mixing | mixing entropy, immune-epithelial ratio | 2 |

The immune classifier is fit **once on the training cohort** and applied unchanged to every cohort, so no target information leaks into the descriptors. See [`docs/uitt_method.md`](docs/uitt_method.md) for the full walk-through.

```python
from uitt import build_immune_classifier, compute_uitt_10dim
import torch

# fit the immune classifier once, on the training cohort (COAD)
immune = build_immune_classifier("data/TCGA_COAD/embeddings/uni2h")

# per slide: append the 10-dim prior to the frozen embeddings
d = torch.load("data/TCGA_STAD/embeddings/uni2h/PID.pt", map_location="cpu")
uitt = compute_uitt_10dim(d["features"], d["coords"], d["slide_dims"], *immune)
x = torch.cat([d["features"], uitt], dim=1)   # [N, 1536 + 10] -> any MIL head
```

---

## Results

### 1. Internal TCGA-COAD — UNI2-h (5-fold stratified cross-validation)

MSI AUC, mean ± std over 5 folds. On this near-saturated cohort UITT is neutral, confirming the descriptors do not disturb an already-sufficient backbone. UNI2-h is the primary backbone used for all cross-cancer experiments.

| Aggregator | Baseline | + UITT |
|------------|:--------:|:------:|
| ABMIL    | 0.9337 ± 0.0166 | 0.9348 ± 0.0197 |
| CLAM-SB  | 0.9469 ± 0.0233 | 0.9447 ± 0.0205 |
| TransMIL | 0.9398 ± 0.0274 | **0.9571 ± 0.0112** |

### 2. Internal TCGA-COAD — CONCH (5-fold stratified cross-validation)

CONCH baselines for reference. CONCH+UITT is left to future work (its more compact 512-d embedding gave inconsistent cross-cohort behavior).

| Aggregator | Baseline |
|------------|:--------:|
| ABMIL    | 0.9145 ± 0.0499 |
| CLAM-SB  | 0.9181 ± 0.0563 |
| TransMIL | 0.9198 ± 0.0615 |

### 3. Zero-shot cross-cancer TCGA-STAD (UNI2-h)

COAD-trained models applied to gastric cancer with **zero gastric examples**. UITT improves every aggregator; all improvements are significant under the paired DeLong test. Parentheses give the 95% bootstrap CI for the UITT AUC.

| Aggregator | Baseline | + UITT | 95% CI (UITT) | DeLong _p_ |
|------------|:--------:|:------:|:-------------:|:----------:|
| ABMIL    | 0.5681 | **0.6313** | 0.542 – 0.712 | < 0.001 |
| CLAM-SB  | 0.6242 | **0.6414** | 0.546 – 0.725 | 0.037   |
| TransMIL | 0.6627 | **0.7161** | 0.633 – 0.789 | 0.003   |

### Figures

<div align="center">

| Generalization gap | Forest plot | ROC curve |
|:------------------:|:-----------:|:---------:|
| <img src="assets/plot1_gap.png" width="270"/> | <img src="assets/plot2_forest.png" width="270"/> | <img src="assets/plot3_roc.png" width="270"/> |

</div>

<p align="center">
  <em><b>Left:</b> the UITT gain grows under distribution shift (internal +0.017 vs cross-cancer +0.053, TransMIL).
  <b>Middle:</b> forest plot of UITT vs baseline AUC with 95% CIs and DeLong p-values, all three aggregators.
  <b>Right:</b> ROC for the strongest aggregator (TransMIL); UITT (orange) dominates the baseline (navy).</em>
</p>
---

## Installation

```bash
git clone https://github.com/raajuuu1998/uitt-msi.git
cd uitt-msi
pip install -r requirements.txt
pip install -e .          # exposes the `uitt` package
```

Python ≥ 3.9, PyTorch ≥ 2.0. A single GPU is sufficient: the backbone is frozen and features are precomputed, so every experiment runs on modest hardware.

---

## Data

This repository operates on **precomputed tile embeddings** (raw WSIs and the FM forward pass are upstream and not included). Slides are from TCGA; MSI labels are from cBioPortal.

**Precomputed embeddings (UNI2-h & CONCH) for TCGA-COAD and TCGA-STAD, plus results:**

📁 **[Google Drive — embeddings + results](https://drive.google.com/drive/folders/1L5e5_AE0Hsm1wKlBqfx3_KoQmvH_amne?usp=sharing)**

Download and arrange as:

```
data/
  TCGA_COAD/
    labels_tcga_coad.csv          # columns: patient_id, msi_label  (MSI-H / MSS)
    embeddings/uni2h/{pid}.pt      # dict: features[N,1536], coords[N,2], slide_dims=(W,H)
    embeddings/conch/{pid}.pt      # dict: features[N,512],  coords[N,2], slide_dims=(W,H)
  TCGA_STAD/
    labels_tcga_stad.csv
    embeddings/uni2h/{pid}.pt
```

Each `.pt` is a dict with keys `features`, `coords`, `slide_dims`, `patient_id`, `msi_label`, `cohort`. Tiles: 224 × 224 at 20× (0.5 µm/px), Otsu tissue mask, capped at 8000 tiles, no stain normalization. Backbones: [UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h) (1536-d) and [CONCH](https://huggingface.co/MahmoodLab/CONCH) (512-d).

---

## Reproducing the paper

```bash
# 1. Internal COAD 5-fold cross-validation (baseline + UITT), UNI2-h
python scripts/train_internal.py --fm UNI2h --config baseline
python scripts/train_internal.py --fm UNI2h --config uitt

# 2. Zero-shot cross-cancer evaluation on STAD, with bootstrap CIs + DeLong
python scripts/eval_cross_cancer.py \
    --coad_root data/TCGA_COAD --stad_root data/TCGA_STAD \
    --results_dir results --aggregators ABMIL CLAM_SB TransMIL

# 3. Regenerate the figures
python scripts/make_figures.py --results_dir results --assets_dir assets
```

Defaults — 5-fold stratified CV, 20 epochs, Adam, lr 1e-4, weight decay 1e-5, hidden 256, seed 42 — match the paper; see [`configs/default.yaml`](configs/default.yaml).

---

## What's in the code

| File | Contents |
|------|----------|
| `uitt/uitt_transform.py` | **The method.** `build_immune_classifier` (unsupervised immune-tile ID), `compute_uitt_10dim` (the 10-dim transform), plus `sinusoidal_encoding` and `peripheral_encoding` comparison priors |
| `uitt/models.py` | MIL aggregators: `ABMIL`, `CLAM_SB`, `TransMIL`, `MeanPool`, `MaxPool`, and `get_model` |
| `uitt/data.py` | `MILDataset` (loads embeddings, optionally appends the prior) and `mil_collate` |
| `uitt/engine.py` | `train_one_epoch` and `evaluate` (slide-level AUC) |
| `uitt/stats.py` | `bootstrap_ci` (percentile bootstrap) and `delong_test` (fast paired DeLong, Sun & Xu 2014) |
| `scripts/train_internal.py` | COAD 5-fold CV training, baseline or UITT, per-fold checkpoints + mean ± std AUC |
| `scripts/eval_cross_cancer.py` | Zero-shot STAD evaluation, ensemble over folds, AUC + CI + DeLong |
| `scripts/make_figures.py` | Regenerates the gap, forest, and ROC figures |

---

## Repository structure

```
uitt-msi/
├── uitt/                     # installable package
│   ├── models.py             # ABMIL, CLAM-SB, TransMIL, Mean/Max pool
│   ├── uitt_transform.py     # immune classifier + 10-dim UITT (the method)
│   ├── data.py               # MIL dataset / collate
│   ├── engine.py             # train / evaluate loops
│   └── stats.py              # bootstrap CI + paired DeLong test
├── scripts/
│   ├── train_internal.py     # COAD 5-fold CV (baseline / UITT)
│   ├── eval_cross_cancer.py  # zero-shot STAD eval + statistics
│   └── make_figures.py       # regenerate the result figures
├── configs/default.yaml      # all hyperparameters + data layout
├── docs/uitt_method.md       # detailed method description
├── assets/                   # figures
├── requirements.txt
├── setup.py
└── LICENSE
```

---

## Notes and scope

- **Scope.** This release covers the cross-cancer setting reported in the paper (COAD → STAD). The UITT transform is backbone-agnostic; the reported cross-cancer results use UNI2-h.
- **Determinism.** The unsupervised immune classifier samples tiles per slide; for bitwise reproducibility, fit it once and reuse the returned object across all evaluations (as the scripts do).
- **Not included.** Raw WSIs and the FM forward pass are not redistributed; obtain TCGA slides from their original sources. Precomputed embeddings are provided via the Drive link above.

---

## License & citation

Released under the MIT License — see [LICENSE](LICENSE).

```bibtex
@misc{uitt2026,
  title  = {Beyond Appearance: Biologically-Grounded Spatial Priors for
            Zero-Shot Cross-Cancer MSI Prediction},
  author = {Dasari Naga Raju},
  year   = {2026},
  note   = {Under review}
}
```
