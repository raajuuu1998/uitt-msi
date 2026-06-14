# The UITT Transform

This note documents the Universal Immune-Tumor Topology (UITT) descriptors in
more detail than the README. It is meant to make the method readable directly
from code.

## Motivation

A foundation-model embedding of an H&E tile captures *appearance*: texture,
stain, tissue type. That appearance varies with scanner, protocol, and cancer
of origin, so a classifier built on it transfers poorly across cancer types.
The immune response to MSI-high tumors, by contrast, has a *spatial* signature
that is conserved across organs because it reflects mismatch-repair-deficient
immunobiology rather than tissue appearance: a dense peritumoral lymphocytic
band (the Crohn's-like reaction), abundant tumor-infiltrating lymphocytes, and
tertiary lymphoid structures. UITT makes that spatial signature explicit.

## Step 1: unsupervised immune-tile identification

Lymphocytes are small, round, and densely packed, far more morphologically
uniform than heterogeneous tumor epithelium and stroma. That regularity
survives in FM embedding space: lymphocytic tiles form a few tight, internally
coherent clusters. We exploit this with K-means (K = 50) on tiles sampled from
the **training cohort only**, scoring each cluster by tightness (inverse mean
distance to its centroid) and flagging the top 30% as immune (15 of 50 in
practice). This classifier is fit once and applied unchanged to every cohort,
so no target information enters the descriptors.

At inference, each tile gets a binary immune indicator `m_i` and a continuous
immune score `s_i` in [0, 1].

## Step 2: ten spatial descriptors (four groups)

All descriptors are computed from normalized tile coordinates and the immune
mask, then min-max normalized per slide. Column indices in the returned
`[N, 10]` tensor are given in brackets.

**G1 — Tertiary lymphoid structures** (DBSCAN on immune-tile coordinates,
eps = 0.02, min 5 points):
- `[0]` membership: is the tile inside a detected aggregate
- `[1]` distance: normalized distance to the nearest aggregate centroid
- `[2]` size: size of the nearest aggregate, normalized by the largest

**G2 — Peritumoral margin reaction** (local density as interior/boundary proxy):
- `[3]` margin distance: complement of local tile density (k = 20)
- `[4]` peri-immune band: immune score x margin distance (large for immune
  tiles near the boundary, the Crohn's-like signature)

**G3 — Multi-scale TIL density** (fraction of immune neighbors at three scales):
- `[5]` k = 10 (immediate), `[6]` k = 30 (local), `[7]` k = 100 (regional)

**G4 — Immune-tumor mixing**:
- `[8]` mixing entropy: binary entropy of the local immune fraction (k = 20),
  maximal at an even interleaving of immune and tumor tiles
- `[9]` immune-epithelial ratio: immune-to-tumor balance in the k = 30
  neighborhood

## Step 3: integration with MIL

The ten descriptors are concatenated with the frozen embedding to form
`[N, d + 10]` inputs. No change is made to the aggregator, loss, or
hyperparameters; the only difference between a baseline and a UITT model is the
ten appended dimensions, isolating the effect of the prior.

## Why it transfers

The backbone and UITT encode different parts of the signal. The backbone
encodes appearance, discriminative within a cohort but unstable across cancer
types. UITT encodes the spatial architecture of the immune response, determined
by tumor immunobiology and conserved across organs. When appearance becomes
unreliable under shift, the conserved spatial signal remains, which is why the
margin in favor of UITT widens under distribution shift rather than vanishing.
