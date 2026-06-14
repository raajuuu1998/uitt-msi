"""
The UITT (Universal Immune-Tumor Topology) transform.

Given frozen foundation-model tile embeddings F in R^{N x d} and tile
coordinates, UITT produces a 10-dimensional, tissue-agnostic spatial
descriptor per tile, encoding the immune architecture conserved across
MSI-high tumors. The descriptors are appended to each tile embedding
before MIL aggregation.

Descriptor layout (column order in the returned [N, 10] tensor):
  G1 Tertiary lymphoid structures : [0] tls_membership, [1] tls_distance, [2] tls_size
  G2 Peritumoral margin reaction  : [3] margin_distance, [4] periimmune_band
  G3 Multi-scale TIL density      : [5] til_10, [6] til_30, [7] til_100
  G4 Immune-tumor mixing          : [8] mix_entropy, [9] immune_epithelial_ratio

This module also provides the sinusoidal and peripheral-distance
encodings used as comparison priors in the journal version.
"""
import warnings
import numpy as np
import torch
from pathlib import Path
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------
# Unsupervised immune-tile identification
# --------------------------------------------------------------------------
def build_immune_classifier(feat_dir, n_clusters=50, top_frac=0.3,
                            max_sample=500, seed=42):
    """
    Fit K-means on tile embeddings sampled from the TRAINING cohort only, and
    flag the tightest clusters as immune. Returns (kmeans, immune_clusters,
    tightness). Fit once and reused unchanged across all cohorts so no target
    information leaks into the descriptors.
    """
    np.random.seed(seed)
    files = list(Path(feat_dir).glob("*.pt"))
    all_feats = []
    for f in files:
        d = torch.load(f, map_location="cpu")
        feats = d["features"].float().numpy()
        n = min(len(feats), max_sample)
        idx = np.random.choice(len(feats), n, replace=False)
        all_feats.append(feats[idx])
    all_feats_norm = normalize(np.vstack(all_feats), norm="l2")

    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=3)
    kmeans.fit(all_feats_norm)

    tightness = np.zeros(n_clusters)
    for c in range(n_clusters):
        mask = kmeans.labels_ == c
        if mask.sum() > 0:
            dists = np.linalg.norm(all_feats_norm[mask] - kmeans.cluster_centers_[c], axis=1)
            tightness[c] = 1.0 / (dists.mean() + 1e-6)

    threshold = np.percentile(tightness, (1 - top_frac) * 100)
    immune_clusters = set(np.where(tightness >= threshold)[0])
    return kmeans, immune_clusters, tightness


def get_immune_scores(feats_norm, kmeans, immune_clusters, tightness):
    """Per-tile continuous immune score in [0,1] and a binary immune mask."""
    labels = kmeans.predict(feats_norm)
    scores = tightness[labels] / (tightness.max() + 1e-6)
    immune_mask = np.array([l in immune_clusters for l in labels])
    scores = scores * (1 + immune_mask.astype(float))
    scores = scores / (scores.max() + 1e-6)
    return scores, immune_mask


# --------------------------------------------------------------------------
# The UITT 10-dim transform
# --------------------------------------------------------------------------
def compute_uitt_10dim(feats, coords, slide_dims, kmeans, immune_clusters, tightness):
    """
    Compute the 10-dim UITT descriptor for one slide.

    Args:
        feats        : torch.Tensor [N, d]  frozen FM embeddings
        coords       : torch.Tensor [N, 2]  tile (x, y) coordinates
        slide_dims   : (W, H) slide size in pixels
        kmeans, immune_clusters, tightness : output of build_immune_classifier

    Returns:
        torch.Tensor [N, 10]  per-tile UITT descriptor (see module docstring
        for column layout).
    """
    N = len(feats)
    W, H = slide_dims
    feats_norm = normalize(feats.float().numpy(), norm="l2")
    coords_norm = coords.float().numpy() / np.array([W, H])

    immune_scores, immune_mask = get_immune_scores(
        feats_norm, kmeans, immune_clusters, tightness)

    max_k = min(101, N)
    nbrs = NearestNeighbors(n_neighbors=max_k, algorithm="ball_tree").fit(coords_norm)
    dists_all, idx_all = nbrs.kneighbors(coords_norm)
    idx_all = idx_all[:, 1:]      # drop self
    dists_all = dists_all[:, 1:]

    k10 = min(10, idx_all.shape[1])
    k30 = min(30, idx_all.shape[1])
    k100 = min(100, idx_all.shape[1])
    idx_10 = idx_all[:, :k10]
    idx_30 = idx_all[:, :k30]
    idx_100 = idx_all[:, :k100]
    idx_20 = idx_all[:, :min(20, idx_all.shape[1])]

    # ----- G1: Tertiary lymphoid structures (3 dims) -----
    immune_coords = coords_norm[immune_mask]
    tls_membership = np.zeros(N)
    tls_distance = np.ones(N)
    tls_size = np.zeros(N)
    if len(immune_coords) >= 5:
        db = DBSCAN(eps=0.02, min_samples=5).fit(immune_coords)
        tls_labels = db.labels_
        n_tls = len(set(tls_labels)) - (1 if -1 in tls_labels else 0)
        if n_tls > 0:
            tls_centroids, tls_sizes = [], []
            for tid in set(tls_labels):
                if tid == -1:
                    continue
                m = tls_labels == tid
                tls_centroids.append(immune_coords[m].mean(axis=0))
                tls_sizes.append(m.sum())
            tls_centroids = np.array(tls_centroids)
            tls_sizes = np.array(tls_sizes, dtype=float)
            tls_sizes_norm = tls_sizes / (tls_sizes.max() + 1e-6)
            diff = coords_norm[:, None, :] - tls_centroids[None, :, :]
            dists = np.linalg.norm(diff, axis=-1)
            nearest = dists.argmin(axis=1)
            tls_distance = dists[np.arange(N), nearest]
            tls_size = tls_sizes_norm[nearest]
            for j, i in enumerate(np.where(immune_mask)[0]):
                if tls_labels[j] != -1:
                    tls_membership[i] = 1.0
    tls_distance = tls_distance / (tls_distance.max() + 1e-6)

    # ----- G2: Peritumoral margin reaction (2 dims) -----
    local_density = 1.0 / (dists_all[:, :20].mean(axis=1) + 1e-6)
    local_density = local_density / (local_density.max() + 1e-6)
    margin_distance = 1.0 - local_density
    periimmune_band = immune_scores * margin_distance
    periimmune_band = periimmune_band / (periimmune_band.max() + 1e-6)

    # ----- G3: Multi-scale TIL density (3 dims) -----
    til_10 = immune_mask[idx_10].mean(axis=1).astype(float)
    til_30 = immune_mask[idx_30].mean(axis=1).astype(float)
    til_100 = immune_mask[idx_100].mean(axis=1).astype(float)

    # ----- G4: Immune-tumor mixing (2 dims) -----
    p_immune = immune_mask[idx_20].mean(axis=1)
    p_tumor = 1.0 - p_immune
    mix_ent = np.where(
        (p_immune > 0) & (p_tumor > 0),
        -p_immune * np.log2(p_immune + 1e-9) - p_tumor * np.log2(p_tumor + 1e-9),
        0.0,
    )
    n_immune_30 = immune_mask[idx_30].sum(axis=1).astype(float)
    ie_ratio = n_immune_30 / (k30 - n_immune_30 + 1.0)
    ie_ratio = ie_ratio / (ie_ratio.max() + 1e-6)

    uitt = np.stack([
        tls_membership, tls_distance, tls_size,    # G1
        margin_distance, periimmune_band,          # G2
        til_10, til_30, til_100,                   # G3
        mix_ent, ie_ratio,                         # G4
    ], axis=1)
    return torch.tensor(uitt, dtype=torch.float32)


# --------------------------------------------------------------------------
# Comparison priors (journal version)
# --------------------------------------------------------------------------
def sinusoidal_encoding(coords, slide_dims, n_freqs=3):
    """Generic sinusoidal positional encoding (4 * n_freqs dims)."""
    W, H = slide_dims
    xy = coords.clone().float()
    xy[:, 0] /= W
    xy[:, 1] /= H
    freqs = torch.pow(2, torch.linspace(0, n_freqs - 1, n_freqs)) * torch.pi
    return torch.cat([
        torch.sin(xy[:, 0:1] * freqs),
        torch.cos(xy[:, 0:1] * freqs),
        torch.sin(xy[:, 1:2] * freqs),
        torch.cos(xy[:, 1:2] * freqs),
    ], dim=1)


def peripheral_encoding(coords, slide_dims):
    """Distance-to-nearest-slide-edge encoding (1 dim)."""
    W, H = slide_dims
    cx = coords[:, 0].float() / W
    cy = coords[:, 1].float() / H
    dist = torch.stack([cx, 1 - cx, cy, 1 - cy], dim=1).min(dim=1).values.unsqueeze(1)
    return dist
