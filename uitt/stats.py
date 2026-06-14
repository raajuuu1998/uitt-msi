"""
Statistical analysis for the cross-cancer results:
  - bootstrap_ci  : percentile bootstrap 95% CI for an AUC
  - delong_test   : paired DeLong test for two correlated AUCs (fast, scipy-only)

The DeLong implementation follows Sun & Xu (2014), "Fast Implementation of
DeLong's Algorithm for Comparing the Areas Under Correlated Receiver Operating
Characteristic Curves", and the original DeLong et al. (1988).
"""
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score


def bootstrap_ci(labels, probs, n_boot=1000, alpha=0.05, seed=42):
    """Percentile bootstrap CI for AUC. Returns (auc, lo, hi)."""
    rng = np.random.RandomState(seed)
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    n = len(labels)
    aucs = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(np.unique(labels[idx])) < 2:
            continue
        aucs.append(roc_auc_score(labels[idx], probs[idx]))
    auc = roc_auc_score(labels, probs)
    lo = np.percentile(aucs, 100 * alpha / 2)
    hi = np.percentile(aucs, 100 * (1 - alpha / 2))
    return auc, lo, hi


# --------------------------------------------------------------------------
# Fast DeLong (Sun & Xu, 2014)
# --------------------------------------------------------------------------
def _compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def _fast_delong(predictions_sorted_transposed, label_1_count):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive = predictions_sorted_transposed[:, :m]
    negative = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = _compute_midrank(positive[r, :])
        ty[r, :] = _compute_midrank(negative[r, :])
        tz[r, :] = _compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_test(labels, probs_a, probs_b):
    """
    Paired DeLong test comparing two correlated AUCs on the same labels.
    Returns (auc_a, auc_b, p_value) for the two-sided test that the AUCs differ.
    """
    labels = np.asarray(labels)
    order = np.argsort(-labels)  # positives first
    label_1_count = int(labels.sum())
    preds = np.vstack((np.asarray(probs_a), np.asarray(probs_b)))[:, order]
    aucs, cov = _fast_delong(preds, label_1_count)
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        z = 0.0
    else:
        z = (aucs[0] - aucs[1]) / np.sqrt(var)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return float(aucs[0]), float(aucs[1]), float(p)
