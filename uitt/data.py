"""
Dataset and collate utilities for slide-level MIL.

Each .pt embedding file is expected to be a dict with keys:
    features   : torch.Tensor [N, d]   frozen FM tile embeddings
    coords     : torch.Tensor [N, 2]   tile (x, y) coordinates in pixels
    slide_dims : (W, H)                slide size in pixels
    patient_id : str
    msi_label  : str  ("MSI-H" or "MSS")
    cohort     : str
"""
import os
import torch
from torch.utils.data import Dataset


class MILDataset(Dataset):
    """
    Loads precomputed tile embeddings per slide. If `extra_features` is
    provided (a dict pid -> [N, e] tensor), it is concatenated to the
    backbone features, giving [N, d + e] inputs (this is how UITT, sinpe,
    and peripheral priors are injected).
    """
    def __init__(self, pids, labels, feat_dir, extra_features=None):
        self.pids = list(pids)
        self.labels = list(labels)
        self.feat_dir = feat_dir
        self.extra_features = extra_features  # optional dict pid -> tensor

    def __len__(self):
        return len(self.pids)

    def __getitem__(self, i):
        pid = self.pids[i]
        label = self.labels[i]
        d = torch.load(os.path.join(self.feat_dir, f"{pid}.pt"), map_location="cpu")
        feats = d["features"].float()
        if self.extra_features is not None and pid in self.extra_features:
            feats = torch.cat([feats, self.extra_features[pid]], dim=1)
        return feats, torch.tensor(float(label)), pid


def mil_collate(batch):
    """Batch size is 1 (variable tile counts per slide)."""
    feats, label, pid = batch[0]
    return feats, label, pid
