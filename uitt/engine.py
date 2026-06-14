"""
Training and evaluation loops.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    for feats, label, _ in loader:
        feats = feats.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        logit = model(feats)
        loss = criterion(logit.unsqueeze(0), label.unsqueeze(0))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, device):
    """Return (AUC, probs, labels) over the loader."""
    model.eval()
    probs, labels = [], []
    for feats, label, _ in loader:
        feats = feats.to(device)
        prob = torch.sigmoid(model(feats)).item()
        probs.append(prob)
        labels.append(int(label.item()))
    auc = roc_auc_score(labels, probs) if len(set(labels)) > 1 else float("nan")
    return auc, np.array(probs), np.array(labels)
