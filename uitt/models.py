"""
MIL aggregators used in the paper.

All models take tile features of shape [N, in_dim] and produce a single
slide-level logit. For UITT runs, in_dim = backbone_dim + 10
(e.g. 1536 + 10 = 1546 for UNI2-h). For baselines, in_dim = backbone_dim.
"""
import torch
import torch.nn as nn


class ABMIL(nn.Module):
    """Attention-based MIL (Ilse et al., 2018), gated attention pooling."""
    def __init__(self, in_dim, hidden=256):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        A = self.attention(x)
        A = torch.softmax(A, dim=0)
        z = (A * x).sum(dim=0)
        return self.classifier(z).squeeze(-1)


class CLAM_SB(nn.Module):
    """CLAM single-branch (Lu et al., 2021)."""
    def __init__(self, in_dim, hidden=256):
        super().__init__()
        self.feat_proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.25),
        )
        self.attention_V = nn.Sequential(nn.Linear(hidden, 128), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden, 128), nn.Sigmoid())
        self.attention_w = nn.Linear(128, 1)
        self.classifier = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.feat_proj(x)
        A = self.attention_w(self.attention_V(h) * self.attention_U(h))
        A = torch.softmax(A, dim=0)
        z = (A * h).sum(dim=0)
        return self.classifier(z).squeeze(-1)


class TransMIL(nn.Module):
    """TransMIL (Shao et al., 2021), inter-tile self-attention with a CLS token."""
    def __init__(self, in_dim, hidden=256, nhead=8, num_layers=2):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=nhead,
            dim_feedforward=hidden * 2,
            dropout=0.1, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden))
        self.classifier = nn.Linear(hidden, 1)

    def forward(self, x):
        x = self.proj(x).unsqueeze(0)            # [1, N, hidden]
        cls = self.cls_token.expand(1, -1, -1)   # [1, 1, hidden]
        x = torch.cat([cls, x], dim=1)           # [1, N+1, hidden]
        x = self.transformer(x)
        return self.classifier(x[0, 0]).squeeze(-1)


class MeanPool(nn.Module):
    """Permutation-invariant mean pooling baseline."""
    def __init__(self, in_dim, hidden=256):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.classifier(x.mean(dim=0)).squeeze(-1)


class MaxPool(nn.Module):
    """Permutation-invariant max pooling baseline."""
    def __init__(self, in_dim, hidden=256):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.classifier(x.max(dim=0).values).squeeze(-1)


def get_model(agg, in_dim):
    """Dispatch an aggregator by name."""
    if agg == "ABMIL":    return ABMIL(in_dim)
    if agg == "CLAM_SB":  return CLAM_SB(in_dim)
    if agg == "TransMIL": return TransMIL(in_dim)
    if agg == "MeanPool": return MeanPool(in_dim)
    if agg == "MaxPool":  return MaxPool(in_dim)
    raise ValueError(f"Unknown aggregator: {agg}")
