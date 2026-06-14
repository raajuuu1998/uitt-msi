"""
UITT: Universal Immune-Tumor Topology
Biologically-grounded spatial priors for zero-shot cross-cancer MSI prediction.
"""

__version__ = "1.0.0"

from .models import ABMIL, CLAM_SB, TransMIL, MeanPool, MaxPool, get_model
from .uitt_transform import (
    build_immune_classifier,
    get_immune_scores,
    compute_uitt_10dim,
    sinusoidal_encoding,
    peripheral_encoding,
)
from .data import MILDataset, mil_collate

__all__ = [
    "ABMIL", "CLAM_SB", "TransMIL", "MeanPool", "MaxPool", "get_model",
    "build_immune_classifier", "get_immune_scores", "compute_uitt_10dim",
    "sinusoidal_encoding", "peripheral_encoding",
    "MILDataset", "mil_collate",
]
