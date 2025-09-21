import torch
import torch.nn as nn
import torch.nn.functional as F

class CosineHead(nn.Module):
    """
    Cosine classifier against a set of normalized prototypes.
    """
    def __init__(self, prototypes: torch.Tensor):
        super().__init__()
        self.register_buffer("proto", F.normalize(prototypes, dim=-1))

    def forward(self, feats: torch.Tensor):
        feats = F.normalize(feats, dim=-1)
        return feats @ self.proto.t()
