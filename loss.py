"""Combined segmentation loss from the historical training script."""

import torch
from torch import nn


class CombinedLoss(nn.Module):
    def __init__(self, weight_dice: float = 0.33, weight_bce: float = 0.33, weight_iou: float = 0.33):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.weight_dice = weight_dice
        self.weight_bce = weight_bce
        self.weight_iou = weight_iou

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probability = torch.sigmoid(logits)
        intersection = (probability * targets).sum()
        total = probability.sum() + targets.sum()
        union = total - intersection
        dice = 1 - (2 * intersection + 1e-6) / (total + 1e-6)
        iou = 1 - (intersection + 1e-6) / (union + 1e-6)
        return self.weight_bce * self.bce(logits, targets) + self.weight_dice * dice + self.weight_iou * iou
