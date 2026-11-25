"""Binary pixel metrics for bacterial segmentation."""

import math

import torch


class SegmentationMetrics:
    def __init__(self, logit_threshold: float = 0.5) -> None:
        self.threshold = logit_threshold
        self.tp = self.fp = self.fn = self.tn = 0
        self.min_overlap_sum = 0.0
        self.patch_count = 0

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        predictions = logits.detach() > self.threshold
        truth = targets.detach() > 0.5
        self.tp += int((predictions & truth).sum().item())
        self.fp += int((predictions & ~truth).sum().item())
        self.fn += int((~predictions & truth).sum().item())
        self.tn += int((~predictions & ~truth).sum().item())

        intersections = (predictions & truth).sum(dim=(1, 2, 3)).cpu().tolist()
        pred_areas = predictions.sum(dim=(1, 2, 3)).cpu().tolist()
        true_areas = truth.sum(dim=(1, 2, 3)).cpu().tolist()
        for intersection, pred_area, true_area in zip(intersections, pred_areas, true_areas):
            denominator = min(pred_area, true_area)
            self.min_overlap_sum += intersection / denominator if denominator else float(
                pred_area == true_area
            )
            self.patch_count += 1

    def as_dict(self) -> dict[str, float]:
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
        return {
            "accuracy": (tp + tn) / max(tp + fp + fn + tn, 1),
            "precision": tp / max(tp + fp, 1),
            "recall": tp / max(tp + fn, 1),
            "f1": 2 * tp / max(2 * tp + fp + fn, 1),
            "moc_legacy": tp / max(math.sqrt((tp + fp) * (tp + fn)), 1),
            "moc_min_area_patch": self.min_overlap_sum / max(self.patch_count, 1),
        }
