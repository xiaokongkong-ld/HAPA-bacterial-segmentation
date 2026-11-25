"""Shared epoch and checkpoint helpers for the v2 experiment."""

import torch

from metrics import SegmentationMetrics


METRICS = ("accuracy", "f1", "moc_legacy", "precision", "recall",
           "moc_min_area_patch")


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    metrics = SegmentationMetrics(logit_threshold=0.5)
    loss_total = 0.0
    sample_total = 0
    with torch.set_grad_enabled(training):
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, masks)
            if training:
                loss.backward()
                optimizer.step()
            metrics.update(logits, masks)
            loss_total += float(loss.item()) * len(images)
            sample_total += len(images)
    return {"loss": loss_total / sample_total, **metrics.as_dict()}


def save_atomic(value, path):
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)
