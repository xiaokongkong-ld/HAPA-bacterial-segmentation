"""Train the HAPA bacterial U-Net and save five metric curves."""

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import PairedPatchDataset
from loss import CombinedLoss
from metrics import SegmentationMetrics
from model import U_Net
from plot_curves import write_curves


METRICS = ("accuracy", "f1", "moc_legacy", "precision", "recall", "moc_min_area_patch")


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


def save_checkpoint(path, model, optimizer, epoch, best_f1, best_epoch, settings):
    save_atomic(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_f1": best_f1,
            "best_epoch": best_epoch,
            "settings": settings,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing train/ and test/")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/current"))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--train-crop", type=int, default=512)
    parser.add_argument("--test-crop", type=int, default=512)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--train-stride", type=int, default=40)
    parser.add_argument("--test-stride", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--resume", action="store_true", help="Resume from output-dir/checkpoint_latest.pth")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.checkpoint_every < 1:
        parser.error("epochs, batch-size, and checkpoint-every must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    train_data = PairedPatchDataset(
        args.data_dir, "train", args.train_crop, args.patch_size, args.train_stride
    )
    test_data = PairedPatchDataset(
        args.data_dir, "test", args.test_crop, args.patch_size, args.test_stride
    )
    overlapping_groups = train_data.group_ids & test_data.group_ids
    if overlapping_groups:
        raise ValueError(
            "Train/test image series overlap: "
            + ", ".join(sorted(overlapping_groups))
            + ". Create a group-disjoint split before training."
        )
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    print(f"Device: {device}; train patches: {len(train_data)}; test patches: {len(test_data)}")

    model = U_Net(in_channels=1, out_channels=1).to(device)
    criterion = CombinedLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "training_metrics.csv"
    checkpoint_path = args.output_dir / "checkpoint_latest.pth"
    settings = {
        "data_dir": str(args.data_dir.resolve()),
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "train_crop": args.train_crop,
        "test_crop": args.test_crop,
        "patch_size": args.patch_size,
        "train_stride": args.train_stride,
        "test_stride": args.test_stride,
        "seed": args.seed,
        "train_names": train_data.image_names,
        "test_names": test_data.image_names,
    }
    fields = ["epoch"] + [f"{split}_{name}" for split in ("train", "test") for name in ("loss", *METRICS)]
    first_epoch = 1
    best_f1 = -1.0
    best_epoch = 0
    previous_rows = []
    if args.resume:
        if not checkpoint_path.is_file() or not metrics_path.is_file():
            parser.error("Resume requires checkpoint_latest.pth and training_metrics.csv")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if checkpoint["settings"] != settings:
            parser.error("Current data or training settings differ from the checkpoint")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        if torch.cuda.is_available() and checkpoint["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng"]])
        np.random.set_state(checkpoint["numpy_rng"])
        random.setstate(checkpoint["python_rng"])
        first_epoch = int(checkpoint["epoch"]) + 1
        best_f1 = float(checkpoint["best_f1"])
        best_epoch = int(checkpoint["best_epoch"])
        with metrics_path.open(newline="", encoding="utf-8") as stream:
            previous_rows = [row for row in csv.DictReader(stream) if int(row["epoch"]) < first_epoch]
        if len(previous_rows) != first_epoch - 1:
            parser.error("Metrics CSV is shorter than the saved checkpoint")
        print(f"Resuming after epoch {first_epoch - 1}; best validation F1={best_f1:.4f} at epoch {best_epoch}")
    elif metrics_path.exists() or checkpoint_path.exists():
        parser.error("Output directory contains a previous run; use --resume or a new --output-dir")

    with metrics_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(previous_rows)
        stream.flush()
        for epoch in range(first_epoch, args.epochs + 1):
            train_scores = run_epoch(model, train_loader, criterion, device, optimizer)
            test_scores = run_epoch(model, test_loader, criterion, device)
            row = {"epoch": epoch}
            row.update({f"train_{name}": value for name, value in train_scores.items()})
            row.update({f"test_{name}": value for name, value in test_scores.items()})
            writer.writerow(row)
            stream.flush()
            if test_scores["f1"] > best_f1:
                best_f1 = test_scores["f1"]
                best_epoch = epoch
                save_atomic(model.state_dict(), args.output_dir / "trained_best_validation.pth")
            if epoch % args.checkpoint_every == 0 or epoch == args.epochs:
                save_checkpoint(checkpoint_path, model, optimizer, epoch, best_f1, best_epoch, settings)
            write_curves(metrics_path, args.output_dir)
            (args.output_dir / "latest_epoch.txt").write_text(f"{epoch}\n", encoding="utf-8")
            print(f"Epoch {epoch}/{args.epochs}: train F1={train_scores['f1']:.4f}; test F1={test_scores['f1']:.4f}; best={best_f1:.4f} (epoch {best_epoch})")

    save_atomic(model.state_dict(), args.output_dir / "trained_final.pth")
    write_curves(metrics_path, args.output_dir)
    print(f"Metrics, five JPG plots, and combined PNG saved in {args.output_dir}")


if __name__ == "__main__":
    main()
