"""Fine-tune the U-Net on a versioned paired-image dataset."""

import argparse
import csv
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import PairedPatchDataset
from grouping import group_id
from loss import CombinedLoss
from model_v2 import U_Net
from plot_train import update_plots
from training_utils import METRICS, run_epoch, save_atomic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--initial-weights", type=Path)
    source.add_argument("--resume-state", type=Path)
    parser.add_argument("--epoch-offset", type=int, default=0,
                        help="Completed epochs before loading weights-only checkpoint")
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--cosine-min-lr", type=float, default=None,
                        help="Cosine-decay learning rate to this value over the requested run")
    parser.add_argument("--save-weights-every", type=int, default=0,
                        help="Keep an additional model-weight snapshot every N total epochs")
    parser.add_argument("--test-each-epoch", action="store_true",
                        help="Record deterministic test metrics after each training epoch")
    parser.add_argument("--append-existing", action="store_true",
                        help="Continue metrics and plots in the existing output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size) < 1:
        parser.error("epochs and batch-size must be positive")
    if args.cosine_min_lr is not None and not (0 < args.cosine_min_lr <= args.learning_rate):
        parser.error("cosine-min-lr must be positive and <= learning-rate")
    if args.save_weights_every < 0:
        parser.error("save-weights-every must be nonnegative")
    existing_metrics = args.output_dir / "training_metrics.csv"
    if existing_metrics.exists() and not args.append_existing:
        parser.error("Output directory already has metrics; choose a new directory")
    if args.append_existing and not (args.resume_state and existing_metrics.exists()):
        parser.error("append-existing requires resume-state and existing metrics")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    train_data = PairedPatchDataset(args.data_dir, "train", 512, 256, 40,
                                    augment="light")
    test_groups = {group_id(p.name) for p in (args.data_dir / "test").iterdir()
                   if p.is_dir()}
    if train_data.group_ids & test_groups:
        raise ValueError("Train and held-out test image series must be disjoint")
    loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = None
    if args.test_each_epoch:
        test_data = PairedPatchDataset(args.data_dir, "test", 512, 256, 256,
                                       augment=False)
        test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    test_status = "test metrics recorded without optimization" if test_loader is not None else "test remains unused"
    print(f"Device={device}; train={len(train_data)} patches; {test_status}", flush=True)

    model = U_Net().to(device)
    criterion = CombinedLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=1e-4)
    if args.resume_state:
        checkpoint = torch.load(args.resume_state, map_location=device,
                                weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate
        args.epoch_offset = int(checkpoint["epoch"])
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        if device == "cuda" and checkpoint["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng"]])
        np.random.set_state(checkpoint["numpy_rng"])
        random.setstate(checkpoint["python_rng"])
    else:
        model.load_state_dict(torch.load(args.initial_weights, map_location=device,
                                         weights_only=True))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    source_path = args.resume_state or args.initial_weights
    with source_path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    settings = {
        "source_checkpoint_sha256": digest.hexdigest(),
        "source_kind": "full_resume_state" if args.resume_state else "model_weights",
        "epoch_offset": args.epoch_offset,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "cosine_min_lr": args.cosine_min_lr,
        "save_weights_every": args.save_weights_every,
        "seed": args.seed,
        "train_images": len(train_data.images),
        "train_patches_per_epoch": len(train_data),
        "validation_during_training": False,
        "test_during_training": args.test_each_epoch,
        "test_used_for_checkpoint_selection": False,
        "augmentation": "90-degree rotations and horizontal flips",
        "sampling": "49 random full-image crops per image per epoch",
    }
    settings_path = (args.output_dir / f"run_config_resume_epoch{args.epoch_offset}.json"
                     if args.append_existing else args.output_dir / "run_config.json")
    settings_path.write_text(
        json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    path = args.output_dir / "training_metrics.csv"
    fields = ["epoch", "learning_rate"] + ["train_" + name for name in ("loss", *METRICS)]
    if test_loader is not None:
        fields += ["test_" + name for name in ("loss", *METRICS)]
    if args.append_existing:
        with path.open(newline="", encoding="utf-8") as old_stream:
            old_reader = csv.DictReader(old_stream)
            if old_reader.fieldnames != fields:
                raise ValueError("Existing metrics columns differ from current run")
            old_rows = list(old_reader)
        if not old_rows or int(old_rows[-1]["epoch"]) != args.epoch_offset:
            raise ValueError("Existing metrics end at a different epoch than resume-state")
    with path.open("a" if args.append_existing else "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if not args.append_existing:
            writer.writeheader()
        for step in range(1, args.epochs + 1):
            epoch = args.epoch_offset + step
            if args.cosine_min_lr is not None:
                fraction = (step - 1) / max(args.epochs - 1, 1)
                current_lr = args.cosine_min_lr + 0.5 * (args.learning_rate - args.cosine_min_lr) * (1 + math.cos(math.pi * fraction))
                for group in optimizer.param_groups:
                    group["lr"] = current_lr
            else:
                current_lr = args.learning_rate
            scores = run_epoch(model, loader, criterion, device, optimizer)
            test_scores = run_epoch(model, test_loader, criterion, device) if test_loader is not None else None
            row = {"epoch": epoch, "learning_rate": current_lr,
                   **{"train_" + key: value for key, value in scores.items()}}
            if test_scores is not None:
                row.update({"test_" + key: value for key, value in test_scores.items()})
            writer.writerow(row)
            stream.flush()
            save_atomic(model.state_dict(), args.output_dir / "checkpoint_latest.pth")
            if args.save_weights_every and epoch % args.save_weights_every == 0:
                save_atomic(model.state_dict(), args.output_dir / f"checkpoint_epoch{epoch:04d}.pth")
            save_atomic({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "numpy_rng": np.random.get_state(),
                "python_rng": random.getstate(),
            }, args.output_dir / "resume_state.pth")
            update_plots(path, args.output_dir)
            (args.output_dir / "latest_epoch.txt").write_text(
                f"{epoch}\n", encoding="utf-8")
            test_message = f"; test F1={test_scores['f1']:.4f}" if test_scores is not None else ""
            print(f"Epoch {epoch} (step {step}/{args.epochs}): train F1={scores['f1']:.4f}{test_message}", flush=True)
    save_atomic(model.state_dict(), args.output_dir / "final_model.pth")


if __name__ == "__main__":
    main()
