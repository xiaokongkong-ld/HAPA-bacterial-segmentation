"""Plot ACC, F1, MOC, Precision, and Recall from a training_metrics.csv file."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PLOTS = (
    ("Acc", "accuracy"),
    ("F1", "f1"),
    ("moc", "moc_legacy"),
    ("Precision", "precision"),
    ("Recall", "recall"),
)


def save_figure_atomic(figure, path: Path, dpi: int = 200) -> None:
    temporary = path.with_name(path.stem + ".partial" + path.suffix)
    figure.savefig(temporary, dpi=dpi)
    temporary.replace(path)


def write_text_atomic(path: Path, value: str) -> None:
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def write_curves(metrics_csv: Path, output_dir: Path) -> None:
    with metrics_csv.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"No epoch records in {metrics_csv}")
    epochs = [int(row["epoch"]) for row in rows]
    marker = "o" if len(epochs) == 1 else None
    output_dir.mkdir(parents=True, exist_ok=True)
    combined, axes = plt.subplots(2, 3, figsize=(11, 6), constrained_layout=True)
    for index, (title, key) in enumerate(PLOTS):
        train = [float(row[f"train_{key}"]) for row in rows]
        test = [float(row[f"test_{key}"]) for row in rows]
        for axis in (axes.flat[index],):
            axis.plot(epochs, train, color="#2f6db3", marker=marker, label="Train")
            axis.plot(epochs, test, color="#b99c6c", marker=marker, label="Test")
            axis.set_xlabel("Epoch")
            axis.set_ylabel(title.upper() if title in ("Acc", "moc") else title)
            axis.legend(frameon=False)
        individual, individual_axis = plt.subplots(figsize=(5, 4), constrained_layout=True)
        individual_axis.plot(epochs, train, color="#2f6db3", marker=marker, label="Train")
        individual_axis.plot(epochs, test, color="#b99c6c", marker=marker, label="Test")
        individual_axis.set(xlabel="Epoch", ylabel=title.upper() if title in ("Acc", "moc") else title)
        individual_axis.legend(frameon=False)
        save_figure_atomic(individual, output_dir / f"{title}_plot.jpg")
        plt.close(individual)
        for split, values in (("train", train), ("test", test)):
            write_text_atomic(
                output_dir / f"{title}_{split}.txt",
                "\n".join(str(value) for value in values) + "\n",
            )
    axes.flat[5].axis("off")
    save_figure_atomic(combined, output_dir / "segmentation_training_curves.png")
    plt.close(combined)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    write_curves(args.metrics_csv, args.output_dir or args.metrics_csv.parent)


if __name__ == "__main__":
    main()
