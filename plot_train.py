"""Update five training-only diagnostic curves after each epoch."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PLOTS = (("Acc", "accuracy"), ("F1", "f1"), ("moc", "moc_legacy"),
         ("Precision", "precision"), ("Recall", "recall"))


def update_plots(csv_path: Path, output_dir: Path) -> None:
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    epochs = [int(row["epoch"]) for row in rows]
    combined, axes = plt.subplots(2, 3, figsize=(11, 6), constrained_layout=True)
    for index, (label, key) in enumerate(PLOTS):
        values = [float(row["train_" + key]) for row in rows]
        test_values = ([float(row["test_" + key]) for row in rows]
                       if rows and "test_" + key in rows[0] else None)
        train_percent = [value * 100 for value in values]
        test_percent = [value * 100 for value in test_values] if test_values is not None else None
        display_label = label.upper() if label in ("Acc", "moc") else label
        axis = axes.flat[index]
        axis.plot(epochs, train_percent, color="#2f6db3", label="Train")
        if test_percent is not None:
            axis.plot(epochs, test_percent, color="#b59b6a", label="Test")
            axis.legend(loc="best", frameon=False)
        axis.set(xlabel="Epoch", ylabel=f"{display_label} (%)", ylim=(0, 100), yticks=range(0, 101, 10))
        axis.grid(axis="y", color="#d8dce2", linewidth=0.6)
        single, single_axis = plt.subplots(figsize=(5, 4), constrained_layout=True)
        single_axis.plot(epochs, train_percent, color="#2f6db3", label="Train")
        if test_percent is not None:
            single_axis.plot(epochs, test_percent, color="#b59b6a", label="Test")
            single_axis.legend(loc="best", frameon=False)
        single_axis.set(xlabel="Epoch", ylabel=f"{display_label} (%)", ylim=(0, 100), yticks=range(0, 101, 10))
        single_axis.grid(axis="y", color="#d8dce2", linewidth=0.6)
        figure_path = output_dir / f"{label}_plot.jpg"
        temporary = figure_path.with_name(figure_path.stem + ".partial.jpg")
        single.savefig(temporary, dpi=150)
        temporary.replace(figure_path)
        plt.close(single)
        text_path = output_dir / f"{label}_train.txt"
        temporary = text_path.with_name(text_path.name + ".partial")
        temporary.write_text("\n".join(map(str, values)) + "\n", encoding="utf-8")
        temporary.replace(text_path)
        if test_values is not None:
            test_path = output_dir / f"{label}_test.txt"
            test_temporary = test_path.with_name(test_path.name + ".partial")
            test_temporary.write_text("\n".join(map(str, test_values)) + "\n", encoding="utf-8")
            test_temporary.replace(test_path)
    axes.flat[5].axis("off")
    path = output_dir / "training_curves.png"
    temporary = path.with_name("training_curves.partial.png")
    combined.savefig(temporary, dpi=150)
    temporary.replace(path)
    plt.close(combined)
