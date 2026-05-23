"""Training and evaluation helper tests."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from src.training.evaluate_imitation import format_confusion_matrix
from src.training.train_imitation import (
    compute_class_distribution,
    save_training_curves,
    train_imitation,
    write_metrics_csv,
)


def test_training_metrics_and_curves_are_written(tmp_path: Path) -> None:
    metrics = [
        {
            "epoch": "1",
            "train_loss": "1.000000",
            "train_accuracy": "0.500000",
            "val_loss": "1.200000",
            "val_accuracy": "0.250000",
        },
        {
            "epoch": "2",
            "train_loss": "0.500000",
            "train_accuracy": "0.750000",
            "val_loss": "0.900000",
            "val_accuracy": "0.500000",
        },
    ]
    metrics_csv = tmp_path / "metrics.csv"
    curves_png = tmp_path / "curves.png"

    write_metrics_csv(metrics_csv, metrics)
    save_training_curves(curves_png, metrics)

    assert metrics_csv.is_file()
    assert curves_png.is_file()


def test_class_distribution_counts_actions() -> None:
    rows = [
        {"action_name": "no_op"},
        {"action_name": "no_op"},
        {"action_name": "press_f1"},
    ]

    distribution = compute_class_distribution(rows)

    assert distribution["no_op"] == 2
    assert distribution["press_f1"] == 1


def test_confusion_matrix_format_contains_action_names() -> None:
    formatted = format_confusion_matrix(
        confusion=[[2, 1], [0, 3]],
        action_id_to_name={0: "no_op", 1: "press_f1"},
    )

    assert "true\\pred" in formatted
    assert "no_op" in formatted
    assert "press_f1" in formatted


def test_training_cli_help_runs_without_torch_import() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "src.training.train_imitation", "--help"],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--epochs" in result.stdout


def test_evaluation_cli_help_runs_without_torch_import() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "src.training.evaluate_imitation", "--help"],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--profile" in result.stdout


def test_policy_training_and_evaluation_smoke_if_torch_available(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from src.training.evaluate_imitation import evaluate_imitation

    datasets_root = _create_tiny_dataset(tmp_path)
    models_dir = tmp_path / "models"
    reports_dir = tmp_path / "reports"

    artifacts = train_imitation(
        profile_name="lineage2_private",
        epochs=1,
        batch_size=2,
        image_size=32,
        datasets_root=datasets_root,
        models_dir=models_dir,
        reports_dir=reports_dir,
        device_name="cpu",
    )
    result = evaluate_imitation(
        profile_name="lineage2_private",
        batch_size=2,
        datasets_root=datasets_root,
        models_dir=models_dir,
        reports_dir=reports_dir,
        device_name="cpu",
    )

    assert artifacts.model_path.is_file()
    assert artifacts.metrics_csv.is_file()
    assert artifacts.curves_png.is_file()
    assert result.report_path.is_file()


def _create_tiny_dataset(tmp_path: Path) -> Path:
    profile_name = "lineage2_private"
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir(parents=True)
    dataset_dir = tmp_path / "datasets" / profile_name
    dataset_dir.mkdir(parents=True)
    dataset_csv = dataset_dir / "dataset.csv"

    splits = ["train"] * 8 + ["val"] * 2 + ["test"] * 2
    with dataset_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["frame_path", "action_id", "action_name", "timestamp", "episode", "split"],
        )
        writer.writeheader()
        for index, split in enumerate(splits):
            image_path = frames_dir / f"{index:06d}.png"
            color = (30 + index * 10, 80, 120) if index % 2 == 0 else (120, 80, 30)
            Image.new("RGB", (40, 40), color=color).save(image_path)
            action_id = 0 if index % 2 == 0 else 1
            writer.writerow(
                {
                    "frame_path": str(image_path),
                    "action_id": str(action_id),
                    "action_name": "no_op" if action_id == 0 else "press_f1",
                    "timestamp": f"{index / 10:.6f}",
                    "episode": "tiny_demo",
                    "split": split,
                }
            )

    return tmp_path / "datasets"
