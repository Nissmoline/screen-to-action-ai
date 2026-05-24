"""Train a CNN policy with supervised imitation learning."""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.config import PROJECT_ROOT, ProfileValidationError, load_profile
from src.dataset.build_dataset import REQUIRED_DATASET_COLUMNS, resolve_dataset_frame_path
from src.input_control.action_space import ActionSpace


DEFAULT_IMAGE_SIZE = 84
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 0.001
NO_OP_PREDICTION_WARNING_THRESHOLD = 0.80


class TrainingError(RuntimeError):
    """Raised when imitation training cannot run."""


@dataclass(frozen=True)
class TrainingArtifacts:
    """Paths and metrics produced by training."""

    model_path: Path
    metrics_csv: Path
    curves_png: Path
    final_train_accuracy: float
    final_val_accuracy: float | None


class ImitationDataset:
    """PyTorch-compatible dataset backed by rows from dataset.csv."""

    def __init__(self, rows: list[dict[str, str]], image_size: int, torch_module: Any) -> None:
        self.rows = rows
        self.image_size = image_size
        self.torch = torch_module

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image = load_image_tensor(row["frame_path"], self.image_size, self.torch)
        target = self.torch.tensor(int(row["action_id"]), dtype=self.torch.long)
        return image, target


def train_imitation(
    profile_name: str,
    epochs: int = 10,
    batch_size: int = DEFAULT_BATCH_SIZE,
    image_size: int = DEFAULT_IMAGE_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    datasets_root: str | Path | None = None,
    models_dir: str | Path | None = None,
    reports_dir: str | Path | None = None,
    device_name: str | None = None,
) -> TrainingArtifacts:
    """Train a CNN policy for a profile from ``dataset.csv``."""
    if epochs <= 0:
        raise TrainingError("epochs must be greater than 0")
    if batch_size <= 0:
        raise TrainingError("batch_size must be greater than 0")

    torch = import_torch()
    from src.models.policy_cnn import PolicyCNN

    profile = load_profile(profile_name)
    action_space = ActionSpace.from_profile(profile)
    num_actions = len(action_space)
    dataset_csv = profile_dataset_csv(profile.profile_name, datasets_root)
    rows = read_dataset_rows(dataset_csv)
    validate_action_ids(rows, num_actions)

    train_rows = filter_split(rows, "train")
    val_rows = filter_split(rows, "val")
    if not train_rows:
        raise TrainingError(f"No train rows found in dataset: {dataset_csv}")

    print_dataset_distribution(train_rows, split_name="train")
    print_dataset_distribution(val_rows, split_name="val")

    device = select_device(torch, device_name)
    model = PolicyCNN(num_actions=num_actions).to(device)
    class_weights = compute_class_weights(train_rows, num_actions, torch).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loader = make_data_loader(train_rows, image_size, batch_size, torch, shuffle=True)
    val_loader = make_data_loader(val_rows, image_size, batch_size, torch, shuffle=False) if val_rows else None

    metrics: list[dict[str, str]] = []
    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            torch,
        )
        if val_loader is not None:
            val_loss, val_accuracy = evaluate_loss_accuracy(model, val_loader, criterion, device, torch)
        else:
            val_loss, val_accuracy = None, None

        metrics.append(
            {
                "epoch": str(epoch),
                "train_loss": f"{train_loss:.6f}",
                "train_accuracy": f"{train_accuracy:.6f}",
                "val_loss": "" if val_loss is None else f"{val_loss:.6f}",
                "val_accuracy": "" if val_accuracy is None else f"{val_accuracy:.6f}",
            }
        )
        val_text = "n/a" if val_accuracy is None else f"{val_accuracy:.3f}"
        print(
            f"epoch {epoch}/{epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.3f} val_acc={val_text}"
        )

    models_path = Path(models_dir) if models_dir is not None else PROJECT_ROOT / "models"
    reports_path = Path(reports_dir) if reports_dir is not None else PROJECT_ROOT / "outputs" / "reports"
    models_path.mkdir(parents=True, exist_ok=True)
    reports_path.mkdir(parents=True, exist_ok=True)

    model_path = models_path / f"{profile.profile_name}_imitation_policy.pt"
    metrics_csv = reports_path / f"{profile.profile_name}_training_metrics.csv"
    curves_png = reports_path / f"{profile.profile_name}_training_curves.png"

    action_id_to_name = {action.id: action.name for action in action_space}
    checkpoint = {
        "state_dict": model.state_dict(),
        "profile_name": profile.profile_name,
        "num_actions": num_actions,
        "image_size": image_size,
        "action_id_to_name": action_id_to_name,
        "action_name_to_id": {name: action_id for action_id, name in action_id_to_name.items()},
    }
    torch.save(checkpoint, model_path)
    write_metrics_csv(metrics_csv, metrics)
    save_training_curves(curves_png, metrics)

    print(f"Model saved to: {model_path}")
    print(f"Metrics saved to: {metrics_csv}")
    print(f"Training curves saved to: {curves_png}")

    return TrainingArtifacts(
        model_path=model_path,
        metrics_csv=metrics_csv,
        curves_png=curves_png,
        final_train_accuracy=float(metrics[-1]["train_accuracy"]),
        final_val_accuracy=None if metrics[-1]["val_accuracy"] == "" else float(metrics[-1]["val_accuracy"]),
    )


def import_torch():
    """Import torch with a clear installation error."""
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise TrainingError(
            "PyTorch is required for imitation training. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc
    return torch


def profile_dataset_csv(profile_name: str, datasets_root: str | Path | None = None) -> Path:
    root = Path(datasets_root) if datasets_root is not None else PROJECT_ROOT / "data" / "datasets"
    return root / profile_name / "dataset.csv"


def read_dataset_rows(path: str | Path) -> list[dict[str, str]]:
    """Read and validate dataset.csv rows."""
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise TrainingError(f"Dataset file does not exist: {dataset_path}")

    with dataset_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        missing = [
            column
            for column in REQUIRED_DATASET_COLUMNS
            if column not in (reader.fieldnames or [])
        ]
        if missing:
            raise TrainingError(f"Dataset is missing required columns: {', '.join(missing)}")
        rows = list(reader)

    if not rows:
        raise TrainingError(f"Dataset has no rows: {dataset_path}")
    return rows


def filter_split(rows: list[dict[str, str]], split: str) -> list[dict[str, str]]:
    return [row for row in rows if row.get("split") == split]


def validate_action_ids(rows: list[dict[str, str]], num_actions: int) -> None:
    """Ensure all dataset action ids fit the profile action space."""
    for row in rows:
        try:
            action_id = int(row["action_id"])
        except ValueError as exc:
            raise TrainingError(f"Invalid action_id in dataset: {row['action_id']!r}") from exc
        if action_id < 0 or action_id >= num_actions:
            raise TrainingError(
                f"action_id {action_id} is outside the profile action range 0..{num_actions - 1}"
            )


def load_image_tensor(frame_path: str, image_size: int, torch_module: Any):
    """Load a frame image as a normalized CHW float tensor."""
    resolved_path = resolve_dataset_frame_path(frame_path)
    if not resolved_path.is_file():
        raise TrainingError(f"Frame path does not exist: {resolved_path}")

    with Image.open(resolved_path) as image:
        rgb = image.convert("RGB").resize((image_size, image_size), Image.BILINEAR)
        array = np.asarray(rgb, dtype=np.float32) / 255.0
    chw = np.transpose(array, (2, 0, 1))
    return torch_module.from_numpy(np.ascontiguousarray(chw))


def compute_class_distribution(rows: list[dict[str, str]]) -> Counter[str]:
    return Counter(row["action_name"] for row in rows)


def print_dataset_distribution(rows: list[dict[str, str]], split_name: str) -> None:
    if not rows:
        print(f"Class distribution for {split_name}: empty")
        return

    distribution = compute_class_distribution(rows)
    print(f"Class distribution for {split_name}:")
    for action_name, count in distribution.most_common():
        percent = (count / len(rows)) * 100
        print(f"  {action_name}: {count} ({percent:.1f}%)")

    no_op_ratio = distribution.get("no_op", 0) / len(rows)
    if no_op_ratio >= NO_OP_PREDICTION_WARNING_THRESHOLD:
        print(
            "WARNING: no_op dominates this split. High accuracy may only mean "
            "the model learned to predict no_op."
        )


def compute_class_weights(rows: list[dict[str, str]], num_actions: int, torch_module: Any):
    """Compute inverse-frequency class weights for CrossEntropyLoss."""
    counts = Counter(int(row["action_id"]) for row in rows)
    total = sum(counts.values())
    weights = []
    for action_id in range(num_actions):
        count = counts.get(action_id, 0)
        if count == 0:
            weights.append(0.0)
        else:
            weights.append(total / (num_actions * count))
    return torch_module.tensor(weights, dtype=torch_module.float32)


def make_data_loader(
    rows: list[dict[str, str]],
    image_size: int,
    batch_size: int,
    torch_module: Any,
    shuffle: bool,
):
    dataset = ImitationDataset(rows, image_size=image_size, torch_module=torch_module)
    return torch_module.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )


def select_device(torch_module: Any, device_name: str | None):
    if device_name is not None:
        return torch_module.device(device_name)
    return torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")


def train_one_epoch(model, data_loader, criterion, optimizer, device, torch_module: Any) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, targets in data_loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == targets).sum().item()
        total += batch_size

    return total_loss / total, correct / total


def evaluate_loss_accuracy(model, data_loader, criterion, device, torch_module: Any) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch_module.no_grad():
        for images, targets in data_loader:
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            loss = criterion(logits, targets)

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            correct += (logits.argmax(dim=1) == targets).sum().item()
            total += batch_size

    return total_loss / total, correct / total


def write_metrics_csv(path: str | Path, metrics: list[dict[str, str]]) -> None:
    metrics_path = Path(path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy"],
        )
        writer.writeheader()
        writer.writerows(metrics)


def save_training_curves(path: str | Path, metrics: list[dict[str, str]]) -> None:
    """Save loss/accuracy curves, using matplotlib when available and Pillow fallback."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        save_training_curves_with_pillow(output_path, metrics)
        return

    epochs = [int(row["epoch"]) for row in metrics]
    train_loss = [float(row["train_loss"]) for row in metrics]
    train_accuracy = [float(row["train_accuracy"]) for row in metrics]
    val_loss = [float(row["val_loss"]) if row["val_loss"] else math.nan for row in metrics]
    val_accuracy = [float(row["val_accuracy"]) if row["val_accuracy"] else math.nan for row in metrics]

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, train_loss, label="train")
    axes[0].plot(epochs, val_loss, label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, train_accuracy, label="train")
    axes[1].plot(epochs, val_accuracy, label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0, 1)
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


def save_training_curves_with_pillow(path: Path, metrics: list[dict[str, str]]) -> None:
    """Small fallback chart writer when matplotlib is not installed."""
    width, height = 900, 360
    image = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((20, 20), "Training metrics", fill=(0, 0, 0), font=font)

    y = 60
    for row in metrics:
        text = (
            f"epoch {row['epoch']}: "
            f"train_loss={row['train_loss']} train_acc={row['train_accuracy']} "
            f"val_loss={row['val_loss'] or 'n/a'} val_acc={row['val_accuracy'] or 'n/a'}"
        )
        draw.text((20, y), text, fill=(0, 0, 0), font=font)
        y += 18
        if y > height - 30:
            break
    image.save(path, format="PNG")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an imitation policy from dataset.csv.")
    parser.add_argument("--profile", required=True, help="Profile name from configs/profiles.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of supervised epochs.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--device", default=None, help="Optional torch device, e.g. cpu or cuda.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        train_imitation(
            profile_name=args.profile,
            epochs=args.epochs,
            batch_size=args.batch_size,
            image_size=args.image_size,
            learning_rate=args.learning_rate,
            device_name=args.device,
        )
    except (ProfileValidationError, TrainingError, ValueError) as exc:
        parser.exit(status=2, message=f"Error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
