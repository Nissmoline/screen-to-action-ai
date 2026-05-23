"""Evaluate a trained imitation policy on the test split."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config import PROJECT_ROOT, ProfileValidationError, load_profile
from src.input_control.action_space import ActionSpace
from src.training.train_imitation import (
    DEFAULT_BATCH_SIZE,
    ImitationDataset,
    TrainingError,
    filter_split,
    import_torch,
    profile_dataset_csv,
    read_dataset_rows,
    validate_action_ids,
)


NO_OP_PREDICTION_WARNING_THRESHOLD = 0.80


class EvaluationError(RuntimeError):
    """Raised when a trained policy cannot be evaluated."""


@dataclass(frozen=True)
class EvaluationResult:
    """Summary of one evaluation run."""

    accuracy: float
    top3_accuracy: float
    report_path: Path


def evaluate_imitation(
    profile_name: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    datasets_root: str | Path | None = None,
    models_dir: str | Path | None = None,
    reports_dir: str | Path | None = None,
    device_name: str | None = None,
) -> EvaluationResult:
    """Evaluate a saved imitation policy on the test split."""
    torch = import_torch()
    from src.models.policy_cnn import PolicyCNN

    profile = load_profile(profile_name)
    action_space = ActionSpace.from_profile(profile)
    num_actions = len(action_space)

    dataset_csv = profile_dataset_csv(profile.profile_name, datasets_root)
    rows = read_dataset_rows(dataset_csv)
    validate_action_ids(rows, num_actions)
    test_rows = filter_split(rows, "test")
    if not test_rows:
        raise EvaluationError(f"No test rows found in dataset: {dataset_csv}")

    model_path = profile_model_path(profile.profile_name, models_dir)
    if not model_path.is_file():
        raise EvaluationError(f"Model file does not exist: {model_path}")

    device = select_device(torch, device_name)
    checkpoint = torch.load(model_path, map_location=device)
    image_size = int(checkpoint.get("image_size", 84))
    checkpoint_num_actions = int(checkpoint.get("num_actions", num_actions))
    if checkpoint_num_actions != num_actions:
        raise EvaluationError(
            f"Model num_actions={checkpoint_num_actions} does not match profile num_actions={num_actions}"
        )

    action_id_to_name = normalize_action_map(
        checkpoint.get("action_id_to_name"),
        fallback={action.id: action.name for action in action_space},
    )

    model = PolicyCNN(num_actions=num_actions).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    data_loader = torch.utils.data.DataLoader(
        ImitationDataset(test_rows, image_size=image_size, torch_module=torch),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    eval_stats = run_evaluation_loop(
        model=model,
        data_loader=data_loader,
        num_actions=num_actions,
        action_id_to_name=action_id_to_name,
        device=device,
        torch_module=torch,
        source_rows=test_rows,
    )

    report_path = profile_report_path(profile.profile_name, reports_dir)
    report_text = build_evaluation_report(
        profile_name=profile.profile_name,
        model_path=model_path,
        dataset_csv=dataset_csv,
        action_id_to_name=action_id_to_name,
        stats=eval_stats,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text, encoding="utf-8")

    print(report_text)
    print(f"Evaluation report saved to: {report_path}")

    return EvaluationResult(
        accuracy=eval_stats["accuracy"],
        top3_accuracy=eval_stats["top3_accuracy"],
        report_path=report_path,
    )


def run_evaluation_loop(
    model,
    data_loader,
    num_actions: int,
    action_id_to_name: dict[int, str],
    device,
    torch_module: Any,
    source_rows: list[dict[str, str]],
) -> dict[str, Any]:
    """Run model evaluation and collect confusion matrix plus top-3 examples."""
    confusion = [[0 for _ in range(num_actions)] for _ in range(num_actions)]
    prediction_counts: Counter[int] = Counter()
    correct = 0
    top3_correct = 0
    total = 0
    top3_examples: list[dict[str, str]] = []
    row_offset = 0

    with torch_module.no_grad():
        for images, targets in data_loader:
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            probabilities = torch_module.softmax(logits, dim=1)
            predictions = probabilities.argmax(dim=1)
            k = min(3, num_actions)
            top_values, top_indices = torch_module.topk(probabilities, k=k, dim=1)

            batch_size = targets.size(0)
            for batch_index in range(batch_size):
                true_id = int(targets[batch_index].item())
                predicted_id = int(predictions[batch_index].item())
                confusion[true_id][predicted_id] += 1
                prediction_counts[predicted_id] += 1
                correct += int(predicted_id == true_id)

                top_ids = [int(value.item()) for value in top_indices[batch_index]]
                top_probs = [float(value.item()) for value in top_values[batch_index]]
                top3_correct += int(true_id in top_ids)

                if len(top3_examples) < 20:
                    source_row = source_rows[row_offset + batch_index]
                    top3_examples.append(
                        {
                            "frame_path": source_row["frame_path"],
                            "true": action_id_to_name[true_id],
                            "top3": ", ".join(
                                f"{action_id_to_name[action_id]}:{probability:.3f}"
                                for action_id, probability in zip(top_ids, top_probs)
                            ),
                        }
                    )

            total += batch_size
            row_offset += batch_size

    return {
        "accuracy": correct / total if total else 0.0,
        "top3_accuracy": top3_correct / total if total else 0.0,
        "confusion": confusion,
        "prediction_counts": prediction_counts,
        "total": total,
        "top3_examples": top3_examples,
    }


def build_evaluation_report(
    profile_name: str,
    model_path: Path,
    dataset_csv: Path,
    action_id_to_name: dict[int, str],
    stats: dict[str, Any],
) -> str:
    """Build a text evaluation report."""
    lines = [
        f"Profile: {profile_name}",
        f"Model: {model_path}",
        f"Dataset: {dataset_csv}",
        f"Test samples: {stats['total']}",
        f"Accuracy: {stats['accuracy']:.4f}",
        f"Top-3 accuracy: {stats['top3_accuracy']:.4f}",
        "",
        "Prediction distribution:",
    ]

    prediction_counts: Counter[int] = stats["prediction_counts"]
    for action_id, action_name in action_id_to_name.items():
        count = prediction_counts.get(action_id, 0)
        percent = count / stats["total"] * 100 if stats["total"] else 0
        lines.append(f"  {action_name}: {count} ({percent:.1f}%)")

    no_op_id = next((action_id for action_id, name in action_id_to_name.items() if name == "no_op"), None)
    if no_op_id is not None and stats["total"]:
        no_op_ratio = prediction_counts.get(no_op_id, 0) / stats["total"]
        if no_op_ratio >= NO_OP_PREDICTION_WARNING_THRESHOLD:
            lines.append(
                "WARNING: model predicts no_op for most test samples; accuracy may be misleading."
            )

    lines.extend(["", "Confusion matrix:", format_confusion_matrix(stats["confusion"], action_id_to_name)])
    lines.extend(["", "Top-3 prediction examples:"])
    for example in stats["top3_examples"]:
        lines.append(
            f"  true={example['true']} top3=[{example['top3']}] frame={example['frame_path']}"
        )

    return "\n".join(lines) + "\n"


def format_confusion_matrix(confusion: list[list[int]], action_id_to_name: dict[int, str]) -> str:
    """Format a compact text confusion matrix."""
    header = ["true\\pred"] + [action_id_to_name[index] for index in range(len(confusion))]
    rows = ["\t".join(header)]
    for true_id, row in enumerate(confusion):
        rows.append("\t".join([action_id_to_name[true_id]] + [str(value) for value in row]))
    return "\n".join(rows)


def normalize_action_map(raw_map: object, fallback: dict[int, str]) -> dict[int, str]:
    """Normalize checkpoint action maps that may have string keys after serialization."""
    if not isinstance(raw_map, dict):
        return fallback
    normalized: dict[int, str] = {}
    for key, value in raw_map.items():
        try:
            normalized[int(key)] = str(value)
        except (TypeError, ValueError):
            continue
    return normalized or fallback


def select_device(torch_module: Any, device_name: str | None):
    if device_name is not None:
        return torch_module.device(device_name)
    return torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")


def profile_model_path(profile_name: str, models_dir: str | Path | None = None) -> Path:
    root = Path(models_dir) if models_dir is not None else PROJECT_ROOT / "models"
    return root / f"{profile_name}_imitation_policy.pt"


def profile_report_path(profile_name: str, reports_dir: str | Path | None = None) -> Path:
    root = Path(reports_dir) if reports_dir is not None else PROJECT_ROOT / "outputs" / "reports"
    return root / f"{profile_name}_evaluation_report.txt"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate an imitation policy on the test split.")
    parser.add_argument("--profile", required=True, help="Profile name from configs/profiles.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--device", default=None, help="Optional torch device, e.g. cpu or cuda.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        evaluate_imitation(
            profile_name=args.profile,
            batch_size=args.batch_size,
            device_name=args.device,
        )
    except (EvaluationError, ProfileValidationError, TrainingError, ValueError) as exc:
        parser.exit(status=2, message=f"Error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
