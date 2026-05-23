"""Plot dry-run predicted action distribution."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from src.analysis.analyze_predictions import PredictionAnalysisError, read_prediction_log
from src.config import PROJECT_ROOT, ProfileValidationError, load_profile
from src.inference.dry_run_agent import prediction_log_path


def plot_action_distribution(
    profile_name: str,
    logs_dir: str | Path | None = None,
    reports_dir: str | Path | None = None,
) -> Path:
    """Create a PNG bar chart of dry-run predicted action distribution."""
    profile = load_profile(profile_name)
    log_path = prediction_log_path(profile.profile_name, logs_dir=logs_dir)
    rows = read_prediction_log(log_path)
    counts = Counter(
        row.get("effective_action_name") or row.get("predicted_action_name", "")
        for row in rows
    )

    output_path = action_distribution_plot_path(profile.profile_name, reports_dir=reports_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_action_distribution_plot(counts, output_path, title=f"{profile.profile_name} action distribution")
    print(f"Action distribution plot saved to: {output_path}")
    return output_path


def save_action_distribution_plot(
    counts: Counter[str],
    output_path: str | Path,
    title: str,
) -> None:
    """Save action counts as PNG, using matplotlib when available and Pillow fallback."""
    if not counts:
        raise PredictionAnalysisError("No actions available to plot")

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        save_action_distribution_with_pillow(counts, Path(output_path), title)
        return

    labels = list(counts.keys())
    values = [counts[label] for label in labels]
    figure_width = max(8, len(labels) * 1.2)
    figure, axis = plt.subplots(figsize=(figure_width, 4.5))
    axis.bar(labels, values, color="#3867d6")
    axis.set_title(title)
    axis.set_ylabel("count")
    axis.tick_params(axis="x", rotation=35)
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


def save_action_distribution_with_pillow(
    counts: Counter[str],
    output_path: Path,
    title: str,
) -> None:
    """Small fallback bar chart renderer."""
    items = counts.most_common()
    width = max(720, 140 * len(items))
    height = 420
    margin_left = 70
    margin_bottom = 90
    margin_top = 60
    chart_width = width - margin_left - 30
    chart_height = height - margin_top - margin_bottom
    max_count = max(counts.values())

    image = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((20, 20), title, fill=(0, 0, 0), font=font)
    draw.line((margin_left, margin_top, margin_left, margin_top + chart_height), fill=(0, 0, 0))
    draw.line(
        (margin_left, margin_top + chart_height, width - 20, margin_top + chart_height),
        fill=(0, 0, 0),
    )

    bar_gap = 16
    bar_width = max(24, (chart_width - bar_gap * (len(items) + 1)) // max(1, len(items)))
    x = margin_left + bar_gap
    for label, count in items:
        bar_height = int((count / max_count) * chart_height) if max_count else 0
        y0 = margin_top + chart_height - bar_height
        y1 = margin_top + chart_height
        draw.rectangle((x, y0, x + bar_width, y1), fill=(56, 103, 214))
        draw.text((x, y0 - 16), str(count), fill=(0, 0, 0), font=font)
        draw.text((x, y1 + 8), _short_label(label), fill=(0, 0, 0), font=font)
        x += bar_width + bar_gap

    image.save(output_path, format="PNG")


def action_distribution_plot_path(profile_name: str, reports_dir: str | Path | None = None) -> Path:
    root = Path(reports_dir) if reports_dir is not None else PROJECT_ROOT / "outputs" / "reports"
    return root / f"{profile_name}_action_distribution.png"


def _short_label(label: str, max_chars: int = 16) -> str:
    if len(label) <= max_chars:
        return label
    return f"{label[: max_chars - 3]}..."


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot dry-run action distribution.")
    parser.add_argument("--profile", required=True, help="Profile name from configs/profiles.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        plot_action_distribution(profile_name=args.profile)
    except (PredictionAnalysisError, ProfileValidationError, ValueError) as exc:
        parser.exit(status=2, message=f"Error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
