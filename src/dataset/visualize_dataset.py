"""Create visual previews of dataset samples."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from src.config import PROJECT_ROOT
from src.dataset.build_dataset import REQUIRED_DATASET_COLUMNS, resolve_dataset_frame_path


class DatasetVisualizationError(RuntimeError):
    """Raised when a dataset preview cannot be generated."""


def visualize_dataset(
    profile_name: str,
    samples: int = 20,
    dataset_csv: str | Path | None = None,
    output_path: str | Path | None = None,
) -> Path:
    """Save a grid preview image for a profile dataset."""
    dataset_path = (
        Path(dataset_csv)
        if dataset_csv is not None
        else PROJECT_ROOT / "data" / "datasets" / profile_name / "dataset.csv"
    )
    preview_path = (
        Path(output_path)
        if output_path is not None
        else PROJECT_ROOT / "outputs" / "debug" / f"{profile_name}_dataset_samples.png"
    )

    rows = read_dataset_rows(dataset_path)
    selected_rows = select_sample_rows(rows, samples)
    save_grid_preview(selected_rows, preview_path)
    print(f"Dataset preview written to: {preview_path}")
    return preview_path


def read_dataset_rows(path: str | Path) -> list[dict[str, str]]:
    """Read dataset.csv rows."""
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise DatasetVisualizationError(f"Dataset file does not exist: {dataset_path}")

    with dataset_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        missing_columns = [
            column
            for column in REQUIRED_DATASET_COLUMNS
            if column not in (reader.fieldnames or [])
        ]
        if missing_columns:
            raise DatasetVisualizationError(
                f"Dataset file is missing required columns: {', '.join(missing_columns)}"
            )
        return list(reader)


def select_sample_rows(rows: list[dict[str, str]], samples: int) -> list[dict[str, str]]:
    """Select evenly spaced rows for preview."""
    if samples <= 0:
        raise DatasetVisualizationError("samples must be greater than 0")
    if not rows:
        raise DatasetVisualizationError("Dataset has no rows to visualize")
    if len(rows) <= samples:
        return rows
    if samples == 1:
        return [rows[0]]

    indexes = [
        round(index * (len(rows) - 1) / (samples - 1))
        for index in range(samples)
    ]
    return [rows[index] for index in indexes]


def save_grid_preview(
    rows: list[dict[str, str]],
    output_path: str | Path,
    thumb_size: tuple[int, int] = (160, 120),
    label_height: int = 28,
    padding: int = 8,
) -> None:
    """Save sampled frames as a labeled image grid."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ModuleNotFoundError as exc:
        raise DatasetVisualizationError(
            "Pillow is required to create dataset previews. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    columns = min(5, len(rows))
    rows_count = math.ceil(len(rows) / columns)
    cell_width = thumb_size[0] + padding
    cell_height = thumb_size[1] + label_height + padding
    canvas_width = columns * cell_width + padding
    canvas_height = rows_count * cell_height + padding

    canvas = Image.new("RGB", (canvas_width, canvas_height), color=(24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for index, row in enumerate(rows):
        column = index % columns
        row_index = index // columns
        x = padding + column * cell_width
        y = padding + row_index * cell_height

        frame = _load_frame_image(row["frame_path"], thumb_size)
        canvas.paste(frame, (x, y))

        label = _truncate_label(format_sample_label(row), max_chars=24)
        draw.text((x, y + thumb_size[1] + 4), label, fill=(235, 235, 235), font=font)

    preview_path = Path(output_path)
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(preview_path, format="PNG")


def _load_frame_image(frame_path: str, thumb_size: tuple[int, int]):
    from PIL import Image

    resolved_path = resolve_dataset_frame_path(frame_path)
    if not resolved_path.is_file():
        raise DatasetVisualizationError(f"Frame path does not exist: {resolved_path}")

    with Image.open(resolved_path) as image:
        frame = image.convert("RGB")
        frame.thumbnail(thumb_size)
        background = Image.new("RGB", thumb_size, color=(0, 0, 0))
        x = (thumb_size[0] - frame.width) // 2
        y = (thumb_size[1] - frame.height) // 2
        background.paste(frame, (x, y))
        return background


def _truncate_label(label: str, max_chars: int) -> str:
    if len(label) <= max_chars:
        return label
    return f"{label[: max_chars - 3]}..."


def format_sample_label(row: dict[str, str]) -> str:
    label = row.get("action_name", "")
    mouse_x = row.get("mouse_x", "")
    mouse_y = row.get("mouse_y", "")
    if mouse_x and mouse_y:
        return f"{label} @ {mouse_x},{mouse_y}"
    return label


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a PNG preview of dataset samples.")
    parser.add_argument("--profile", required=True, help="Profile name from configs/profiles.")
    parser.add_argument("--samples", type=int, default=20, help="Number of samples to show.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        visualize_dataset(profile_name=args.profile, samples=args.samples)
    except DatasetVisualizationError as exc:
        parser.exit(status=2, message=f"Error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
