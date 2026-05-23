"""Simple terminal-based manual labeling for imported video frames."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Callable

from src.config import PROJECT_ROOT, ProfileValidationError, load_profile
from src.dataset.build_dataset import SUPPORTED_FRAME_EXTENSIONS
from src.input_control.action_space import ActionDefinition, ActionSpace
from src.recorder.record_demo import ACTION_CSV_COLUMNS, build_episode_paths
from src.video.import_video import UNKNOWN_ACTION_ID, UNKNOWN_ACTION_NAME


class ManualLabelError(RuntimeError):
    """Raised when manual labeling cannot proceed."""


def manual_label_episode(
    profile_name: str,
    episode: str,
    demos_root: str | Path | None = None,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> Path:
    """Interactively label frames in an imported episode."""
    profile = load_profile(profile_name)
    action_space = ActionSpace.from_profile(profile)
    paths = build_episode_paths(profile.profile_name, episode, data_root=demos_root)
    frames = discover_episode_frames(paths.frames_dir)
    if not frames:
        raise ManualLabelError(f"No frames found in episode: {paths.frames_dir}")

    existing_rows = read_existing_action_rows(paths.actions_csv)
    output_fn(f"Manual labeling episode: {paths.episode_dir}")
    output_fn("Available actions:")
    for action in action_space:
        output_fn(f"  {action.id}: {action.name}")
    output_fn("  u: unknown")
    output_fn("  Enter: keep current label")
    output_fn("  q: quit and save")

    labeled_rows: list[dict[str, object]] = []
    for frame_path in frames:
        frame_id = parse_frame_id(frame_path)
        current = existing_rows.get(frame_id) or unknown_row(frame_id)
        current_label = current.get("action_name", UNKNOWN_ACTION_NAME)
        output_fn(f"Frame: {frame_path}")
        output_fn(f"Current label: {current_label}")

        selection = prompt_for_label(input_fn, action_space)
        if selection == "quit":
            labeled_rows.append(current)
            remaining_frames = frames[frames.index(frame_path) + 1 :]
            for remaining_frame in remaining_frames:
                remaining_id = parse_frame_id(remaining_frame)
                labeled_rows.append(existing_rows.get(remaining_id) or unknown_row(remaining_id))
            break
        if selection == "keep":
            labeled_rows.append(current)
        elif selection == "unknown":
            labeled_rows.append(unknown_row(frame_id, timestamp=current.get("timestamp", "")))
        else:
            action = selection
            labeled_rows.append(manual_label_row(frame_id, action, timestamp=current.get("timestamp", "")))
    else:
        pass

    write_action_rows(paths.actions_csv, labeled_rows)
    output_fn(f"Labels saved to: {paths.actions_csv}")
    return paths.actions_csv


def prompt_for_label(
    input_fn: Callable[[str], str],
    action_space: ActionSpace,
) -> ActionDefinition | str:
    """Prompt until the user enters a valid action id, unknown, keep, or quit."""
    while True:
        raw_value = input_fn("action_id> ").strip().casefold()
        if raw_value == "":
            return "keep"
        if raw_value == "q":
            return "quit"
        if raw_value == "u":
            return "unknown"
        try:
            action_id = int(raw_value)
            return action_space.get_by_id(action_id)
        except (ValueError, KeyError):
            print("Invalid action id. Enter a listed id, u, q, or blank.")


def discover_episode_frames(frames_dir: str | Path) -> list[Path]:
    frame_dir = Path(frames_dir)
    if not frame_dir.is_dir():
        return []
    return sorted(
        path
        for path in frame_dir.iterdir()
        if path.is_file() and path.suffix.casefold() in SUPPORTED_FRAME_EXTENSIONS
    )


def read_existing_action_rows(path: str | Path) -> dict[int, dict[str, object]]:
    actions_path = Path(path)
    if not actions_path.is_file():
        return {}
    rows: dict[int, dict[str, object]] = {}
    with actions_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            try:
                frame_id = int(row.get("frame_id", ""))
            except ValueError:
                continue
            rows[frame_id] = row
    return rows


def write_action_rows(path: str | Path, rows: list[dict[str, object]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda row: int(row["frame_id"]))
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=ACTION_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def manual_label_row(
    frame_id: int,
    action: ActionDefinition,
    timestamp: object = "",
) -> dict[str, object]:
    return {
        "timestamp": timestamp,
        "frame_id": frame_id,
        "action_name": action.name,
        "action_id": action.id,
        "key_or_button": action.key or "",
        "event_type": "manual_label",
        "mouse_x": "",
        "mouse_y": "",
    }


def unknown_row(frame_id: int, timestamp: object = "") -> dict[str, object]:
    return {
        "timestamp": timestamp,
        "frame_id": frame_id,
        "action_name": UNKNOWN_ACTION_NAME,
        "action_id": UNKNOWN_ACTION_ID,
        "key_or_button": "",
        "event_type": "manual_label_unknown",
        "mouse_x": "",
        "mouse_y": "",
    }


def parse_frame_id(path: Path) -> int:
    try:
        return int(path.stem)
    except ValueError as exc:
        raise ManualLabelError(f"Frame file must use numeric stem: {path}") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manually label imported video frames.")
    parser.add_argument("--profile", required=True, help="Profile name from configs/profiles.")
    parser.add_argument("--episode", required=True, help="Episode name under data/demos/{profile}.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        manual_label_episode(profile_name=args.profile, episode=args.episode)
    except (ManualLabelError, ProfileValidationError, ValueError) as exc:
        parser.exit(status=2, message=f"Error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
