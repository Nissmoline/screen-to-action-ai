"""Build supervised datasets from recorded desktop demonstrations."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from src.config import PROJECT_ROOT, ProfileValidationError, load_profile
from src.input_control.action_space import ActionDefinition, ActionSpace


REQUIRED_DATASET_COLUMNS = [
    "frame_path",
    "action_id",
    "action_name",
    "timestamp",
    "episode",
    "split",
]

DATASET_COLUMNS = REQUIRED_DATASET_COLUMNS + [
    "key_or_button",
    "event_type",
    "mouse_x",
    "mouse_y",
]

SUPPORTED_FRAME_EXTENSIONS = {".png", ".jpg", ".jpeg"}


class DatasetBuildError(RuntimeError):
    """Raised when recorded demonstrations cannot be converted into a dataset."""


@dataclass(frozen=True)
class FrameSample:
    """One captured frame with an approximate timestamp."""

    path: Path
    frame_id: int
    timestamp: float | None
    episode: str


@dataclass(frozen=True)
class RecordedAction:
    """One action event from actions.csv that is valid for the active profile."""

    action: ActionDefinition
    frame_id: int | None
    timestamp: float | None
    key_or_button: str = ""
    event_type: str = ""
    mouse_x: str = ""
    mouse_y: str = ""


@dataclass(frozen=True)
class DatasetBuildResult:
    """Summary of a dataset build."""

    dataset_csv: Path
    row_count: int
    class_distribution: Counter[str]


def build_dataset(
    profile_name: str,
    demos_root: str | Path | None = None,
    datasets_root: str | Path | None = None,
    no_op_warning_threshold: float = 0.75,
) -> DatasetBuildResult:
    """Build ``dataset.csv`` for a profile from all recorded episodes."""
    profile = load_profile(profile_name)
    action_space = ActionSpace.from_profile(profile)
    profile_demos_dir = _profile_demos_dir(profile.profile_name, demos_root)
    output_dir = _profile_dataset_dir(profile.profile_name, datasets_root)

    rows: list[dict[str, str]] = []
    for episode_dir in find_episode_dirs(profile_demos_dir):
        rows.extend(build_episode_rows(episode_dir, action_space, default_fps=profile.fps))

    if not rows:
        raise DatasetBuildError(
            f"No frames found for profile {profile.profile_name!r} in {profile_demos_dir}"
        )

    assign_splits(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_csv = output_dir / "dataset.csv"
    write_dataset_csv(dataset_csv, rows)

    distribution = Counter(row["action_name"] for row in rows)
    print_class_distribution(distribution, total=len(rows), no_op_warning_threshold=no_op_warning_threshold)
    print(f"Dataset written to: {dataset_csv}")
    return DatasetBuildResult(
        dataset_csv=dataset_csv,
        row_count=len(rows),
        class_distribution=distribution,
    )


def find_episode_dirs(profile_demos_dir: str | Path) -> list[Path]:
    """Return episode directories under ``data/demos/{profile}``."""
    demos_dir = Path(profile_demos_dir)
    if not demos_dir.is_dir():
        raise DatasetBuildError(f"Demo directory does not exist: {demos_dir}")
    return sorted(path for path in demos_dir.iterdir() if path.is_dir())


def build_episode_rows(
    episode_dir: str | Path,
    action_space: ActionSpace,
    default_fps: int,
) -> list[dict[str, str]]:
    """Build dataset rows for one recorded episode."""
    episode_path = Path(episode_dir)
    frames_dir = episode_path / "frames"
    metadata = read_metadata(episode_path / "metadata.json")
    fps = _metadata_fps(metadata, default_fps)
    frame_interval = 1.0 / fps
    start_epoch = _parse_start_time_epoch(metadata.get("start_time"))

    frames = discover_frames(frames_dir, episode=episode_path.name, fps=fps, start_epoch=start_epoch)
    actions = read_actions_csv(episode_path / "actions.csv", action_space)

    rows: list[dict[str, str]] = []
    for frame in frames:
        recorded_action = choose_action_for_frame(
            frame,
            actions,
            action_space.no_op,
            max_time_distance=frame_interval / 2.0,
        )
        action = recorded_action.action if recorded_action is not None else action_space.no_op
        rows.append(
            {
                "frame_path": serialize_frame_path(frame.path),
                "action_id": str(action.id),
                "action_name": action.name,
                "timestamp": "" if frame.timestamp is None else f"{frame.timestamp:.6f}",
                "episode": frame.episode,
                "split": "",
                "key_or_button": "" if recorded_action is None else recorded_action.key_or_button,
                "event_type": "" if recorded_action is None else recorded_action.event_type,
                "mouse_x": "" if recorded_action is None else recorded_action.mouse_x,
                "mouse_y": "" if recorded_action is None else recorded_action.mouse_y,
            }
        )
    return rows


def read_metadata(path: str | Path) -> dict[str, object]:
    """Read episode metadata if present."""
    metadata_path = Path(path)
    if not metadata_path.is_file():
        return {}
    with metadata_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        return {}
    return data


def discover_frames(
    frames_dir: str | Path,
    episode: str,
    fps: int,
    start_epoch: float | None,
) -> list[FrameSample]:
    """Find frame image files and attach approximate timestamps."""
    frame_dir = Path(frames_dir)
    if not frame_dir.is_dir():
        return []

    frame_paths = sorted(
        path
        for path in frame_dir.iterdir()
        if path.is_file() and path.suffix.casefold() in SUPPORTED_FRAME_EXTENSIONS
    )
    frames: list[FrameSample] = []
    for sequence_index, frame_path in enumerate(frame_paths):
        frame_id = _parse_frame_id(frame_path, default=sequence_index)
        timestamp = None if start_epoch is None else start_epoch + (frame_id / fps)
        frames.append(
            FrameSample(
                path=frame_path,
                frame_id=frame_id,
                timestamp=timestamp,
                episode=episode,
            )
        )
    return frames


def read_actions_csv(path: str | Path, action_space: ActionSpace) -> list[RecordedAction]:
    """Read allowed actions from a recorder actions.csv file."""
    actions_path = Path(path)
    if not actions_path.is_file():
        return []

    actions: list[RecordedAction] = []
    with actions_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if (row.get("event_type") or "").strip().casefold() == "mouse_release":
                continue
            action = _resolve_action(row, action_space)
            if action is None:
                continue
            actions.append(
                RecordedAction(
                    action=action,
                    frame_id=_parse_optional_int(row.get("frame_id")),
                    timestamp=_parse_optional_float(row.get("timestamp")),
                    key_or_button=(row.get("key_or_button") or "").strip(),
                    event_type=(row.get("event_type") or "").strip(),
                    mouse_x=(row.get("mouse_x") or "").strip(),
                    mouse_y=(row.get("mouse_y") or "").strip(),
                )
            )
    return actions


def choose_action_for_frame(
    frame: FrameSample,
    actions: list[RecordedAction],
    no_op_action: ActionDefinition,
    max_time_distance: float,
) -> RecordedAction | None:
    """Choose the nearest allowed action for a frame, or ``no_op``."""
    if not actions:
        return None

    if frame.timestamp is not None:
        timestamp_actions = [action for action in actions if action.timestamp is not None]
        if timestamp_actions:
            nearest = min(timestamp_actions, key=lambda action: abs(action.timestamp - frame.timestamp))
            if abs(nearest.timestamp - frame.timestamp) <= max_time_distance:
                return nearest

    frame_id_actions = [action for action in actions if action.frame_id is not None]
    if frame_id_actions:
        nearest = min(frame_id_actions, key=lambda action: abs(action.frame_id - frame.frame_id))
        if nearest.frame_id == frame.frame_id:
            return nearest

    return None


def assign_splits(
    rows: list[dict[str, str]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> None:
    """Assign deterministic train/val/test splits in-place."""
    if not rows:
        return

    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)

    if len(rows) == 1:
        split_by_index = {indices[0]: "train"}
    elif len(rows) == 2:
        split_by_index = {indices[0]: "train", indices[1]: "val"}
    else:
        test_count = max(1, round(len(rows) * (1.0 - train_ratio - val_ratio)))
        val_count = max(1, round(len(rows) * val_ratio))
        if test_count + val_count >= len(rows):
            test_count = 1
            val_count = 1
        train_count = len(rows) - val_count - test_count

        split_by_index = {}
        for index in indices[:train_count]:
            split_by_index[index] = "train"
        for index in indices[train_count : train_count + val_count]:
            split_by_index[index] = "val"
        for index in indices[train_count + val_count :]:
            split_by_index[index] = "test"

    for row_index, split in split_by_index.items():
        rows[row_index]["split"] = split


def write_dataset_csv(path: str | Path, rows: list[dict[str, str]]) -> None:
    """Write dataset rows to CSV."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=DATASET_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def print_class_distribution(
    distribution: Counter[str],
    total: int,
    no_op_warning_threshold: float,
) -> None:
    """Print action class counts and no_op imbalance warning."""
    print("Class distribution:")
    for action_name, count in distribution.most_common():
        percent = (count / total) * 100 if total else 0
        print(f"  {action_name}: {count} ({percent:.1f}%)")

    no_op_count = distribution.get("no_op", 0)
    no_op_ratio = no_op_count / total if total else 0
    if no_op_ratio >= no_op_warning_threshold:
        print(
            "WARNING: no_op is very frequent. Do not delete data automatically; "
            "review action/frame alignment and collect more active demonstrations first."
        )


def serialize_frame_path(path: str | Path) -> str:
    """Store frame paths relative to the project root when possible."""
    frame_path = Path(path).resolve()
    try:
        return frame_path.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(frame_path)


def resolve_dataset_frame_path(frame_path: str | Path) -> Path:
    """Resolve a frame path from dataset.csv."""
    path = Path(frame_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _profile_demos_dir(profile_name: str, demos_root: str | Path | None) -> Path:
    root = Path(demos_root) if demos_root is not None else PROJECT_ROOT / "data" / "demos"
    return root / profile_name


def _profile_dataset_dir(profile_name: str, datasets_root: str | Path | None) -> Path:
    root = Path(datasets_root) if datasets_root is not None else PROJECT_ROOT / "data" / "datasets"
    return root / profile_name


def _metadata_fps(metadata: dict[str, object], default_fps: int) -> int:
    raw_fps = metadata.get("fps")
    if isinstance(raw_fps, int) and raw_fps > 0:
        return raw_fps
    return default_fps


def _parse_start_time_epoch(value: object) -> float | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        return None


def _parse_frame_id(path: Path, default: int) -> int:
    try:
        return int(path.stem)
    except ValueError:
        return default


def _parse_optional_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_optional_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _resolve_action(row: dict[str, str], action_space: ActionSpace) -> ActionDefinition | None:
    action_name = (row.get("action_name") or "").strip()
    if action_name:
        try:
            return action_space.get_by_name(action_name)
        except KeyError:
            return None

    action_id = _parse_optional_int(row.get("action_id"))
    if action_id is None:
        return None
    try:
        return action_space.get_by_id(action_id)
    except KeyError:
        return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a training dataset from recorded demos.")
    parser.add_argument("--profile", required=True, help="Profile name from configs/profiles.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        build_dataset(profile_name=args.profile)
    except (DatasetBuildError, ProfileValidationError, ValueError) as exc:
        parser.exit(status=2, message=f"Error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
