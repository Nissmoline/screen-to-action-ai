"""Dataset build and visualization tests."""

from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image

from src.dataset.build_dataset import build_dataset
from src.dataset.visualize_dataset import visualize_dataset
from src.input_control.action_space import ActionSpace
from src.config import load_profile


def test_dataset_csv_is_created(tmp_path: Path) -> None:
    demos_root = _create_demo_root(tmp_path)
    datasets_root = tmp_path / "datasets"

    result = build_dataset(
        profile_name="lineage2_private",
        demos_root=demos_root,
        datasets_root=datasets_root,
    )

    assert result.dataset_csv.is_file()


def test_dataset_frame_paths_exist(tmp_path: Path) -> None:
    demos_root = _create_demo_root(tmp_path)
    datasets_root = tmp_path / "datasets"
    result = build_dataset(
        profile_name="lineage2_private",
        demos_root=demos_root,
        datasets_root=datasets_root,
    )

    rows = _read_dataset_rows(result.dataset_csv)

    assert rows
    assert all(Path(row["frame_path"]).is_file() for row in rows)


def test_dataset_action_ids_are_valid(tmp_path: Path) -> None:
    demos_root = _create_demo_root(tmp_path)
    datasets_root = tmp_path / "datasets"
    result = build_dataset(
        profile_name="lineage2_private",
        demos_root=demos_root,
        datasets_root=datasets_root,
    )
    profile = load_profile("lineage2_private")
    action_space = ActionSpace.from_profile(profile)
    valid_action_ids = {str(action.id) for action in action_space}

    rows = _read_dataset_rows(result.dataset_csv)

    assert all(row["action_id"] in valid_action_ids for row in rows)


def test_dataset_splits_exist(tmp_path: Path) -> None:
    demos_root = _create_demo_root(tmp_path)
    datasets_root = tmp_path / "datasets"
    result = build_dataset(
        profile_name="lineage2_private",
        demos_root=demos_root,
        datasets_root=datasets_root,
    )

    rows = _read_dataset_rows(result.dataset_csv)
    splits = {row["split"] for row in rows}

    assert splits == {"train", "val", "test"}


def test_dataset_preserves_mouse_click_coordinates(tmp_path: Path) -> None:
    demos_root = _create_demo_root(tmp_path)
    datasets_root = tmp_path / "datasets"
    result = build_dataset(
        profile_name="lineage2_private",
        demos_root=demos_root,
        datasets_root=datasets_root,
    )

    rows = _read_dataset_rows(result.dataset_csv)
    click_rows = [row for row in rows if row["action_name"] == "left_click"]

    assert click_rows
    assert click_rows[0]["key_or_button"] == "left"
    assert click_rows[0]["event_type"] == "mouse_press"
    assert click_rows[0]["mouse_x"] == "640"
    assert click_rows[0]["mouse_y"] == "360"


def test_dataset_ignores_mouse_release_labels(tmp_path: Path) -> None:
    demos_root = _create_demo_root(tmp_path)
    datasets_root = tmp_path / "datasets"
    result = build_dataset(
        profile_name="lineage2_private",
        demos_root=demos_root,
        datasets_root=datasets_root,
    )

    rows = _read_dataset_rows(result.dataset_csv)

    assert not any(row["event_type"] == "mouse_release" for row in rows)


def test_dataset_preview_is_created(tmp_path: Path) -> None:
    demos_root = _create_demo_root(tmp_path)
    datasets_root = tmp_path / "datasets"
    result = build_dataset(
        profile_name="lineage2_private",
        demos_root=demos_root,
        datasets_root=datasets_root,
    )
    preview_path = tmp_path / "preview.png"

    visualize_dataset(
        profile_name="lineage2_private",
        samples=6,
        dataset_csv=result.dataset_csv,
        output_path=preview_path,
    )

    assert preview_path.is_file()


def _create_demo_root(tmp_path: Path) -> Path:
    profile = load_profile("lineage2_private")
    demo_dir = tmp_path / "demos" / profile.profile_name / "demo_001"
    frames_dir = demo_dir / "frames"
    frames_dir.mkdir(parents=True)

    for frame_id in range(12):
        image = Image.new("RGB", (64, 48), color=(frame_id * 10, 80, 120))
        image.save(frames_dir / f"{frame_id:06d}.png")

    (demo_dir / "metadata.json").write_text(
        """
{
  "profile_name": "lineage2_private",
  "start_time": "2026-05-22T00:00:00+00:00",
  "end_time": "2026-05-22T00:00:02+00:00",
  "fps": 10,
  "capture_region": {"left": 0, "top": 0, "width": 1280, "height": 720},
  "allowed_window_title": "Lineage II Private Sandbox",
  "total_frames": 12,
  "total_actions": 2
}
""".lstrip(),
        encoding="utf-8",
    )

    with (demo_dir / "actions.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "timestamp",
                "frame_id",
                "action_name",
                "action_id",
                "key_or_button",
                "event_type",
                "mouse_x",
                "mouse_y",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "timestamp": "",
                "frame_id": "3",
                "action_name": "press_f1",
                "action_id": "1",
                "key_or_button": "f1",
                "event_type": "key_press",
                "mouse_x": "",
                "mouse_y": "",
            }
        )
        writer.writerow(
            {
                "timestamp": "",
                "frame_id": "8",
                "action_name": "press_f2",
                "action_id": "2",
                "key_or_button": "f2",
                "event_type": "key_press",
                "mouse_x": "",
                "mouse_y": "",
            }
        )
        writer.writerow(
            {
                "timestamp": "",
                "frame_id": "9",
                "action_name": "left_click",
                "action_id": "14",
                "key_or_button": "left",
                "event_type": "mouse_press",
                "mouse_x": "640",
                "mouse_y": "360",
            }
        )
        writer.writerow(
            {
                "timestamp": "",
                "frame_id": "10",
                "action_name": "left_click",
                "action_id": "14",
                "key_or_button": "left",
                "event_type": "mouse_release",
                "mouse_x": "640",
                "mouse_y": "360",
            }
        )

    return tmp_path / "demos"


def _read_dataset_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))
