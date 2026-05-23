"""Video import and manual labeling tests."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from src.dataset.manual_label_tool import manual_label_episode
from src.video import import_video as import_video_module
from src.video.import_video import UNKNOWN_ACTION_NAME, import_video


def test_video_import_creates_frames_actions_and_metadata(monkeypatch, tmp_path: Path) -> None:
    video_path = tmp_path / "demo.mp4"
    video_path.write_bytes(b"fake-video")

    monkeypatch.setattr(import_video_module, "import_cv2", lambda: FakeCV2())

    result = import_video(
        profile_name="lineage2_private",
        video_path=video_path,
        every_n_frames=2,
        demos_root=tmp_path / "demos",
        episode_name="video_import_test",
    )

    assert result.frames_dir.is_dir()
    assert len(list(result.frames_dir.glob("*.png"))) == 3
    assert result.actions_csv.is_file()
    assert result.metadata_json.is_file()

    rows = _read_csv(result.actions_csv)
    assert len(rows) == 3
    assert {row["action_name"] for row in rows} == {UNKNOWN_ACTION_NAME}
    assert {row["action_id"] for row in rows} == {"-1"}

    metadata = json.loads(result.metadata_json.read_text(encoding="utf-8"))
    assert metadata["source"] == "video_import"
    assert metadata["total_frames"] == 3
    assert metadata["label_state"] == "unknown"


def test_manual_label_tool_writes_selected_actions(tmp_path: Path) -> None:
    episode_dir = tmp_path / "demos" / "lineage2_private" / "video_import_test"
    frames_dir = episode_dir / "frames"
    frames_dir.mkdir(parents=True)
    for frame_id in range(3):
        Image.new("RGB", (32, 32), color=(frame_id * 20, 40, 80)).save(
            frames_dir / f"{frame_id:06d}.png"
        )

    actions_csv = episode_dir / "actions.csv"
    with actions_csv.open("w", encoding="utf-8", newline="") as file:
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
        for frame_id in range(3):
            writer.writerow(
                {
                    "timestamp": f"{frame_id / 10:.6f}",
                    "frame_id": frame_id,
                    "action_name": "unknown",
                    "action_id": -1,
                    "key_or_button": "",
                    "event_type": "video_import_unknown",
                    "mouse_x": "",
                    "mouse_y": "",
                }
            )

    answers = iter(["1", "0", "u"])
    manual_label_episode(
        profile_name="lineage2_private",
        episode="video_import_test",
        demos_root=tmp_path / "demos",
        input_fn=lambda prompt: next(answers),
        output_fn=lambda text: None,
    )

    rows = _read_csv(actions_csv)
    assert [row["action_name"] for row in rows] == ["press_f1", "no_op", "unknown"]
    assert [row["action_id"] for row in rows] == ["1", "0", "-1"]


def test_video_import_cli_help_runs() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "src.video.import_video", "--help"],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--video-path" in result.stdout


def test_manual_label_cli_help_runs() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "src.dataset.manual_label_tool", "--help"],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--episode" in result.stdout


class FakeCV2:
    CAP_PROP_FPS = 5
    CAP_PROP_FRAME_COUNT = 7
    CAP_PROP_POS_MSEC = 9
    COLOR_BGR2RGB = 11

    def VideoCapture(self, path: str):
        return FakeVideoCapture()

    def cvtColor(self, frame, code):
        return frame[:, :, ::-1]


class FakeVideoCapture:
    def __init__(self) -> None:
        self.frames = [
            np.full((16, 16, 3), fill_value=index * 20, dtype=np.uint8)
            for index in range(6)
        ]
        self.index = 0

    def isOpened(self) -> bool:
        return True

    def get(self, prop: int) -> float:
        if prop == FakeCV2.CAP_PROP_FPS:
            return 30.0
        if prop == FakeCV2.CAP_PROP_FRAME_COUNT:
            return float(len(self.frames))
        if prop == FakeCV2.CAP_PROP_POS_MSEC:
            return self.index * 33.333
        return 0.0

    def read(self):
        if self.index >= len(self.frames):
            return False, None
        frame = self.frames[self.index]
        self.index += 1
        return True, frame

    def release(self) -> None:
        pass


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))
