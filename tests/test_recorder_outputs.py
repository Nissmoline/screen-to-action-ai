"""Recorder output schema tests."""

from __future__ import annotations

import csv
import json
import sys
import types
from pathlib import Path

from src.config import load_profile
from src.recorder import record_demo
from src.recorder.record_demo import (
    ACTION_CSV_COLUMNS,
    RECORDER_OVERLAY_TITLE,
    build_episode_paths,
    build_metadata,
    prepare_episode_folder,
    require_recording_window,
    run_recording,
    write_metadata,
)


def test_episode_folder_is_created(tmp_path: Path) -> None:
    profile = load_profile("lineage2_private")
    paths = build_episode_paths(profile.profile_name, "demo_001", data_root=tmp_path)

    prepare_episode_folder(paths)

    assert paths.episode_dir.is_dir()
    assert paths.frames_dir.is_dir()
    assert paths.actions_csv.is_file()


def test_metadata_schema_is_correct(tmp_path: Path) -> None:
    profile = load_profile("lineage2_private")
    paths = build_episode_paths(profile.profile_name, "demo_001", data_root=tmp_path)
    metadata = build_metadata(
        profile=profile,
        start_time="2026-05-22T00:00:00+00:00",
        end_time="2026-05-22T00:00:01+00:00",
        total_frames=10,
        total_actions=3,
    )

    write_metadata(paths.metadata_json, metadata)
    loaded = json.loads(paths.metadata_json.read_text(encoding="utf-8"))

    assert set(loaded) == {
        "profile_name",
        "start_time",
        "end_time",
        "fps",
        "capture_region",
        "allowed_window_title",
        "total_frames",
        "total_actions",
    }
    assert loaded["capture_region"] == {
        "left": 0,
        "top": 0,
        "width": 1280,
        "height": 720,
    }


def test_actions_csv_has_expected_columns(tmp_path: Path) -> None:
    profile = load_profile("lineage2_private")
    paths = build_episode_paths(profile.profile_name, "demo_001", data_root=tmp_path)

    prepare_episode_folder(paths)

    with paths.actions_csv.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        assert reader.fieldnames == ACTION_CSV_COLUMNS


def test_recording_window_guard_allows_status_overlay(monkeypatch) -> None:
    profile = load_profile("lineage2_private")
    monkeypatch.setattr(record_demo, "get_active_window_title", lambda: RECORDER_OVERLAY_TITLE)

    assert require_recording_window(profile) == RECORDER_OVERLAY_TITLE


def test_recording_loop_creates_episode_outputs(tmp_path: Path, monkeypatch) -> None:
    profile = load_profile("lineage2_private")
    paths = build_episode_paths(profile.profile_name, "demo_001", data_root=tmp_path)

    class FakeCapture:
        def __init__(self, capture_region, fps: int) -> None:
            self.frame_interval_seconds = 1.0 / fps

        def grab_frame(self):
            return object()

        def save_frame(self, frame, path: Path) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fake-png")

    class FakeHotkeys:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def start(self) -> None:
            pass

        def stop(self) -> None:
            pass

    class FakeOverlay:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def start(self) -> None:
            pass

        def update(self, *args, **kwargs) -> None:
            pass

        def close(self) -> None:
            pass

    class FakeListener:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def start(self) -> None:
            pass

        def stop(self) -> None:
            pass

    fake_pynput = types.ModuleType("pynput")
    fake_pynput.keyboard = types.SimpleNamespace(Listener=FakeListener)
    fake_pynput.mouse = types.SimpleNamespace(Listener=FakeListener)

    monkeypatch.setitem(sys.modules, "pynput", fake_pynput)
    monkeypatch.setattr(record_demo, "ScreenCapture", FakeCapture)
    monkeypatch.setattr(record_demo, "PynputHotkeyController", FakeHotkeys)
    monkeypatch.setattr(record_demo, "RecordingStatusOverlay", FakeOverlay)
    monkeypatch.setattr(record_demo, "require_active_window", lambda title: title)

    metadata = run_recording(
        profile=profile,
        episode="demo_001",
        paths=paths,
        countdown_seconds=0,
        duration_seconds=0.02,
    )

    assert paths.frames_dir.is_dir()
    assert list(paths.frames_dir.glob("*.png"))
    assert paths.actions_csv.is_file()
    assert paths.metadata_json.is_file()
    assert metadata["total_frames"] >= 1
