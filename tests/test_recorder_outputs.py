"""Recorder output schema tests."""

from __future__ import annotations

import csv
import json
import sys
import types
from pathlib import Path

from src.config import load_profile
from src.input_control.action_space import ActionDefinition
from src.recorder import record_demo
from src.recorder.record_demo import (
    ACTION_CSV_COLUMNS,
    RECORDER_OVERLAY_TITLE,
    PolledInputEvent,
    Win32InputPoller,
    build_episode_paths,
    build_metadata,
    prepare_episode_folder,
    require_recording_window,
    run_recording,
    virtual_key_for_key_name,
    write_metadata,
)
from src.safety.hotkeys import HotkeyState


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


def test_prepare_episode_folder_clears_stale_frames_and_metadata(tmp_path: Path) -> None:
    profile = load_profile("lineage2_private")
    paths = build_episode_paths(profile.profile_name, "demo_001", data_root=tmp_path)
    paths.frames_dir.mkdir(parents=True)
    stale_frame = paths.frames_dir / "000999.png"
    keep_file = paths.frames_dir / "notes.txt"
    stale_frame.write_bytes(b"old-frame")
    keep_file.write_text("keep", encoding="utf-8")
    paths.metadata_json.parent.mkdir(parents=True, exist_ok=True)
    paths.metadata_json.write_text("{}", encoding="utf-8")

    prepare_episode_folder(paths)

    assert not stale_frame.exists()
    assert keep_file.exists()
    assert not paths.metadata_json.exists()


def test_recording_window_guard_allows_status_overlay(monkeypatch) -> None:
    profile = load_profile("lineage2_private")
    monkeypatch.setattr(record_demo, "get_active_window_title", lambda: RECORDER_OVERLAY_TITLE)

    assert require_recording_window(profile) == RECORDER_OVERLAY_TITLE


def test_win32_input_poller_records_allowed_key_and_mouse_once() -> None:
    pressed_keys: set[int] = set()
    f1_vk = virtual_key_for_key_name("f1")
    left_vk = virtual_key_for_key_name("left")
    assert f1_vk is not None
    assert left_vk is not None

    poller = Win32InputPoller(
        key_actions={"f1": ActionDefinition(id=1, name="press_f1", type="key", key="f1")},
        mouse_actions={
            "left": ActionDefinition(
                id=14,
                name="left_click",
                type="mouse_click",
                params={"button": "left"},
            )
        },
        emergency_stop_hotkey="ctrl+shift+q",
        pause_hotkey="ctrl+shift+p",
        is_pressed=lambda virtual_key: virtual_key in pressed_keys,
        cursor_position=lambda: (640, 360),
    )

    pressed_keys.add(f1_vk)
    key_events = poller.poll_actions()
    assert [(event.action.name, event.event_type) for event in key_events] == [
        ("press_f1", "key_press")
    ]
    assert poller.poll_actions() == []

    pressed_keys.clear()
    poller.poll_actions()
    pressed_keys.add(left_vk)
    mouse_events = poller.poll_actions()
    assert len(mouse_events) == 1
    assert mouse_events[0].action.name == "left_click"
    assert mouse_events[0].event_type == "mouse_press"
    assert mouse_events[0].mouse_x == 640
    assert mouse_events[0].mouse_y == 360


def test_win32_input_poller_handles_pause_and_emergency_hotkeys() -> None:
    pressed_keys: set[int] = set()
    state = HotkeyState()
    pause_changes: list[bool] = []
    poller = Win32InputPoller(
        key_actions={},
        mouse_actions={},
        emergency_stop_hotkey="ctrl+shift+q",
        pause_hotkey="ctrl+shift+p",
        is_pressed=lambda virtual_key: virtual_key in pressed_keys,
        cursor_position=lambda: (0, 0),
    )

    for key_name in ("ctrl", "shift", "p"):
        pressed_keys.add(virtual_key_for_key_name(key_name))
    poller.poll_hotkeys(state, on_pause_change=pause_changes.append)
    poller.poll_hotkeys(state, on_pause_change=pause_changes.append)

    assert state.paused is True
    assert pause_changes == [True]

    pressed_keys.clear()
    poller.poll_hotkeys(state, on_pause_change=pause_changes.append)
    for key_name in ("ctrl", "shift", "q"):
        pressed_keys.add(virtual_key_for_key_name(key_name))
    poller.poll_hotkeys(state, on_pause_change=pause_changes.append)

    assert state.emergency_stopped is True


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
    monkeypatch.setattr(record_demo.Win32InputPoller, "from_action_maps", lambda *args, **kwargs: None)

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


def test_recording_loop_writes_events_from_win32_poller(tmp_path: Path, monkeypatch) -> None:
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

    class FakeOverlay:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def start(self) -> None:
            pass

        def update(self, *args, **kwargs) -> None:
            pass

        def close(self) -> None:
            pass

    class FakePoller:
        def __init__(self) -> None:
            self.emitted = False

        def poll_hotkeys(self, hotkey_state, on_pause_change=None) -> None:
            pass

        def poll_actions(self, emit: bool = True):
            if self.emitted or not emit:
                return []
            self.emitted = True
            return [
                PolledInputEvent(
                    action=ActionDefinition(id=1, name="press_f1", type="key", key="f1"),
                    key_or_button="f1",
                    event_type="key_press",
                )
            ]

    monkeypatch.setattr(record_demo, "ScreenCapture", FakeCapture)
    monkeypatch.setattr(record_demo, "RecordingStatusOverlay", FakeOverlay)
    monkeypatch.setattr(record_demo, "require_active_window", lambda title: title)
    monkeypatch.setattr(record_demo, "require_recording_window", lambda profile_arg: profile_arg.allowed_window_title)
    monkeypatch.setattr(record_demo.Win32InputPoller, "from_action_maps", lambda *args, **kwargs: FakePoller())

    metadata = run_recording(
        profile=profile,
        episode="demo_001",
        paths=paths,
        countdown_seconds=0,
        duration_seconds=0.05,
    )

    with paths.actions_csv.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    assert metadata["total_actions"] == 1
    assert rows[0]["action_name"] == "press_f1"
    assert rows[0]["event_type"] == "key_press"
