"""Record user demonstrations as screen frames plus input events."""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Event, Lock, Thread
from typing import TextIO

from src.capture.screen_capture import ScreenCapture, ScreenCaptureError
from src.config import PROJECT_ROOT, ProfileConfig, ProfileValidationError, load_profile
from src.input_control.action_space import ActionDefinition, ActionSpace
from src.safety.hotkeys import HotkeyError, HotkeyState, PynputHotkeyController
from src.safety.window_guard import (
    WindowGuardError,
    get_active_window_title,
    is_window_title_allowed,
    require_active_window,
)


ACTION_CSV_COLUMNS = [
    "timestamp",
    "frame_id",
    "action_name",
    "action_id",
    "key_or_button",
    "event_type",
    "mouse_x",
    "mouse_y",
]

EPISODE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
RECORDER_OVERLAY_TITLE = "Desktop AI Trainer Recording Status"


@dataclass
class EpisodePaths:
    """Filesystem locations for one recorded demonstration episode."""

    episode_dir: Path
    frames_dir: Path
    actions_csv: Path
    metadata_json: Path


@dataclass
class RecordingCounters:
    """Counters shared between the frame loop and input callbacks."""

    current_frame_id: int = -1
    total_frames: int = 0
    total_actions: int = 0


@dataclass
class OverlayState:
    """State displayed by the recording status overlay."""

    status: str
    total_frames: int = 0
    total_actions: int = 0


class RecordingStatusOverlay:
    """Small always-on-top status overlay that does not steal focus."""

    def __init__(self, profile: ProfileConfig, episode: str) -> None:
        self.profile = profile
        self.episode = episode
        self._state = OverlayState(status="RECORDING")
        self._lock = Lock()
        self._thread: Thread | None = None
        self._root = None
        self._labels = {}
        self._stop_event = Event()

    def start(self) -> None:
        self._thread = Thread(target=self._run, name="recording-status-overlay", daemon=True)
        self._thread.start()

    def update(self, total_frames: int, total_actions: int, paused: bool) -> None:
        with self._lock:
            self._state.total_frames = total_frames
            self._state.total_actions = total_actions
            self._state.status = "PAUSED" if paused else "RECORDING"

    def close(self) -> None:
        self._stop_event.set()
        root = self._root
        if root is not None:
            try:
                root.after(0, root.quit)
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _run(self) -> None:
        try:
            import tkinter as tk

            root = tk.Tk()
            root.withdraw()
            self._root = root
            root.title(RECORDER_OVERLAY_TITLE)
            root.overrideredirect(True)
            root.attributes("-topmost", True)
            root.attributes("-alpha", 0.88)
            root.configure(bg="#111827")

            width = 330
            height = 190
            screen_width = root.winfo_screenwidth()
            screen_height = root.winfo_screenheight()
            preferred_x = self.profile.capture_region.left + self.profile.capture_region.width + 10
            x = preferred_x if preferred_x + width < screen_width else max(10, screen_width - width - 18)
            y = max(40, min(screen_height - height - 20, self.profile.capture_region.top + 80))
            root.geometry(f"{width}x{height}+{x}+{y}")

            self._make_noactivate_on_windows(root)
            self._build_overlay_widgets(root, tk)
            root.update_idletasks()
            self._show_noactivate_on_windows(root)
            root.after(250, self._refresh)
            root.after(100, lambda: self._make_noactivate_on_windows(root))
            root.mainloop()
        except Exception as exc:
            print(f"Recording overlay is unavailable: {exc}")
        finally:
            root = self._root
            if root is not None:
                try:
                    root.destroy()
                except Exception:
                    pass

    def _build_overlay_widgets(self, root, tk_module) -> None:
        title = tk_module.Label(
            root,
            text="SCREEN RECORDING ACTIVE",
            fg="#f9fafb",
            bg="#b91c1c",
            font=("Segoe UI", 12, "bold"),
            padx=10,
            pady=6,
        )
        title.pack(fill="x")

        body = tk_module.Frame(root, bg="#111827", padx=12, pady=10)
        body.pack(fill="both", expand=True)

        self._labels["status"] = tk_module.Label(body, fg="#f9fafb", bg="#111827", anchor="w")
        self._labels["episode"] = tk_module.Label(body, fg="#d1d5db", bg="#111827", anchor="w")
        self._labels["counters"] = tk_module.Label(body, fg="#d1d5db", bg="#111827", anchor="w")
        self._labels["keys"] = tk_module.Label(
            body,
            fg="#fde68a",
            bg="#111827",
            anchor="w",
            justify="left",
            text=(
                f"Stop: {self.profile.emergency_stop_hotkey}\n"
                f"Pause/resume: {self.profile.pause_hotkey}"
            ),
        )
        self._labels["window"] = tk_module.Label(
            body,
            fg="#93c5fd",
            bg="#111827",
            anchor="w",
            justify="left",
            text=f"Window: {self.profile.allowed_window_title}",
        )

        for label in self._labels.values():
            label.pack(fill="x", pady=2)

    def _refresh(self) -> None:
        root = self._root
        if root is None:
            return
        if self._stop_event.is_set():
            root.quit()
            return

        with self._lock:
            state = OverlayState(
                status=self._state.status,
                total_frames=self._state.total_frames,
                total_actions=self._state.total_actions,
            )

        status_text = "PAUSED" if state.status == "PAUSED" else "RECORDING"
        self._labels["status"].configure(text=f"Status: {status_text}")
        self._labels["episode"].configure(text=f"Episode: {self.episode}")
        self._labels["counters"].configure(
            text=f"Frames: {state.total_frames}   Actions: {state.total_actions}"
        )
        root.after(250, self._refresh)

    @staticmethod
    def _make_noactivate_on_windows(root) -> None:
        try:
            import ctypes

            hwnd = root.winfo_id()
            user32 = ctypes.windll.user32
            gwl_exstyle = -20
            ws_ex_topmost = 0x00000008
            ws_ex_toolwindow = 0x00000080
            ws_ex_transparent = 0x00000020
            ws_ex_layered = 0x00080000
            ws_ex_noactivate = 0x08000000
            style = user32.GetWindowLongW(hwnd, gwl_exstyle)
            style |= (
                ws_ex_topmost
                | ws_ex_toolwindow
                | ws_ex_transparent
                | ws_ex_layered
                | ws_ex_noactivate
            )
            user32.SetWindowLongW(hwnd, gwl_exstyle, style)
        except Exception:
            pass

    @staticmethod
    def _show_noactivate_on_windows(root) -> None:
        try:
            import ctypes

            sw_shownoactivate = 4
            ctypes.windll.user32.ShowWindow(root.winfo_id(), sw_shownoactivate)
        except Exception:
            try:
                root.deiconify()
            except Exception:
                pass


def require_recording_window(profile: ProfileConfig) -> str:
    """Allow the target window and the recorder's own status overlay."""
    active_title = get_active_window_title()
    if is_window_title_allowed(active_title, profile.allowed_window_title):
        return active_title
    if RECORDER_OVERLAY_TITLE in active_title:
        return active_title
    raise WindowGuardError(
        "Active window is not allowed for this profile. "
        f"Expected title containing {profile.allowed_window_title!r}, got {active_title!r}."
    )


def validate_episode_name(episode: str) -> str:
    """Validate an episode name before using it as a folder name."""
    if not EPISODE_NAME_PATTERN.fullmatch(episode):
        raise ValueError(
            "Episode name may contain only letters, numbers, underscores, and hyphens"
        )
    return episode


def build_episode_paths(
    profile_name: str,
    episode: str,
    data_root: str | Path | None = None,
) -> EpisodePaths:
    """Return the standard output paths for a profile episode."""
    safe_episode = validate_episode_name(episode)
    root = Path(data_root) if data_root is not None else PROJECT_ROOT / "data" / "demos"
    episode_dir = root / profile_name / safe_episode
    return EpisodePaths(
        episode_dir=episode_dir,
        frames_dir=episode_dir / "frames",
        actions_csv=episode_dir / "actions.csv",
        metadata_json=episode_dir / "metadata.json",
    )


def prepare_episode_folder(paths: EpisodePaths) -> None:
    """Create the episode folder and empty CSV with the expected columns."""
    paths.frames_dir.mkdir(parents=True, exist_ok=True)
    write_actions_header(paths.actions_csv)


def write_actions_header(path: str | Path) -> None:
    """Create an actions CSV file with the recorder schema."""
    with Path(path).open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=ACTION_CSV_COLUMNS)
        writer.writeheader()


def build_metadata(
    profile: ProfileConfig,
    start_time: str,
    end_time: str,
    total_frames: int,
    total_actions: int,
) -> dict[str, object]:
    """Build metadata for one recorded episode."""
    return {
        "profile_name": profile.profile_name,
        "start_time": start_time,
        "end_time": end_time,
        "fps": profile.fps,
        "capture_region": asdict(profile.capture_region),
        "allowed_window_title": profile.allowed_window_title,
        "total_frames": total_frames,
        "total_actions": total_actions,
    }


def write_metadata(path: str | Path, metadata: dict[str, object]) -> None:
    """Write episode metadata as formatted JSON."""
    metadata_path = Path(path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
        file.write("\n")


def print_recording_warning(profile: ProfileConfig, episode: str) -> None:
    """Warn the user about privacy and safety before recording starts."""
    print("WARNING: demonstration recording captures screenshots and allowed input events.")
    print("Close password fields, private chats, documents, and any sensitive windows first.")
    print("This recorder does not control the computer; it only observes your actions.")
    print(f"Profile: {profile.profile_name}")
    print(f"Episode: {episode}")
    print(f"Allowed active window title must contain: {profile.allowed_window_title!r}")
    print(f"Emergency stop: {profile.emergency_stop_hotkey}")
    print(f"Pause/resume: {profile.pause_hotkey}")


def countdown(seconds: int) -> None:
    """Print a blocking countdown before recording."""
    for remaining in range(seconds, 0, -1):
        print(f"Recording starts in {remaining}...")
        time.sleep(1)


def build_key_action_lookup(action_space: ActionSpace) -> dict[str, ActionDefinition]:
    """Map normalized keyboard names to configured key actions."""
    lookup: dict[str, ActionDefinition] = {}
    for action in action_space:
        if action.type == "key" and action.key:
            lookup[normalize_key_name(action.key)] = action
    return lookup


def build_mouse_action_lookup(action_space: ActionSpace) -> dict[str, ActionDefinition]:
    """Map normalized mouse button names to configured mouse actions."""
    lookup: dict[str, ActionDefinition] = {}
    for action in action_space:
        if action.type not in {"mouse", "mouse_button", "mouse_click"}:
            continue
        params = action.params or {}
        button = params.get("button") or params.get("key") or action.key
        if isinstance(button, str) and button.strip():
            lookup[normalize_button_name(button)] = action
    return lookup


def normalize_key_name(key: object) -> str:
    """Normalize pynput and profile key names to the same compact form."""
    text = str(key).strip().casefold()
    if text.startswith("'") and text.endswith("'") and len(text) >= 3:
        text = text[1:-1]
    if text.startswith("key."):
        text = text[4:]
    aliases = {
        "escape": "esc",
        "return": "enter",
        " ": "space",
    }
    return aliases.get(text, text)


def normalize_button_name(button: object) -> str:
    """Normalize pynput mouse button names."""
    text = str(button).strip().casefold()
    if text.startswith("button."):
        text = text[7:]
    return text


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _timestamp() -> str:
    return f"{time.time():.6f}"


def _write_action_row(
    writer: csv.DictWriter,
    csv_file: TextIO,
    csv_lock: Lock,
    counters: RecordingCounters,
    counters_lock: Lock,
    action: ActionDefinition,
    key_or_button: str,
    event_type: str,
    mouse_x: int | None = None,
    mouse_y: int | None = None,
) -> None:
    with counters_lock:
        frame_id = counters.current_frame_id
        counters.total_actions += 1

    row = {
        "timestamp": _timestamp(),
        "frame_id": frame_id,
        "action_name": action.name,
        "action_id": action.id,
        "key_or_button": key_or_button,
        "event_type": event_type,
        "mouse_x": "" if mouse_x is None else mouse_x,
        "mouse_y": "" if mouse_y is None else mouse_y,
    }
    with csv_lock:
        writer.writerow(row)
        csv_file.flush()


def run_recording(
    profile: ProfileConfig,
    episode: str,
    paths: EpisodePaths,
    countdown_seconds: int = 5,
    duration_seconds: float | None = None,
    show_overlay: bool = True,
) -> dict[str, object]:
    """Record one demonstration episode."""
    print_recording_warning(profile, episode)
    countdown(countdown_seconds)

    active_title = require_active_window(profile.allowed_window_title)
    print(f"Active window accepted: {active_title!r}")

    prepare_episode_folder(paths)

    action_space = ActionSpace.from_profile(profile)
    key_actions = build_key_action_lookup(action_space)
    mouse_actions = build_mouse_action_lookup(action_space)
    capture = ScreenCapture(profile.capture_region, profile.fps)
    hotkey_state = HotkeyState()
    counters = RecordingCounters()
    counters_lock = Lock()
    csv_lock = Lock()
    start_time = _utc_now_iso()
    started_at = time.monotonic()
    next_frame_at = started_at
    last_window_check_at = started_at
    overlay = RecordingStatusOverlay(profile, episode) if show_overlay else None

    def on_pause_change(paused: bool) -> None:
        print("Recording paused." if paused else "Recording resumed.")
        if overlay is not None:
            with counters_lock:
                overlay.update(counters.total_frames, counters.total_actions, paused)

    hotkeys = PynputHotkeyController(
        hotkey_state,
        profile.emergency_stop_hotkey,
        profile.pause_hotkey,
        on_pause_change=on_pause_change,
    )

    try:
        from pynput import keyboard, mouse
    except ModuleNotFoundError as exc:
        raise HotkeyError(
            "pynput is required to record keyboard and mouse events. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    with paths.actions_csv.open("a", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=ACTION_CSV_COLUMNS)

        def on_press(key) -> None:
            if hotkey_state.paused or hotkey_state.emergency_stopped:
                return
            normalized_key = normalize_key_name(key)
            action = key_actions.get(normalized_key)
            if action is None:
                return
            _write_action_row(
                writer,
                csv_file,
                csv_lock,
                counters,
                counters_lock,
                action,
                normalized_key,
                "key_press",
            )
            if overlay is not None:
                with counters_lock:
                    overlay.update(counters.total_frames, counters.total_actions, hotkey_state.paused)

        def on_click(x, y, button, pressed) -> None:
            if hotkey_state.paused or hotkey_state.emergency_stopped:
                return
            normalized_button = normalize_button_name(button)
            action = mouse_actions.get(normalized_button)
            if action is None:
                return
            _write_action_row(
                writer,
                csv_file,
                csv_lock,
                counters,
                counters_lock,
                action,
                normalized_button,
                "mouse_press" if pressed else "mouse_release",
                mouse_x=int(x),
                mouse_y=int(y),
            )
            if overlay is not None:
                with counters_lock:
                    overlay.update(counters.total_frames, counters.total_actions, hotkey_state.paused)

        keyboard_listener = keyboard.Listener(on_press=on_press)
        mouse_listener = mouse.Listener(on_click=on_click)
        hotkeys.start()
        keyboard_listener.start()
        mouse_listener.start()
        if overlay is not None:
            overlay.start()

        try:
            print("Recording now. Use emergency stop hotkey to finish.")
            if overlay is not None:
                overlay.update(counters.total_frames, counters.total_actions, hotkey_state.paused)
            while not hotkey_state.emergency_stopped:
                now = time.monotonic()

                if duration_seconds is not None and now - started_at >= duration_seconds:
                    break

                if now - last_window_check_at >= 1.0:
                    try:
                        require_recording_window(profile)
                    except WindowGuardError as exc:
                        print(f"Stopping recording: {exc}")
                        hotkey_state.emergency_stop()
                        break
                    last_window_check_at = now

                if hotkey_state.paused:
                    time.sleep(0.05)
                    next_frame_at = time.monotonic()
                    continue

                if now < next_frame_at:
                    time.sleep(min(0.02, next_frame_at - now))
                    continue

                frame_id = counters.total_frames
                frame = capture.grab_frame()
                frame_path = paths.frames_dir / f"{frame_id:06d}.png"
                capture.save_frame(frame, frame_path)
                with counters_lock:
                    counters.current_frame_id = frame_id
                    counters.total_frames += 1
                    if overlay is not None:
                        overlay.update(counters.total_frames, counters.total_actions, hotkey_state.paused)
                next_frame_at += capture.frame_interval_seconds
        finally:
            if overlay is not None:
                overlay.close()
            hotkeys.stop()
            keyboard_listener.stop()
            mouse_listener.stop()

    end_time = _utc_now_iso()
    metadata = build_metadata(
        profile=profile,
        start_time=start_time,
        end_time=end_time,
        total_frames=counters.total_frames,
        total_actions=counters.total_actions,
    )
    write_metadata(paths.metadata_json, metadata)
    print(f"Recording saved to: {paths.episode_dir}")
    return metadata


def record_demo(
    profile_name: str,
    episode: str,
    countdown_seconds: int = 5,
    duration_seconds: float | None = None,
    data_root: str | Path | None = None,
    show_overlay: bool = True,
) -> dict[str, object]:
    """Load a profile and record a demonstration episode."""
    profile = load_profile(profile_name)
    paths = build_episode_paths(profile.profile_name, episode, data_root=data_root)
    return run_recording(
        profile=profile,
        episode=episode,
        paths=paths,
        countdown_seconds=countdown_seconds,
        duration_seconds=duration_seconds,
        show_overlay=show_overlay,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record a desktop demonstration episode.")
    parser.add_argument("--profile", required=True, help="Profile name from configs/profiles.")
    parser.add_argument("--episode", required=True, help="Episode folder name, e.g. demo_001.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        record_demo(profile_name=args.profile, episode=args.episode)
    except (
        HotkeyError,
        ProfileValidationError,
        ScreenCaptureError,
        ValueError,
        WindowGuardError,
    ) as exc:
        parser.exit(status=2, message=f"Error: {exc}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
