"""Hotkey safety state for pause and emergency stop controls."""

from __future__ import annotations

from dataclasses import dataclass


class HotkeyError(RuntimeError):
    """Raised when global hotkeys cannot be started."""


@dataclass
class HotkeyState:
    """Mutable safety state shared by recording and agent loops."""

    paused: bool = False
    emergency_stopped: bool = False

    def pause(self) -> None:
        self.paused = True

    def resume(self) -> None:
        self.paused = False

    def toggle_pause(self) -> bool:
        self.paused = not self.paused
        return self.paused

    def emergency_stop(self) -> None:
        self.emergency_stopped = True


def normalize_hotkey_for_pynput(hotkey: str) -> str:
    """Convert ``ctrl+shift+q`` style strings to pynput GlobalHotKeys syntax."""
    aliases = {
        "ctrl": "<ctrl>",
        "control": "<ctrl>",
        "shift": "<shift>",
        "alt": "<alt>",
        "cmd": "<cmd>",
        "win": "<cmd>",
        "esc": "<esc>",
        "escape": "<esc>",
        "space": "<space>",
        "enter": "<enter>",
        "return": "<enter>",
    }
    parts = [part.strip().casefold() for part in hotkey.split("+") if part.strip()]
    if not parts:
        raise HotkeyError("Hotkey must not be empty")
    return "+".join(aliases.get(part, part) for part in parts)


class PynputHotkeyController:
    """Register global pause and emergency-stop hotkeys with pynput."""

    def __init__(
        self,
        state: HotkeyState,
        emergency_stop_hotkey: str,
        pause_hotkey: str,
        on_pause_change=None,
    ) -> None:
        self.state = state
        self.emergency_stop_hotkey = emergency_stop_hotkey
        self.pause_hotkey = pause_hotkey
        self.on_pause_change = on_pause_change
        self._listener = None

    def start(self) -> None:
        try:
            from pynput import keyboard
        except ModuleNotFoundError as exc:
            raise HotkeyError(
                "pynput is required for recorder hotkeys. "
                "Install dependencies with: pip install -r requirements.txt"
            ) from exc

        hotkeys = {
            normalize_hotkey_for_pynput(self.emergency_stop_hotkey): self.state.emergency_stop,
            normalize_hotkey_for_pynput(self.pause_hotkey): self._toggle_pause,
        }
        self._listener = keyboard.GlobalHotKeys(hotkeys)
        self._listener.start()

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None

    def _toggle_pause(self) -> None:
        paused = self.state.toggle_pause()
        if self.on_pause_change is not None:
            self.on_pause_change(paused)
