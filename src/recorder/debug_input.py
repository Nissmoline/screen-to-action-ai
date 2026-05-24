"""Debug whether allowed keyboard/mouse inputs are visible to Python."""

from __future__ import annotations

import argparse
import time

from src.config import ProfileValidationError, load_profile
from src.input_control.action_space import ActionSpace
from src.recorder.record_demo import (
    Win32InputPoller,
    build_key_action_lookup,
    build_mouse_action_lookup,
)


class InputDebugError(RuntimeError):
    """Raised when the input debug command cannot run."""


def debug_input(profile_name: str, seconds: float = 10.0) -> int:
    """Poll profile actions and print detected input events."""
    if seconds <= 0:
        raise InputDebugError("seconds must be greater than 0")

    profile = load_profile(profile_name)
    action_space = ActionSpace.from_profile(profile)
    key_actions = build_key_action_lookup(action_space)
    mouse_actions = build_mouse_action_lookup(action_space)
    poller = Win32InputPoller.from_action_maps(
        key_actions=key_actions,
        mouse_actions=mouse_actions,
        emergency_stop_hotkey=profile.emergency_stop_hotkey,
        pause_hotkey=profile.pause_hotkey,
    )
    if poller is None:
        raise InputDebugError("Win32 input polling is only available on Windows")

    print(f"Input debug for profile: {profile.profile_name}")
    print(f"Duration: {seconds:.1f}s")
    print("Press allowed keys/click mouse now. Detected events will appear below.")
    print("Allowed keys:", ", ".join(sorted(key_actions)) or "none")
    print("Allowed mouse buttons:", ", ".join(sorted(mouse_actions)) or "none")

    detected = 0
    started_at = time.monotonic()
    while time.monotonic() - started_at < seconds:
        for event in poller.poll_actions(emit=True):
            detected += 1
            coords = ""
            if event.mouse_x is not None and event.mouse_y is not None:
                coords = f" at {event.mouse_x},{event.mouse_y}"
            print(f"DETECTED: {event.action.name} {event.event_type}{coords}")
        time.sleep(0.005)

    print(f"Detected events: {detected}")
    return detected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Debug Win32 input capture for a profile.")
    parser.add_argument("--profile", required=True, help="Profile name from configs/profiles.")
    parser.add_argument("--seconds", type=float, default=10.0, help="How long to poll input.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        debug_input(profile_name=args.profile, seconds=args.seconds)
    except (InputDebugError, ProfileValidationError, ValueError) as exc:
        parser.exit(status=2, message=f"Error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
