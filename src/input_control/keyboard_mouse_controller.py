"""Keyboard and mouse execution constrained by the active profile action space."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.input_control.action_space import ActionDefinition, ActionSpace


class InputControlError(RuntimeError):
    """Raised when an action cannot be safely executed."""


@dataclass
class KeyboardMouseController:
    """Execute only actions explicitly present in a profile ``ActionSpace``."""

    action_space: ActionSpace

    def execute(self, action: ActionDefinition) -> str:
        """Execute one allowed action and return a short execution description."""
        self._require_allowed_action(action)

        if action.type == "none" or action.name == "no_op":
            return "no_op"

        try:
            import pyautogui
        except ModuleNotFoundError as exc:
            raise InputControlError(
                "pyautogui is required for live keyboard/mouse control. "
                "Install dependencies with: pip install -r requirements.txt"
            ) from exc

        if action.type == "key":
            if not action.key:
                raise InputControlError(f"Key action {action.name!r} has no key configured")
            pyautogui.press(action.key)
            return f"key:{action.key}"

        if action.type in {"mouse", "mouse_button", "mouse_click"}:
            button = self._mouse_button_for_action(action)
            pyautogui.click(button=button)
            return f"mouse_click:{button}"

        raise InputControlError(
            f"Action {action.name!r} has unsupported live-control type {action.type!r}"
        )

    def _require_allowed_action(self, action: ActionDefinition) -> None:
        try:
            allowed = self.action_space.get_by_id(action.id)
        except KeyError as exc:
            raise InputControlError(f"Action id {action.id} is not allowed by this profile") from exc

        if (
            allowed.name != action.name
            or allowed.type != action.type
            or allowed.key != action.key
        ):
            raise InputControlError(
                f"Action {action.name!r} does not match the configured profile action"
            )

    @staticmethod
    def _mouse_button_for_action(action: ActionDefinition) -> str:
        params: dict[str, Any] = action.params or {}
        button = params.get("button") or params.get("mouse_button") or action.key
        if not isinstance(button, str) or not button.strip():
            raise InputControlError(
                f"Mouse action {action.name!r} must explicitly define a button"
            )
        normalized = button.strip().casefold()
        if normalized not in {"left", "right", "middle"}:
            raise InputControlError(
                f"Mouse action {action.name!r} uses unsupported button {button!r}"
            )
        return normalized
