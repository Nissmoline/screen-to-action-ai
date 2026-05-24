"""Profile-driven action space for desktop imitation learning."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any
from typing import Iterator

from src.config import ActionConfig, ProfileConfig, ProfileValidationError, load_profile


@dataclass(frozen=True)
class ActionDefinition:
    """An action with a stable integer id for model training and inference."""

    id: int
    name: str
    type: str
    key: str | None = None
    params: dict[str, Any] | None = None

    @classmethod
    def from_config(cls, action_id: int, config: ActionConfig) -> "ActionDefinition":
        return cls(
            id=action_id,
            name=config.name,
            type=config.type,
            key=config.key,
            params=dict(config.params),
        )


class ActionSpace:
    """Index user-configured actions from a validated profile."""

    def __init__(self, actions: list[ActionDefinition]) -> None:
        if not actions:
            raise ValueError("ActionSpace requires at least one action")
        self._actions = actions
        self._by_id = {action.id: action for action in actions}
        self._by_name = {action.name: action for action in actions}

        if "no_op" not in self._by_name:
            raise ValueError("ActionSpace requires a no_op action")

    @classmethod
    def from_profile(cls, profile: ProfileConfig) -> "ActionSpace":
        actions = [
            ActionDefinition.from_config(action_id, action)
            for action_id, action in enumerate(profile.actions)
        ]
        return cls(actions)

    def __iter__(self) -> Iterator[ActionDefinition]:
        return iter(self._actions)

    def __len__(self) -> int:
        return len(self._actions)

    @property
    def no_op(self) -> ActionDefinition:
        return self._by_name["no_op"]

    def get_by_id(self, action_id: int) -> ActionDefinition:
        return self._by_id[action_id]

    def get_by_name(self, name: str) -> ActionDefinition:
        return self._by_name[name]

    def to_cli_lines(self) -> list[str]:
        return [format_action(action) for action in self._actions]


def format_action(action: ActionDefinition) -> str:
    """Return a compact human-readable action line."""
    if action.key:
        return f"{action.id}: {action.name} ({action.type}, key={action.key})"
    if action.type in {"mouse", "mouse_button", "mouse_click"} and action.params:
        button = action.params.get("button") or action.params.get("mouse_button")
        if button:
            return f"{action.id}: {action.name} ({action.type}, button={button})"
    return f"{action.id}: {action.name} ({action.type})"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect a Desktop AI Trainer action profile.")
    parser.add_argument(
        "--profile",
        default="default",
        help="Profile name from configs/profiles without extension, or a YAML path.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        profile = load_profile(args.profile)
        action_space = ActionSpace.from_profile(profile)
    except (ProfileValidationError, ValueError) as exc:
        parser.exit(status=2, message=f"Error: {exc}\n")

    print(f"Profile name: {profile.profile_name}")
    print(f"Allowed window title: {profile.allowed_window_title}")
    print("Actions:")
    for line in action_space.to_cli_lines():
        print(f"  {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
