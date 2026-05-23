"""Profile loading and validation for Desktop AI Trainer."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROFILES_DIR = PROJECT_ROOT / "configs" / "profiles"

REQUIRED_PROFILE_FIELDS = (
    "profile_name",
    "allowed_window_title",
    "capture_region",
    "fps",
    "action_delay",
    "confidence_threshold",
    "emergency_stop_hotkey",
    "pause_hotkey",
    "actions",
)

REQUIRED_CAPTURE_REGION_FIELDS = ("left", "top", "width", "height")


class ProfileValidationError(ValueError):
    """Raised when a profile file is missing fields or has invalid values."""


@dataclass(frozen=True)
class CaptureRegion:
    """Screen region captured for demonstrations and inference."""

    left: int
    top: int
    width: int
    height: int


@dataclass(frozen=True)
class ActionConfig:
    """One user-configured action allowed for a profile."""

    name: str
    type: str
    key: str | None = None
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProfileConfig:
    """Validated profile configuration."""

    profile_name: str
    allowed_window_title: str
    capture_region: CaptureRegion
    fps: int
    action_delay: float
    confidence_threshold: float
    emergency_stop_hotkey: str
    pause_hotkey: str
    actions: list[ActionConfig]


def resolve_profile_path(profile: str | Path, profiles_dir: str | Path | None = None) -> Path:
    """Resolve a profile name like ``default`` to its YAML path."""
    profile_path = Path(profile)
    if profile_path.suffix in {".yaml", ".yml"} or profile_path.exists():
        return profile_path

    base_dir = Path(profiles_dir) if profiles_dir is not None else PROFILES_DIR
    return base_dir / f"{profile}.yaml"


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file as a dictionary."""
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise ProfileValidationError(
            "PyYAML is required to load profile files. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    config_path = Path(path)
    if not config_path.is_file():
        raise ProfileValidationError(f"Profile file does not exist: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    if not isinstance(data, dict):
        raise ProfileValidationError(f"Profile must be a YAML mapping: {config_path}")

    return data


def load_profile(profile: str | Path, profiles_dir: str | Path | None = None) -> ProfileConfig:
    """Load and validate a YAML profile."""
    profile_path = resolve_profile_path(profile, profiles_dir=profiles_dir)
    data = load_yaml_config(profile_path)
    return validate_profile(data, source=profile_path)


def load_config(path: str | Path) -> ProfileConfig:
    """Backward-compatible alias for loading a profile from a path."""
    return load_profile(path)


def validate_profile(data: dict[str, Any], source: str | Path = "<profile>") -> ProfileConfig:
    """Validate profile dictionary data and return a typed profile config."""
    source_label = str(source)
    for field_name in REQUIRED_PROFILE_FIELDS:
        if field_name not in data:
            raise ProfileValidationError(
                f"Profile {source_label} is missing required field: {field_name}"
            )

    capture_region = _parse_capture_region(data["capture_region"], source_label)
    actions = _parse_actions(data["actions"], source_label)

    profile = ProfileConfig(
        profile_name=_require_non_empty_string(data["profile_name"], "profile_name", source_label),
        allowed_window_title=_require_non_empty_string(
            data["allowed_window_title"], "allowed_window_title", source_label
        ),
        capture_region=capture_region,
        fps=_require_positive_int(data["fps"], "fps", source_label),
        action_delay=_require_non_negative_number(data["action_delay"], "action_delay", source_label),
        confidence_threshold=_require_confidence(
            data["confidence_threshold"], "confidence_threshold", source_label
        ),
        emergency_stop_hotkey=_require_non_empty_string(
            data["emergency_stop_hotkey"], "emergency_stop_hotkey", source_label
        ),
        pause_hotkey=_require_non_empty_string(data["pause_hotkey"], "pause_hotkey", source_label),
        actions=actions,
    )

    if not any(action.name == "no_op" and action.type == "none" for action in profile.actions):
        raise ProfileValidationError(
            f"Profile {source_label} must define a no_op action with type: none"
        )

    return profile


def _parse_capture_region(value: Any, source: str) -> CaptureRegion:
    if not isinstance(value, dict):
        raise ProfileValidationError(f"Profile {source} field capture_region must be a mapping")

    for field_name in REQUIRED_CAPTURE_REGION_FIELDS:
        if field_name not in value:
            raise ProfileValidationError(
                f"Profile {source} capture_region is missing required field: {field_name}"
            )

    left = _require_int(value["left"], "capture_region.left", source)
    top = _require_int(value["top"], "capture_region.top", source)
    width = _require_positive_int(value["width"], "capture_region.width", source)
    height = _require_positive_int(value["height"], "capture_region.height", source)
    return CaptureRegion(left=left, top=top, width=width, height=height)


def _parse_actions(value: Any, source: str) -> list[ActionConfig]:
    if not isinstance(value, list) or not value:
        raise ProfileValidationError(f"Profile {source} field actions must be a non-empty list")

    actions: list[ActionConfig] = []
    seen_names: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ProfileValidationError(f"Profile {source} action #{index} must be a mapping")
        for field_name in ("name", "type"):
            if field_name not in item:
                raise ProfileValidationError(
                    f"Profile {source} action #{index} is missing required field: {field_name}"
                )

        name = _require_non_empty_string(item["name"], f"actions[{index}].name", source)
        action_type = _require_non_empty_string(item["type"], f"actions[{index}].type", source)
        if name in seen_names:
            raise ProfileValidationError(f"Profile {source} has duplicate action name: {name}")
        seen_names.add(name)

        key = item.get("key")
        if action_type == "key":
            key = _require_non_empty_string(key, f"actions[{index}].key", source)
        elif action_type == "none":
            key = None

        params = {key_name: raw_value for key_name, raw_value in item.items() if key_name not in {"name", "type", "key"}}
        actions.append(ActionConfig(name=name, type=action_type, key=key, params=params))

    return actions


def _require_non_empty_string(value: Any, field_name: str, source: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProfileValidationError(f"Profile {source} field {field_name} must be a non-empty string")
    return value.strip()


def _require_int(value: Any, field_name: str, source: str) -> int:
    if not isinstance(value, int):
        raise ProfileValidationError(f"Profile {source} field {field_name} must be an integer")
    return value


def _require_positive_int(value: Any, field_name: str, source: str) -> int:
    parsed = _require_int(value, field_name, source)
    if parsed <= 0:
        raise ProfileValidationError(f"Profile {source} field {field_name} must be greater than 0")
    return parsed


def _require_non_negative_number(value: Any, field_name: str, source: str) -> float:
    if not isinstance(value, (int, float)):
        raise ProfileValidationError(f"Profile {source} field {field_name} must be a number")
    parsed = float(value)
    if parsed < 0:
        raise ProfileValidationError(f"Profile {source} field {field_name} must be 0 or greater")
    return parsed


def _require_confidence(value: Any, field_name: str, source: str) -> float:
    parsed = _require_non_negative_number(value, field_name, source)
    if parsed > 1:
        raise ProfileValidationError(f"Profile {source} field {field_name} must be between 0 and 1")
    return parsed
