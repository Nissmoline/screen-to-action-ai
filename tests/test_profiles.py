"""Profile loading and action-space tests."""

from pathlib import Path

import pytest

from src.config import ProfileValidationError, load_profile
from src.input_control.action_space import ActionSpace


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_lineage2_private_profile_loads() -> None:
    profile = load_profile("lineage2_private")

    assert profile.profile_name == "lineage2_private"
    assert profile.allowed_window_title == "Lineage II"
    assert profile.fps == 10
    assert profile.confidence_threshold == 0.90


def test_no_op_action_exists() -> None:
    profile = load_profile("lineage2_private")
    action_space = ActionSpace.from_profile(profile)

    assert action_space.no_op.name == "no_op"
    assert action_space.no_op.type == "none"


def test_actions_receive_ids() -> None:
    profile = load_profile("lineage2_private")
    action_space = ActionSpace.from_profile(profile)

    action_ids = [action.id for action in action_space]
    assert action_ids == list(range(len(profile.actions)))
    assert action_space.get_by_name("press_f1").id == 1


def test_lineage_profile_allows_function_keys_and_mouse_clicks() -> None:
    profile = load_profile("lineage2_private")
    action_space = ActionSpace.from_profile(profile)

    for index in range(1, 13):
        action = action_space.get_by_name(f"press_f{index}")
        assert action.type == "key"
        assert action.key == f"f{index}"

    assert action_space.get_by_name("left_click").params == {"button": "left"}
    assert action_space.get_by_name("right_click").params == {"button": "right"}
    assert action_space.get_by_name("middle_click").params == {"button": "middle"}


def test_lineage_profile_does_not_use_wasd_movement() -> None:
    profile = load_profile("lineage2_private")
    action_names = {action.name for action in profile.actions}

    assert "move_forward" not in action_names
    assert "move_backward" not in action_names
    assert "turn_left" not in action_names
    assert "turn_right" not in action_names


def test_missing_required_field_has_clear_error(tmp_path: Path) -> None:
    invalid_profile = tmp_path / "missing_fps.yaml"
    invalid_profile.write_text(
        """
profile_name: broken_profile
allowed_window_title: "Sandbox"
capture_region:
  left: 0
  top: 0
  width: 1280
  height: 720
action_delay: 0.2
confidence_threshold: 0.85
emergency_stop_hotkey: "ctrl+alt+esc"
pause_hotkey: "ctrl+alt+p"
actions:
  - name: no_op
    type: none
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(ProfileValidationError, match="missing required field: fps"):
        load_profile(invalid_profile)
