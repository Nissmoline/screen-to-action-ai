"""Live-control safety and execution tests."""

from __future__ import annotations

import csv
import subprocess
import sys
import types
from pathlib import Path

import pytest

from src.config import load_profile
from src.inference import run_agent as live_module
from src.inference.dry_run_agent import PredictionResult
from src.inference.run_agent import (
    LIVE_ACTION_LOG_COLUMNS,
    SAFETY_LOW_CONFIDENCE_NO_OP,
    SAFETY_OK,
    SAFETY_WRONG_WINDOW_NO_OP,
    apply_live_safety,
    prepare_live_action_log,
    run_live_agent,
)
from src.input_control.action_space import ActionDefinition, ActionSpace
from src.input_control.keyboard_mouse_controller import InputControlError, KeyboardMouseController
from src.safety.rate_limiter import RateLimiter


def test_keyboard_controller_executes_allowed_key(monkeypatch) -> None:
    profile = load_profile("lineage2_private")
    action_space = ActionSpace.from_profile(profile)
    calls: list[tuple[str, str]] = []
    fake_pyautogui = types.SimpleNamespace(press=lambda key: calls.append(("press", key)))
    monkeypatch.setitem(sys.modules, "pyautogui", fake_pyautogui)

    controller = KeyboardMouseController(action_space)
    result = controller.execute(action_space.get_by_name("press_f1"))

    assert result == "key:f1"
    assert calls == [("press", "f1")]


def test_keyboard_controller_no_op_does_nothing(monkeypatch) -> None:
    profile = load_profile("lineage2_private")
    action_space = ActionSpace.from_profile(profile)
    calls: list[str] = []
    fake_pyautogui = types.SimpleNamespace(press=lambda key: calls.append(key))
    monkeypatch.setitem(sys.modules, "pyautogui", fake_pyautogui)

    controller = KeyboardMouseController(action_space)
    result = controller.execute(action_space.no_op)

    assert result == "no_op"
    assert calls == []


def test_keyboard_controller_rejects_unlisted_action() -> None:
    profile = load_profile("lineage2_private")
    action_space = ActionSpace.from_profile(profile)
    controller = KeyboardMouseController(action_space)
    unlisted = ActionDefinition(id=999, name="press_f9", type="key", key="f9")

    with pytest.raises(InputControlError, match="not allowed"):
        controller.execute(unlisted)


def test_keyboard_controller_executes_explicit_mouse_click(monkeypatch) -> None:
    action_space = ActionSpace(
        [
            ActionDefinition(id=0, name="no_op", type="none"),
            ActionDefinition(
                id=1,
                name="left_click",
                type="mouse_click",
                params={"button": "left"},
            ),
        ]
    )
    calls: list[tuple[str, str]] = []
    fake_pyautogui = types.SimpleNamespace(click=lambda button: calls.append(("click", button)))
    monkeypatch.setitem(sys.modules, "pyautogui", fake_pyautogui)

    controller = KeyboardMouseController(action_space)
    result = controller.execute(action_space.get_by_name("left_click"))

    assert result == "mouse_click:left"
    assert calls == [("click", "left")]


def test_rate_limiter_uses_action_delay() -> None:
    limiter = RateLimiter.from_action_delay(10.0)

    assert limiter.allow() is True
    assert limiter.allow() is False
    assert limiter.seconds_until_allowed() > 0


def test_live_safety_wrong_window_forces_no_op() -> None:
    profile = load_profile("lineage2_private")
    action_space = ActionSpace.from_profile(profile)
    prediction = PredictionResult(
        predicted_action=action_space.get_by_name("press_f1"),
        confidence=0.99,
        top3=[(action_space.get_by_name("press_f1"), 0.99)],
    )

    decision = apply_live_safety(
        prediction=prediction,
        action_space=action_space,
        confidence_threshold=profile.confidence_threshold,
        active_window="Wrong Window",
        window_allowed=False,
    )

    assert decision.executed_action.name == "no_op"
    assert decision.safety_state == SAFETY_WRONG_WINDOW_NO_OP


def test_live_safety_low_confidence_forces_no_op() -> None:
    profile = load_profile("lineage2_private")
    action_space = ActionSpace.from_profile(profile)
    prediction = PredictionResult(
        predicted_action=action_space.get_by_name("press_f1"),
        confidence=0.10,
        top3=[(action_space.get_by_name("press_f1"), 0.10)],
    )

    decision = apply_live_safety(
        prediction=prediction,
        action_space=action_space,
        confidence_threshold=profile.confidence_threshold,
        active_window=profile.allowed_window_title,
        window_allowed=True,
    )

    assert decision.executed_action.name == "no_op"
    assert decision.safety_state == SAFETY_LOW_CONFIDENCE_NO_OP


def test_live_safety_ok_executes_prediction() -> None:
    profile = load_profile("lineage2_private")
    action_space = ActionSpace.from_profile(profile)
    prediction = PredictionResult(
        predicted_action=action_space.get_by_name("press_f1"),
        confidence=0.99,
        top3=[(action_space.get_by_name("press_f1"), 0.99)],
    )

    decision = apply_live_safety(
        prediction=prediction,
        action_space=action_space,
        confidence_threshold=profile.confidence_threshold,
        active_window=profile.allowed_window_title,
        window_allowed=True,
    )

    assert decision.executed_action.name == "press_f1"
    assert decision.safety_state == SAFETY_OK


def test_live_action_log_has_expected_columns(tmp_path: Path) -> None:
    log_path = tmp_path / "live.csv"

    prepare_live_action_log(log_path)

    with log_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        assert reader.fieldnames == LIVE_ACTION_LOG_COLUMNS


def test_live_loop_executes_only_safe_fake_actions(monkeypatch, tmp_path: Path) -> None:
    profile = load_profile("lineage2_private")
    executed_actions: list[str] = []

    class FakeCapture:
        def __init__(self, capture_region, fps: int) -> None:
            pass

        def grab_frame(self):
            return object()

    class FakeHotkeys:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def start(self) -> None:
            pass

        def stop(self) -> None:
            pass

    class FakeRateLimiter:
        @classmethod
        def from_action_delay(cls, action_delay: float):
            return cls()

        def wait_until_allowed(self) -> None:
            pass

    class FakeController:
        def __init__(self, action_space: ActionSpace) -> None:
            self.action_space = action_space

        def execute(self, action: ActionDefinition) -> str:
            configured = self.action_space.get_by_id(action.id)
            assert configured.name == action.name
            executed_actions.append(action.name)
            return f"fake:{action.name}"

    def fake_predict_frame(frame, loaded_policy, action_space: ActionSpace):
        return PredictionResult(
            predicted_action=action_space.get_by_name("press_f1"),
            confidence=0.99,
            top3=[
                (action_space.get_by_name("press_f1"), 0.99),
                (action_space.no_op, 0.01),
            ],
        )

    monkeypatch.setattr(live_module, "ScreenCapture", FakeCapture)
    monkeypatch.setattr(live_module, "PynputHotkeyController", FakeHotkeys)
    monkeypatch.setattr(live_module, "RateLimiter", FakeRateLimiter)
    monkeypatch.setattr(live_module, "KeyboardMouseController", FakeController)
    monkeypatch.setattr(live_module, "load_policy", lambda *args, **kwargs: object())
    monkeypatch.setattr(live_module, "predict_frame", fake_predict_frame)
    monkeypatch.setattr(live_module, "read_window_state", lambda profile_arg: (profile.allowed_window_title, True))
    monkeypatch.setattr(live_module, "countdown", lambda seconds: None)

    log_path = run_live_agent(
        profile_name="lineage2_private",
        logs_dir=tmp_path,
        countdown_seconds=0,
        max_iterations=2,
    )

    with log_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    assert executed_actions == ["press_f1", "press_f1"]
    assert [row["safety_state"] for row in rows] == [SAFETY_OK, SAFETY_OK]
    assert all(row["executed_action"] == "press_f1" for row in rows)


def test_run_agent_cli_help_runs() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "src.inference.run_agent", "--help"],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--profile" in result.stdout
