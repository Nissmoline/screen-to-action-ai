"""Dry-run inference tests."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

from src.config import load_profile
from src.inference import dry_run_agent as dry_run_module
from src.inference.dry_run_agent import (
    PREDICTION_LOG_COLUMNS,
    PredictionResult,
    apply_dry_run_safety,
    prepare_prediction_log,
    write_prediction_row,
)
from src.input_control.action_space import ActionSpace


def test_low_confidence_forces_no_op() -> None:
    profile = load_profile("lineage2_private")
    action_space = ActionSpace.from_profile(profile)
    prediction = PredictionResult(
        predicted_action=action_space.get_by_name("press_f1"),
        confidence=0.10,
        top3=[(action_space.get_by_name("press_f1"), 0.10)],
    )

    safe = apply_dry_run_safety(
        prediction=prediction,
        action_space=action_space,
        confidence_threshold=profile.confidence_threshold,
        window_title=profile.allowed_window_title,
        window_allowed=True,
    )

    assert safe.effective_action.name == "no_op"
    assert safe.forced_no_op is True
    assert safe.reason == "below_confidence_threshold"


def test_window_mismatch_forces_no_op() -> None:
    profile = load_profile("lineage2_private")
    action_space = ActionSpace.from_profile(profile)
    prediction = PredictionResult(
        predicted_action=action_space.get_by_name("press_f1"),
        confidence=0.99,
        top3=[(action_space.get_by_name("press_f1"), 0.99)],
    )

    safe = apply_dry_run_safety(
        prediction=prediction,
        action_space=action_space,
        confidence_threshold=profile.confidence_threshold,
        window_title="Some Other Window",
        window_allowed=False,
    )

    assert safe.effective_action.name == "no_op"
    assert safe.forced_no_op is True
    assert safe.reason == "window_not_allowed"


def test_prediction_log_csv_has_expected_columns(tmp_path: Path) -> None:
    profile = load_profile("lineage2_private")
    action_space = ActionSpace.from_profile(profile)
    prediction = PredictionResult(
        predicted_action=action_space.get_by_name("press_f1"),
        confidence=0.99,
        top3=[
            (action_space.get_by_name("press_f1"), 0.99),
            (action_space.no_op, 0.01),
        ],
    )
    safe = apply_dry_run_safety(
        prediction=prediction,
        action_space=action_space,
        confidence_threshold=profile.confidence_threshold,
        window_title=profile.allowed_window_title,
        window_allowed=True,
    )
    log_path = tmp_path / "predictions.csv"

    prepare_prediction_log(log_path)
    with log_path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=PREDICTION_LOG_COLUMNS)
        write_prediction_row(writer, file, profile, safe)

    with log_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    assert reader.fieldnames == PREDICTION_LOG_COLUMNS
    assert rows[0]["predicted_action_name"] == "press_f1"
    assert rows[0]["effective_action_name"] == "press_f1"


def test_dry_run_loop_writes_predictions_without_control(monkeypatch, tmp_path: Path) -> None:
    profile = load_profile("lineage2_private")
    action_space = ActionSpace.from_profile(profile)

    class FakeCapture:
        frame_interval_seconds = 0.0

        def __init__(self, capture_region, fps: int) -> None:
            pass

        def grab_frame(self):
            return object()

    class FakeHotkeys:
        def __init__(self, state, *args, **kwargs) -> None:
            self.state = state

        def start(self) -> None:
            pass

        def stop(self) -> None:
            pass

    def fake_predict_frame(frame, loaded_policy, action_space_arg):
        return PredictionResult(
            predicted_action=action_space_arg.get_by_name("press_f1"),
            confidence=0.99,
            top3=[
                (action_space_arg.get_by_name("press_f1"), 0.99),
                (action_space_arg.no_op, 0.01),
            ],
        )

    monkeypatch.setattr(dry_run_module, "ScreenCapture", FakeCapture)
    monkeypatch.setattr(dry_run_module, "PynputHotkeyController", FakeHotkeys)
    monkeypatch.setattr(dry_run_module, "load_policy", lambda *args, **kwargs: object())
    monkeypatch.setattr(dry_run_module, "read_window_state", lambda profile_arg: (profile.allowed_window_title, True))
    monkeypatch.setattr(dry_run_module, "predict_frame", fake_predict_frame)

    log_path = dry_run_module.dry_run_agent(
        profile_name="lineage2_private",
        logs_dir=tmp_path,
        max_iterations=2,
    )

    with log_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 2
    assert all(row["effective_action_name"] == "press_f1" for row in rows)


def test_dry_run_source_does_not_import_input_controller() -> None:
    source = Path(dry_run_module.__file__).read_text(encoding="utf-8")

    assert "KeyboardMouseController" not in source
    assert "pyautogui.click" not in source
    assert "pyautogui.press" not in source
    assert "pyautogui.moveTo" not in source


def test_dry_run_cli_help_runs() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "src.inference.dry_run_agent", "--help"],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--profile" in result.stdout
