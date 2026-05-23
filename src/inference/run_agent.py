"""Live-control agent loop with strict safety gates."""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path

from src.capture.screen_capture import ScreenCapture, ScreenCaptureError
from src.config import PROJECT_ROOT, ProfileConfig, ProfileValidationError, load_profile
from src.inference.dry_run_agent import DryRunError, PredictionResult, load_policy, predict_frame, read_window_state
from src.input_control.action_space import ActionDefinition, ActionSpace
from src.input_control.keyboard_mouse_controller import InputControlError, KeyboardMouseController
from src.safety.hotkeys import HotkeyError, HotkeyState, PynputHotkeyController
from src.safety.rate_limiter import RateLimiter


SAFETY_OK = "OK"
SAFETY_LOW_CONFIDENCE_NO_OP = "LOW_CONFIDENCE_NO_OP"
SAFETY_WRONG_WINDOW_NO_OP = "WRONG_WINDOW_NO_OP"
SAFETY_PAUSED_NO_OP = "PAUSED_NO_OP"
SAFETY_EMERGENCY_STOP = "EMERGENCY_STOP"

LIVE_ACTION_LOG_COLUMNS = [
    "timestamp",
    "profile",
    "predicted_action",
    "executed_action",
    "confidence",
    "active_window",
    "safety_state",
    "predicted_action_id",
    "executed_action_id",
    "top3_actions",
    "top3_confidences",
    "execution_result",
]


class LiveControlError(RuntimeError):
    """Raised when live-control cannot run safely."""


@dataclass(frozen=True)
class LiveDecision:
    """Action selected after live-control safety checks."""

    prediction: PredictionResult
    executed_action: ActionDefinition
    safety_state: str
    active_window: str
    window_allowed: bool


def run_live_agent(
    profile_name: str,
    models_dir: str | Path | None = None,
    logs_dir: str | Path | None = None,
    device_name: str | None = None,
    countdown_seconds: int = 5,
    max_iterations: int | None = None,
) -> Path:
    """Run live-control inference until emergency stop and return the log path."""
    profile = load_profile(profile_name)
    action_space = ActionSpace.from_profile(profile)
    loaded_policy = load_policy(
        profile=profile,
        num_actions=len(action_space),
        models_dir=models_dir,
        device_name=device_name,
    )

    log_path = live_action_log_path(profile.profile_name, logs_dir=logs_dir)
    prepare_live_action_log(log_path)
    print_live_control_warning(profile, log_path)
    countdown(countdown_seconds)

    capture = ScreenCapture(profile.capture_region, profile.fps)
    controller = KeyboardMouseController(action_space=action_space)
    rate_limiter = RateLimiter.from_action_delay(profile.action_delay)
    hotkey_state = HotkeyState()

    def on_pause_change(paused: bool) -> None:
        print("Live-control paused." if paused else "Live-control resumed.")

    hotkeys = PynputHotkeyController(
        state=hotkey_state,
        emergency_stop_hotkey=profile.emergency_stop_hotkey,
        pause_hotkey=profile.pause_hotkey,
        on_pause_change=on_pause_change,
    )

    iteration = 0
    hotkeys.start()
    try:
        with log_path.open("a", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=LIVE_ACTION_LOG_COLUMNS)
            while True:
                if hotkey_state.emergency_stopped:
                    write_emergency_stop_row(writer, file, profile, action_space)
                    break
                if max_iterations is not None and iteration >= max_iterations:
                    break

                rate_limiter.wait_until_allowed()

                if hotkey_state.emergency_stopped:
                    write_emergency_stop_row(writer, file, profile, action_space)
                    break

                if hotkey_state.paused:
                    decision = paused_decision(profile, action_space)
                else:
                    frame = capture.grab_frame()
                    prediction = predict_frame(frame, loaded_policy, action_space)
                    active_window, window_allowed = read_window_state(profile)
                    decision = apply_live_safety(
                        prediction=prediction,
                        action_space=action_space,
                        confidence_threshold=profile.confidence_threshold,
                        active_window=active_window,
                        window_allowed=window_allowed,
                    )

                execution_result = controller.execute(decision.executed_action)
                write_live_action_row(writer, file, profile, decision, execution_result)
                print_live_decision(decision, execution_result)
                iteration += 1
    finally:
        hotkeys.stop()

    print(f"Live-control stopped. Action log saved to: {log_path}")
    return log_path


def apply_live_safety(
    prediction: PredictionResult,
    action_space: ActionSpace,
    confidence_threshold: float,
    active_window: str,
    window_allowed: bool,
) -> LiveDecision:
    """Apply window and confidence safety checks before execution."""
    if not window_allowed:
        return LiveDecision(
            prediction=prediction,
            executed_action=action_space.no_op,
            safety_state=SAFETY_WRONG_WINDOW_NO_OP,
            active_window=active_window,
            window_allowed=False,
        )

    if prediction.confidence < confidence_threshold:
        return LiveDecision(
            prediction=prediction,
            executed_action=action_space.no_op,
            safety_state=SAFETY_LOW_CONFIDENCE_NO_OP,
            active_window=active_window,
            window_allowed=True,
        )

    return LiveDecision(
        prediction=prediction,
        executed_action=prediction.predicted_action,
        safety_state=SAFETY_OK,
        active_window=active_window,
        window_allowed=True,
    )


def paused_decision(profile: ProfileConfig, action_space: ActionSpace) -> LiveDecision:
    """Return a logged no_op decision while pause mode is active."""
    no_op_prediction = PredictionResult(
        predicted_action=action_space.no_op,
        confidence=0.0,
        top3=[(action_space.no_op, 1.0)],
    )
    return LiveDecision(
        prediction=no_op_prediction,
        executed_action=action_space.no_op,
        safety_state=SAFETY_PAUSED_NO_OP,
        active_window="<paused>",
        window_allowed=False,
    )


def print_live_control_warning(profile: ProfileConfig, log_path: Path) -> None:
    """Print a blocking warning before live control starts."""
    print("Live control will send keyboard/mouse input to your computer.")
    print("Run dry-run first and inspect prediction logs before using this mode.")
    print(f"Profile: {profile.profile_name}")
    print(f"Allowed active window title must contain: {profile.allowed_window_title!r}")
    print(f"Confidence threshold: {profile.confidence_threshold}")
    print(f"Action delay: {profile.action_delay} seconds")
    print(f"Emergency stop: {profile.emergency_stop_hotkey}")
    print(f"Pause/resume: {profile.pause_hotkey}")
    print(f"Live action log: {log_path}")


def countdown(seconds: int) -> None:
    """Print a blocking countdown before live control starts."""
    for remaining in range(seconds, 0, -1):
        print(f"Live control starts in {remaining}...")
        time.sleep(1)


def print_live_decision(decision: LiveDecision, execution_result: str) -> None:
    """Print one live-control decision."""
    top3_text = ", ".join(
        f"{action.name}:{confidence:.3f}"
        for action, confidence in decision.prediction.top3
    )
    print(
        f"safety={decision.safety_state} "
        f"predicted={decision.prediction.predicted_action.name} "
        f"confidence={decision.prediction.confidence:.3f} "
        f"executed={decision.executed_action.name} "
        f"result={execution_result} "
        f"top3=[{top3_text}]"
    )


def prepare_live_action_log(path: str | Path) -> None:
    """Create or replace the live action CSV log."""
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=LIVE_ACTION_LOG_COLUMNS)
        writer.writeheader()


def write_live_action_row(
    writer: csv.DictWriter,
    file_handle,
    profile: ProfileConfig,
    decision: LiveDecision,
    execution_result: str,
) -> None:
    """Append one live-control decision to the CSV log."""
    writer.writerow(
        {
            "timestamp": f"{time.time():.6f}",
            "profile": profile.profile_name,
            "predicted_action": decision.prediction.predicted_action.name,
            "executed_action": decision.executed_action.name,
            "confidence": f"{decision.prediction.confidence:.6f}",
            "active_window": decision.active_window,
            "safety_state": decision.safety_state,
            "predicted_action_id": decision.prediction.predicted_action.id,
            "executed_action_id": decision.executed_action.id,
            "top3_actions": "|".join(action.name for action, _ in decision.prediction.top3),
            "top3_confidences": "|".join(
                f"{confidence:.6f}" for _, confidence in decision.prediction.top3
            ),
            "execution_result": execution_result,
        }
    )
    file_handle.flush()


def write_emergency_stop_row(
    writer: csv.DictWriter,
    file_handle,
    profile: ProfileConfig,
    action_space: ActionSpace,
) -> None:
    """Log emergency stop as a final no_op row."""
    prediction = PredictionResult(
        predicted_action=action_space.no_op,
        confidence=0.0,
        top3=[(action_space.no_op, 1.0)],
    )
    decision = LiveDecision(
        prediction=prediction,
        executed_action=action_space.no_op,
        safety_state=SAFETY_EMERGENCY_STOP,
        active_window="<emergency_stop>",
        window_allowed=False,
    )
    write_live_action_row(writer, file_handle, profile, decision, execution_result="no_op")


def live_action_log_path(profile_name: str, logs_dir: str | Path | None = None) -> Path:
    root = Path(logs_dir) if logs_dir is not None else PROJECT_ROOT / "outputs" / "logs"
    return root / f"{profile_name}_live_actions.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run live-control policy inference with safety gates.")
    parser.add_argument("--profile", required=True, help="Profile name from configs/profiles.")
    parser.add_argument("--device", default=None, help="Optional torch device, e.g. cpu or cuda.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        run_live_agent(profile_name=args.profile, device_name=args.device)
    except (
        DryRunError,
        HotkeyError,
        InputControlError,
        LiveControlError,
        ProfileValidationError,
        ScreenCaptureError,
        ValueError,
    ) as exc:
        parser.exit(status=2, message=f"Error: {exc}\n")
    return 0


def run_agent() -> None:
    """Backward-compatible entrypoint."""
    run_live_agent(profile_name="default")


if __name__ == "__main__":
    raise SystemExit(main())
