"""Dry-run policy inference loop.

This module never sends keyboard or mouse input. It only captures the screen,
runs the trained policy, prints predictions, and writes prediction logs.
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.capture.screen_capture import ScreenCapture, ScreenCaptureError
from src.config import PROJECT_ROOT, ProfileConfig, ProfileValidationError, load_profile
from src.input_control.action_space import ActionDefinition, ActionSpace
from src.safety.hotkeys import HotkeyError, HotkeyState, PynputHotkeyController
from src.safety.window_guard import WindowGuardError, get_active_window_title, is_window_title_allowed
from src.training.evaluate_imitation import normalize_action_map, profile_model_path
from src.training.train_imitation import import_torch


PREDICTION_LOG_COLUMNS = [
    "timestamp",
    "profile",
    "predicted_action_id",
    "predicted_action_name",
    "confidence",
    "effective_action_id",
    "effective_action_name",
    "top3_actions",
    "top3_confidences",
    "window_title",
    "window_allowed",
    "forced_no_op",
    "reason",
]


class DryRunError(RuntimeError):
    """Raised when dry-run inference cannot run."""


@dataclass(frozen=True)
class PredictionResult:
    """Raw model prediction before safety filtering."""

    predicted_action: ActionDefinition
    confidence: float
    top3: list[tuple[ActionDefinition, float]]


@dataclass(frozen=True)
class SafePrediction:
    """Prediction after dry-run safety rules are applied."""

    raw: PredictionResult
    effective_action: ActionDefinition
    forced_no_op: bool
    reason: str
    window_title: str
    window_allowed: bool


@dataclass(frozen=True)
class LoadedPolicy:
    """Loaded model plus preprocessing metadata."""

    model: Any
    image_size: int
    action_id_to_name: dict[int, str]
    device: Any
    torch: Any


def dry_run_agent(
    profile_name: str,
    models_dir: str | Path | None = None,
    logs_dir: str | Path | None = None,
    device_name: str | None = None,
    max_iterations: int | None = None,
) -> Path:
    """Run dry-run inference until emergency stop and return the log path."""
    profile = load_profile(profile_name)
    action_space = ActionSpace.from_profile(profile)
    loaded_policy = load_policy(
        profile=profile,
        num_actions=len(action_space),
        models_dir=models_dir,
        device_name=device_name,
    )

    log_path = prediction_log_path(profile.profile_name, logs_dir=logs_dir)
    prepare_prediction_log(log_path)

    capture = ScreenCapture(profile.capture_region, profile.fps)
    hotkey_state = HotkeyState()

    def on_pause_change(paused: bool) -> None:
        print("Dry-run paused." if paused else "Dry-run resumed.")

    hotkeys = PynputHotkeyController(
        state=hotkey_state,
        emergency_stop_hotkey=profile.emergency_stop_hotkey,
        pause_hotkey=profile.pause_hotkey,
        on_pause_change=on_pause_change,
    )

    print("Dry-run mode: predictions are logged only. No keyboard or mouse input is sent.")
    print(f"Emergency stop: {profile.emergency_stop_hotkey}")
    print(f"Pause/resume: {profile.pause_hotkey}")
    print(f"Prediction log: {log_path}")

    iteration = 0
    next_frame_at = time.monotonic()
    hotkeys.start()
    try:
        with log_path.open("a", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=PREDICTION_LOG_COLUMNS)
            while not hotkey_state.emergency_stopped:
                if max_iterations is not None and iteration >= max_iterations:
                    break

                if hotkey_state.paused:
                    time.sleep(0.05)
                    next_frame_at = time.monotonic()
                    continue

                now = time.monotonic()
                if now < next_frame_at:
                    time.sleep(min(0.02, next_frame_at - now))
                    continue

                frame = capture.grab_frame()
                raw_prediction = predict_frame(frame, loaded_policy, action_space)
                window_title, window_allowed = read_window_state(profile)
                safe_prediction = apply_dry_run_safety(
                    prediction=raw_prediction,
                    action_space=action_space,
                    confidence_threshold=profile.confidence_threshold,
                    window_title=window_title,
                    window_allowed=window_allowed,
                )
                print_prediction(safe_prediction)
                write_prediction_row(writer, file, profile, safe_prediction)

                iteration += 1
                next_frame_at += capture.frame_interval_seconds
    finally:
        hotkeys.stop()

    print(f"Dry-run stopped. Predictions saved to: {log_path}")
    return log_path


def load_policy(
    profile: ProfileConfig,
    num_actions: int,
    models_dir: str | Path | None = None,
    device_name: str | None = None,
) -> LoadedPolicy:
    """Load a trained policy checkpoint for a profile."""
    torch = import_torch()
    from src.models.policy_cnn import PolicyCNN

    model_path = profile_model_path(profile.profile_name, models_dir=models_dir)
    if not model_path.is_file():
        raise DryRunError(f"Model file does not exist: {model_path}")

    device = select_device(torch, device_name)
    checkpoint = torch.load(model_path, map_location=device)
    checkpoint_num_actions = int(checkpoint.get("num_actions", num_actions))
    if checkpoint_num_actions != num_actions:
        raise DryRunError(
            f"Model num_actions={checkpoint_num_actions} does not match profile num_actions={num_actions}"
        )

    image_size = int(checkpoint.get("image_size", 84))
    action_id_to_name = normalize_action_map(
        checkpoint.get("action_id_to_name"),
        fallback={action_id: action.name for action_id, action in enumerate(profile.actions)},
    )
    model = PolicyCNN(num_actions=num_actions).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    return LoadedPolicy(
        model=model,
        image_size=image_size,
        action_id_to_name=action_id_to_name,
        device=device,
        torch=torch,
    )


def predict_frame(
    frame: np.ndarray,
    loaded_policy: LoadedPolicy,
    action_space: ActionSpace,
) -> PredictionResult:
    """Run one model prediction for an RGB frame."""
    tensor = frame_to_tensor(frame, loaded_policy.image_size, loaded_policy.torch).unsqueeze(0)
    tensor = tensor.to(loaded_policy.device)

    with loaded_policy.torch.no_grad():
        logits = loaded_policy.model(tensor)
        probabilities = loaded_policy.torch.softmax(logits, dim=1)[0]
        k = min(3, len(action_space))
        top_values, top_indices = loaded_policy.torch.topk(probabilities, k=k)

    predicted_id = int(top_indices[0].item())
    predicted_action = action_space.get_by_id(predicted_id)
    confidence = float(top_values[0].item())
    top3 = [
        (action_space.get_by_id(int(action_id.item())), float(probability.item()))
        for action_id, probability in zip(top_indices, top_values)
    ]
    return PredictionResult(
        predicted_action=predicted_action,
        confidence=confidence,
        top3=top3,
    )


def frame_to_tensor(frame: np.ndarray, image_size: int, torch_module: Any):
    """Convert an RGB numpy frame to a normalized CHW tensor."""
    if frame.ndim != 3 or frame.shape[2] < 3:
        raise DryRunError("Captured frame must have shape HxWx3")
    rgb = frame[:, :, :3]
    image = Image.fromarray(rgb.astype("uint8"), mode="RGB").resize(
        (image_size, image_size),
        Image.BILINEAR,
    )
    array = np.asarray(image, dtype=np.float32) / 255.0
    chw = np.transpose(array, (2, 0, 1))
    return torch_module.from_numpy(np.ascontiguousarray(chw))


def read_window_state(profile: ProfileConfig) -> tuple[str, bool]:
    """Return active window title and whether it matches the profile."""
    try:
        title = get_active_window_title()
    except WindowGuardError as exc:
        return f"<window unavailable: {exc}>", False
    return title, is_window_title_allowed(title, profile.allowed_window_title)


def apply_dry_run_safety(
    prediction: PredictionResult,
    action_space: ActionSpace,
    confidence_threshold: float,
    window_title: str,
    window_allowed: bool,
) -> SafePrediction:
    """Force unsafe or low-confidence predictions to no_op."""
    if not window_allowed:
        return SafePrediction(
            raw=prediction,
            effective_action=action_space.no_op,
            forced_no_op=True,
            reason="window_not_allowed",
            window_title=window_title,
            window_allowed=False,
        )

    if prediction.confidence < confidence_threshold:
        return SafePrediction(
            raw=prediction,
            effective_action=action_space.no_op,
            forced_no_op=True,
            reason="below_confidence_threshold",
            window_title=window_title,
            window_allowed=True,
        )

    return SafePrediction(
        raw=prediction,
        effective_action=prediction.predicted_action,
        forced_no_op=False,
        reason="ok",
        window_title=window_title,
        window_allowed=True,
    )


def print_prediction(prediction: SafePrediction) -> None:
    """Print one dry-run prediction line."""
    top3_text = ", ".join(
        f"{action.name}:{confidence:.3f}"
        for action, confidence in prediction.raw.top3
    )
    warning = ""
    if prediction.forced_no_op:
        warning = f" WARNING forced no_op reason={prediction.reason}"
    print(
        f"predicted={prediction.raw.predicted_action.name} "
        f"confidence={prediction.raw.confidence:.3f} "
        f"effective={prediction.effective_action.name} "
        f"top3=[{top3_text}]{warning}"
    )


def prepare_prediction_log(path: str | Path) -> None:
    """Create or replace a dry-run prediction log CSV."""
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=PREDICTION_LOG_COLUMNS)
        writer.writeheader()


def write_prediction_row(
    writer: csv.DictWriter,
    file_handle,
    profile: ProfileConfig,
    prediction: SafePrediction,
) -> None:
    """Append one prediction row to the dry-run CSV log."""
    writer.writerow(
        {
            "timestamp": f"{time.time():.6f}",
            "profile": profile.profile_name,
            "predicted_action_id": prediction.raw.predicted_action.id,
            "predicted_action_name": prediction.raw.predicted_action.name,
            "confidence": f"{prediction.raw.confidence:.6f}",
            "effective_action_id": prediction.effective_action.id,
            "effective_action_name": prediction.effective_action.name,
            "top3_actions": "|".join(action.name for action, _ in prediction.raw.top3),
            "top3_confidences": "|".join(f"{confidence:.6f}" for _, confidence in prediction.raw.top3),
            "window_title": prediction.window_title,
            "window_allowed": str(prediction.window_allowed),
            "forced_no_op": str(prediction.forced_no_op),
            "reason": prediction.reason,
        }
    )
    file_handle.flush()


def prediction_log_path(profile_name: str, logs_dir: str | Path | None = None) -> Path:
    root = Path(logs_dir) if logs_dir is not None else PROJECT_ROOT / "outputs" / "logs"
    return root / f"{profile_name}_dry_run_predictions.csv"


def select_device(torch_module: Any, device_name: str | None):
    if device_name is not None:
        return torch_module.device(device_name)
    return torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run dry-run policy inference without desktop control.")
    parser.add_argument("--profile", required=True, help="Profile name from configs/profiles.")
    parser.add_argument("--device", default=None, help="Optional torch device, e.g. cpu or cuda.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        dry_run_agent(profile_name=args.profile, device_name=args.device)
    except (
        DryRunError,
        HotkeyError,
        ProfileValidationError,
        ScreenCaptureError,
        ValueError,
    ) as exc:
        parser.exit(status=2, message=f"Error: {exc}\n")
    return 0


def run_dry_run_agent() -> None:
    """Backward-compatible placeholder entrypoint."""
    dry_run_agent(profile_name="default")


if __name__ == "__main__":
    raise SystemExit(main())
