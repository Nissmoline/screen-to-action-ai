"""Import video frames as unlabeled demonstration frames."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.config import PROJECT_ROOT, ProfileValidationError, load_profile
from src.recorder.record_demo import ACTION_CSV_COLUMNS, EpisodePaths, build_episode_paths


UNKNOWN_ACTION_ID = -1
UNKNOWN_ACTION_NAME = "unknown"


class VideoImportError(RuntimeError):
    """Raised when video import cannot be completed."""


@dataclass(frozen=True)
class VideoImportResult:
    """Summary of a video import episode."""

    episode: str
    episode_dir: Path
    frames_dir: Path
    actions_csv: Path
    metadata_json: Path
    total_frames: int


def import_video(
    profile_name: str,
    video_path: str | Path,
    every_n_frames: int,
    demos_root: str | Path | None = None,
    episode_name: str | None = None,
) -> VideoImportResult:
    """Extract every Nth frame from a video into a demo episode."""
    if every_n_frames <= 0:
        raise VideoImportError("every_n_frames must be greater than 0")

    profile = load_profile(profile_name)
    source_path = Path(video_path)
    if not source_path.is_file():
        raise VideoImportError(f"Video file does not exist: {source_path}")

    cv2 = import_cv2()
    episode = episode_name or f"video_import_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    paths = build_episode_paths(profile.profile_name, episode, data_root=demos_root)
    paths.frames_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise VideoImportError(f"Could not open video file: {source_path}")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    source_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    imported_rows: list[dict[str, object]] = []
    original_frame_index = 0
    imported_frame_id = 0

    try:
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break

            if original_frame_index % every_n_frames == 0:
                frame_path = paths.frames_dir / f"{imported_frame_id:06d}.png"
                save_video_frame(frame_bgr, frame_path, cv2)
                timestamp = video_frame_timestamp(capture, cv2, original_frame_index, source_fps)
                imported_rows.append(
                    {
                        "timestamp": f"{timestamp:.6f}",
                        "frame_id": imported_frame_id,
                        "action_name": UNKNOWN_ACTION_NAME,
                        "action_id": UNKNOWN_ACTION_ID,
                        "key_or_button": "",
                        "event_type": "video_import_unknown",
                        "mouse_x": "",
                        "mouse_y": "",
                    }
                )
                imported_frame_id += 1

            original_frame_index += 1
    finally:
        capture.release()

    if not imported_rows:
        raise VideoImportError(f"No frames were extracted from video: {source_path}")

    write_import_actions_csv(paths.actions_csv, imported_rows)
    metadata = build_video_import_metadata(
        profile=profile,
        source_path=source_path,
        source_fps=source_fps,
        source_frame_count=source_frame_count,
        every_n_frames=every_n_frames,
        total_frames=len(imported_rows),
    )
    write_json(paths.metadata_json, metadata)

    print(f"Imported {len(imported_rows)} frames to: {paths.episode_dir}")
    print(f"Labels are '{UNKNOWN_ACTION_NAME}' until manually labeled.")
    return VideoImportResult(
        episode=episode,
        episode_dir=paths.episode_dir,
        frames_dir=paths.frames_dir,
        actions_csv=paths.actions_csv,
        metadata_json=paths.metadata_json,
        total_frames=len(imported_rows),
    )


def import_cv2():
    """Import OpenCV with a clear installation error."""
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise VideoImportError(
            "opencv-python is required to import video frames. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc
    return cv2


def save_video_frame(frame_bgr: np.ndarray, path: str | Path, cv2_module: Any) -> None:
    """Save one BGR OpenCV frame as RGB PNG."""
    rgb = cv2_module.cvtColor(frame_bgr, cv2_module.COLOR_BGR2RGB)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(output_path, format="PNG")


def video_frame_timestamp(
    capture: Any,
    cv2_module: Any,
    original_frame_index: int,
    source_fps: float,
) -> float:
    """Return video timestamp in seconds for the current captured frame."""
    pos_msec = float(capture.get(cv2_module.CAP_PROP_POS_MSEC) or 0.0)
    if pos_msec > 0:
        return pos_msec / 1000.0
    if source_fps > 0:
        return original_frame_index / source_fps
    return float(original_frame_index)


def write_import_actions_csv(path: str | Path, rows: list[dict[str, object]]) -> None:
    """Write video-import labels in the recorder actions.csv schema."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=ACTION_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def build_video_import_metadata(
    profile,
    source_path: Path,
    source_fps: float,
    source_frame_count: int,
    every_n_frames: int,
    total_frames: int,
) -> dict[str, object]:
    """Build metadata for a video import episode."""
    now = datetime.now(timezone.utc).isoformat()
    effective_fps = source_fps / every_n_frames if source_fps > 0 else 0.0
    return {
        "profile_name": profile.profile_name,
        "source": "video_import",
        "source_video_path": str(source_path),
        "start_time": now,
        "end_time": now,
        "fps": max(1, int(round(effective_fps))) if effective_fps > 0 else profile.fps,
        "source_video_fps": source_fps,
        "effective_import_fps": effective_fps,
        "source_frame_count": source_frame_count,
        "every_n_frames": every_n_frames,
        "capture_region": asdict(profile.capture_region),
        "allowed_window_title": profile.allowed_window_title,
        "total_frames": total_frames,
        "total_actions": 0,
        "label_state": "unknown",
    }


def write_json(path: str | Path, data: dict[str, object]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)
        file.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import video frames as an unlabeled demo episode.")
    parser.add_argument("--profile", required=True, help="Profile name from configs/profiles.")
    parser.add_argument("--video-path", required=True, help="Path to a video file.")
    parser.add_argument("--every-n-frames", type=int, default=5, help="Extract every Nth frame.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        import_video(
            profile_name=args.profile,
            video_path=args.video_path,
            every_n_frames=args.every_n_frames,
        )
    except (ProfileValidationError, ValueError, VideoImportError) as exc:
        parser.exit(status=2, message=f"Error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
