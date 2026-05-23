"""Screen capture helpers for demonstration recording."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.config import CaptureRegion

if TYPE_CHECKING:
    import numpy as np


class ScreenCaptureError(RuntimeError):
    """Raised when screen capture dependencies or the desktop session are unavailable."""


@dataclass
class ScreenCapture:
    """Capture a configured screen region with mss."""

    capture_region: CaptureRegion
    fps: int

    @property
    def frame_interval_seconds(self) -> float:
        """Return the delay between frames for the configured FPS."""
        return 1.0 / self.fps

    @property
    def mss_region(self) -> dict[str, int]:
        """Return capture region in the format expected by mss."""
        return {
            "left": self.capture_region.left,
            "top": self.capture_region.top,
            "width": self.capture_region.width,
            "height": self.capture_region.height,
        }

    def grab_frame(self) -> "np.ndarray[Any, Any]":
        """Capture one frame and return it as an RGB numpy array."""
        try:
            import mss
            import numpy as np
        except ModuleNotFoundError as exc:
            raise ScreenCaptureError(
                "mss and numpy are required for screen capture. "
                "Install dependencies with: pip install -r requirements.txt"
            ) from exc

        try:
            with mss.mss() as screen:
                raw_frame = screen.grab(self.mss_region)
        except Exception as exc:
            raise ScreenCaptureError(f"Failed to capture screen region: {exc}") from exc

        bgra_frame = np.array(raw_frame)
        rgb_frame = bgra_frame[:, :, :3][:, :, ::-1]
        return np.ascontiguousarray(rgb_frame)

    def save_frame(self, frame: "np.ndarray[Any, Any]", path: str | Path) -> None:
        """Save a captured RGB frame as a PNG image."""
        try:
            from PIL import Image
        except ModuleNotFoundError as exc:
            raise ScreenCaptureError(
                "Pillow is required to save captured frames. "
                "Install dependencies with: pip install -r requirements.txt"
            ) from exc

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(frame).save(output_path, format="PNG")
