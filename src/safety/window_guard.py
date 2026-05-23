"""Window title allowlist checks."""

from __future__ import annotations


class WindowGuardError(RuntimeError):
    """Raised when active-window validation fails."""


def get_active_window_title() -> str:
    """Return the active desktop window title using pyautogui."""
    try:
        import pyautogui
    except ModuleNotFoundError as exc:
        raise WindowGuardError(
            "pyautogui is required to check the active window. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    try:
        active_window = pyautogui.getActiveWindow()
    except Exception as exc:
        raise WindowGuardError(f"Could not read active window title: {exc}") from exc

    if active_window is None or not getattr(active_window, "title", ""):
        raise WindowGuardError("Could not determine the active window title")

    return str(active_window.title)


def is_window_title_allowed(title: str, allowlist: str | list[str]) -> bool:
    """Return True when the active window title matches the configured allowlist."""
    allowed_titles = [allowlist] if isinstance(allowlist, str) else allowlist
    normalized_title = title.casefold()
    return any(allowed.casefold() in normalized_title for allowed in allowed_titles if allowed)


def require_active_window(allowed_window_title: str) -> str:
    """Return the active title or raise when it does not match the allowed title."""
    active_title = get_active_window_title()
    if not is_window_title_allowed(active_title, allowed_window_title):
        raise WindowGuardError(
            "Active window is not allowed for this profile. "
            f"Expected title containing {allowed_window_title!r}, got {active_title!r}."
        )
    return active_title
