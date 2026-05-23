"""Project scaffold tests."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


EXPECTED_FILES = [
    "README.md",
    "requirements.txt",
    ".gitignore",
    "configs/profiles/default.yaml",
    "configs/profiles/lineage2_private.yaml",
    "data/videos/.gitkeep",
    "src/__init__.py",
    "src/main.py",
    "src/config.py",
    "src/analysis/__init__.py",
    "src/analysis/analyze_predictions.py",
    "src/analysis/plot_action_distribution.py",
    "src/safety/__init__.py",
    "src/safety/hotkeys.py",
    "src/safety/window_guard.py",
    "src/safety/rate_limiter.py",
    "src/capture/__init__.py",
    "src/capture/screen_capture.py",
    "src/input_control/__init__.py",
    "src/input_control/action_space.py",
    "src/input_control/keyboard_mouse_controller.py",
    "src/recorder/__init__.py",
    "src/recorder/record_demo.py",
    "src/dataset/__init__.py",
    "src/dataset/build_dataset.py",
    "src/dataset/manual_label_tool.py",
    "src/dataset/visualize_dataset.py",
    "src/video/__init__.py",
    "src/video/import_video.py",
    "src/models/__init__.py",
    "src/models/policy_cnn.py",
    "src/training/__init__.py",
    "src/training/train_imitation.py",
    "src/training/evaluate_imitation.py",
    "src/inference/__init__.py",
    "src/inference/dry_run_agent.py",
    "src/inference/run_agent.py",
    "src/utils/__init__.py",
    "src/utils/logger.py",
    "src/utils/paths.py",
    "tests/test_project_structure.py",
]


EXPECTED_DIRS = [
    "configs/profiles",
    "data/demos",
    "data/datasets",
    "data/videos",
    "models",
    "outputs/debug",
    "outputs/logs",
    "outputs/reports",
    "src/analysis",
    "src/safety",
    "src/capture",
    "src/input_control",
    "src/recorder",
    "src/dataset",
    "src/video",
    "src/models",
    "src/training",
    "src/inference",
    "src/utils",
    "tests",
]


def test_expected_project_files_exist() -> None:
    missing = [path for path in EXPECTED_FILES if not (PROJECT_ROOT / path).is_file()]
    assert missing == []


def test_expected_project_directories_exist() -> None:
    missing = [path for path in EXPECTED_DIRS if not (PROJECT_ROOT / path).is_dir()]
    assert missing == []


def test_main_readiness_message() -> None:
    main_file = PROJECT_ROOT / "src" / "main.py"
    assert 'print("Desktop AI Trainer is ready.")' in main_file.read_text(encoding="utf-8")


def test_readme_contains_safety_boundaries() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8").casefold()
    required_phrases = [
        "dry-run",
        "emergency stop hotkey",
        "pause hotkey",
        "allowlist window title",
        "action rate limit",
        "confidence threshold",
        "full action logging",
        "process memory reading",
        "packet sniffing",
        "code injection",
        "anti-cheat",
    ]
    missing = [phrase for phrase in required_phrases if phrase not in readme]
    assert missing == []
