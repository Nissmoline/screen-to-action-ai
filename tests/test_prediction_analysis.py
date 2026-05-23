"""Dry-run prediction analysis tests."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

from src.analysis.analyze_predictions import (
    analyze_predictions,
    compute_prediction_analysis,
    read_prediction_log,
)
from src.analysis.plot_action_distribution import plot_action_distribution
from src.inference.dry_run_agent import PREDICTION_LOG_COLUMNS


def test_prediction_analysis_computes_metrics(tmp_path: Path) -> None:
    log_path = _write_prediction_log(tmp_path)
    rows = read_prediction_log(log_path)

    analysis = compute_prediction_analysis(
        profile_name="lineage2_private",
        rows=rows,
        confidence_threshold=0.90,
    )

    assert analysis.total_predictions == 10
    assert analysis.no_op_percentage == 0.6
    assert round(analysis.average_confidence, 3) == 0.800
    assert analysis.low_confidence_rate == 0.6
    assert analysis.action_counts["no_op"] == 6
    assert analysis.action_switches_per_minute > 0


def test_prediction_analysis_warnings_for_bad_model(tmp_path: Path) -> None:
    log_path = _write_prediction_log(tmp_path, actions=["no_op"] * 20, confidence=0.40)
    rows = read_prediction_log(log_path)

    analysis = compute_prediction_analysis(
        profile_name="lineage2_private",
        rows=rows,
        confidence_threshold=0.90,
    )

    assert any("no_op > 95%" in warning for warning in analysis.warnings)
    assert any("One action dominates" in warning for warning in analysis.warnings)
    assert any("Average confidence" in warning for warning in analysis.warnings)


def test_prediction_analysis_report_is_saved(tmp_path: Path) -> None:
    _write_prediction_log(tmp_path / "logs")

    report_path = analyze_predictions(
        profile_name="lineage2_private",
        logs_dir=tmp_path / "logs",
        reports_dir=tmp_path / "reports",
    )

    assert report_path.is_file()
    text = report_path.read_text(encoding="utf-8")
    assert "Average confidence" in text
    assert "Most frequent actions" in text


def test_action_distribution_plot_is_saved(tmp_path: Path) -> None:
    _write_prediction_log(tmp_path / "logs")

    plot_path = plot_action_distribution(
        profile_name="lineage2_private",
        logs_dir=tmp_path / "logs",
        reports_dir=tmp_path / "reports",
    )

    assert plot_path.is_file()


def test_analysis_cli_help_runs() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "src.analysis.analyze_predictions", "--help"],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--profile" in result.stdout


def test_plot_cli_help_runs() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "src.analysis.plot_action_distribution", "--help"],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--profile" in result.stdout


def _write_prediction_log(
    root: Path,
    actions: list[str] | None = None,
    confidence: float | None = None,
) -> Path:
    log_dir = root / "logs" if root.name != "logs" else root
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "lineage2_private_dry_run_predictions.csv"
    actions = actions or [
        "no_op",
        "no_op",
        "press_f1",
        "no_op",
        "press_f1",
        "turn_left",
        "no_op",
        "no_op",
        "turn_left",
        "no_op",
    ]

    with log_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=PREDICTION_LOG_COLUMNS)
        writer.writeheader()
        for index, action_name in enumerate(actions):
            row_confidence = confidence if confidence is not None else (0.95 if action_name != "no_op" else 0.70)
            writer.writerow(
                {
                    "timestamp": f"{index * 1.0:.6f}",
                    "profile": "lineage2_private",
                    "predicted_action_id": "0" if action_name == "no_op" else "1",
                    "predicted_action_name": action_name,
                    "confidence": f"{row_confidence:.6f}",
                    "effective_action_id": "0" if action_name == "no_op" else "1",
                    "effective_action_name": action_name,
                    "top3_actions": f"{action_name}|no_op|press_f2",
                    "top3_confidences": f"{row_confidence:.6f}|0.100000|0.050000",
                    "window_title": "Lineage II Private Sandbox",
                    "window_allowed": "True",
                    "forced_no_op": "False",
                    "reason": "ok",
                }
            )
    return log_path
