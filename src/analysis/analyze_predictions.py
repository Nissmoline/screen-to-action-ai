"""Analyze dry-run prediction logs before live-control."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from src.config import PROJECT_ROOT, ProfileValidationError, load_profile
from src.inference.dry_run_agent import PREDICTION_LOG_COLUMNS, prediction_log_path


NO_OP_WARNING_THRESHOLD = 0.95
SINGLE_ACTION_WARNING_THRESHOLD = 0.90
SWITCHES_PER_MINUTE_WARNING_THRESHOLD = 60.0


class PredictionAnalysisError(RuntimeError):
    """Raised when dry-run prediction analysis cannot be completed."""


@dataclass(frozen=True)
class PredictionAnalysis:
    """Computed dry-run prediction quality metrics."""

    profile_name: str
    total_predictions: int
    no_op_percentage: float
    average_confidence: float
    low_confidence_rate: float
    action_counts: Counter[str]
    action_switches_per_minute: float
    warnings: list[str]


def analyze_predictions(
    profile_name: str,
    logs_dir: str | Path | None = None,
    reports_dir: str | Path | None = None,
) -> Path:
    """Analyze a profile dry-run CSV log and save a text report."""
    profile = load_profile(profile_name)
    log_path = prediction_log_path(profile.profile_name, logs_dir=logs_dir)
    rows = read_prediction_log(log_path)
    analysis = compute_prediction_analysis(
        profile_name=profile.profile_name,
        rows=rows,
        confidence_threshold=profile.confidence_threshold,
    )

    report_path = prediction_analysis_report_path(profile.profile_name, reports_dir=reports_dir)
    report_text = format_prediction_analysis_report(
        analysis=analysis,
        log_path=log_path,
        confidence_threshold=profile.confidence_threshold,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text, encoding="utf-8")
    print(report_text)
    print(f"Prediction analysis saved to: {report_path}")
    return report_path


def read_prediction_log(path: str | Path) -> list[dict[str, str]]:
    """Read dry-run predictions CSV."""
    log_path = Path(path)
    if not log_path.is_file():
        raise PredictionAnalysisError(f"Dry-run prediction log does not exist: {log_path}")

    with log_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        missing = [column for column in PREDICTION_LOG_COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            raise PredictionAnalysisError(
                f"Prediction log is missing required columns: {', '.join(missing)}"
            )
        rows = list(reader)

    if not rows:
        raise PredictionAnalysisError(f"Prediction log has no rows: {log_path}")
    return rows


def compute_prediction_analysis(
    profile_name: str,
    rows: list[dict[str, str]],
    confidence_threshold: float,
) -> PredictionAnalysis:
    """Compute dry-run quality metrics and warning conditions."""
    total = len(rows)
    actions = [row.get("effective_action_name") or row.get("predicted_action_name", "") for row in rows]
    action_counts = Counter(actions)
    no_op_percentage = action_counts.get("no_op", 0) / total

    confidences = [_parse_float(row.get("confidence")) for row in rows]
    confidences = [value for value in confidences if value is not None]
    average_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    low_confidence_count = sum(1 for value in confidences if value < confidence_threshold)
    low_confidence_rate = low_confidence_count / total
    switches_per_minute = compute_action_switches_per_minute(rows, actions)
    warnings = build_warnings(
        total=total,
        action_counts=action_counts,
        no_op_percentage=no_op_percentage,
        average_confidence=average_confidence,
        confidence_threshold=confidence_threshold,
        action_switches_per_minute=switches_per_minute,
    )

    return PredictionAnalysis(
        profile_name=profile_name,
        total_predictions=total,
        no_op_percentage=no_op_percentage,
        average_confidence=average_confidence,
        low_confidence_rate=low_confidence_rate,
        action_counts=action_counts,
        action_switches_per_minute=switches_per_minute,
        warnings=warnings,
    )


def compute_action_switches_per_minute(rows: list[dict[str, str]], actions: list[str]) -> float:
    """Compute action switches per minute from timestamped prediction rows."""
    if len(actions) < 2:
        return 0.0

    switches = sum(1 for previous, current in zip(actions, actions[1:]) if previous != current)
    timestamps = [_parse_float(row.get("timestamp")) for row in rows]
    timestamps = [value for value in timestamps if value is not None]
    if len(timestamps) >= 2:
        duration_seconds = max(timestamps) - min(timestamps)
        if duration_seconds > 0:
            return switches / (duration_seconds / 60.0)

    return float(switches)


def build_warnings(
    total: int,
    action_counts: Counter[str],
    no_op_percentage: float,
    average_confidence: float,
    confidence_threshold: float,
    action_switches_per_minute: float,
) -> list[str]:
    """Return warning messages for unsafe model behavior."""
    warnings: list[str] = []
    if no_op_percentage > NO_OP_WARNING_THRESHOLD:
        warnings.append("no_op > 95%; model may be too passive for live-control.")

    most_common_action, most_common_count = action_counts.most_common(1)[0]
    if most_common_count / total > SINGLE_ACTION_WARNING_THRESHOLD:
        warnings.append(
            f"One action dominates > 90%: {most_common_action}; model may be collapsed."
        )

    if average_confidence < confidence_threshold:
        warnings.append(
            "Average confidence is below the profile threshold; live-control should stay disabled."
        )

    if action_switches_per_minute > SWITCHES_PER_MINUTE_WARNING_THRESHOLD:
        warnings.append(
            "Actions switch too frequently; predictions may be unstable or noisy."
        )

    return warnings


def format_prediction_analysis_report(
    analysis: PredictionAnalysis,
    log_path: Path,
    confidence_threshold: float,
) -> str:
    """Format a text analysis report."""
    lines = [
        f"Profile: {analysis.profile_name}",
        f"Dry-run log: {log_path}",
        f"Total predictions: {analysis.total_predictions}",
        f"no_op percentage: {analysis.no_op_percentage * 100:.2f}%",
        f"Average confidence: {analysis.average_confidence:.4f}",
        f"Confidence threshold: {confidence_threshold:.4f}",
        f"Low confidence rate: {analysis.low_confidence_rate * 100:.2f}%",
        f"Action switches per minute: {analysis.action_switches_per_minute:.2f}",
        "",
        "Most frequent actions:",
    ]
    for action_name, count in analysis.action_counts.most_common():
        percent = count / analysis.total_predictions * 100
        lines.append(f"  {action_name}: {count} ({percent:.2f}%)")

    lines.append("")
    if analysis.warnings:
        lines.append("Warnings:")
        for warning in analysis.warnings:
            lines.append(f"  WARNING: {warning}")
    else:
        lines.append("Warnings: none")

    return "\n".join(lines) + "\n"


def prediction_analysis_report_path(profile_name: str, reports_dir: str | Path | None = None) -> Path:
    root = Path(reports_dir) if reports_dir is not None else PROJECT_ROOT / "outputs" / "reports"
    return root / f"{profile_name}_prediction_analysis.txt"


def _parse_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze dry-run prediction logs.")
    parser.add_argument("--profile", required=True, help="Profile name from configs/profiles.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        analyze_predictions(profile_name=args.profile)
    except (PredictionAnalysisError, ProfileValidationError, ValueError) as exc:
        parser.exit(status=2, message=f"Error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
