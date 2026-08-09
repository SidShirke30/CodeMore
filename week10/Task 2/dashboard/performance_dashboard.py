from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from monitoring import calculate_metrics

root = Path(__file__).resolve().parents[1]
report = root / "reports" / "monitoring_report.md"
metrics = calculate_metrics()

accuracy = "N/A — no actual labels logged"
if metrics["accuracy"] is not None:
    accuracy = f"{metrics['accuracy']:.2%}"

latency = "N/A"
if metrics["average_latency_ms"] is not None:
    latency = f"{metrics['average_latency_ms']:.3f} ms"

text = "# Model Monitoring Report\n\n"
text += "## Current Metrics\n\n"
text += "| Metric | Value |\n|---|---:|\n"
text += f"| Prediction requests | {metrics['request_count']} |\n"
text += f"| Accuracy | {accuracy} |\n"
text += f"| Average inference latency | {latency} |\n\n"
text += "## Prediction Distribution\n\n"
text += "```text\n" + str(metrics["prediction_distribution"]) + "\n```\n\n"
text += "## Drift Detection\n\n"
text += "The monitoring module includes a PSI-style function for comparing a reference prediction distribution with the current production distribution. A drift signal should trigger investigation rather than automatic retraining.\n"

report.parent.mkdir(parents=True, exist_ok=True)
report.write_text(text, encoding="utf-8")
print(f"Report written to {report}")
